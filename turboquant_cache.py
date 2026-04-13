"""
TurboQuant KV cache for HuggingFace Transformers.

Algorithm (arXiv:2504.19874):
  1. Random orthogonal rotation of K/V vectors (energy-spreading pre-processing)
  2. Per-token scalar quantization to `bits` bits (min-max approx of Lloyd-Max)
  3. Full-precision residual buffer for the most-recent `residual_length` tokens
  4. Dequantization + inverse rotation on retrieval

Applied only to full-attention layers in Gemma 4 (layers 5,11,17,23,29).
Sliding-attention layers stay in standard BF16 (they're bounded at 1024 tokens).
"""

import math
import torch
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    DynamicLayer,
    DynamicSlidingWindowLayer,
)
from transformers.configuration_utils import PreTrainedConfig


# ── Rotation helpers ─────────────────────────────────────────────────────────

_rotation_cache: dict[tuple[int, str], torch.Tensor] = {}

def _get_rotation_matrix(head_dim: int, device: torch.device) -> torch.Tensor:
    """
    Return a fixed random orthogonal matrix R of shape [head_dim, head_dim].
    Computed via QR decomposition with a fixed seed; cached after first call.
    """
    key = (head_dim, str(device))
    if key not in _rotation_cache:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20250405)          # deterministic across runs
        R_raw = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
        R, _ = torch.linalg.qr(R_raw)     # orthogonal
        _rotation_cache[key] = R.to(device)
    return _rotation_cache[key]


# ── Per-layer quantized cache ─────────────────────────────────────────────────

class TurboQuantLayer(DynamicLayer):
    """
    One layer of TurboQuant-compressed KV cache.

    Storage layout:
      _q_keys / _q_values : uint8  [B, H, S_quantized, D]  (3 bits of info per byte)
      _k_min  / _v_min    : fp16   [B, H, S_quantized]     per-token range minimum
      _k_scale/ _v_scale  : fp16   [B, H, S_quantized]     per-token range scale
      keys    / values    : bfloat16 [B, H, S_residual, D] recent tokens in full precision
    """

    is_sliding = False

    def __init__(self, bits: int = 3, residual_length: int = 128):
        super().__init__()
        self.bits = bits
        self.levels = 2 ** bits           # 8 for 3-bit
        self.residual_length = residual_length

        # Quantized history
        self._q_keys: torch.Tensor | None = None
        self._q_values: torch.Tensor | None = None
        self._k_min: torch.Tensor | None = None
        self._k_scale: torch.Tensor | None = None
        self._v_min: torch.Tensor | None = None
        self._v_scale: torch.Tensor | None = None

        # Cumulative sequence length (for mask and stats)
        self.cumulative_length: int = 0

    # ── quantization helpers ──────────────────────────────────────────────────

    def _quantize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-token min-max scalar quantization.
        x : [B, H, S, D] float32
        returns (indices uint8 [B,H,S,D], min fp16 [B,H,S], scale fp16 [B,H,S])
        """
        x_min = x.amin(dim=-1)                              # [B, H, S]
        x_max = x.amax(dim=-1)
        scale = (x_max - x_min).clamp(min=1e-6) / (self.levels - 1)
        indices = ((x - x_min.unsqueeze(-1)) / scale.unsqueeze(-1)).round_()
        indices.clamp_(0, self.levels - 1)
        return indices.to(torch.uint8), x_min.half(), scale.half()

    def _dequantize(
        self,
        indices: torch.Tensor,
        x_min: torch.Tensor,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct float32 from quantized storage.
        indices : [B,H,S,D] uint8
        x_min, x_scale : [B,H,S] fp16
        """
        return (
            indices.float() * x_scale.float().unsqueeze(-1)
            + x_min.float().unsqueeze(-1)
        )

    # ── flush residual buffer to quantized storage ────────────────────────────

    def _flush_to_quantized(self, n_flush: int) -> None:
        """
        Move the first `n_flush` tokens from the full-precision residual buffer
        into the quantized history store.
        """
        to_flush_k = self.keys[..., :n_flush, :].contiguous()
        to_flush_v = self.values[..., :n_flush, :].contiguous()
        head_dim = to_flush_k.shape[-1]

        R = _get_rotation_matrix(head_dim, to_flush_k.device)

        k_rot = to_flush_k.float() @ R
        v_rot = to_flush_v.float() @ R

        k_idx, k_min, k_scale = self._quantize(k_rot)
        v_idx, v_min, v_scale = self._quantize(v_rot)

        if self._q_keys is None:
            self._q_keys  = k_idx
            self._q_values = v_idx
            self._k_min   = k_min;  self._k_scale = k_scale
            self._v_min   = v_min;  self._v_scale = v_scale
        else:
            self._q_keys  = torch.cat([self._q_keys,   k_idx],  dim=2)
            self._q_values = torch.cat([self._q_values, v_idx],  dim=2)
            self._k_min   = torch.cat([self._k_min,   k_min],  dim=2)
            self._k_scale = torch.cat([self._k_scale, k_scale], dim=2)
            self._v_min   = torch.cat([self._v_min,   v_min],  dim=2)
            self._v_scale = torch.cat([self._v_scale, v_scale], dim=2)

        # Trim residual
        self.keys   = self.keys[...,   n_flush:, :]
        self.values = self.values[..., n_flush:, :]

    # ── CacheLayerMixin interface ─────────────────────────────────────────────

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        self.dtype  = key_states.dtype
        self.device = key_states.device
        self.keys   = torch.empty(0, dtype=self.dtype, device=self.device)
        self.values = torch.empty(0, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        new_len = key_states.shape[-2]
        self.cumulative_length += new_len

        # Grow residual buffer
        self.keys   = torch.cat([self.keys,   key_states],   dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        # If residual exceeds 2x capacity, flush the excess to quantized storage
        cur_res = self.keys.shape[-2]
        if cur_res > 2 * self.residual_length:
            self._flush_to_quantized(cur_res - self.residual_length)

        # Reconstruct full sequence for attention: dequantized history + residual
        if self._q_keys is not None:
            head_dim = self.keys.shape[-1]
            R = _get_rotation_matrix(head_dim, self.keys.device)
            k_deq = (self._dequantize(self._q_keys,   self._k_min, self._k_scale) @ R.T).to(self.dtype)
            v_deq = (self._dequantize(self._q_values, self._v_min, self._v_scale) @ R.T).to(self.dtype)
            k_full = torch.cat([k_deq, self.keys],   dim=-2)
            v_full = torch.cat([v_deq, self.values], dim=-2)
        else:
            k_full = self.keys
            v_full = self.values

        return k_full, v_full

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1   # dynamic, no maximum

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cumulative_length + query_length, 0

    def reset(self) -> None:
        super().reset()
        self._q_keys = self._q_values = None
        self._k_min  = self._k_scale = None
        self._v_min  = self._v_scale = None
        self.cumulative_length = 0

    # ── memory accounting ─────────────────────────────────────────────────────

    def memory_bytes(self) -> int:
        """Actual bytes consumed by this layer's TurboQuant storage."""
        total = 0
        for t in (self._q_keys, self._q_values):
            if t is not None:
                total += t.numel()          # uint8 → 1 byte/element
        for t in (self._k_min, self._k_scale, self._v_min, self._v_scale):
            if t is not None:
                total += t.numel() * 2      # fp16  → 2 bytes/element
        if self.is_initialized and self.keys.numel() > 0:
            total += (self.keys.numel() + self.values.numel()) * 2   # bf16
        return total

    def baseline_memory_bytes(self) -> int:
        """Bytes this layer would use in standard BF16 (no quantization)."""
        if self._q_keys is not None:
            B, H, S_q, D = self._q_keys.shape
            S_res = self.keys.shape[-2] if self.keys.numel() > 0 else 0
            return (S_q + S_res) * H * D * 2 * 2   # K+V, 2 bytes BF16
        if self.keys.numel() > 0:
            B, H, S, D = self.keys.shape
            return S * H * D * 2 * 2
        return 0

    def theoretical_3bit_bytes(self) -> int:
        """
        True 3-bit packed memory (not what we store, but what the paper achieves
        with bit-packing: 8 values into 3 bytes).
        """
        seq = self.cumulative_length
        if self._q_keys is not None:
            H, D = self._q_keys.shape[1], self._q_keys.shape[3]
        elif self.keys.numel() > 0:
            H, D = self.keys.shape[1], self.keys.shape[3]
        else:
            return 0
        # K+V packed: ceil(seq * H * D * 3 / 8) bytes each
        packed = math.ceil(seq * H * D * 3 / 8) * 2    # K + V
        # Scales (min+scale per head per token, fp16): 4 tensors, fp16
        scales = seq * H * 2 * 2 * 2                   # 4 scalars per token per head
        return packed + scales


# ── Top-level cache container ─────────────────────────────────────────────────

class TurboQuantCache(Cache):
    """
    Drop-in replacement for DynamicCache that uses TurboQuantLayer for
    full-attention layers and the standard DynamicSlidingWindowLayer for
    sliding-attention layers.

    Usage::

        cache = TurboQuantCache(config=model.config, bits=3)
        outputs = model(**inputs, past_key_values=cache, use_cache=True)
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        bits: int = 3,
        residual_length: int = 128,
    ):
        decoder_cfg = config.get_text_config(decoder=True)
        sliding_window = getattr(decoder_cfg, "sliding_window", None) or getattr(
            decoder_cfg, "attention_chunk_size", None
        )
        layer_types = list(getattr(decoder_cfg, "layer_types", []))

        # Trim shared layers (e.g. Gemma 3n)
        num_shared = getattr(decoder_cfg, "num_kv_shared_layers", 0)
        if num_shared:
            layer_types = layer_types[:-num_shared]

        layers: list[CacheLayerMixin] = []
        for lt in layer_types:
            if lt in ("sliding_attention", "chunked_attention"):
                layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
            elif lt == "full_attention":
                layers.append(TurboQuantLayer(bits=bits, residual_length=residual_length))
            else:
                layers.append(DynamicLayer())

        # Fall back to lazy init if config gave us nothing
        if not layers:
            super().__init__(layer_class_to_replicate=TurboQuantLayer)
        else:
            super().__init__(layers=layers)

    # ── stats helpers ─────────────────────────────────────────────────────────

    def compression_stats(self) -> dict:
        """
        Return a dict with byte counts across all TurboQuantLayer layers.
        Useful for reporting compression ratio in benchmarks.
        """
        actual_bytes   = 0
        baseline_bytes = 0
        theoretical_bytes = 0
        n_quantized    = 0

        for layer in self.layers:
            if isinstance(layer, TurboQuantLayer):
                actual_bytes      += layer.memory_bytes()
                baseline_bytes    += layer.baseline_memory_bytes()
                theoretical_bytes += layer.theoretical_3bit_bytes()
                n_quantized       += 1

        ratio = baseline_bytes / actual_bytes if actual_bytes > 0 else 0.0
        theoretical_ratio = baseline_bytes / theoretical_bytes if theoretical_bytes > 0 else 0.0

        return {
            "actual_bytes":        actual_bytes,
            "baseline_bytes":      baseline_bytes,
            "theoretical_bytes":   theoretical_bytes,
            "uint8_ratio":         ratio,
            "theoretical_ratio":   theoretical_ratio,
            "n_quantized_layers":  n_quantized,
        }
