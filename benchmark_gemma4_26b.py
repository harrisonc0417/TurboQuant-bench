#!/usr/bin/env python3
"""
benchmark_gemma4_26b.py — Gemma 4 26B-A4B-IT: baseline vs TurboQuant KV-cache compression

Metrics collected at each context length:
  • Prefill time        — seconds to process the full prompt (chunked to cap peak RAM)
  • Time-to-first-token — same as prefill for the first run
  • Generation speed    — tokens/second for NEW_TOKENS of autoregressive decoding
  • Peak memory         — torch.cuda.max_memory_allocated() peak over the trial
  • KV cache bytes      — actual bytes used by each cache implementation
  • Compression ratio   — baseline BF16 / TurboQuant uint8 (and theoretical 3-bit)

Chunked prefill (PREFILL_CHUNK_SIZE) is used to avoid materialising activations for
the full sequence at once, which OOM-kills the system at long contexts even though
the final KV cache fits comfortably in memory.
"""

import gc
import sys
import time
import json
import math
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from turboquant_cache import TurboQuantCache
from context_generator import generate_context

# ── configuration ─────────────────────────────────────────────────────────────
MODEL_PATH         = "/home/harrison/Projects/LocalAI/models/gemma-4-26b-a4b-it"
CONTEXT_LENGTHS    = [8_192, 32_768, 65_536, 131_072]   # tokens
NEW_TOKENS         = 50     # tokens to generate per trial (latency + throughput)
TURBOQUANT_BITS    = 3      # quantization bits
RESIDUAL_LEN       = 128    # full-precision tail buffer for TurboQuant
WARMUP_TOKENS      = 128    # small warmup run to prime CUDA kernels (done once)
PREFILL_CHUNK_SIZE = 2_048  # tokens processed per forward pass during prefill
                            # keeps peak activation RAM low regardless of total seq_len

# ── helpers ───────────────────────────────────────────────────────────────────

def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.3f} GB"

def _reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3

def _current_gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3


def warmup(model, processor):
    """Short forward pass to initialise CUDA kernels before timing."""
    print("  [warmup] priming CUDA kernels …", end=" ", flush=True)
    toks = processor(text="Hello world!", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model(**toks, use_cache=False)
    _sync()
    print("done")


def _count_dynamic_cache_bytes(cache) -> int:
    """
    Robustly count bytes used by a DynamicCache regardless of internal layout.
    Handles both the legacy key_cache/value_cache list API and the newer
    layer-object API introduced in recent transformers versions.
    """
    kv_bytes = 0
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if hasattr(layer, "keys") and layer.keys is not None and layer.keys.numel() > 0:
                kv_bytes += layer.keys.numel() * 2    # bfloat16 = 2 bytes
                kv_bytes += layer.values.numel() * 2
    elif hasattr(cache, "key_cache"):
        for k, v in zip(cache.key_cache, cache.value_cache):
            if k is not None and k.numel() > 0:
                kv_bytes += k.numel() * 2
                kv_bytes += v.numel() * 2
    return kv_bytes


# ── single trial ─────────────────────────────────────────────────────────────

def run_trial(
    model,
    processor,
    context_ids: torch.Tensor,   # [1, S] on CPU
    use_turboquant: bool,
    label: str,
) -> dict:
    """
    Run one prefill + generation trial. Returns a stats dict.
    context_ids is moved to model device inside.
    """
    _reset_memory()

    device = next(model.parameters()).device
    input_ids = context_ids.to(device)
    seq_len = input_ids.shape[1]

    # ── create cache ──────────────────────────────────────────────────────────
    if use_turboquant:
        past_key_values = TurboQuantCache(
            config=model.config,
            bits=TURBOQUANT_BITS,
            residual_length=RESIDUAL_LEN,
        )
    else:
        past_key_values = DynamicCache()

    # ── chunked prefill ───────────────────────────────────────────────────────
    # Process the context in PREFILL_CHUNK_SIZE slices so that intermediate
    # activations (proportional to chunk_len × hidden_dim × n_layers) never
    # exceed ~chunk_size's worth of memory, regardless of total sequence length.
    _sync()
    t_prefill_start = time.perf_counter()

    next_token = None
    n_chunks = math.ceil(seq_len / PREFILL_CHUNK_SIZE)
    for i in range(n_chunks):
        chunk_start = i * PREFILL_CHUNK_SIZE
        chunk_end   = min(chunk_start + PREFILL_CHUNK_SIZE, seq_len)
        chunk_ids   = input_ids[:, chunk_start:chunk_end]

        with torch.inference_mode():
            out = model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        del out
        # Release chunk activations immediately so they don't accumulate
        torch.cuda.empty_cache()

    _sync()
    prefill_time = time.perf_counter() - t_prefill_start

    # ── generation loop ───────────────────────────────────────────────────────
    _sync()
    t_gen_start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(NEW_TOKENS):
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            past_key_values = out.past_key_values
            del out

    _sync()
    gen_time = time.perf_counter() - t_gen_start

    peak_mem_gb = _peak_gb()

    # ── KV cache accounting ───────────────────────────────────────────────────
    if use_turboquant:
        stats = past_key_values.compression_stats()
        kv_actual_bytes      = stats["actual_bytes"]
        kv_baseline_bytes    = stats["baseline_bytes"]
        kv_theoretical_bytes = stats["theoretical_bytes"]
        uint8_ratio          = stats["uint8_ratio"]
        theoretical_ratio    = stats["theoretical_ratio"]
    else:
        kv_bytes             = _count_dynamic_cache_bytes(past_key_values)
        kv_actual_bytes      = kv_bytes
        kv_baseline_bytes    = kv_bytes
        kv_theoretical_bytes = kv_bytes
        uint8_ratio          = 1.0
        theoretical_ratio    = 1.0

    # ── cleanup ───────────────────────────────────────────────────────────────
    del past_key_values, input_ids, next_token
    _reset_memory()

    return {
        "label":                  label,
        "seq_len":                seq_len,
        "prefill_s":              prefill_time,
        "gen_s":                  gen_time,
        "throughput_tok_s":       NEW_TOKENS / gen_time,
        "peak_mem_gb":            peak_mem_gb,
        "kv_actual_gb":           kv_actual_bytes      / 1024**3,
        "kv_baseline_gb":         kv_baseline_bytes    / 1024**3,
        "kv_theoretical_gb":      kv_theoretical_bytes / 1024**3,
        "uint8_compression_ratio":       uint8_ratio,
        "theoretical_compression_ratio": theoretical_ratio,
    }


# ── results reporting ─────────────────────────────────────────────────────────

HEADER = """
╔══════════════════════════════════════════════════════════════════════════════════════╗
║         Gemma 4 26B-A4B-IT  ·  KV-Cache Benchmark  ·  TurboQuant vs Baseline        ║
║  GPU: NVIDIA GB10 (119 GB unified)  ·  Quantization: {bits}-bit  ·  Residual: {res} tok  ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""".format(bits=TURBOQUANT_BITS, res=RESIDUAL_LEN)

ROW = (
    "  {label:<22}  prefill {prefill_s:>6.2f}s  "
    "gen {throughput_tok_s:>6.1f} tok/s  "
    "peak {peak_mem_gb:>6.2f} GB  "
    "KV {kv_actual_gb:>6.3f} GB"
)


def print_comparison(baseline: dict, turboquant: dict):
    ctx = baseline["seq_len"]
    print(f"\n  Context: {ctx:,} tokens")
    print(f"  {'─'*80}")
    print(ROW.format(**baseline))
    print(ROW.format(**turboquant))

    kv_saved_gb = baseline["kv_actual_gb"] - turboquant["kv_actual_gb"]
    speed_delta  = turboquant["throughput_tok_s"] - baseline["throughput_tok_s"]
    speed_pct    = speed_delta / baseline["throughput_tok_s"] * 100

    print(
        f"\n  → uint8 KV compression:      {turboquant['uint8_compression_ratio']:.2f}×  "
        f"(saved {kv_saved_gb:.3f} GB)"
    )
    print(
        f"  → theoretical 3-bit ratio:   {turboquant['theoretical_compression_ratio']:.2f}×  "
        f"(if bit-packed: {turboquant['kv_theoretical_gb']:.3f} GB)"
    )
    print(
        f"  → throughput delta:          {speed_delta:+.1f} tok/s  ({speed_pct:+.1f}%)"
    )
    print(
        f"  → peak memory saved:         "
        f"{baseline['peak_mem_gb'] - turboquant['peak_mem_gb']:+.2f} GB"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(HEADER)

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    print("Loading model … (this reads ~52 GB of weights)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",   # memory-efficient attention; no flash-attn install needed
    )
    model.eval()
    print(f"Model on: {next(model.parameters()).device}\n")

    warmup(model, processor)

    all_results: list[dict] = []

    # ── iterate over context lengths ──────────────────────────────────────────
    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'═'*84}")
        print(f"  Generating {ctx_len:,}-token context …")

        ctx = generate_context(tokenizer, target_tokens=ctx_len)
        actual_len = ctx["actual_tokens"]
        print(
            f"  Tokenized: {actual_len:,} tokens  |  "
            f'Preview: "{ctx["text_preview"][:80].strip()} …"'
        )

        context_ids = ctx["input_ids"]          # [1, actual_len], on CPU

        # baseline ─────────────────────────────────────────────────────────────
        print(f"\n  Running baseline (DynamicCache BF16) …")
        try:
            base = run_trial(
                model, processor, context_ids,
                use_turboquant=False,
                label="Baseline BF16",
            )
            print(f"    prefill {base['prefill_s']:.2f}s  |  "
                  f"{base['throughput_tok_s']:.1f} tok/s  |  "
                  f"peak {base['peak_mem_gb']:.2f} GB  |  "
                  f"KV {base['kv_actual_gb']:.3f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"    ✗ OOM at {ctx_len:,} tokens — skipping baseline")
            base = None

        # TurboQuant ───────────────────────────────────────────────────────────
        print(f"  Running TurboQuant ({TURBOQUANT_BITS}-bit, residual={RESIDUAL_LEN}) …")
        try:
            tq = run_trial(
                model, processor, context_ids,
                use_turboquant=True,
                label=f"TurboQuant {TURBOQUANT_BITS}-bit",
            )
            print(f"    prefill {tq['prefill_s']:.2f}s  |  "
                  f"{tq['throughput_tok_s']:.1f} tok/s  |  "
                  f"peak {tq['peak_mem_gb']:.2f} GB  |  "
                  f"KV {tq['kv_actual_gb']:.3f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"    ✗ OOM at {ctx_len:,} tokens — skipping TurboQuant")
            tq = None

        if base is not None and tq is not None:
            print_comparison(base, tq)
            all_results.append({"context_len": actual_len, "baseline": base, "turboquant": tq})

    # ── summary table ─────────────────────────────────────────────────────────
    if all_results:
        print(f"\n\n{'═'*84}")
        print("  SUMMARY — KV compression ratio by context length")
        print(f"  {'─'*80}")
        print(f"  {'Context':>12}  {'uint8 ratio':>12}  {'3-bit ratio':>12}  "
              f"{'KV saved':>10}  {'Speed Δ':>10}")
        print(f"  {'─'*80}")
        for r in all_results:
            b, t = r["baseline"], r["turboquant"]
            ctx  = r["context_len"]
            saved = b["kv_actual_gb"] - t["kv_actual_gb"]
            spd   = (t["throughput_tok_s"] - b["throughput_tok_s"]) / b["throughput_tok_s"] * 100
            print(
                f"  {ctx:>12,}  "
                f"{t['uint8_compression_ratio']:>11.2f}×  "
                f"{t['theoretical_compression_ratio']:>11.2f}×  "
                f"{saved:>9.3f} GB  "
                f"{spd:>+9.1f}%"
            )

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "benchmark_gemma4_26b_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
