# Gemma 4 26B — TurboQuant KV Cache Benchmark

A local benchmark comparing standard BF16 KV-cache inference against a
TurboQuant-compressed KV cache on Gemma 4 26B-A4B-IT, using an NVIDIA GB10
(119 GB unified memory). Results are collected at 8K, 32K, 64K, and 128K
token context lengths.

---

## Table of Contents

1. [Background: The KV Cache Memory Problem](#1-background-the-kv-cache-memory-problem)
2. [TurboQuant: The Algorithm Explained](#2-turboquant-the-algorithm-explained)
3. [Gemma 4 Architecture Notes](#3-gemma-4-architecture-notes)
4. [File Reference](#4-file-reference)
   - [turboquant_cache.py](#turboquant_cachepy)
   - [context_generator.py](#context_generatorpy)
   - [benchmark.py](#benchmarkpy)
5. [Running the Benchmark](#5-running-the-benchmark)
6. [Reading the Output](#6-reading-the-output)
7. [Configuration Knobs](#7-configuration-knobs)

---

## 1. Background: The KV Cache Memory Problem

### What attention actually does

Every transformer layer runs scaled dot-product attention:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

- **Q** (queries) — "what am I looking for?"
- **K** (keys)   — "what does each past token offer?"
- **V** (values)  — "what information does each past token carry?"

For each new token the model generates, it computes a query vector and uses it
to attend over *every* key and value vector from *all* preceding tokens. That
attending operation is what lets the model refer back to anything in the context
window — the model literally reads all of the past to produce each new word.

### Why the cache exists

Without a cache, generating token N would require re-running the full forward
pass over tokens 1…N-1 just to re-compute their key and value vectors. That
would make generation O(N²) in compute. Instead, after each token is processed,
its key and value vectors are saved. On the next token, the model just appends
the new token's K and V and runs attention over the saved + new vectors. This is
the **KV cache**: a growing buffer of (K, V) pairs, one entry per token per
attention layer.

### The memory cost

Memory usage scales as:

```
KV cache bytes = sequence_length × num_layers × num_KV_heads × head_dim × 2 (K+V) × bytes_per_value
```

For Gemma 4 26B's five **full-attention** layers (the ones whose cache grows
without bound):

| Metric             | Value            |
|--------------------|------------------|
| Full-attention layers | 5             |
| KV heads per layer | 2                |
| Head dimension     | 512              |
| Precision          | BF16 (2 bytes)   |
| **Per-token cost** | **20 KB**        |

At **128 K tokens** that is 128,000 × 20 KB ≈ **2.5 GB** just for the KV
cache of the full-attention layers — on top of the ~52 GB already occupied by
the model weights. That 2.5 GB also has to be *read from memory on every
generated token*, creating a bandwidth bottleneck that limits generation speed
at long contexts.

---

## 2. TurboQuant: The Algorithm Explained

TurboQuant (arXiv:2504.19874, ICLR 2026) compresses KV vectors to approximately
3 bits per value while keeping attention quality nearly identical to full
precision. There are three key ideas.

### 2.1 Why naive quantization fails

**Quantization** means replacing a high-precision number (e.g. a 16-bit float)
with a low-precision approximation using fewer bits (e.g. a 3-bit integer out of
8 possible levels: 0–7).

The problem with doing this naively to KV vectors is that attention quality
depends on **dot products** (Q · K), not on individual values. Even if each
value is only slightly wrong after quantization, those errors can compound
badly in the dot product — especially when some dimensions of K carry most of
the "energy" (i.e. have much larger magnitudes than the others). Quantizing a
dimension that spans the range [−10, 10] with 8 levels gives granularity of
2.5 per step; quantizing a dimension that spans [−0.01, 0.01] gives granularity
of 0.003. If both use the same quantization scheme, the big dimension dominates
the error and distorts the dot product.

### 2.2 Step 1 — Random orthogonal rotation

Before quantizing, TurboQuant multiplies every key vector by a fixed random
**orthogonal matrix R**:

```
k_rotated = k · R
```

An orthogonal matrix is a square matrix where every row is a unit vector and
all rows are perpendicular to each other — think of it as a rigid rotation in
high-dimensional space, like rotating a 3D object. Crucially:

- It **preserves dot products**: (k₁ · R) · (k₂ · R)ᵀ = k₁ · k₂. The
  rotation is invisible to attention — the final scores are unchanged.
- It **spreads energy evenly** across all dimensions. If K had one big dimension
  and many tiny ones before, after rotation every dimension tends to carry a
  roughly equal share. This is called *isotropic* distribution.

When energy is spread evenly, per-dimension min-max quantization wastes much
less precision on dimensions that don't matter, and the quantization error per
dot product becomes predictable and small.

In code (`turboquant_cache.py`):

```python
R_raw = torch.randn(head_dim, head_dim, generator=gen)
R, _ = torch.linalg.qr(R_raw)   # QR decomposition produces an orthogonal R
k_rotated = key_states.float() @ R
```

The same fixed R is reused for every token in every layer that shares the same
`head_dim`, so the inverse rotation `R.T` (= R⁻¹ for orthogonal matrices)
reverses it exactly on retrieval.

### 2.3 Step 2 — Per-token min-max quantization

After rotation, each token's key vector is a 1D array of `head_dim` values.
TurboQuant quantizes each token independently using its own min and max:

```
scale   = (max(v) − min(v)) / (2^bits − 1)
index_i = round( (v_i − min(v)) / scale )    ← integer in [0, 2^bits − 1]
```

For 3 bits, `2^3 = 8` levels (indices 0–7). Each float32 value is replaced by
a `uint8` integer (which uses one byte of storage, so the 3 bits of information
sit inside a byte that also has 5 unused bits).

To reconstruct the original value later:

```
v_i ≈ index_i × scale + min(v)
```

Only two small numbers — `min` and `scale` — need to be stored alongside the
integer indices for each token vector. Because they are *per-token* (not
per-value), their overhead is negligible for large `head_dim`.

**Memory comparison for one full-attention layer at 64K tokens:**

| Storage            | Formula                                 | Size     |
|--------------------|-----------------------------------------|----------|
| BF16 (baseline)    | 64K × 2 heads × 512 dim × 2 (K+V) × 2B | 256 MB   |
| uint8 indices      | 64K × 2 × 512 × 2 × 1B                 | 128 MB   |
| min + scale (fp16) | 64K × 2 × 2 (K+V) × 2 × 2B             | 1 MB     |
| **Total TQ**       |                                         | **129 MB** |
| **Compression**    |                                         | **~2×** |

> The reason it's only 2× (not the paper's claimed 5×) is that we store one
> uint8 **byte** per value even though we only need 3 **bits**. Packing 8
> values into 3 bytes instead of 8 bytes would give the full **5.2×**
> compression. The benchmark reports both the actual uint8 ratio and the
> theoretical bit-packed ratio side by side.

### 2.4 Step 3 — Full-precision residual buffer

Quantization introduces small errors. Those errors are tolerable for tokens
the model saw hundreds of steps ago, but can noticeably degrade quality for
the most recent few hundred tokens — tokens the model is still "thinking
about" and whose keys and values dominate the current attention scores.

TurboQuant keeps the last `residual_length` tokens (default: 128) in full
BF16, and only moves tokens into quantized storage once they fall off the
back of this window. In code, the residual buffer is flushed to the quantized
store whenever it exceeds twice the configured capacity:

```
if residual_buffer_length > 2 × residual_length:
    flush (residual_length tokens) → quantized store
    keep last (residual_length tokens) in full precision
```

This way the attention computation always sees:
```
[dequantized old history] + [full-precision recent tokens]
```

### 2.5 Why dot products are preserved

Combining all three steps: rotate → quantize → store → dequantize → unrotate.
The rotation is exact (orthogonal, no information loss). The quantization
introduces a small per-token error ε. After unrotating, this error is spread
back across all dimensions of the key vector. The dot product Q · K̂ differs
from Q · K only by Q · (R · ε), which is small because ε is small (3-bit
quantization after rotation is near-optimal) and the rotation spreads it
uniformly, preventing any single dimension from concentrating the error.

The paper demonstrates ~0.985 cosine similarity between quantized and
full-precision attention scores — essentially indistinguishable in practice.

---

## 3. Gemma 4 Architecture Notes

Understanding why TurboQuant is applied selectively requires a brief look at
how Gemma 4 is structured.

### Mixture-of-Experts (MoE)

Gemma 4 26B uses a **Mixture-of-Experts** feedforward layer. The model has 26B
total parameters but only activates ~4B per token — eight of 128 expert FFN
sub-networks are selected per token by a learned router. This means inference
memory bandwidth is much lower than the parameter count suggests, which is why
the model fits comfortably in 119 GB of unified memory.

### Sliding attention vs full attention

Gemma 4 has **30 transformer layers**, but they come in two flavours that
alternate in a 5:1 pattern:

| Type               | Count | KV heads | Head dim | Cache grows with context? |
|--------------------|-------|----------|----------|---------------------------|
| Sliding attention  | 25    | 8        | 256      | No — bounded at 1,024 tokens |
| Full attention     | 5     | 2        | 512      | Yes — grows linearly       |

**Sliding attention** layers only attend to the last 1,024 tokens. Their KV
cache is a fixed-size ring buffer: new tokens overwrite the oldest. Memory is
constant regardless of context length (~200 MB total across all 25 layers).

**Full attention** layers attend to the *entire* context. These are the layers
whose KV cache grows linearly with sequence length and is responsible for
essentially all the long-context memory pressure.

TurboQuant is therefore applied **only to the 5 full-attention layers**. The
25 sliding-attention layers stay in standard BF16 because (a) their memory
footprint is already capped and (b) quantizing them would add compute overhead
with no memory benefit.

---

## 4. File Reference

### `turboquant_cache.py`

The core implementation. Contains two classes and one helper function.

#### `_get_rotation_matrix(head_dim, device)`

Builds and caches a random orthogonal matrix of shape `[head_dim, head_dim]`
using QR decomposition with a fixed seed. Called once per unique `head_dim`,
then reused. The fixed seed ensures the same rotation is used across every
layer and every token — critical because the same R must be used for both
quantization (write) and dequantization (read).

#### `TurboQuantLayer`

Subclasses HuggingFace's `DynamicLayer` (the standard per-layer KV cache
object). The standard `DynamicLayer.update()` simply concatenates new K and V
tensors onto a growing list. `TurboQuantLayer.update()` does this:

```
new K/V arrive
      │
      ▼
append to full-precision residual buffer
      │
      ├─ if buffer > 2 × residual_length ──► flush excess to quantized store:
      │                                         rotate → quantize → store uint8 + fp16 scales
      │
      ▼
reconstruct full sequence for attention:
    dequantize stored history → unrotate → concat with residual buffer
      │
      ▼
return (full K, full V) to attention module
```

**Internal storage tensors:**

| Tensor          | Dtype   | Shape                   | Contains                          |
|-----------------|---------|-------------------------|-----------------------------------|
| `_q_keys`       | uint8   | [B, H, S_hist, D]       | Quantized key indices             |
| `_q_values`     | uint8   | [B, H, S_hist, D]       | Quantized value indices           |
| `_k_min`        | fp16    | [B, H, S_hist]          | Per-token min for keys            |
| `_k_scale`      | fp16    | [B, H, S_hist]          | Per-token scale for keys          |
| `_v_min`        | fp16    | [B, H, S_hist]          | Per-token min for values          |
| `_v_scale`      | fp16    | [B, H, S_hist]          | Per-token scale for values        |
| `keys`          | bfloat16 | [B, H, S_res, D]       | Full-precision residual buffer    |
| `values`        | bfloat16 | [B, H, S_res, D]       | Full-precision residual buffer    |

Where `S_hist + S_res = total sequence length` and `S_res ≤ residual_length`.

**Memory accounting methods:**

- `memory_bytes()` — actual bytes used (uint8 indices + fp16 scales + bf16 residual).
- `baseline_memory_bytes()` — what this sequence would cost in standard BF16.
- `theoretical_3bit_bytes()` — bytes if indices were bit-packed (3 bits per value,
  8 values per 3 bytes). This is what the original TurboQuant CUDA kernel achieves.

#### `TurboQuantCache`

A container that subclasses HuggingFace's `Cache`. It reads the model config
to build a list of per-layer objects:

```python
for layer_type in config.text_config.layer_types:
    if layer_type == "sliding_attention":
        layers.append(DynamicSlidingWindowLayer(sliding_window))   # unchanged BF16
    elif layer_type == "full_attention":
        layers.append(TurboQuantLayer(bits, residual_length))      # TurboQuant
```

This means the object is a **drop-in replacement for `DynamicCache`**:

```python
# Standard inference (baseline)
cache = DynamicCache(config=model.config)
out   = model(**inputs, past_key_values=cache, use_cache=True)

# TurboQuant inference — only this line changes
cache = TurboQuantCache(config=model.config, bits=3)
out   = model(**inputs, past_key_values=cache, use_cache=True)
```

No modifications to the model are required.

`compression_stats()` sums `memory_bytes()`, `baseline_memory_bytes()`, and
`theoretical_3bit_bytes()` across all `TurboQuantLayer` instances and returns
a dict with the actual and theoretical compression ratios.

---

### `context_generator.py`

Generates a deterministic long-form text prompt and tokenizes it to an exact
token count.

#### Corpus

`_PASSAGES` is a hand-curated list of 16 dense, diverse paragraphs covering:
computer science, mathematics, physics, history, biology, philosophy, and
economics. Diversity matters because a benchmark that repeats the same sentence
10,000 times would produce an artificially easy (or artificially hard) attention
pattern that doesn't reflect real-world usage.

#### `generate_context(tokenizer, target_tokens, seed, question)`

1. **Shuffle** the passages using a seeded RNG (reproducible across runs).
2. **Cycle** through the shuffled passages, concatenating them until the total
   character count reaches `target_tokens × 6`. The factor 6 is a conservative
   buffer: Gemma's tokenizer produces about 4.7 characters per token on average
   for English text, so generating 6× the target characters guarantees the
   tokenized length will exceed the target.
3. **Tokenize** the full concatenated text in one call (no chunking).
4. **Trim** the token tensor to exactly `target_tokens` tokens by slicing:
   `tokens[:, :target_tokens]`.
5. Append a question at the end so the model has a genuine answering task.

Returns a dict with `input_ids` (shape `[1, target_tokens]`), `actual_tokens`,
and a 200-character preview of the text.

---

### `benchmark.py`

The main entry point. Loads the model once and runs paired trials at each
context length.

#### Constants (top of file)

| Constant           | Default     | Meaning                                             |
|--------------------|-------------|-----------------------------------------------------|
| `MODEL_PATH`       | (local path)| Where Gemma 4 weights are stored on disk            |
| `CONTEXT_LENGTHS`  | 8K–128K     | Token counts to benchmark at                        |
| `NEW_TOKENS`       | 50          | Tokens generated per trial (for throughput timing)  |
| `TURBOQUANT_BITS`  | 3           | Quantization depth                                  |
| `RESIDUAL_LEN`     | 128         | Full-precision tail buffer size (tokens)            |

#### `warmup(model, processor)`

Runs a single short forward pass before any timing begins. CUDA kernels are
JIT-compiled on their first invocation, which can take several seconds. The
warmup absorbs this cost so it doesn't contaminate the timing of the first
real trial.

#### `run_trial(model, processor, context_ids, use_turboquant, label)`

Runs one complete benchmark trial:

1. **Reset memory** — clears the CUDA memory allocator's peak-stats counter
   so we get the peak for this trial only.
2. **Create cache** — either `DynamicCache` or `TurboQuantCache`.
3. **Prefill** — runs `model(input_ids=full_context, ...)` in one forward pass.
   This processes all input tokens simultaneously and fills the KV cache.
   Time recorded with `time.perf_counter()` + `torch.cuda.synchronize()` to
   ensure the GPU has actually finished before stopping the clock.
4. **Generation loop** — feeds one token at a time for `NEW_TOKENS` steps,
   each time passing `past_key_values` back to avoid recomputing anything.
   Records total wall-clock time for the loop.
5. **KV cache accounting** — for TurboQuant, calls `compression_stats()`;
   for baseline, walks the `DynamicCache.layers` list and sums tensor sizes.
6. **Cleanup** — deletes all tensors and calls `gc.collect()` +
   `torch.cuda.empty_cache()` so the next trial starts with a clean slate.

Returns a dict of timing and memory metrics.

#### Two-phase timing model

```
│◄────────── prefill_s ───────────────►│◄── gen_s (50 tokens) ──►│
│                                       │                          │
Input: [tok₁, tok₂, ... tok_N]        │ tok_{N+1} ... tok_{N+50} │
        (full context, one pass)       │ (one token per step)     │
```

- **`prefill_s`** = time to process the entire context and fill the KV cache.
  Dominated by parallelism over the sequence; scales roughly linearly with
  context length.
- **`throughput_tok_s`** = `NEW_TOKENS / gen_s`. Each generation step reads
  the full KV cache from memory to compute one attention output. This is
  where memory bandwidth pressure shows up: a smaller cache means faster reads.

#### `print_comparison(baseline, turboquant)`

Prints side-by-side results for one context length, then the derived stats:
uint8 compression ratio, theoretical 3-bit ratio, throughput delta, and peak
memory saved.

#### Output file

Results are saved as `benchmark_results.json` in the same directory — a list
of dicts, one per context length, with all raw numbers for offline analysis.

---

## 5. Running the Benchmark

```bash
cd /home/harrison/Projects/LocalAI

# Activate the virtual environment if needed
source .venv-oss/bin/activate

# Run (model loading takes a few minutes; full benchmark takes ~30–90 min)
python3 benchmark.py
```

The benchmark loads the model **once** and reuses it across all context
lengths. If a context length causes an out-of-memory error it is skipped
gracefully and the benchmark continues with the next size.

---

## 6. Reading the Output

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║         Gemma 4 26B-A4B-IT  ·  KV-Cache Benchmark  ·  TurboQuant vs Baseline        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════════════
  Generating 65,536-token context …
  Tokenized: 65,536 tokens  |  Preview: "The transformer architecture …"

  Running baseline (DynamicCache BF16) …
    prefill 18.42s  |  31.2 tok/s  |  peak 68.14 GB  |  KV 1.284 GB

  Running TurboQuant (3-bit, residual=128) …
    prefill 18.51s  |  34.7 tok/s  |  peak 66.62 GB  |  KV 0.651 GB

  Context: 65,536 tokens
  ────────────────────────────────────────────────────────────────
  Baseline BF16           prefill  18.42s  gen   31.2 tok/s  peak  68.14 GB  KV  1.284 GB
  TurboQuant 3-bit        prefill  18.51s  gen   34.7 tok/s  peak  66.62 GB  KV  0.651 GB

  → uint8 KV compression:      1.97×  (saved 0.633 GB)
  → theoretical 3-bit ratio:   5.19×  (if bit-packed: 0.247 GB)
  → throughput delta:          +3.5 tok/s  (+11.2%)
  → peak memory saved:         -1.52 GB
```

**Column guide:**

| Column           | Meaning                                                          |
|------------------|------------------------------------------------------------------|
| `prefill`        | Seconds to process the full prompt (one forward pass)           |
| `gen tok/s`      | Autoregressive generation speed over 50 new tokens              |
| `peak GB`        | Peak GPU memory allocated during the trial                       |
| `KV GB`          | Bytes actually stored in the KV cache (measured, not estimated) |

**Derived metrics:**

| Metric                    | Meaning                                                         |
|---------------------------|-----------------------------------------------------------------|
| `uint8 KV compression`    | Baseline BF16 cache ÷ TurboQuant uint8 cache (actual storage)  |
| `theoretical 3-bit ratio` | What the compression would be with full bit-packing             |
| `throughput delta`        | Generation speed difference (positive = TurboQuant is faster)  |
| `peak memory saved`       | Negative = TurboQuant used less peak memory overall             |

**What to expect:**

- Prefill times should be nearly identical — quantization happens *during*
  the forward pass, not before it, and adds minimal overhead.
- Generation throughput should improve at longer contexts because each step
  reads less data from memory (the quantized cache is smaller).
- The compression ratio is stable across context lengths (it depends on
  head_dim and bits, not on sequence length). The absolute memory saved
  grows linearly with context length.

---

## 7. Configuration Knobs

All tuneable constants are at the top of `benchmark.py`:

```python
CONTEXT_LENGTHS = [8_192, 32_768, 65_536, 131_072]
NEW_TOKENS      = 50
TURBOQUANT_BITS = 3      # try 4 for higher quality, 2 for more compression
RESIDUAL_LEN    = 128    # try 64 or 256 to trade quality vs memory
```

And in `TurboQuantCache` directly:

```python
cache = TurboQuantCache(config=model.config, bits=3, residual_length=128)
```

| Parameter        | Lower value                           | Higher value                          |
|------------------|---------------------------------------|---------------------------------------|
| `bits`           | More compression, more error          | Less compression, less error          |
| `residual_length`| Less memory for tail buffer           | Better quality for recent tokens      |

The rotation matrix seed is fixed at `20250405` in `_get_rotation_matrix`.
Changing it produces a different (but equally valid) orthogonal matrix; results
should be statistically similar.
