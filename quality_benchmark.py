#!/usr/bin/env python3
"""
quality_benchmark.py — KV-cache quality: TurboQuant vs Baseline

Embeds specific "needle" facts at known fractional depths inside a long filler
context, then asks both the baseline (DynamicCache BF16) and TurboQuant cache
to answer questions about those facts.  Measures whether KV quantization
degrades recall accuracy without touching model weights.

Model-agnostic: works with any HuggingFace AutoModelForCausalLM.

Usage:
    python quality_benchmark.py --model /path/to/model
    python quality_benchmark.py --model /path/to/model \\
        --context-lengths 8192 32768 --bits 3 --residual-length 128
"""

import argparse
import gc
import json
import math
import re
import sys
import time
import torch
from difflib import SequenceMatcher
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, str(Path(__file__).parent))
from turboquant_cache import TurboQuantCache


# ── Needle facts ──────────────────────────────────────────────────────────────
# Each needle is a short sentence with a unique, unambiguous fact.
# `depth` is the fractional position in the context where it is inserted
# (0.0 = very start, 1.0 = very end).

NEEDLES = [
    {
        "id":       "override_code",
        "text":     "SYSTEM NOTE: The emergency override code for facility Alpha is ZETA-4471-KRYPTON.",
        "question": "What is the emergency override code for facility Alpha?",
        "expected": "ZETA-4471-KRYPTON",
        "depth":    0.10,
    },
    {
        "id":       "batch_size",
        "text":     "EXPERIMENT LOG: The optimal mini-batch size found in ablation study 7 was 3712 samples.",
        "question": "What was the optimal mini-batch size found in ablation study 7?",
        "expected": "3712",
        "depth":    0.30,
    },
    {
        "id":       "meeting_date",
        "text":     "MEMO: The architecture review for project Starling is scheduled for November 3rd at 14:00 UTC.",
        "question": "When is the architecture review for project Starling scheduled?",
        "expected": "November 3",
        "depth":    0.50,
    },
    {
        "id":       "api_key",
        "text":     "CONFIGURATION: The read-only API key for the telemetry service is sk-telemetry-RQ9281XZ.",
        "question": "What is the read-only API key for the telemetry service?",
        "expected": "sk-telemetry-RQ9281XZ",
        "depth":    0.70,
    },
    {
        "id":       "threshold",
        "text":     "CALIBRATION REPORT: The anomaly detection threshold was set to 0.00847 after cross-validation.",
        "question": "What anomaly detection threshold was set after cross-validation?",
        "expected": "0.00847",
        "depth":    0.90,
    },
]

# ── Filler passages ───────────────────────────────────────────────────────────
# Diverse technical text used as surrounding context to fill the target length.

_FILLER = [
    "The transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), replaced recurrent networks with self-attention mechanisms that attend to every token simultaneously. Multi-head attention runs this in parallel h times with different learned projections, allowing the model to jointly attend to information from different representation subspaces. Position is injected via sinusoidal encodings added to the input embeddings.",
    "The CAP theorem states that a distributed system can guarantee at most two of: Consistency (every read receives the most recent write), Availability (every request receives a response), and Partition tolerance (the system operates despite network partitions). Cassandra favors AP, HBase favors CP, and traditional RDBMS targets CA in the absence of partitions.",
    "Gradient descent minimizes a loss function L(θ) by iterating θ ← θ − η∇L(θ). Stochastic gradient descent approximates the full gradient using mini-batches. Adam combines momentum with adaptive learning rates, making it robust to the choice of η. Weight decay adds L2 regularization to prevent overfitting by penalizing large parameter magnitudes.",
    "Maxwell's equations unify electricity, magnetism, and optics. In vacuum they imply wave equations for E and B with speed c = 1/√(μ₀ε₀). The displacement current term μ₀ε₀∂E/∂t, added by Maxwell, was essential for consistency with charge conservation and for predicting electromagnetic radiation.",
    "General relativity describes gravity as spacetime curvature caused by energy and momentum. Einstein's field equations Gμν + Λgμν = (8πG/c⁴)Tμν relate spacetime curvature to the distribution of matter. Solutions include the Schwarzschild metric for black holes and gravitational waves first detected by LIGO in 2015.",
    "CRISPR-Cas9 repurposes a bacterial immune system for precision gene editing. A guide RNA of ~20 nucleotides directs Cas9 to a complementary genomic sequence adjacent to a PAM motif. Cas9 creates a double-strand break; the cell repairs via NHEJ (introducing indels) or HDR (incorporating a supplied template). Applications include correcting the HBB mutation causing sickle cell disease.",
    "The efficient market hypothesis asserts that asset prices fully reflect all available information. The weak form claims prices incorporate all historical price data, making technical analysis ineffective. Empirical challenges include momentum effects, the value premium, and calendar anomalies, motivating behavioral finance models invoking cognitive biases.",
    "Quicksort achieves O(n log n) average-case time through divide-and-conquer: select a pivot, partition the array, then recursively sort both sub-arrays. Random pivot selection avoids the O(n²) worst case that occurs with sorted input. In-place partitioning uses O(log n) stack space, making quicksort practical for large datasets.",
    "Quantum entanglement produces correlations between separated particles that cannot be explained by local hidden variables, as Bell's theorem proves. The CHSH inequality bounds local-realist correlations at |S| ≤ 2, while quantum mechanics predicts S = 2√2 ≈ 2.83. Experiments confirm the quantum prediction, enabling quantum cryptography whose security rests on measurement disturbance.",
    "The Human Genome Project, completed in 2003, sequenced approximately 3.2 billion base pairs and identified around 20,000–25,000 protein-coding genes. Only ~1.5% of the genome encodes proteins; the rest includes regulatory elements, non-coding RNA, and repetitive sequences. Next-generation sequencing subsequently reduced the sequencing cost from $2.7 billion to under $200.",
    "Kant's Critique of Pure Reason distinguishes analytic judgments (predicate contained in subject) from synthetic judgments (which extend knowledge). He argues mathematics and Newtonian physics are synthetic a priori — extending knowledge yet knowable independently of experience. Space and time are forms of intuition; the categories of the understanding structure experience rather than being derived from it.",
    "Natural selection requires heritable variation, differential reproductive success linked to that variation, and time. Modern evolutionary synthesis integrates Mendelian genetics: mutations supply new alleles, recombination shuffles existing variation, genetic drift causes random frequency changes (especially in small populations), and gene flow introduces new variants from other populations.",
    "Keynesian economics argues that aggregate demand drives output and employment in the short run. When private demand collapses, government fiscal stimulus fills the gap through the multiplier effect. The liquidity trap — when interest rates near zero neutralize monetary policy — justifies deficit spending because the marginal propensity to consume exceeds the marginal propensity to save.",
    "The Riemann Hypothesis conjectures that all non-trivial zeros of the Riemann zeta function ζ(s) lie on the critical line Re(s) = 1/2. Proven consequences include sharp estimates for the distribution of prime numbers. Despite over 10¹³ zeros verified computationally on the critical line, a proof remains elusive and the problem is one of the Millennium Prize Problems.",
    "P vs NP is the central open problem in computational complexity. P is the class of problems solvable in polynomial time; NP is the class where solutions are verifiable in polynomial time. NP-complete problems are the hardest in NP: if any has a polynomial-time algorithm then P = NP. Most researchers believe P ≠ NP, motivating the study of heuristics and approximation algorithms.",
    "Fourier analysis decomposes a periodic function into a sum of sinusoids: f(x) = a₀/2 + Σ[aₙcos(nx) + bₙsin(nx)]. The convolution theorem states that convolution in the time domain equals pointwise multiplication in the frequency domain, underpinning fast convolution via the FFT, which computes the DFT in O(n log n) operations.",
    "The Industrial Revolution, beginning in Britain around 1760, transformed manufacturing through steam-powered mechanization. James Watt's separate condenser dramatically improved engine efficiency. By 1850, Britain produced half the world's iron and cotton cloth. The revolution spread to Europe and North America, reshaping social structures and laying infrastructure for global trade.",
]


# ── Context builder ───────────────────────────────────────────────────────────

def build_needle_context(
    tokenizer,
    target_tokens: int,
    needles: list[dict],
    seed: int = 42,
) -> dict:
    """
    Assemble a context of approximately `target_tokens` tokens by cycling through
    filler passages, then splice each needle at its specified fractional depth.

    Returns a dict with:
        input_ids      : LongTensor [1, L]
        actual_tokens  : int
        needle_info    : list of {id, token_position, depth_fraction}
    """
    import random
    rng = random.Random(seed)
    passages = list(_FILLER)
    rng.shuffle(passages)

    # Build enough filler text (generous overshoot, trim later)
    parts: list[str] = []
    total_chars = 0
    target_chars = target_tokens * 4   # ~4 chars/token for most tokenizers
    i = 0
    while total_chars < target_chars * 1.2:
        p = passages[i % len(passages)].strip()
        parts.append(p)
        total_chars += len(p)
        i += 1

    filler_text = "\n\n".join(parts)
    filler_ids = tokenizer(
        filler_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"]

    # Trim to target length
    if filler_ids.shape[1] > target_tokens:
        filler_ids = filler_ids[:, :target_tokens]
    filler_len = filler_ids.shape[1]

    # Tokenize each needle
    needle_tok: list[torch.Tensor] = []
    for n in needles:
        nids = tokenizer(
            f"\n\n{n['text']}\n\n", return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        needle_tok.append(nids)

    # Insert needles in depth order, tracking cumulative offset
    assembled = filler_ids.clone()
    sorted_pairs = sorted(zip(needles, needle_tok), key=lambda x: x[0]["depth"])
    needle_info: list[dict] = []
    offset = 0

    for needle, nids in sorted_pairs:
        insert_at = int(needle["depth"] * filler_len) + offset
        insert_at = max(0, min(insert_at, assembled.shape[1]))
        assembled = torch.cat(
            [assembled[:, :insert_at], nids, assembled[:, insert_at:]], dim=1
        )
        needle_info.append({
            "id":             needle["id"],
            "token_position": insert_at,
            "depth_fraction": insert_at / assembled.shape[1],
        })
        offset += nids.shape[1]

    # Prepend BOS if the tokenizer expects it
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        assembled = torch.cat([torch.tensor([[bos_id]]), assembled], dim=1)

    return {
        "input_ids":     assembled,
        "actual_tokens": assembled.shape[1],
        "needle_info":   needle_info,
    }


# ── Inference helpers ─────────────────────────────────────────────────────────

def _reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_inference(
    model,
    tokenizer,
    context_ids: torch.Tensor,    # [1, S]  the context on CPU
    question: str,
    use_turboquant: bool,
    max_new_tokens: int,
    chunk_size: int,
    bits: int,
    residual_length: int,
) -> tuple[str, float]:
    """
    Chunked prefill of (context + question suffix), then greedy generation.
    Returns (answer_string, prefill_seconds).
    """
    _reset_memory()
    device = next(model.parameters()).device

    # Build question suffix and concatenate with context
    suffix = (
        f"\n\n---\n"
        f"Using only the information in the document above, answer briefly.\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    suffix_ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = torch.cat([context_ids, suffix_ids], dim=1).to(device)
    seq_len = full_ids.shape[1]

    # Create cache
    if use_turboquant:
        cache = TurboQuantCache(
            config=model.config, bits=bits, residual_length=residual_length
        )
    else:
        cache = DynamicCache()

    # Chunked prefill
    _sync()
    t0 = time.perf_counter()
    next_token = None
    for i in range(math.ceil(seq_len / chunk_size)):
        start = i * chunk_size
        end   = min(start + chunk_size, seq_len)
        with torch.inference_mode():
            out = model(
                input_ids=full_ids[:, start:end],
                past_key_values=cache,
                use_cache=True,
            )
        cache      = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        del out
        torch.cuda.empty_cache()
    _sync()
    prefill_s = time.perf_counter() - t0

    # Determine EOS token id(s)
    eos_ids: set[int] = set()
    cfg_eos = getattr(model.config, "eos_token_id", None)
    if isinstance(cfg_eos, list):
        eos_ids.update(cfg_eos)
    elif cfg_eos is not None:
        eos_ids.add(cfg_eos)
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if tok_eos is not None:
        eos_ids.add(tok_eos)

    # Greedy generation until EOS or double-newline
    generated_ids: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            cache = out.past_key_values
            del out
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if tok_id in eos_ids:
                break
            # Stop at a blank line — answer is complete
            decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "\n\n" in decoded_so_far:
                break

    del cache, full_ids, next_token
    _reset_memory()

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # Trim at first newline — keep only the first line of the answer
    answer = answer.split("\n")[0].strip()
    return answer, prefill_s


# ── Comparison metrics ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def answer_correct(answer: str, expected: str) -> bool:
    """True if expected (normalised) appears anywhere in the answer (normalised)."""
    return _normalize(expected) in _normalize(answer)

def token_f1(a: str, b: str) -> float:
    """Token-level F1 between two answer strings — measures agreement."""
    a_toks = set(_normalize(a).split())
    b_toks = set(_normalize(b).split())
    if not a_toks or not b_toks:
        return 0.0
    common    = a_toks & b_toks
    precision = len(common) / len(a_toks)
    recall    = len(common) / len(b_toks)
    denom     = precision + recall
    return 2 * precision * recall / denom if denom else 0.0

def char_sim(a: str, b: str) -> float:
    """Character-level similarity (SequenceMatcher ratio)."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


# ── Reporting ─────────────────────────────────────────────────────────────────

def _trunc(s: str, n: int = 55) -> str:
    return (s[:n] + "…") if len(s) > n else s

def print_result(needle: dict, base_ans: str, tq_ans: str, base_s: float, tq_s: float):
    correct_base = answer_correct(base_ans, needle["expected"])
    correct_tq   = answer_correct(tq_ans,   needle["expected"])
    f1  = token_f1(base_ans, tq_ans)
    sim = char_sim(base_ans, tq_ans)

    mark = lambda ok: "✓" if ok else "✗"
    print(f"\n  [{needle['id']}]  depth={needle['depth']:.0%}  "
          f"expected: \"{needle['expected']}\"")
    print(f"    Baseline  ({base_s:5.2f}s prefill) [{mark(correct_base)}]: "
          f"{_trunc(base_ans)}")
    print(f"    TurboQuant({tq_s:5.2f}s prefill) [{mark(correct_tq)}]: "
          f"{_trunc(tq_ans)}")
    print(f"    Agreement: token-F1={f1:.3f}  char-sim={sim:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV-cache quality benchmark (needle-in-haystack)")
    p.add_argument("--model",           required=True,
                   help="Path or HuggingFace repo ID of the model")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[8_192, 32_768],
                   help="Context lengths to test (tokens). Default: 8192 32768")
    p.add_argument("--bits",            type=int,   default=3,
                   help="TurboQuant quantization bits (default: 3)")
    p.add_argument("--residual-length", type=int,   default=128,
                   help="TurboQuant full-precision tail length (default: 128)")
    p.add_argument("--chunk-size",      type=int,   default=2_048,
                   help="Prefill chunk size in tokens (default: 2048)")
    p.add_argument("--max-answer-tokens", type=int, default=80,
                   help="Max new tokens to generate per answer (default: 80)")
    p.add_argument("--output",          type=str,   default=None,
                   help="Path to save JSON results (default: quality_benchmark_results.json)")
    return p.parse_args()


def load_tokenizer(model_path: str):
    """Try AutoProcessor first (multimodal models), fall back to AutoTokenizer."""
    try:
        proc = AutoProcessor.from_pretrained(model_path)
        # Use the text tokenizer component if it exists
        tok = getattr(proc, "tokenizer", proc)
        print(f"  Loaded AutoProcessor (using .tokenizer component for text)")
        return tok
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path)
        print(f"  Loaded AutoTokenizer")
        return tok


def main():
    args = parse_args()
    out_path = Path(args.output) if args.output else Path(__file__).parent / "quality_benchmark_results.json"

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        KV-Cache Quality Benchmark  ·  Needle-in-Haystack  ·  TurboQuant     ║
╚══════════════════════════════════════════════════════════════════════════════╝
  Model  : {args.model}
  Bits   : {args.bits}-bit  |  Residual: {args.residual_length} tok  |  Chunk: {args.chunk_size} tok
  Lengths: {args.context_lengths}
""")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading tokenizer …")
    tokenizer = load_tokenizer(args.model)

    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model on: {next(model.parameters()).device}\n")

    # Warmup
    print("  [warmup] priming CUDA kernels …", end=" ", flush=True)
    w_ids = tokenizer("Hello!", return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        model(**w_ids, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("done\n")

    all_results: list[dict] = []

    # ── Iterate over context lengths ──────────────────────────────────────────
    for ctx_len in args.context_lengths:
        print(f"{'═'*78}")
        print(f"  Context length: {ctx_len:,} tokens")

        ctx = build_needle_context(tokenizer, ctx_len, NEEDLES)
        actual_len = ctx["actual_tokens"]
        print(f"  Assembled: {actual_len:,} tokens  ({len(NEEDLES)} needles embedded)")
        for ni in ctx["needle_info"]:
            print(f"    [{ni['id']}] at token {ni['token_position']:,} "
                  f"(depth {ni['depth_fraction']:.1%})")

        context_ids = ctx["input_ids"]   # [1, L] on CPU
        ctx_results: list[dict] = []

        for needle in NEEDLES:
            print(f"\n  ── Question: {needle['question']}")

            # Baseline
            base_ans, base_s = run_inference(
                model, tokenizer, context_ids,
                needle["question"],
                use_turboquant=False,
                max_new_tokens=args.max_answer_tokens,
                chunk_size=args.chunk_size,
                bits=args.bits,
                residual_length=args.residual_length,
            )

            # TurboQuant
            tq_ans, tq_s = run_inference(
                model, tokenizer, context_ids,
                needle["question"],
                use_turboquant=True,
                max_new_tokens=args.max_answer_tokens,
                chunk_size=args.chunk_size,
                bits=args.bits,
                residual_length=args.residual_length,
            )

            print_result(needle, base_ans, tq_ans, base_s, tq_s)

            ctx_results.append({
                "needle_id":       needle["id"],
                "depth":           needle["depth"],
                "expected":        needle["expected"],
                "baseline_answer": base_ans,
                "tq_answer":       tq_ans,
                "baseline_correct": answer_correct(base_ans, needle["expected"]),
                "tq_correct":      answer_correct(tq_ans,   needle["expected"]),
                "agreement_f1":    token_f1(base_ans, tq_ans),
                "agreement_sim":   char_sim(base_ans, tq_ans),
                "baseline_prefill_s": base_s,
                "tq_prefill_s":    tq_s,
            })

        # ── Per-context summary ───────────────────────────────────────────────
        n       = len(ctx_results)
        b_corr  = sum(r["baseline_correct"] for r in ctx_results)
        tq_corr = sum(r["tq_correct"]       for r in ctx_results)
        avg_f1  = sum(r["agreement_f1"]     for r in ctx_results) / n
        avg_sim = sum(r["agreement_sim"]    for r in ctx_results) / n

        print(f"\n  {'─'*74}")
        print(f"  Summary @ {actual_len:,} tokens")
        print(f"    Baseline correct : {b_corr}/{n}")
        print(f"    TurboQuant correct: {tq_corr}/{n}")
        print(f"    Avg agreement    : token-F1={avg_f1:.3f}  char-sim={avg_sim:.3f}")

        all_results.append({
            "context_length": actual_len,
            "target_length":  ctx_len,
            "needles":        ctx_results,
            "summary": {
                "baseline_correct":  b_corr,
                "tq_correct":        tq_corr,
                "total":             n,
                "avg_agreement_f1":  avg_f1,
                "avg_agreement_sim": avg_sim,
            },
        })

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n\n{'═'*78}")
    print("  OVERALL SUMMARY")
    print(f"  {'─'*74}")
    print(f"  {'Context':>10}  {'Base✓':>6}  {'TQ✓':>6}  {'Agree F1':>10}  {'Agree Sim':>10}")
    print(f"  {'─'*74}")
    for r in all_results:
        s = r["summary"]
        total = s["total"]
        print(
            f"  {r['context_length']:>10,}  "
            f"{s['baseline_correct']:>3}/{total}   "
            f"{s['tq_correct']:>3}/{total}   "
            f"{s['avg_agreement_f1']:>10.3f}  "
            f"{s['avg_agreement_sim']:>10.3f}"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "config": vars(args), "results": all_results}, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
