"""
Generates long deterministic text suitable for KV-cache context-filling benchmarks.
Builds a diverse corpus from short seed passages, then concatenates until the
tokenized result reaches the requested token count.
"""

import random
from transformers import PreTrainedTokenizerBase


# ── Seed corpus ───────────────────────────────────────────────────────────────
# Diverse passages to avoid pure repetition (which can cause degenerate caching).

_PASSAGES = [
    # ── computer science ──────────────────────────────────────────────────────
    """
    The transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017),
    replaced recurrent networks with self-attention mechanisms that attend to every token in the
    sequence simultaneously. The key insight is that the scaled dot-product attention
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    computes a weighted sum of values, where the weights are determined by the similarity between
    queries and keys. Multi-head attention runs this in parallel h times with different learned
    projections, allowing the model to jointly attend to information from different representation
    subspaces. Position is injected via sinusoidal encodings added to the input embeddings.
    """,
    """
    The CAP theorem states that a distributed system can guarantee at most two of the following
    three properties simultaneously: Consistency (every read receives the most recent write or an
    error), Availability (every request receives a response—not necessarily the most recent write),
    and Partition tolerance (the system continues to operate despite network partitions). Modern
    databases navigate these trade-offs differently: Cassandra favors AP, HBase favors CP, and
    traditional RDBMS systems target CA in the absence of partitions.
    """,
    """
    Quicksort achieves O(n log n) average-case time complexity through divide-and-conquer: select
    a pivot, partition the array so all elements less than the pivot precede it and all greater
    elements follow it, then recursively sort the two sub-arrays. The choice of pivot dramatically
    affects performance—random pivot selection or the median-of-three heuristic avoids the O(n²)
    worst case that occurs with sorted input and a fixed pivot. In-place partitioning gives O(log n)
    stack space, making quicksort practical for large datasets despite its worst-case behavior.
    """,
    """
    Gradient descent minimizes a loss function L(θ) by iteratively updating parameters in the
    direction of the negative gradient: θ ← θ − η∇L(θ). Stochastic gradient descent (SGD)
    approximates the full gradient using a mini-batch of m samples, trading accuracy for speed.
    Adam combines momentum (exponential moving average of gradients) with adaptive learning rates
    (per-parameter scaling by the root mean square of recent gradients), making it robust to the
    choice of global learning rate η. Weight decay adds L2 regularization to prevent overfitting
    by penalizing large parameter magnitudes.
    """,
    # ── mathematics ───────────────────────────────────────────────────────────
    """
    The Riemann Hypothesis, one of the Millennium Prize Problems, conjectures that all non-trivial
    zeros of the Riemann zeta function ζ(s) = Σ n^{-s} lie on the critical line Re(s) = 1/2.
    Proven consequences include sharp estimates for the distribution of prime numbers via the
    prime-counting function π(x). The explicit formula connecting primes to zeta zeros,
    π(x) ≈ Li(x) − Σ_ρ Li(x^ρ) + ..., shows that each zero ρ introduces oscillations in the
    prime distribution. Despite enormous computational evidence (over 10^13 zeros verified on the
    critical line), a proof remains elusive.
    """,
    """
    Fourier analysis decomposes a periodic function f(x) into a sum of sinusoids:
        f(x) = a₀/2 + Σ_{n=1}^∞ [aₙ cos(nx) + bₙ sin(nx)]
    where the coefficients aₙ, bₙ are computed by integrating f against the basis functions.
    The Fourier transform extends this to non-periodic functions, yielding the frequency-domain
    representation F(ω) = ∫ f(t) e^{-iωt} dt. The convolution theorem states that convolution
    in the time domain corresponds to pointwise multiplication in the frequency domain, underpinning
    fast convolution algorithms via the FFT, which computes the DFT in O(n log n) operations.
    """,
    """
    P versus NP is the central open problem in computational complexity theory. P is the class of
    decision problems solvable in polynomial time; NP is the class where a proposed solution can
    be verified in polynomial time. NP-complete problems—like SAT, Graph Coloring, and the
    Travelling Salesman decision variant—are the hardest problems in NP: if any NP-complete problem
    has a polynomial-time algorithm, then P = NP. Most computer scientists believe P ≠ NP, but
    no proof exists. The practical consequence is that many combinatorial optimization problems
    probably have no efficient exact algorithm, motivating heuristics and approximation algorithms.
    """,
    # ── physics ───────────────────────────────────────────────────────────────
    """
    Maxwell's equations unify electricity, magnetism, and optics into four relations:
        ∇·E = ρ/ε₀             (Gauss's law)
        ∇·B = 0                 (no magnetic monopoles)
        ∇×E = −∂B/∂t            (Faraday's law)
        ∇×B = μ₀J + μ₀ε₀∂E/∂t  (Ampère-Maxwell law)
    In vacuum (ρ=0, J=0), these imply wave equations for E and B with speed c = 1/√(μ₀ε₀),
    predicting electromagnetic waves. The displacement current term μ₀ε₀∂E/∂t, added by Maxwell,
    was essential: without it Ampère's law would be inconsistent with charge conservation and would
    not predict electromagnetic radiation.
    """,
    """
    General relativity describes gravity as the curvature of spacetime caused by energy and
    momentum. Einstein's field equations, Gμν + Λgμν = (8πG/c⁴)Tμν, relate the Einstein tensor
    Gμν (encoding spacetime curvature) to the stress-energy tensor Tμν (encoding the distribution
    of matter and energy). Solutions include the Schwarzschild metric describing spherically
    symmetric black holes, the FLRW metric describing the expanding universe, and gravitational
    waves—ripples in spacetime first directly detected by LIGO in 2015 from a binary black hole
    merger 1.3 billion light-years away.
    """,
    """
    Quantum entanglement produces correlations between measurements on spatially separated particles
    that cannot be explained by any local hidden variable theory, as Bell's theorem proves. For a
    pair of spin-1/2 particles in the singlet state |ψ⟩ = (|↑↓⟩ − |↓↑⟩)/√2, measuring along any
    axis always yields anti-correlated results. The CHSH inequality quantifies the maximum
    correlations local realism allows (|S| ≤ 2), while quantum mechanics predicts S = 2√2 ≈ 2.83.
    Repeated experiments confirm the quantum prediction, ruling out local hidden variables and
    enabling quantum cryptography protocols whose security rests on the impossibility of
    eavesdropping without disturbing entangled states.
    """,
    # ── history ───────────────────────────────────────────────────────────────
    """
    The Industrial Revolution, beginning in Britain around 1760, transformed manufacturing through
    mechanization powered by steam engines. James Watt's separate condenser (1769) dramatically
    improved steam engine efficiency; the resulting machines drove textile mills, iron foundries,
    and railways. Urbanization accelerated as workers migrated from agriculture to factories.
    By 1850, Britain produced half the world's iron and cotton cloth. The revolution spread to
    Western Europe and North America, reshaping social structures, creating a factory-working
    class, and laying the infrastructure—railways, telegraphs, steamships—for global trade.
    """,
    """
    The Human Genome Project, completed in 2003 after 13 years of international collaboration,
    sequenced approximately 3.2 billion base pairs of human DNA and identified around 20,000–25,000
    protein-coding genes. Shotgun sequencing—fragmenting DNA, sequencing overlapping pieces, and
    assembling by computational overlap detection—enabled rapid progress alongside Sanger
    sequencing. The project revealed that only ~1.5% of the genome encodes proteins; the rest
    includes regulatory elements, non-coding RNA genes, repetitive elements, and regions of
    unknown function. Next-generation sequencing subsequently reduced the cost of a human genome
    from $2.7 billion (HGP) to under $200 today.
    """,
    # ── literature / philosophy ───────────────────────────────────────────────
    """
    Immanuel Kant's Critique of Pure Reason (1781) distinguishes analytic judgments—where the
    predicate is contained in the subject ("all bachelors are unmarried")—from synthetic judgments,
    which extend our knowledge ("the table is brown"). He argues that mathematics and the
    principles of Newtonian physics are synthetic a priori: they extend knowledge yet are known
    independently of experience. Space and time are not properties of things in themselves but
    forms of our intuition; the categories of the understanding (causality, substance, etc.)
    structure experience rather than being derived from it. Things-in-themselves (noumena) remain
    permanently inaccessible to human cognition.
    """,
    """
    In "Crime and Punishment" (1866), Dostoevsky traces the psychological disintegration of
    Raskolnikov, a former student in St. Petersburg who murders a pawnbroker to test his theory
    that extraordinary people are permitted to transgress ordinary moral laws for higher ends.
    Rather than liberation, the act produces mounting psychological torment, paranoia, and
    involuntary confession—an exploration of guilt as an inescapable aspect of human consciousness
    rather than a mere social construct. Through Sonya's compassionate faith, the novel proposes
    that redemption requires suffering, humility, and spiritual renewal rather than rational
    self-justification.
    """,
    # ── biology ───────────────────────────────────────────────────────────────
    """
    CRISPR-Cas9 is a bacterial immune system repurposed as a precision gene-editing tool.
    A guide RNA (gRNA) of ~20 nucleotides directs the Cas9 endonuclease to a complementary
    genomic sequence adjacent to a PAM motif (NGG for SpCas9). Cas9 creates a double-strand
    break; the cell repairs it via non-homologous end joining (introducing small insertions or
    deletions that disrupt gene function) or homology-directed repair (incorporating an externally
    supplied DNA template to introduce precise edits). Therapeutic applications include correcting
    the HBB mutation causing sickle cell disease and knocking out the BCL11A enhancer to
    reactivate fetal hemoglobin—approaches now in clinical trials.
    """,
    """
    Natural selection, the mechanism Darwin identified in "On the Origin of Species" (1859),
    requires heritable variation in traits, differential survival and reproduction linked to those
    traits, and sufficient time. Organisms better adapted to their environment leave more offspring,
    gradually shifting allele frequencies across generations. Modern evolutionary synthesis
    integrates Mendelian genetics: mutations supply new alleles, sexual recombination shuffles
    existing variation, genetic drift causes random allele frequency changes (particularly in small
    populations), and gene flow introduces variation from other populations. Together these forces
    explain the diversity of life and the shared molecular machinery (genetic code, ATP synthesis,
    cell membrane structure) that reveals common ancestry.
    """,
    # ── economics ─────────────────────────────────────────────────────────────
    """
    The efficient market hypothesis (EMH) asserts that asset prices fully reflect all available
    information. The weak form claims prices incorporate historical price data, making technical
    analysis ineffective for generating excess returns. The semi-strong form adds public
    information (earnings announcements, macroeconomic data), ruling out fundamental analysis as
    a source of alpha. The strong form includes insider information. Empirical challenges include
    momentum effects (Jegadeesh & Titman 1993), the value premium (Fama & French 1992), and
    calendar anomalies, leading to behavioral finance models invoking cognitive biases such as
    overconfidence, anchoring, and loss aversion to explain persistent mispricings.
    """,
    """
    Keynesian economics, developed in response to the Great Depression, argues that aggregate
    demand—total spending by households, businesses, and government—is the primary driver of
    economic output and employment in the short run. When private demand collapses (as in a
    recession), government fiscal stimulus (spending increases or tax cuts) can fill the gap
    through the multiplier effect: an initial injection of spending cycles through the economy as
    recipients spend a fraction of their income. The liquidity trap—when nominal interest rates
    near zero make monetary policy ineffective—justifies fiscal expansion even at the cost of
    higher public debt, since the marginal propensity to consume exceeds the marginal propensity
    to save in downturns.
    """,
]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_context(
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
    seed: int = 42,
    question: str = "Based on all of the information above, what is the most important concept discussed?",
) -> dict:
    """
    Build a long context prompt and tokenize it.

    Returns a dict with:
      "input_ids"      : LongTensor [1, L] (L ≤ target_tokens)
      "actual_tokens"  : int
      "text_preview"   : str (first 200 chars)
    """
    rng = random.Random(seed)
    passages = list(_PASSAGES)
    rng.shuffle(passages)

    # Expand corpus by cycling through passages in shuffled order
    corpus_parts: list[str] = []
    target_chars = target_tokens * 6     # conservative: ~4.7 chars/token for Gemma's tokenizer
    total_chars  = 0
    cycle_idx    = 0
    while total_chars < target_chars * 1.1:
        p = passages[cycle_idx % len(passages)]
        corpus_parts.append(p.strip())
        total_chars += len(p)
        cycle_idx   += 1

    full_text = "\n\n".join(corpus_parts)
    full_text += f"\n\n---\n\nQuestion: {question}\n\nAnswer:"

    # Tokenize once at full length, then trim to target
    tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]                         # [1, L_raw]

    if tokens.shape[1] > target_tokens:
        tokens = tokens[:, :target_tokens]

    return {
        "input_ids":     tokens,
        "actual_tokens": tokens.shape[1],
        "text_preview":  full_text[:200],
    }
