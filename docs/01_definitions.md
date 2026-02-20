# Section 1: Definitions & Equivalence Metric

## 1.1 What "1.5T Reasoning Equivalence" Means

We do NOT claim to run 1.5 trillion parameters on consumer hardware. That is physically
impossible at any precision fitting 4 GB VRAM. Instead, we define:

**1.5T Reasoning Equivalence (1.5T-RE):** A composite score on a fixed benchmark suite
such that the system's task-weighted performance matches or exceeds the interpolated
score of a hypothetical dense 1.5T parameter model on that same suite, as projected
from known scaling curves.

### Measurable Proxy Definition

We use a **Composite Reasoning Score (CRS)** defined over 5 benchmarks:

| Benchmark | Weight | Measures |
|-----------|--------|----------|
| MMLU (5-shot) | 0.20 | Factual breadth |
| GSM8K (8-shot) | 0.20 | Multi-step arithmetic reasoning |
| HumanEval (0-shot) | 0.20 | Code generation |
| ARC-Challenge (25-shot) | 0.20 | Scientific reasoning |
| TruthfulQA (0-shot) | 0.20 | Calibration & truthfulness |

**CRS = sum(weight_i * normalized_score_i)** where each score is normalized to [0, 1].

### Scaling Law Projection

From Chinchilla scaling (Hoffmann et al. 2022) and observed scores:
- 7B dense (LLaMA-2-7B): CRS ≈ 0.52
- 13B dense (LLaMA-2-13B): CRS ≈ 0.58
- 70B dense (LLaMA-2-70B): CRS ≈ 0.68
- 1.5T projected (extrapolating log-linear): CRS ≈ 0.78

**Our target: CRS >= 0.78** via the compound system described below.

### How We Close the Gap Without 1.5T Parameters

The system uses a **Mixture of Strategies** (not just a single model):

| Strategy | CRS contribution mechanism |
|----------|---------------------------|
| Sparse Mixture-of-Experts (MoE) with 2-bit quantized experts | Higher effective parameter count per FLOP |
| Retrieval-augmented generation from 2M+ topological memory | Closes factual recall gap (MMLU, ARC) |
| Multi-pass verification with truth grounding | Closes TruthfulQA and calibration gap |
| Scratchpad chain-of-thought with iterative refinement | Closes GSM8K and HumanEval gap |
| Continual learning adaptation | Domain-specific fine-tuning over time |

## 1.2 Baseline Comparison

Reference model: **LLaMA-2-7B-Chat** (Q4_K_M quantization, ~4.08 GB, runs on 4GB VRAM barely).

| Metric | LLaMA-2-7B-Chat (Q4) | OMNIS Target | Delta |
|--------|----------------------|--------------|-------|
| CRS | 0.52 | 0.78 | +0.26 (+50%) |
| MMLU | 0.46 | 0.75 | +0.29 |
| GSM8K | 0.23 | 0.72 | +0.49 |
| HumanEval | 0.13 | 0.55 | +0.42 |
| ARC-Challenge | 0.53 | 0.82 | +0.29 |
| TruthfulQA | 0.41 | 0.80 | +0.39 |
| Active VRAM | ~4000 MB | <=3686 MB | -314 MB |
| Host RAM | ~6 GB | <=28 GB | +22 GB |
| Tokens/s (decode) | ~12 tok/s | >=8 tok/s | -4 tok/s |
| Latency (first token) | ~800 ms | <=2000 ms | +1200 ms |

### Justification of Score Targets

- **MMLU 0.75**: Achievable via RAG over topological memory (retrieval lifts MMLU by
  +0.15 to +0.25 in published RAG studies on 7B models).
- **GSM8K 0.72**: Multi-pass CoT with verification. Published: CoT + self-consistency
  on 7B models reaches ~0.55; adding a verifier adds +0.10 to +0.20.
  [HEURISTIC] Falsifiable test: run GSM8K with and without verifier, measure delta.
- **HumanEval 0.55**: Code-specialized expert routing + iterative refinement.
  [HEURISTIC] Falsifiable test: compare single-pass vs 3-pass refinement on HumanEval.
- **ARC 0.82**: Retrieval-augmented with science corpus in topological memory.
- **TruthfulQA 0.80**: Truth grounding verifier directly targets this metric.

### Honest Limitations

1. The 1.5T-RE claim is a **system-level** claim, not a model-level claim. The base
   model is far smaller; the system compensates with retrieval, verification, and
   multi-pass reasoning.
2. Tokens/s will be lower than the 7B baseline because of multi-pass overhead.
3. First-token latency is higher due to retrieval + routing overhead.
4. The CRS target of 0.78 is ambitious. A realistic floor is CRS >= 0.70 (which still
   exceeds 70B-class performance). We track both targets.

### Failure Criterion for Section 1
If after full implementation, CRS < 0.65 on the benchmark suite, the equivalence
claim is falsified and the architecture must be revised.
