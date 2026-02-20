# § 12 — Validation Plan

> Trust nothing.  Measure everything.

---

## 1  Validation Tiers

| Tier | Name | What it validates | When |
|------|------|------------------|------|
| 0 | Unit tests | Individual component correctness | Every commit |
| 1 | Integration tests | Component interactions | Every PR |
| 2 | System benchmarks | Full-pipeline performance | Weekly |
| 3 | Ablation studies | Contribution of each novel component | Monthly |
| 4 | Stress tests | Failure modes + recovery | Pre-release |
| 5 | Red-team evaluation | Safety + adversarial robustness | Pre-release |

---

## 2  Tier 0: Unit Tests

### 2.1  PDR (Perspective Decay Recurrence)

| Test | Input | Expected | Pass criteria |
|------|-------|----------|--------------|
| Single step | $h \in \mathbb{R}^{4096}$ | $h' = W_p \cdot (\lambda \odot s) + h$ | Element-wise within $\epsilon = 10^{-5}$ of reference |
| Parallel scan consistency | Seq of 1024 tokens | Sequential == parallel scan result | Max absolute diff < $10^{-4}$ |
| State serialisation round-trip | Random state | save → load → compare | Bit-exact |
| Numerical stability | 10K steps, ones input | State norm < 1000 | No NaN/Inf |
| Decay bounds | Any input | $\lambda_i \in (0, 1) \; \forall i$ | Sigmoid output bounded |
| Gradient flow (training) | Loss → W_p grad | Non-zero gradient | $\|\nabla\| > 10^{-8}$ |

### 2.2  Manifold Router

| Test | Input | Expected |
|------|-------|----------|
| Torus wrap-around | Position (0.99, 0.5, 0.99) + delta (0.05, 0, 0.04) | Position (0.04, 0.5, 0.03) |
| Geodesic distance | Pos A, Pos B | $d_T = \sqrt{\min(|x|,1-|x|)^2 + \min(|y|,1-|y|)^2 + \min(|z|,1-|z|)^2}$ |
| Load balance | 10K random tokens | Gini coefficient < 0.15 |
| Deterministic routing | Same input twice | Same expert selected |
| All experts reachable | 100K random inputs | All 128 experts selected ≥ 1× |

### 2.3  Ternary Execution

| Test | Input | Expected |
|------|-------|----------|
| Pack/unpack round-trip | Random ternary matrix | Exact round-trip |
| Ternary GEMM correctness | $A \in \{-1,0,1\}^{m×k}$, $x \in \mathbb{R}^k$ | Within $10^{-6}$ of FP32 reference |
| GEMM speed | 11008 × 4096 | < 0.05 ms on CPU |
| Zero handling | All-zero weight column | Output column = 0 |
| CUDA vs CPU consistency | Same inputs | Max diff < $10^{-5}$ |
| Memory layout | Packed tensor | 5 trits per byte, MSB-first |

### 2.4  Holographic Distributed Memory

| Test | Input | Expected |
|------|-------|----------|
| Binding associativity | $a \oplus (b \oplus c)$ vs $(a \oplus b) \oplus c$ | Bit-exact equal |
| Binding inverse | $a \oplus b \oplus b$ | $= a$ (XOR is self-inverse) |
| Bundling consensus | 5 similar + 1 dissimilar | Majority vote = similar pattern |
| Store/retrieve | Store 100 pairs, query | ≥ 95% correct retrieval |
| Capacity curve | N = 10, 100, 1K, 10K, 100K | Plot accuracy vs N |
| Codebook orthogonality | 4096 codebook vectors | Avg Hamming dist ≈ 5000 ± 50 |

### 2.5  Multi-Perspective Decoding

| Test | Input | Expected |
|------|-------|----------|
| Base perspective = identity | Standard forward pass | Identical logits |
| Jitter sensitivity | Hidden + N(0, 0.01), 100 trials | JSD < 0.05 for confident tokens |
| Agreement threshold | 4 identical distributions | JSD = 0, no disagreement flag |
| Disagreement detection | 1 very different perspective | JSD > threshold |
| ECE computation | 1000 predictions + labels | ECE value ∈ [0, 1] |

### 2.6  FMEA

| Test | Input | Expected |
|------|-------|----------|
| JVP direction | Known function, known direction | Matches finite-diff within $10^{-3}$ |
| LoRA rank | Adapter matrices | Rank ≤ 4 |
| Elastic anchor | Large update | Pulled back toward anchor |
| NES gradient estimate | Sphere function | Points toward minimum |
| Time budget | Full FMEA cycle | < 2.5 s |

### 2.7  Safety Polytope Projection

| Test | Input | Expected |
|------|-------|----------|
| Safe vector unchanged | Vector inside polytope | Projection = input |
| Unsafe vector projected | Vector outside polytope | Projection ∈ polytope |
| Projection idempotent | Project twice | Second projection = identity |
| All anchors inside | Each anchor | All satisfy all half-spaces |
| Adversarial gradient = 0 | Compute $\nabla$ through projection | $\nabla = 0$ (non-differentiable) |

---

## 3  Tier 1: Integration Tests

### 3.1  PDR → Router → Expert → FFN Pipeline

```
Test: Single layer end-to-end
Input: Random hidden state h ∈ R^4096
Expected:
  1. PDR produces valid output (no NaN)
  2. Router selects exactly 1 expert
  3. Expert loads within 5 ms (from NVMe)
  4. Ternary FFN output has correct shape
  5. Residual connection preserves scale
```

### 3.2  Full 80-Layer Forward Pass

```
Test: Complete single-token inference
Input: Token ID = 1 (BOS)
Expected:
  1. Logits shape = [32768]
  2. Logits sum to 1 after softmax
  3. No NaN/Inf in any intermediate
  4. VRAM usage ≤ 2,700 MB
  5. Wall time ≤ 200 ms (first token, cold cache)
```

### 3.3  Expert Loading Pipeline

```
Test: Sequential decode, 100 tokens
Expected:
  1. Double-buffer never stalls (both buffers used alternately)
  2. Delta streaming used when prev_expert set
  3. Cache hit rate matches prediction (±10%)
  4. No expert load failures
```

### 3.4  Safety + Decoding Integration

```
Test: Generate 50 tokens, SPP + MPD active
Expected:
  1. SPP fires on every token (may be no-op for safe tokens)
  2. MPD fires on ~10–30% of tokens
  3. No safety bypass detected
  4. Output quality ≥ base quality
```

---

## 4  Tier 2: System Benchmarks

### 4.1  Throughput Benchmarks

| Test | Setup | Target |
|------|-------|--------|
| Sustained decode | 500 tokens, single stream | ≥ 12 tok/s |
| Prompt processing | 2048-token prompt | ≥ 80 tok/s |
| Long generation | 4096 tokens continuous | ≥ 10 tok/s (no degradation) |
| Cold start | First token after boot | ≤ 20 s |
| Warm start | First token, cache hot | ≤ 200 ms |

### 4.2  Memory Benchmarks

| Test | Target |
|------|--------|
| Peak VRAM (decode) | ≤ 2,700 MB |
| Peak VRAM (prefill, 2048 tokens) | ≤ 3,500 MB |
| Peak RAM (decode) | ≤ 26 GB |
| NVMe read bandwidth utilisation | ≥ 80% of theoretical |

### 4.3  Quality Benchmarks

Run on consumer hardware (RTX 3060, 32 GB RAM):

| Benchmark | PERSPECTIVE target | Comparable model |
|-----------|-------------------|------------------|
| MMLU (5-shot) | ≥ 60% | LLaMA-65B: 63.4% |
| GSM8K (CoT) | ≥ 40% | LLaMA-65B: 50.9% |
| HumanEval (pass@1) | ≥ 35% | LLaMA-65B: 23.7% |
| ARC-Challenge | ≥ 63% | LLaMA-65B: 63.4% |
| TruthfulQA (MC2) | ≥ 55% | LLaMA-65B: 48.7% |
| WinoGrande | ≥ 75% | LLaMA-65B: 77.0% |

Note: Despite being run as a ternary model on consumer hardware,
PERSPECTIVE should be competitive with FP16 dense models at the 65B
scale due to the 14.95B active parameter budget plus expert specialisation.

---

## 5  Tier 3: Ablation Studies

Each ablation removes one novel component and measures the impact:

### 5.1  Ablation Matrix

| ID | Component removed | Replacement | Expected impact |
|----|------------------|-------------|-----------------|
| A1 | PDR | Standard Multi-Head Attention | +30% VRAM (KV cache), -5% quality |
| A2 | Manifold routing | Random top-1 routing | -2% quality, +50% NVMe bandwidth |
| A3 | Delta streaming | Always full expert load | -60% throughput |
| A4 | Ternary quantisation | 4-bit GPTQ | +2% quality, -3× NVMe capacity |
| A5 | HDM | No long-term memory | -10% on multi-turn tasks |
| A6 | MPD | Greedy/nucleus only | +0.05 ECE, +3% hallucination |
| A7 | FMEA | No online adaptation | -5% on domain-specific tasks |
| A8 | SPP | Post-hoc filtering | +15% safety bypass rate |

### 5.2  Ablation Protocol

For each ablation:
1. Disable the component (use replacement or no-op)
2. Run full Tier 2 benchmark suite
3. Compute delta from full PERSPECTIVE
4. Statistical significance: 3 runs, report mean ± std

### 5.3  Expected Contribution Ranking

Predicted most-to-least impactful for overall quality:

1. **Ternary execution** (enables the entire architecture to fit)
2. **PDR** (core sequence processing — replacing it breaks the model)
3. **Delta streaming** (throughput multiplier)
4. **SPP** (safety — critical for deployment)
5. **MPD** (calibration — important for trust)
6. **Manifold routing** (efficiency — moderate impact)
7. **HDM** (memory — valuable but not critical)
8. **FMEA** (adaptation — nice-to-have)

---

## 6  Tier 4: Stress Tests

### 6.1  Memory Stress

| Test | Scenario | Pass criteria |
|------|----------|--------------|
| VRAM exhaustion | Force 4,096 MB allocation | Graceful fallback to CPU |
| RAM exhaustion | Fill to 30 GB | Expert cache eviction, no crash |
| NVMe failure | Simulate read error | Error reported, generation halts cleanly |
| Expert corruption | Flip bits in expert file | Checksum detected, expert reloaded |

### 6.2  Throughput Stress

| Test | Scenario | Pass criteria |
|------|----------|--------------|
| 24-hour soak | Continuous generation | No memory leak, stable tok/s |
| Rapid context switch | Alternate between 8 topics | Correct routing, no stale cache |
| Max context | Fill to 4096 tokens | No OOM, graceful quality degradation |
| Concurrent FMEA | Generate + adapt simultaneously | No race condition |

### 6.3  Numerical Stress

| Test | Scenario | Pass criteria |
|------|----------|--------------|
| PDR 10K steps | Very long conversation | State norm stable |
| Ternary overflow | Maximum activation magnitude | No integer overflow in GEMM |
| Router saturation | All tokens route to same expert | Balance loss kicks in next FMEA |
| HDM capacity | Store 1M associations | Graceful accuracy degradation |

---

## 7  Tier 5: Red-Team Evaluation

### 7.1  Safety Test Suite

| Category | Test count | Source |
|----------|-----------|--------|
| Harmful content generation | 200 | Custom + HarmBench |
| Jailbreak prompts | 150 | AdvBench + custom |
| Indirect prompt injection | 100 | Custom |
| Bias probing | 100 | BBQ benchmark |
| Privacy extraction | 50 | Custom (canary strings) |
| **Total** | **600** | |

### 7.2  SPP Evaluation Metrics

| Metric | Target |
|--------|--------|
| Block rate (harmful prompts) | ≥ 98% |
| False positive rate (safe prompts) | ≤ 2% |
| Latency overhead | ≤ 0.01 ms/token |
| Gradient leakage | 0 (non-differentiable) |

### 7.3  MPD Anti-Hallucination Evaluation

| Metric | Target |
|--------|--------|
| Expected Calibration Error | ≤ 0.08 |
| Perspective disagreement on false claims | ≥ 80% detection |
| Sycophancy reduction (vs baseline) | ≥ 50% |

---

## 8  Automated CI/CD Integration

```yaml
# .github/workflows/validate.yml
on:
  push:
    branches: [main, dev]
  schedule:
    - cron: '0 0 * * 0'  # Weekly full benchmark

jobs:
  tier0:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo test --lib

  tier1:
    runs-on: ubuntu-latest
    needs: tier0
    steps:
      - run: cargo test --test integration

  tier2:
    runs-on: [self-hosted, gpu]
    needs: tier1
    if: github.event_name == 'schedule'
    steps:
      - run: cargo bench --bench system
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/
```

---

*Next: [§ 13 Issue Matrix](13_issue_matrix.md)*
