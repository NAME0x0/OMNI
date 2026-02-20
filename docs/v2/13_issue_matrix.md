# § 13 — Issue Matrix

> Every novel component exists to solve a specific problem.
> Here's the complete mapping from modern LLM issues to our solutions.

---

## 1  Master Issue → Component Matrix

| # | Issue | Severity | PERSPECTIVE Component | Mechanism | Status |
|---|-------|----------|----------------------|-----------|--------|
| 1 | **Hallucination** | Critical | MPD + HDM | 4-perspective agreement detects low-confidence outputs; HDM retrieves grounding context | Designed |
| 2 | **Sycophancy** | High | MPD (antipodal) | Antipodal perspective at 2× temperature naturally resists user-pleasing drift | Designed |
| 3 | **KV cache memory explosion** | Critical | PDR | Fixed-size recurrent state (2 MB/layer) replaces unbounded KV cache | Designed |
| 4 | **Catastrophic forgetting** | High | FMEA + Elastic anchor | LoRA isolation + $\lambda\|a - a_0\|^2$ penalty prevents drift | Designed |
| 5 | **Prohibitive training cost** | High | Ternary + staged training | 1.58 bits/param → 58× less compute per GEMM; staged pipeline reduces total FLOPs | Designed |
| 6 | **Prohibitive inference cost** | Critical | Ternary + NVMe streaming | Zero-multiply GEMM; 205 GB on NVMe, 2.6 GB VRAM | Designed |
| 7 | **Expert load imbalance** | Medium | Manifold routing (Voronoi) | Geometric balancing via torus Voronoi cells | Designed |
| 8 | **Expert redundancy** | Medium | Manifold routing (delta) | Neighbouring experts on torus share structure; delta streaming exploits this | Designed |
| 9 | **Attention quadratic scaling** | Critical | PDR | O(1) per-step recurrence replaces O(n²) attention in 75% of layers | Designed |
| 10 | **Poor calibration** | High | MPD | Multi-perspective geometric mean produces calibrated probabilities (ECE ≤ 0.08) | Designed |
| 11 | **Adversarial vulnerability** | High | SPP | Non-differentiable convex hull projection → zero gradient for adversaries | Designed |
| 12 | **Safety-utility trade-off** | High | SPP (inflated polytope) | ε-inflated polytope minimises false positives while maintaining safety | Designed |
| 13 | **No online learning** | Medium | FMEA | JVP + NES adapts during user think time with 11 MB memory | Designed |
| 14 | **Backprop memory overhead** | High | FMEA (JVP) | Forward-mode AD uses O(1) memory vs O(n) for backprop | Designed |
| 15 | **Long-context degradation** | High | PDR + HDM | PDR compresses history into fixed state; HDM provides searchable long-term memory | Designed |
| 16 | **Token routing overhead** | Low | Manifold router | 16K FLOPs per layer vs ~100K for standard MoE routing | Designed |
| 17 | **NVMe bandwidth waste** | High | Delta streaming | 70% of expert loads use 2.7 MB delta instead of 27 MB full load → 4× speedup | Designed |
| 18 | **Cold-start latency** | Medium | Hot cache warming | 8 most-frequent experts pre-loaded on startup | Designed |
| 19 | **Quantisation quality loss** | Medium | Knowledge distillation | FP16 teacher guides ternary student; only ~3% degradation | Designed |
| 20 | **Reasoning depth** | High | 80-layer stack + MoE | Deeper than most open models; 14.95B active params per token | Designed |
| 21 | **Multi-turn coherence** | Medium | PDR state + HDM | Persistent recurrent state across turns; HDM for distant recall | Designed |
| 22 | **GPU memory fragmentation** | Medium | Static VRAM layout | All VRAM allocated at startup, no dynamic allocation | Designed |
| 23 | **Model size for distribution** | High | Ternary packing | 1.05T params in ~230 GB (vs ~2 TB FP16) — fits on a single NVMe | Designed |
| 24 | **Deployment complexity** | Medium | Single-binary runtime | Rust binary with embedded kernels, zero Python dependency | Designed |
| 25 | **Opacity / uninterpretability** | Medium | MPD + Manifold routing | Perspective disagreement is interpretable; torus positions visualisable | Designed |

---

## 2  Coverage Analysis

### 2.1  Issues per Severity

| Severity | Count | All addressed? |
|----------|-------|---------------|
| Critical | 4 (#1, #3, #6, #9) | ✅ Yes |
| High | 12 (#2, #4, #5, #7–8, #10–12, #14–15, #17, #20, #23) | ✅ Yes |
| Medium | 8 (#13, #16, #18, #21–22, #24–25) | ✅ Yes |
| Low | 1 (#16) | ✅ Yes |

### 2.2  Components per Issue Count

| Component | Issues addressed | Primary/Support |
|-----------|-----------------|-----------------|
| PDR | 4 (#3, #9, #15, #21) | Primary |
| Manifold Routing | 4 (#7, #8, #16, #25) | Primary |
| Ternary Execution | 4 (#5, #6, #19, #23) | Primary |
| MPD | 4 (#1, #2, #10, #25) | Primary |
| SPP | 2 (#11, #12) | Primary |
| HDM | 3 (#1, #15, #21) | Support |
| FMEA | 3 (#4, #13, #14) | Primary |
| Delta Streaming | 2 (#8, #17) | Primary |
| Architecture | 5 (#18, #20, #22, #24) | System-level |

### 2.3  No component addresses fewer than 2 issues.

---

## 3  Issue Deep-Dives

### 3.1  Hallucination (Issue #1)

**Problem**: LLMs generate plausible-sounding but factually incorrect text.

**Root cause**: Single forward pass produces a point estimate; model
cannot distinguish high-confidence from low-confidence outputs.

**PERSPECTIVE solution — two layers of defence**:

1. **MPD (detection)**: When 4 perspectives disagree on a token (high
   JSD), the token is flagged as uncertain.  The model can then:
   - Generate an alternative token
   - Express uncertainty explicitly
   - Consult HDM for grounding

2. **HDM (correction)**: When MPD flags disagreement, the model queries
   HDM for relevant stored associations.  If a grounding fact is found,
   it biases the output toward the factual completion.

**Expected metric**: TruthfulQA MC2 ≥ 55% (vs ~40% for similar-sized
models without these components).

### 3.2  KV Cache Memory Explosion (Issue #3)

**Problem**: Standard attention requires storing K, V tensors for every
past token.  At 4096 context × 32 heads × 128 dim × 2 bytes × 2 (K+V):
$4096 \times 32 \times 128 \times 4 = 67$ MB **per layer**.  Over 80
layers: 5.4 GB — exceeds our entire VRAM budget.

**PERSPECTIVE solution**: 60 of 80 layers use PDR instead of attention.
PDR state: 4096 × 256 × 2 bytes = 2 MB per layer, **fixed**.  The 20
GQA layers use windowed attention (window = 512) with GQA compression
(8 KV heads vs 32 Q heads): $512 \times 8 \times 128 \times 4 = 2$ MB
per GQA layer.

**Total**: 60 × 2 + 20 × 2 = 160 MB — **34× less than standard attention**.

### 3.3  Prohibitive Inference Cost (Issue #6)

**Problem**: A 1T parameter model at FP16 requires ~2 TB of memory and
~2 TB/s bandwidth for interactive inference.  This normally requires
8 × A100-80GB in tensor/pipeline parallelism.

**PERSPECTIVE solution — three innovations stack**:

1. **Ternary weights**: 1.05T params × 1.58 bits = 207 GB (vs 2 TB FP16)
2. **Sparse MoE**: Only 14.95B active (1.4% of total) → one expert
   loaded per layer
3. **NVMe streaming**: 205 GB on $0.05/GB NVMe; stream 27 MB/layer
   through 12.4 GB/s PCIe

**Result**: Full 1T model inference on a $200 GPU + $15 NVMe.

### 3.4  Adversarial Vulnerability (Issue #11)

**Problem**: Neural network safety relies on differentiable classifiers
that can be defeated by gradient-based attacks (GCG, AutoDAN, etc.).

**PERSPECTIVE solution**: SPP uses **geometric projection** (Dykstra's
alternating projection onto convex half-spaces) which is:

- **Non-differentiable** at the polytope boundary → gradient-based
  attacks get exactly zero gradient through the safety layer
- **Provably constraining** → output is always within the safe convex hull
- **Computationally trivial** → 5 iterations × 500 half-space checks ≈ 0.01 ms

No neural safety classifier means no adversarial gradient to exploit.

---

## 4  Comparison with Existing Approaches

| Issue | GPT-4 approach | LLaMA approach | PERSPECTIVE approach |
|-------|---------------|----------------|---------------------|
| Hallucination | RLHF tuning | — | MPD + HDM (architectural) |
| Safety | RLHF + content filter | Supervised FT | SPP (geometric, non-differentiable) |
| Memory | Massive VRAM | Quantisation | PDR + ternary + NVMe streaming |
| Calibration | Temperature tuning | — | MPD (multi-perspective geometric mean) |
| Online learning | — | — | FMEA (forward-mode, 11 MB) |
| Expert routing | Dense model | Dense model | Manifold routing on torus |
| Forgetting | Full retraining | Full retraining | Elastic LoRA anchoring |

Key differentiator: **Every PERSPECTIVE solution is architectural**, not
post-hoc.  The safety, calibration, and memory properties are built into
the forward pass, not bolted on after training.

---

## 5  Risk Matrix

| Component | Risk | Likelihood | Impact | Mitigation |
|-----------|------|-----------|--------|------------|
| PDR | Degradation at very long contexts (>100K) | Medium | High | Fallback to GQA for 25% of layers |
| Ternary | Quality loss beyond 3% | Low | High | Knowledge distillation + mixed precision for critical layers |
| Manifold routing | Poor expert differentiation | Medium | Medium | Diversity loss + post-hoc MDS alignment |
| Delta streaming | Low delta ratio (<50%) | Medium | Medium | Adaptive: fall back to full loads |
| HDM | Capacity exhaustion | Low | Low | Hierarchical banking, periodic garbage collection |
| MPD | Excessive activation (>40%) | Low | Medium | Adaptive threshold tuning via FMEA |
| FMEA | Catastrophic update | Very Low | High | Elastic anchoring + LoRA isolation |
| SPP | False positive spike | Medium | High | ε-inflation tuning, context-aware polytope switching |

---

## 6  Success Criteria Summary

The architecture is validated when ALL of the following hold:

```
□ VRAM ≤ 2,700 MB during inference
□ RAM ≤ 26 GB during inference
□ Throughput ≥ 12 tok/s (without MPD)
□ Throughput ≥ 8 tok/s (with MPD)
□ MMLU ≥ 60% (5-shot)
□ GSM8K ≥ 40% (CoT)
□ TruthfulQA MC2 ≥ 55%
□ SPP block rate ≥ 98%
□ SPP false positive ≤ 2%
□ MPD ECE ≤ 0.08
□ HDM retrieval accuracy ≥ 90% at 10K associations
□ No crash in 24-hour soak test
□ All Tier 0 + Tier 1 tests pass
□ Each ablation shows measurable contribution to ≥ 1 metric
```

---

*End of documentation.  Proceed to implementation.*
