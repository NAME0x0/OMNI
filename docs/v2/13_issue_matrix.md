# § 13 — Issue Matrix

> Every novel component targets a specific problem.
> Here's the complete mapping from modern LLM issues to proposed mitigations.

---

## 1  Master Issue → Component Matrix

**Status:** These are design mappings for an untrained architecture.  None of
the mitigations in this matrix has been empirically validated end-to-end yet;
metrics are targets unless explicitly marked as measured elsewhere.

| # | Issue | Severity | PERSPECTIVE Component | Mechanism | Status |
|---|-------|----------|----------------------|-----------|--------|
| 1 | **Hallucination** | Critical | MPD + HDM | 4-perspective agreement is intended to flag low-confidence outputs; HDM retrieves grounding context | Design mapping (unvalidated) |
| 2 | **Sycophancy** | High | MPD (antipodal) | Antipodal perspective at 2× temperature is intended to test user-pleasing drift | Design mapping (unvalidated) |
| 3 | **KV cache memory explosion** | Critical | PDR | Fixed-size recurrent state (2 MB/layer target) replaces unbounded KV cache in most layers | Design mapping (unvalidated) |
| 4 | **Catastrophic forgetting** | High | FMEA + Elastic anchor | LoRA isolation + $\lambda\|a - a_0\|^2$ penalty is intended to limit drift | Design mapping (unvalidated) |
| 5 | **Prohibitive training cost** | High | Ternary + staged training | 1.6-bit packed storage and ternary arithmetic target lower GEMM cost; staged pipeline targets lower total FLOPs | Design mapping (unvalidated) |
| 6 | **Prohibitive inference cost** | Critical | Ternary + NVMe streaming | Zero-multiply GEMM design; ~208 GB experts on NVMe, ~2.3–2.7 GB VRAM target | Design mapping (unvalidated) |
| 7 | **Expert load imbalance** | Medium | Manifold routing (Voronoi) | Geometric balancing via torus Voronoi cells is the design prior | Design mapping (unvalidated) |
| 8 | **Expert redundancy** | Medium | Manifold routing (delta) | Neighbouring experts on torus are intended to share structure; delta streaming depends on this holding after training | Design mapping (unvalidated) |
| 9 | **Attention quadratic scaling** | Critical | PDR | O(1) per-step recurrence is designed to replace O(n²) attention in 75% of layers | Design mapping (unvalidated) |
| 10 | **Poor calibration** | High | MPD | Multi-perspective geometric mean targets better calibration (ECE ≤ 0.08 target) | Design mapping (unvalidated) |
| 11 | **Adversarial vulnerability** | High | SPP | Projection guarantees polytope membership; it is not attack-immune and remains differentiable almost everywhere | Design mapping (unvalidated) |
| 12 | **Safety-utility trade-off** | High | SPP (inflated polytope) | ε-inflated polytope targets a false-positive/safety trade-off that must be red-team validated | Design mapping (unvalidated) |
| 13 | **No online learning** | Medium | FMEA | JVP + NES targets adaptation during user think time with ~11 MB derived workspace | Design mapping (unvalidated) |
| 14 | **Backprop memory overhead** | High | FMEA (JVP) | Forward-mode AD uses O(1) memory vs O(n) for backprop | Design mapping (unvalidated) |
| 15 | **Long-context degradation** | High | PDR + HDM | PDR compresses history into fixed state; HDM provides searchable long-term memory | Design mapping (unvalidated) |
| 16 | **Token routing overhead** | Low | Manifold router | 16K FLOPs per layer vs ~100K for standard MoE routing | Design mapping (unvalidated) |
| 17 | **NVMe bandwidth waste** | High | Delta streaming | Target: 70% of expert loads use small deltas instead of 27 MB full loads → projected speedup | Design mapping (unvalidated) |
| 18 | **Cold-start latency** | Medium | Hot cache warming | 8 most-frequent experts pre-loaded on startup | Design mapping (unvalidated) |
| 19 | **Quantisation quality loss** | Medium | Knowledge distillation | FP16 teacher guides ternary student; quality degradation target must be measured | Design mapping (unvalidated) |
| 20 | **Reasoning depth** | High | 80-layer stack + MoE | Deeper than most open models; 14.95B active params per token | Design mapping (unvalidated) |
| 21 | **Multi-turn coherence** | Medium | PDR state + HDM | Persistent recurrent state across turns; HDM for distant recall | Design mapping (unvalidated) |
| 22 | **GPU memory fragmentation** | Medium | Static VRAM layout | All VRAM allocated at startup, no dynamic allocation | Design mapping (unvalidated) |
| 23 | **Model size for distribution** | High | Ternary packing | 1.05T params in ~253 GB incl. deltas (vs ~2 TB FP16) — fits on a single NVMe | Design mapping (unvalidated) |
| 24 | **Deployment complexity** | Medium | Single-binary runtime | Rust binary with embedded kernels, zero Python dependency | Design mapping (unvalidated) |
| 25 | **Opacity / uninterpretability** | Medium | MPD + Manifold routing | Perspective disagreement is interpretable; torus positions visualisable | Design mapping (unvalidated) |

---

## 2  Coverage Analysis

### 2.1  Issues per Severity

| Severity | Count | Design mapping? |
|----------|-------|---------------|
| Critical | 4 (#1, #3, #6, #9) | Yes, unvalidated |
| High | 11 (#2, #4, #5, #10–12, #14–15, #17, #20, #23) | Yes, unvalidated |
| Medium | 9 (#7–8, #13, #18–19, #21–22, #24–25) | Yes, unvalidated |
| Low | 1 (#16) | Yes, unvalidated |

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

### 2.3  No component is mapped to fewer than 2 issues.

---

## 3  Issue Deep-Dives

### 3.1  Hallucination (Issue #1)

**Problem**: LLMs generate plausible-sounding but factually incorrect text.

**Root cause**: Single forward pass produces a point estimate; model
cannot distinguish high-confidence from low-confidence outputs.

**PERSPECTIVE design — two layers of defence**:

1. **MPD (detection)**: When 4 perspectives disagree on a token (high
   JSD), the token is flagged as uncertain.  The model can then:
   - Generate an alternative token
   - Express uncertainty explicitly
   - Consult HDM for grounding

2. **HDM (correction)**: When MPD flags disagreement, the model queries
   HDM for relevant stored associations.  If a grounding fact is found,
   it biases the output toward the factual completion.

**Target metric**: TruthfulQA MC2 ≥ 55% (vs ~40% for similar-sized
models without these components).

### 3.2  KV Cache Memory Explosion (Issue #3)

**Problem**: Standard attention requires storing K, V tensors for every
past token.  At 4096 context × 32 heads × 128 dim × 2 bytes × 2 (K+V):
$4096 \times 32 \times 128 \times 4 = 67$ MB **per layer**.  Over 80
layers: 5.4 GB — exceeds our entire VRAM budget.

**PERSPECTIVE design**: 60 of 80 layers use PDR instead of attention.
PDR state: 4096 × 256 × 2 bytes = 2 MB per layer, **fixed**.  The 20
GQA layers use windowed attention (window = 512) with GQA compression
(8 KV heads vs 32 Q heads): $512 \times 8 \times 128 \times 4 = 2$ MB
per GQA layer.

**Derived target total**: 60 × 2 + 20 × 2 = 160 MB — **34× less than standard attention**.

### 3.3  Prohibitive Inference Cost (Issue #6)

**Problem**: A 1T parameter model at FP16 requires ~2 TB of memory and
~2 TB/s bandwidth for interactive inference.  This normally requires
8 × A100-80GB in tensor/pipeline parallelism.

**PERSPECTIVE design — three innovations stack**:

1. **Ternary weights**: 1.05T params × 1.6 stored bits ≈ 210 GB (vs 2 TB FP16)
2. **Sparse MoE**: Only 14.95B active (1.4% of total) → one expert
   loaded per layer
3. **NVMe streaming**: ~208 GB expert files on NVMe; stream 27 MB/layer
   through the projected PCIe budget

**Target result**: Full 1T model inference on consumer hardware.  This has
not been demonstrated end-to-end.

### 3.4  Adversarial Vulnerability (Issue #11)

**Problem**: Neural network safety relies on differentiable classifiers
that can be defeated by gradient-based attacks (GCG, AutoDAN, etc.).

**PERSPECTIVE design**: SPP uses **geometric projection** (Dykstra's
alternating projection onto convex half-spaces) which provides:

- **Membership enforcement** → the projected output satisfies the configured
  half-spaces when projection converges
- **No semantic proof** → membership does not prove the sampled text is safe
- **No attack immunity** → half-space projection is piecewise-linear and
  differentiable almost everywhere; BPDA-style and gradient-free attacks apply
- **Projected cost** → default max 50 Dykstra iterations at tolerance `1e-6`;
  latency is unmeasured end-to-end

No neural safety classifier removes one attack target, but the upstream model
and projection remain analyzable by adaptive attackers.

---

## 4  Comparison with Existing Approaches

| Issue | GPT-4 approach | LLaMA approach | PERSPECTIVE approach |
|-------|---------------|----------------|---------------------|
| Hallucination | RLHF tuning | — | MPD + HDM (architectural) |
| Safety | RLHF + content filter | Supervised FT | SPP (geometric membership constraint) |
| Memory | Massive VRAM | Quantisation | PDR + ternary + NVMe streaming |
| Calibration | Temperature tuning | — | MPD (multi-perspective geometric mean) |
| Online learning | — | — | FMEA (forward-mode, ~11 MB derived workspace) |
| Expert routing | Dense model | Dense model | Manifold routing on torus |
| Forgetting | Full retraining | Full retraining | Elastic LoRA anchoring |

Key differentiator: **Every PERSPECTIVE mitigation is architectural**, not
post-hoc.  The safety, calibration, and memory mechanisms are built into the
forward pass, but their effectiveness remains to be validated after training.

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
| FMEA | Catastrophic update | Unknown | High | Elastic anchoring + LoRA isolation + rollback validation |
| SPP | False positive spike | Medium | High | ε-inflation tuning, context-aware polytope switching |

---

## 6  Success Criteria Summary

The architecture is validated when ALL of the following targets hold:

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
