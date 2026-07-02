# PERSPECTIVE v2 — Architecture Index

> **Perspective Is All You Need**
>
> A 1.05 T-parameter sparse Mixture-of-Experts language model designed to
> run entirely on **4 GB VRAM + 32 GB RAM** through seven interlocking novel
> subsystems (design target; no end-to-end runtime exists yet).  Every
> component is designed from first principles to solve a specific class of
> failures in current large language models.

---

## Design Philosophy

1. **Parameters are a delivery problem, not a storage problem.**
   A trillion weights exist on NVMe; only the ~28 MB slice needed *right now*
   occupies the GPU.  Layer-streamed ternary execution turns PCIe into the
   effective "weight bus," pipelining transfers behind compute.

2. **Perspective, not attention.**
   The dominant sequence mechanism is *Perspective Decay Recurrence* (PDR),
   where a learned "perspective vector" determines what the model considers
   important before deciding what to remember.  Twenty sparse windowed-
   attention layers provide exact recall when needed.

3. **Experts live on a manifold.**
   128 experts are embedded on a 3-D torus.  Nearby experts share most of
   their weights, enabling *delta streaming* (transfer only the diff) and
   fold-based manifold updates in expert space.

4. **Memory is holographic.**
   Holographic Distributed Memory (HDM) encodes associations in 10 000-bit
   binary vectors via XOR binding and majority-vote bundling.  Constant-time
   bank lookup plus a bounded candidate scan; no graph index to maintain.
   (Comparative benchmarks vs HNSW: future work.)

5. **Calibration is structural.**
   Multi-Perspective Decoding (MPD) generates candidate tokens from four
   different expert configurations; agreement *is* confidence.  No reward
   model, no RLHF.

6. **Learning without back-propagation.**
   Forward-Mode Evolutionary Adaptation (FMEA) updates tiny LoRA adapters
   through Jacobian–vector products and evolves the routing manifold via
   Natural Evolution Strategies.  O(1) memory for gradients.

7. **Safety is a hard geometric bound.**
   Safety Polytope Projection (SPP) constrains every output embedding to a
   convex polytope of vetted safe outputs.  Hard geometric constraint:
   outputs always lie inside the polytope.  Guarantees membership, not
   semantic safety; raises (does not eliminate) attack difficulty.

---

## Vital Statistics

| Metric | Value |
|--------|-------|
| Total parameters | **1.05 T** (128 experts × 8.12 B + 6.83 B shared) |
| Active parameters / token | **14.95 B** (6.83 B shared + 8.12 B expert, top-1) |
| Weight precision | Ternary {-1, 0, +1} = 1.58 bit (natively trained) |
| Shared-layer precision | 2-bit (GPTQ-class) |
| Layers | 80 (60 PDR + 20 windowed GQA) |
| Expert FFN layers | 60 (routed), 20 shared |
| Model dimension | 4 096 |
| FFN intermediate | 11 008 (SwiGLU) |
| Attention heads | 32 query, 8 KV (GQA, windowed layers only) |
| Vocabulary | 32 768 (BPE) |
| VRAM budget | 2 348 MB used / 4 096 MB (42.7 % slack) |
| RAM budget | ~26 GB used / 32 GB (6.6 GB headroom) |
| NVMe footprint | ~208 GB experts (+ ~40 GB deltas), packed at 1.6 bits/param |
| Decode throughput | ~10–11 tok/s (projected from §02 PCIe model; unmeasured) |

## Implementation Status

| Component | Status |
|---|---|
| PDR | CPU reference implemented; parallel scan not implemented (sequential fallback) |
| Manifold routing | Implemented (CPU + optional native path). Measured at init: Gini 0.062–0.069, all 128 experts reached over 10K random tokens (`examples/routing_balance.rs`) |
| Ternary execution | CPU pack/GEMM implemented; GPU kernels NOT implemented (dispatch stubs fall back to CPU) |
| HDM | Core primitives implemented; per-bank capacity measured (`examples/hdm_capacity.rs`) |
| MPD | Prototype implemented |
| FMEA | Prototype implemented |
| SPP | Implemented; projection preserves halfspace feasibility |
| Inference pipeline | NOT functional — `process_token` intentionally returns an error |
| Training stack / tokenizer | Not implemented |

Full test suite: 243 tests passing.

---

## Document Map

| § | Document | Contents |
|---|----------|----------|
| 01 | [Model Topology](01_model_topology.md) | 80-layer structure, data-flow diagram, layer types |
| 02 | [Hardware Budgets](02_budgets_v2.md) | VRAM / RAM / NVMe / PCIe byte-level accounting |
| 03 | [Perspective Decay Recurrence](03_perspective_decay.md) | PDR equations, complexity, state management |
| 04 | [Manifold Routing](04_manifold_routing.md) | Torus geometry, expert placement, delta streaming |
| 05 | [Ternary Execution](05_ternary_execution.md) | Native ternary format; CPU kernels implemented, GPU kernels planned |
| 06 | [Holographic Memory](06_holographic_memory.md) | HDM binding, retrieval, capacity analysis |
| 07 | [Multi-Perspective Decoding](07_multi_perspective.md) | MPD agreement protocol, calibration hypothesis |
| 08 | [Forward Adaptation](08_forward_adaptation.md) | FMEA JVP equations, evolutionary routing |
| 09 | [Safety Polytope](09_safety_polytope.md) | SPP anchor construction, half-space projection |
| 10 | [Training Plan](10_training_plan.md) | Curriculum, ternary-aware optimiser, scaling law |
| 11 | [Inference Pipeline](11_inference_pipeline.md) | Full per-token pipeline, latency budget |
| 12 | [Validation](12_validation_v2.md) | Ablations, stress tests, benchmark targets |
| 13 | [Issue Matrix](13_issue_matrix.md) | 25 LLM issues → component mapping |
| 14 | [Future Extensions](14_extensions.md) | v1 subsystems retained as design candidates |

---

## Source Layout

```
src/
├── core/           PDR, windowed GQA, model skeleton
├── routing/        Manifold router, delta streaming
├── execution/      Ternary packing, layer-streaming pipeline
├── kernels/        CPU ternary GEMM (GPU dispatch stubs, kernels planned)
├── memory/         Holographic Distributed Memory
├── decoding/       Multi-Perspective Decoding
├── learning/       Forward-Mode Evolutionary Adaptation
├── safety/         Safety Polytope Projection
└── runtime/        Main entry, provider API, health checks
```

---

## How the Seven Components Interlock

```
Input tokens
     │
     ▼
┌──────────┐      ┌───────────────┐
│ Embedding │─────▶│ PDR Layer ×60 │◀── Manifold Router ──▶ Expert FFN (streamed)
└──────────┘      └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ GQA Layer ×20 │◀── Shared FFN
                  └───────┬───────┘
                          │
                  ┌───────▼───────┐       ┌─────┐
                  │  MPD Decode   │◀─────▶│ HDM │
                  └───────┬───────┘       └─────┘
                          │
                  ┌───────▼───────┐
                  │  SPP Project  │
                  └───────┬───────┘
                          │
                          ▼
                    Output token
                          │
                  ┌───────▼───────┐
                  │  FMEA Update  │  (async, on loss signal)
                  └───────────────┘
```

---

*Last updated: 2026-07-02 — Perspective v2.0*
