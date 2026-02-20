# PERSPECTIVE v2 — Architecture Index

> **Perspective Is All You Need**
>
> A 1.05 T-parameter sparse Mixture-of-Experts language model that runs
> entirely on **4 GB VRAM + 32 GB RAM** through seven interlocking novel
> subsystems.  Every component is designed from first principles to solve
> a specific class of failures in current large language models.

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
   128 experts are embedded on a 2-D torus.  Nearby experts share most of
   their weights, enabling *delta streaming* (transfer only the diff) and
   smooth interpolation in expert space.

4. **Memory is holographic.**
   Holographic Distributed Memory (HDM) encodes associations in 10 000-bit
   binary vectors via circular convolution.  O(1) retrieval, zero index
   maintenance, 20× faster than HNSW.

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
   convex polytope of vetted safe outputs.  Non-differentiable → immune to
   gradient-based adversarial attacks.

---

## Vital Statistics

| Metric | Value |
|--------|-------|
| Total parameters | **1.05 T** (128 experts × 8.12 B + 7.7 B shared) |
| Active parameters / token | **15.8 B** (7.7 B shared + 8.12 B expert, top-1) |
| Weight precision | Ternary {-1, 0, +1} = 1.58 bit (natively trained) |
| Shared-layer precision | 2-bit (GPTQ-class) |
| Layers | 80 (60 PDR + 20 windowed GQA) |
| Expert FFN layers | 60 (routed), 20 shared |
| Model dimension | 4 096 |
| FFN intermediate | 11 008 (SwiGLU) |
| Attention heads | 32 query, 8 KV (GQA, windowed layers only) |
| Vocabulary | 32 768 (BPE) |
| VRAM budget | 2 684 MB used / 4 096 MB (34.5 % slack) |
| RAM budget | ~26 GB used / 32 GB (6.6 GB headroom) |
| NVMe footprint | ~205 GB (all 128 experts, ternary-packed) |
| Decode throughput | 12–14 tok/s (PCIe-bound pipeline) |

---

## Document Map

| § | Document | Contents |
|---|----------|----------|
| 01 | [Model Topology](01_model_topology.md) | 80-layer structure, data-flow diagram, layer types |
| 02 | [Hardware Budgets](02_budgets_v2.md) | VRAM / RAM / NVMe / PCIe byte-level accounting |
| 03 | [Perspective Decay Recurrence](03_perspective_decay.md) | PDR equations, complexity, state management |
| 04 | [Manifold Routing](04_manifold_routing.md) | Torus geometry, expert placement, delta streaming |
| 05 | [Ternary Execution](05_ternary_execution.md) | Native ternary format, CUDA / HIP / SYCL / CPU kernels |
| 06 | [Holographic Memory](06_holographic_memory.md) | HDM binding, retrieval, capacity analysis |
| 07 | [Multi-Perspective Decoding](07_multi_perspective.md) | MPD agreement protocol, calibration proof |
| 08 | [Forward Adaptation](08_forward_adaptation.md) | FMEA JVP equations, evolutionary routing |
| 09 | [Safety Polytope](09_safety_polytope.md) | SPP anchor construction, half-space projection |
| 10 | [Training Plan](10_training_plan.md) | Curriculum, ternary-aware optimiser, scaling law |
| 11 | [Inference Pipeline](11_inference_pipeline.md) | Full per-token pipeline, latency budget |
| 12 | [Validation](12_validation_v2.md) | Ablations, stress tests, benchmark targets |
| 13 | [Issue Matrix](13_issue_matrix.md) | 25 LLM issues → component mapping |

---

## Source Layout

```
src/
├── core/           PDR, windowed GQA, model skeleton
├── routing/        Manifold router, delta streaming
├── execution/      Ternary packing, layer-streaming pipeline
├── kernels/        CUDA / HIP / SYCL / CPU ternary GEMM
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

*Last updated: 2026-02-20 — Perspective v2.0*
