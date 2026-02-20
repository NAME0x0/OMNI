# PERSPECTIVE v2 (OMNI)

PERSPECTIVE v2 is a sparse Mixture-of-Experts language model architecture targeting consumer hardware constraints (4 GB VRAM + 32 GB RAM) through layer streaming, ternary execution, recurrent state, and geometry-based routing.

This repository is an implementation and validation workspace for the architecture described in `docs/v2/`.

## 1. What This Project Is Trying To Build

### Objective
Build a production-capable LLM system that can:
- scale to 1.05T total parameters through sparse expert activation and streaming
- run inference on constrained hardware
- maintain calibrated output confidence
- support bounded online adaptation
- enforce hard geometric output safety constraints

### Why This Exists
Standard dense transformer deployments fail this hardware envelope. This design treats parameter count as a delivery and scheduling problem rather than a full-residency VRAM problem.

## 2. Architecture Snapshot (Current Spec)

| Item | Value |
|---|---|
| Total parameters | 1.05T |
| Experts | 128 |
| Shared parameters | 6.83B |
| Active parameters per token | 14.95B (top-1 routing) |
| Layers | 80 (60 PDR + 20 windowed GQA) |
| Hidden size (`d_model`) | 4096 |
| PDR rank (`d_state`) | 256 |
| Vocabulary | 32768 |
| Expert manifold | 3D torus (`T^3`) on 8x4x4 lattice |
| Expert precision | Ternary (`{-1,0,+1}`) |

## 3. Core Components

| Component | Purpose | Source Modules |
|---|---|---|
| Perspective Decay Recurrence (PDR) | O(1) recurrent sequence processing with perspective-conditioned decay | `src/core/pdr.rs`, `src/core/pdr_state.rs` |
| Manifold Routing + Delta Streaming | Top-1 expert routing over toroidal manifold with neighbor-aware transfer | `src/routing/manifold.rs`, `src/routing/router.rs`, `src/routing/delta_stream.rs` |
| Layer-Streamed Ternary Execution | Add/sub/skip kernels and double-buffered load/compute overlap | `src/execution/`, `src/kernels/` |
| Holographic Distributed Memory (HDM) | Binary hypervector associative memory | `src/memory/` |
| Multi-Perspective Decoding (MPD) | Agreement-based decoding and confidence signal | `src/decoding/` |
| Forward-Mode Evolutionary Adaptation (FMEA) | LoRA + JVP + NES adaptation without backprop graph storage | `src/learning/` |
| Safety Polytope Projection (SPP) | Hard output projection into safe convex region | `src/safety/` |

## 4. Current Maturity (Repository Reality)

| Area | Status | Notes |
|---|---|---|
| Architecture modules | Partial implementation | Most modules exist with tests; some runtime paths are placeholders/fail-fast |
| Inference runtime | Not production-ready | `src/runtime/pipeline.rs` still intentionally returns error in token processing path |
| GPU kernels | Not complete | CUDA/HIP/SYCL dispatch remains fallback-heavy/stubbed in places |
| Training system | Not implemented end-to-end | No full trainer/data/checkpoint/eval pipeline yet |
| Safety/decoding/learning components | Prototype level | Core logic present; full integration and scale validation pending |
| Docs v2 alignment | Improved | Significant consistency fixes applied; further ongoing verification expected |

## 5. Roadmap Checklist (Tracking)

This section is the project execution checklist. Use it as the canonical done/pending tracker.

### 5.1 Completed Recently
- [x] Migrate expert manifold from 2D to 3D torus (`T^3`)
- [x] Implement fold-in-place manifold updates (new evidence folds into existing coordinates, no append semantics)
- [x] Add optional native manifold acceleration path in C/C++ (`native-manifold` feature)
- [x] Wire Rust FFI bridge for native manifold fold + nearest-expert operations
- [x] Add ZeroClaw runtime adapter and CLI task hook (`--swarm-task`)
- [x] Update affected `docs/v2` manifold and adaptation references to 3D/fold language

### 5.2 Required For "True LLM" Readiness
- [ ] Implement full token processing path in `src/runtime/pipeline.rs` (replace fail-fast placeholder with real execution)
- [ ] Resolve full-suite memory crash and stabilize all tests under CI
- [ ] Build end-to-end training stack (data ingest, tokenizer pipeline, trainer loop, checkpoints, resume)
- [ ] Complete GPU backend kernels and remove fallback-only paths for core hot loops
- [ ] Add distributed training orchestration (multi-GPU / multi-node execution and recovery)
- [ ] Validate scaling strategy on smaller models before high-cost runs
- [ ] Execute staged training curriculum end-to-end (dense warmup -> sparse routing/manifold -> ternary/distillation)
- [ ] Define and enforce acceptance gates (perplexity, benchmark thresholds, calibration, safety, throughput)
- [ ] Productionize inference service (packaging, API surface, observability, rollback)
- [ ] Integrate ZeroClaw deeper into training/inference job orchestration (beyond single CLI task call)

### 5.3 Exit Criteria For First Trainable Release
- [ ] Reproducible training run with checkpoint resume
- [ ] Stable convergence on at least one representative benchmark suite
- [ ] End-to-end inference pipeline functional without placeholder failures
- [ ] Safety and calibration metrics validated against documented thresholds
- [ ] Throughput and memory targets met on declared hardware envelope

## 6. Build, Test, and Run

### Build
```bash
cargo build
cargo build --release
```

### Test
```bash
cargo test
```

### Native manifold path (C/C++)
```bash
cargo check --features native-manifold
cargo test routing:: --features native-manifold
```

### Basic CLI run
```bash
cargo run -- --model-dir ./model -n 128 "Your prompt here"
```

### ZeroClaw task dispatch
```bash
cargo run -- --swarm-task "your agentic task" --zeroclaw-bin zeroclaw --swarm-workspace .
```

## 7. Repository Layout

```text
src/
  core/        # PDR, GQA, model skeleton
  routing/     # 3D manifold router and delta streaming
  execution/   # streaming pipeline and weight packing
  kernels/     # CPU kernels, dispatch, optional native manifold C/C++
  memory/      # HDM memory subsystem
  decoding/    # MPD decoding and calibration
  learning/    # JVP, LoRA, NES, FMEA
  safety/      # polytope safety projection
  runtime/     # provider, pipeline orchestration, health, zeroclaw adapter
docs/v2/       # architecture specification and validation documents
```

## 8. Documentation Index (`docs/v2`)

| File | Topic |
|---|---|
| `docs/v2/00_V2_INDEX.md` | v2 architecture index |
| `docs/v2/01_model_topology.md` | model topology |
| `docs/v2/02_budgets_v2.md` | hardware/storage budgets |
| `docs/v2/03_perspective_decay.md` | PDR equations and implementation |
| `docs/v2/04_manifold_routing.md` | 3D manifold routing and delta strategy |
| `docs/v2/05_ternary_execution.md` | ternary execution details |
| `docs/v2/06_holographic_memory.md` | HDM memory model |
| `docs/v2/07_multi_perspective.md` | MPD decoding |
| `docs/v2/08_forward_adaptation.md` | FMEA adaptation |
| `docs/v2/09_safety_polytope.md` | SPP safety model |
| `docs/v2/10_training_plan.md` | staged training plan |
| `docs/v2/11_inference_pipeline.md` | inference pipeline and latency model |
| `docs/v2/12_validation_v2.md` | validation framework |
| `docs/v2/13_issue_matrix.md` | issue-to-component mapping |

## 9. Contribution Standard

Changes are expected to be:
- mathematically consistent with `docs/v2`
- dimension-safe (no hidden magic constants)
- memory-budget aware
- benchmarked on hot paths when performance-sensitive
- accompanied by tests for behavioral changes

## 10. License

MIT
