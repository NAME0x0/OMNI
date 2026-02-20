# OMNI — Omniscient Perspective Is All You Need

> A 1.05 Trillion parameter sparse Mixture-of-Experts architecture that runs on **4 GB VRAM + 32 GB RAM** consumer hardware.

## Overview

OMNI is a novel LLM architecture built from first principles to address the fundamental limitations of modern large language models — hallucination, catastrophic forgetting, adversarial vulnerability, quadratic attention scaling, astronomical compute requirements, and opaque reasoning — through seven interlocking novel components.

**Every component is novel.** There are no transformers here. No standard attention. No backpropagation. No floating-point weights.

## Architecture at a Glance

| Parameter | Value |
|---|---|
| Total parameters | 1.05 Trillion |
| Architecture | Sparse Mixture-of-Experts (top-1) |
| Experts | 128 × 8.12B parameters each |
| Shared backbone | 6.83B parameters |
| Weight precision | 1.58-bit ternary {-1, 0, +1} |
| Active parameters/token | ~15B |
| VRAM usage | ~2.6 GB / 4 GB |
| RAM usage | ~18 GB / 32 GB |
| NVMe storage | ~230 GB |

## The Seven Novel Components

### 1. Parallel Decayed Recurrence (PDR)
Replaces quadratic self-attention with O(1) per-token recurrence. Each layer maintains a decaying state vector:
```
s' = λ ⊙ s + v    where λ = σ(W_λ · h)
```
60 PDR layers + 20 windowed GQA layers in a period-4 interleave (PDR-PDR-PDR-GQA).

### 2. Manifold Routing on Flat Torus T²
Expert selection via continuous coordinates on a 2D flat torus [0,1)², using geodesic distance with wrap-around. Each layer maps the hidden state to a torus point and selects the nearest expert from a 16×8 grid. Enables delta-streaming where nearby experts share most weights.

### 3. Ternary Execution Engine
All expert weights are ternary {-1, 0, +1}, stored as 5 trits per byte (base-3 packing). Matrix-vector products use only add/subtract/skip — **zero multiplications**. A single expert FFN layer (SwiGLU, d=4096→11008→4096) fits in ~17 MB.

### 4. Holographic Distributed Memory (HDM)
Episodic memory using 10,000-bit binary hypervectors. Facts are encoded as role-filler bindings via XOR, stored in a ring buffer, and retrieved by Hamming distance. Working memory uses slot-based LRU with capacity tracking.

### 5. Multi-Perspective Decoding (MPD)
Every token is decoded through 4 perspectives (Base, Jitter, Antipodal, Random) that independently produce distributions. Agreement is measured via Jensen-Shannon Divergence. Tokens are only accepted when perspectives agree, providing built-in hallucination detection.

### 6. Forward-Mode Expert Adaptation (FMEA)
Continual learning without backpropagation. Uses forward-mode Jacobian-Vector Products (JVP) for gradient-free directional derivatives, rank-4 LoRA adapters (~32K params per expert) as the trainable surface, and Natural Evolution Strategies (NES) for routing optimisation on the non-differentiable torus.

### 7. Safety Polytope Projection (SPP)
Every output representation is projected onto a convex safety polytope defined by ~1000 anchor points and ~500 halfspace constraints using Dykstra's alternating projection algorithm. This projection is **non-differentiable** — no gradient can flow through it, making gradient-based jailbreak attacks mathematically impossible.

## Memory Budget

```
VRAM (4 GB):
  Shared params (embeddings + PDR + GQA)    1,359 MB
  Active expert FFN (1 of 128)                  17 MB
  PDR state bank (60 layers × rank 256)        120 MB
  KV cache (20 layers × 512 window)            160 MB
  Double buffer (2 × expert)                    34 MB
  Routing + workspace                          910 MB
  ────────────────────────────────────────────────────
  Total                                      2,599 MB / 4,096 MB

RAM (32 GB):
  Hot expert cache (8 experts)                 137 MB
  Expert index + routing tables                 52 MB
  HDM codebook + episodic memory              125 MB
  OS + runtime overhead                     ~4,000 MB
  ────────────────────────────────────────────────────
  Total                                    ~18,100 MB / 32,768 MB

NVMe:
  128 experts × ~1.8 GB each               ~230 GB
```

## Project Structure

```
src/
├── lib.rs              # Module declarations + global config constants
├── main.rs             # CLI binary entry point
├── core/               # PDR + GQA backbone
│   ├── pdr.rs          # Parallel Decayed Recurrence layer
│   ├── pdr_state.rs    # PDR state management + serialization
│   ├── windowed_gqa.rs # Windowed Grouped-Query Attention
│   └── model.rs        # Full model orchestrator (80-layer forward)
├── routing/            # Manifold routing on T²
│   ├── manifold.rs     # Flat torus geometry + expert grid
│   ├── router.rs       # Per-layer + full-model routing
│   └── delta_stream.rs # Delta-based expert loading strategy
├── execution/          # Ternary compute engine
│   ├── ternary_pack.rs # Base-3 packing (5 trits/byte), SwiGLU FFN
│   ├── double_buffer.rs# Double-buffered DMA for expert streaming
│   └── stream_pipeline.rs # Layer-streamed execution pipeline
├── kernels/            # Hardware-specific compute kernels
│   ├── ternary_gemm_cpu.rs # CPU reference: naive/packed/parallel matvec
│   └── kernel_dispatch.rs  # Backend dispatch (CPU/CUDA/HIP/SYCL)
├── memory/             # Holographic Distributed Memory
│   ├── hdm.rs          # 10,000-bit HyperVector operations
│   ├── hdm_codebook.rs # Token/role/position codebooks
│   ├── hdm_binding.rs  # Fact encoding (S-P-O triples)
│   └── hdm_retrieval.rs# Episodic + working memory
├── decoding/           # Multi-Perspective Decoding
│   ├── perspective_config.rs # 4 perspective types + application
│   ├── agreement.rs    # JSD agreement + acceptance policies
│   ├── mpd.rs          # MPD orchestrator with resampling
│   └── calibration.rs  # ECE/MCE calibration analysis
├── learning/           # Forward-Mode Expert Adaptation
│   ├── jvp.rs          # Forward-mode JVP computation
│   ├── lora.rs         # Rank-4 LoRA adapters
│   ├── evolutionary.rs # Natural Evolution Strategies (NES)
│   └── fmea.rs         # FMEA orchestrator
├── safety/             # Safety Polytope Projection
│   ├── polytope.rs     # Safety Polytope definition
│   ├── anchors.rs      # Safety anchor management
│   ├── halfspace.rs    # Halfspace constraints
│   └── projection.rs   # Dykstra's alternating projection
└── runtime/            # Inference pipeline
    ├── provider.rs     # NVMe weight provider + LRU cache
    ├── pipeline.rs     # Full inference pipeline orchestrator
    └── health.rs       # Health monitoring + diagnostics
```

## Building

```bash
# Debug build
cargo build

# Release build (with LTO)
cargo build --release

# Run tests
cargo test

# Run with specific backend
cargo build --release --features cuda
```

## Usage

```bash
# Basic generation
perspective --model-dir ./model -n 256 "Once upon a time"

# With health diagnostics
perspective --model-dir ./model --health "Explain quantum computing"

# Enable online learning
perspective --model-dir ./model --learn "The capital of France is"
```

## Documentation

Detailed architecture documentation is in [docs/v2/](docs/v2/):

| Document | Description |
|---|---|
| [00_INDEX](docs/v2/00_INDEX.md) | Architecture overview and document index |
| [01_PDR](docs/v2/01_PDR.md) | Parallel Decayed Recurrence specification |
| [02_MANIFOLD_ROUTING](docs/v2/02_MANIFOLD_ROUTING.md) | Flat torus routing on T² |
| [03_TERNARY_EXECUTION](docs/v2/03_TERNARY_EXECUTION.md) | 1.58-bit ternary compute engine |
| [04_HDM](docs/v2/04_HDM.md) | Holographic Distributed Memory |
| [05_MPD](docs/v2/05_MPD.md) | Multi-Perspective Decoding |
| [06_FMEA](docs/v2/06_FMEA.md) | Forward-Mode Expert Adaptation |
| [07_SPP](docs/v2/07_SPP.md) | Safety Polytope Projection |
| [08_MEMORY_BUDGET](docs/v2/08_MEMORY_BUDGET.md) | Detailed memory analysis |
| [09_SAFETY](docs/v2/09_SAFETY.md) | Safety architecture |
| [10_TRAINING](docs/v2/10_training_plan.md) | 7-stage training plan |
| [11_INFERENCE](docs/v2/11_inference_pipeline.md) | Per-token inference pipeline |
| [12_VALIDATION](docs/v2/12_validation_v2.md) | 6-tier validation framework |
| [13_ISSUE_MATRIX](docs/v2/13_issue_matrix.md) | 25 LLM issues × PERSPECTIVE solutions |

## Key Design Principles

1. **No multiplications in the forward pass** — ternary weights mean all compute is add/sub/skip
2. **No backpropagation** — forward-mode JVP + NES provides O(1) memory gradient estimation
3. **Non-differentiable safety** — the safety polytope projection blocks all gradient-based attacks
4. **Built-in hallucination detection** — multi-perspective agreement catches inconsistent outputs
5. **O(1) per-token recurrence** — PDR replaces O(n²) attention with constant-time state updates
6. **Streaming execution** — experts flow from NVMe through RAM to VRAM via double-buffered DMA
7. **Episodic memory** — HDM provides fact storage and retrieval without parameter modification

## License

MIT
