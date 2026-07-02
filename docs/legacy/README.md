# Legacy v1 Architecture — OMNIS-SINGULARITY (Superseded)

> **Status: superseded.** Nothing in this directory describes the current system.
> The active specification is [`docs/v2/`](../v2/00_V2_INDEX.md) ("PERSPECTIVE"),
> which is what `src/` implements.

## What v1 Was

OMNIS-SINGULARITY was the first architecture drafted for this project: a
**~2.5B-parameter MoE** built around **GLA (gated linear attention)** with a large
augmentation stack, targeting the same consumer-hardware envelope (4 GB VRAM +
32 GB RAM). Its distinguishing pieces:

- GLA sub-quadratic attention with windowed attention on the last 2 layers
- 2-bit GPTQ quantization (after an infeasibility analysis of stronger targets)
- HNSW + LSH "topological memory" (2M+ entries, ~10.7 GB RAM)
- Three-tier truth-grounding verifier cascade (150M verifier → main model → evidence vote)
- 4-layer safety stack (sanitizer → classifier → RepE steering → constitutional audit)
- CPU-side multimodal encoders (MobileCLIP-S2 vision, Whisper-Tiny audio)
- Circuit breakers + 5-level degradation ladder ("self-healing")
- An invented "1.5T Reasoning Equivalence (CRS)" headline metric

## Why It Was Superseded

v2 (PERSPECTIVE) replaced the core model with a fundamentally different design:
PDR recurrence instead of GLA, a literal 1.05T-parameter sparse MoE with
layer-streamed ternary experts instead of a 2.5B resident model, HDM holographic
memory instead of HNSW/LSH, and geometric (polytope) output safety instead of a
classifier stack. The two specifications contradict each other in scope, parameter
counts, memory mechanisms, and safety claims — v1 is retained for design history
only and its claims are not maintained.

Some v1 subsystems remain good candidates for v2 extensions (verifier cascade,
degradation ladder, multimodal front-ends); see `docs/v2/14_extensions.md`.

## File Index

| File | Original title |
|---|---|
| [00_ARCHITECTURE_INDEX.md](00_ARCHITECTURE_INDEX.md) | Physically Feasible AGI Runtime Architecture |
| [01_definitions.md](01_definitions.md) | Definitions & Equivalence Metric |
| [02_budgets.md](02_budgets.md) | Budget Tables |
| [03_cognitive_engine.md](03_cognitive_engine.md) | Cognitive Engine (Sub-Quadratic) |
| [04_quantization.md](04_quantization.md) | Quantization Strategy |
| [05_continual_learning.md](05_continual_learning.md) | Continual Learning |
| [06_topological_memory.md](06_topological_memory.md) | Topological Memory |
| [07_truth_grounding.md](07_truth_grounding.md) | Truth Grounding Verifier |
| [08_agent_runtime.md](08_agent_runtime.md) | Agent Runtime (Rust 2021 + C/C++ FFI) |
| [09_validation.md](09_validation.md) | Validation Plan |
| [10_alignment_safety.md](10_alignment_safety.md) | Alignment & Safety System |
| [11_adversarial_robustness.md](11_adversarial_robustness.md) | Adversarial Robustness & Prompt Injection Defense |
| [12_multimodal.md](12_multimodal.md) | Multimodal Capability Under Extreme VRAM Constraints |
| [13_self_healing.md](13_self_healing.md) | Fixing OMNIS Self-Introduced Problems |
| [14_equity_interpretability.md](14_equity_interpretability.md) | Multilingual Equity, Debiasing, Tokenization & Interpretability |
| [15_reasoning_planning.md](15_reasoning_planning.md) | Instruction Following, Temporal Reasoning, Sycophancy & Planning |
| [16_revised_budgets.md](16_revised_budgets.md) | Revised Budget Reconciliation (All Additions Included) |
| [17_OMNIS_COMPLETE.md](17_OMNIS_COMPLETE.md) | Complete Architecture Specification |
