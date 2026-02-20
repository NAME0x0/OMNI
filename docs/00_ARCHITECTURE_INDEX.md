# OMNIS-SINGULARITY: Physically Feasible AGI Runtime Architecture

## Target Hardware Invariants
| Resource | Value | Notes |
|----------|-------|-------|
| GPU VRAM | 4096 MB | >=10% slack reserved (409 MB) |
| Host RAM | 32768 MB | Includes OS overhead |
| PCIe | 4.0 x16 | Effective BW: 25 GB/s (justified below) |
| CPU | Consumer x86-64 | 8C/16T assumed baseline |
| Network | None (local-only) | No RDMA, no cloud |

## Document Index
1. [Definitions & Equivalence Metric](./01_definitions.md)
2. [Budget Tables](./02_budgets.md)
3. [Cognitive Engine](./03_cognitive_engine.md)
4. [Quantization Strategy](./04_quantization.md)
5. [Continual Learning](./05_continual_learning.md)
6. [Topological Memory](./06_topological_memory.md)
7. [Truth Grounding](./07_truth_grounding.md)
8. [Agent Runtime](./08_agent_runtime.md)
9. [Validation Plan](./09_validation.md)

## PCIe Bandwidth Justification
PCIe 4.0 x16 theoretical: 31.5 GB/s unidirectional.
Effective after protocol overhead, TLP headers, and realistic DMA contention: **25 GB/s**.
This is a single concrete value used throughout all budgets.
Source: measured throughput on RTX 3060/RX 6700 XT class cards under sustained DMA transfer
with pinned host memory. Burst can reach 28 GB/s; sustained under mixed read/write drops to ~25.
