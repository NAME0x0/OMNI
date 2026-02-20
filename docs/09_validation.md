# Section 9: Validation Plan

## 9.1 Ablation Studies

Each ablation isolates one subsystem by disabling it and measuring impact on CRS.

### Ablation Matrix

| Ablation ID | Component Disabled | Expected CRS Impact | Measures |
|-------------|-------------------|---------------------|----------|
| A1 | Topological Memory (no retrieval) | CRS drops from 0.78 → ~0.58 | Value of RAG component |
| A2 | Truth Grounding Verifier | CRS drops from 0.78 → ~0.68 | Value of verification |
| A3 | MoE routing (use single expert) | CRS drops from 0.78 → ~0.62 | Value of expert specialization |
| A4 | Continual Learning (freeze LoRA) | CRS unchanged initially; drops over time | Value of adaptation |
| A5 | Lookahead Queue (synchronous) | CRS unchanged; tok/s drops from 8 → ~5 | Value of async speculation |
| A6 | Structured Sparsity (dense 2-bit) | CRS may improve +0.01; VRAM usage +71 MB | Sparsity cost/benefit |
| A7 | LSH Retrieval (HNSW only) | Retrieval latency may increase; CRS ~same | LSH contribution |
| A8 | Multi-pass CoT (single pass) | CRS drops from 0.78 → ~0.65 | Value of iterative reasoning |

### Ablation Protocol

For each ablation:
1. Build system with component disabled (compile-time feature flag)
2. Run full benchmark suite (MMLU, GSM8K, HumanEval, ARC, TruthfulQA)
3. Measure: CRS, tokens/s, VRAM peak, RAM peak, first-token latency
4. Compare against full system baseline
5. Statistical significance: 3 runs per configuration, report mean ± std

### Critical Path Ablations (Must Pass)

| Test | Pass Criterion | If Fails |
|------|---------------|----------|
| A1 (no retrieval) shows CRS drop > 0.10 | ΔΔ CRS > 0.10 | Retrieval is not contributing; redesign memory system |
| A2 (no verifier) shows TruthfulQA drop > 0.10 | ΔΔTQA > 0.10 | Verifier is not working; retrain or redesign |
| A3 (single expert) shows CRS drop > 0.10 | ΔΔCRS > 0.10 | MoE routing is not specializing; check routing collapse |

## 9.2 Stress Tests

### ST-1: PCIe Saturation Test

**Objective**: Verify system behavior when PCIe bandwidth is fully consumed.

```
Setup:
  - Generate input sequences that maximize expert cache misses
    (adversarial routing: every token requests a different expert)
  - Measure: tokens/s, PCIe utilization, GPU stall percentage

Protocol:
  1. Create 1000-token sequences with engineered token patterns
     that trigger different expert routing per token
  2. Run inference at batch_size=1
  3. Monitor: nvidia-smi / rocm-smi dmon for PCIe bandwidth
  4. Record: actual tok/s vs theoretical minimum (0.04 tok/s per §2.4)

Pass criteria:
  - System does not crash or hang
  - Tokens/s degrades gracefully (no cliff)
  - System reports PCIe saturation warning to user
  - After 10 consecutive cache misses, system falls back to
    CPU-only inference for non-resident experts

Failure signature:
  - GPU hangs waiting for PCIe transfer → indicates missing async swap
  - System OOM → indicates double-buffering not working
  - Tokens/s < 0.01 → worse than theoretical minimum, indicates bug
```

### ST-2: Cache Miss Storm Test

**Objective**: Verify topological memory under adversarial access patterns.

```
Setup:
  - Insert 2M entries into topological memory
  - Generate 10,000 queries that are maximally dissimilar to any stored entry
    (random vectors with no semantic content)

Protocol:
  1. Insert 2M entries with realistic embeddings (from Wikipedia)
  2. Generate queries: random unit vectors in R^2048
  3. Run retrieval for all 10K queries
  4. Measure: retrieval latency p50/p95/p99, returned cosine similarities,
     HNSW nodes visited, LSH bucket sizes

Pass criteria:
  - p99 retrieval latency < 10 ms (5x normal)
  - System correctly returns INSUFFICIENT for all queries (low similarity)
  - No memory corruption or segfault
  - LSH bucket sizes within 3x of expected

Failure signature:
  - p99 > 50 ms → HNSW graph is degenerate, needs rebuild
  - Segfault → memory corruption in edge list management
  - All queries return high similarity → embedding space collapsed
```

### ST-3: VRAM Exhaustion Test

**Objective**: Verify system behavior when VRAM approaches limit.

```
Setup:
  - Artificially reduce VRAM limit to 3000 MB (below minimum)
  - Run standard inference workload

Protocol:
  1. Set provider.vram_limit = 3000 MB
  2. Attempt to load model (should fail gracefully)
  3. Set provider.vram_limit = 3686 MB
  4. Run inference with progressively longer contexts (128, 256, 512, ..., 8192)
  5. Monitor: actual VRAM usage vs budget

Pass criteria:
  - At 3000 MB: system refuses to start, clear error message
  - At 3686 MB: system runs successfully up to 4096 context length
  - At context > 4096: system either truncates or reports error
  - VRAM never exceeds 4096 MB (10% slack preserved)

Failure signature:
  - CUDA/HIP OOM error during inference → VRAM budget calculation wrong
  - Silent corruption (wrong outputs) → buffer overflow in activation scratch
```

### ST-4: Concurrent Agent Load Test

**Objective**: Verify agent memory isolation under maximum load.

```
Setup:
  - Spawn 8 agents simultaneously (maximum designed capacity)
  - Each agent runs a tool-heavy workload (code execution + memory queries)

Protocol:
  1. Spawn 8 Agent instances, each with 5 MB RSS limit
  2. Each agent: execute 100 tool calls (mix of code exec, memory insert/query)
  3. Monitor: per-agent RSS, total system RAM, SQLite lock contention
  4. After completion: verify agent memory isolation (agent A cannot see agent B's data)

Pass criteria:
  - No agent exceeds 5 MB RSS (kill policy activates if violated)
  - Total additional RAM for 8 agents < 40 MB
  - No cross-agent data leakage in SQLite
  - SQLite WAL contention does not cause deadlock (timeout after 5s)

Failure signature:
  - Agent RSS > 5 MB → memory leak in tool execution
  - Cross-agent data visible → SQLite database path not isolated
  - Deadlock → WAL checkpoint contention; need busy_timeout tuning
```

### ST-5: Continual Learning Stability Test

**Objective**: Verify no catastrophic forgetting after 10K learning updates.

```
Setup:
  - Run baseline CRS evaluation (pre-learning)
  - Perform 10,000 OGP-LoRA-Stream updates with diverse training samples
  - Re-evaluate CRS

Protocol:
  1. Evaluate CRS on benchmark suite → record as CRS_0
  2. Feed 10K diverse samples (mix of domains) through continual learning
  3. Every 1000 updates, evaluate CRS → record as CRS_i
  4. After 10K updates, evaluate CRS → record as CRS_10K
  5. Plot CRS_i over time

Pass criteria:
  - CRS_10K >= CRS_0 - 0.02 (at most 0.02 degradation)
  - No CRS_i < CRS_0 - 0.05 at any intermediate checkpoint
  - Domain-specific accuracy improves on trained domains
  - Projection basis P does not become rank-deficient (condition number < 1e6)

Failure signature:
  - CRS drops > 0.05 → OGP not protecting important directions; increase P dimension
  - CRS improves on one domain but crashes on others → replay buffer too small
  - P condition number > 1e6 → orthogonalization failing; re-initialize P
```

### ST-6: End-to-End Latency Test

**Objective**: Verify first-token latency and decode throughput targets.

```
Setup:
  - Standard hardware (8C/16T CPU, 4GB VRAM GPU, 32GB RAM)
  - 100 diverse prompts (short, medium, long)

Protocol:
  1. For each prompt:
     a. Measure time from prompt submission to first output token (TTFT)
     b. Measure tokens/second for decode phase
     c. Measure total wall time for complete response
  2. Report p50, p90, p99 for each metric

Pass criteria:
  - TTFT p50 < 1500 ms, p99 < 3000 ms
  - Decode tok/s p50 > 8, p99 > 3
  - No prompt causes > 10s TTFT (timeout)

Failure signature:
  - TTFT p99 > 5000 ms → retrieval or expert loading too slow on cold start
  - Decode p50 < 5 tok/s → PCIe swapping too frequent; routing cache hit rate too low
  - Timeout → deadlock or infinite loop in generation
```

## 9.3 Success Criteria Summary

| Criterion | Metric | Target | Hard Fail |
|-----------|--------|--------|-----------|
| Reasoning quality | CRS | >= 0.78 (stretch), >= 0.70 (minimum) | < 0.65 |
| VRAM usage | Peak VRAM | <= 3686 MB | > 4096 MB |
| RAM usage | Peak RSS | <= 28672 MB | > 32768 MB |
| Decode speed | tok/s | >= 8 (target), >= 3 (minimum) | < 1 |
| First token latency | TTFT | <= 2000 ms (target) | > 5000 ms |
| Expert cache hit rate | % | >= 90% | < 70% |
| Verifier calibration | ECE | <= 0.10 | > 0.20 |
| Agent memory | RSS per agent | <= 5 MB | > 10 MB |
| Continual learning stability | CRS drift | <= -0.02 | < -0.05 |
| Retrieval latency | p99 | <= 5 ms | > 50 ms |

## 9.4 Failure Signatures (Design Not Feasible)

If ANY of the following occur, the architecture must be fundamentally revised:

### F1: VRAM Cannot Fit Minimum Configuration
**Signature**: Even after maximum quantization and sparsity, the minimum model
(2 experts + shared + verifier) exceeds 3686 MB.
**Root cause**: Model dimensions too large for VRAM budget.
**Remediation**: Reduce d from 2048 to 1536 or 1024. This proportionally reduces
all VRAM components but also reduces model capacity (lower CRS ceiling).

### F2: PCIe Bottleneck Prevents Usable Speed
**Signature**: Measured cache hit rate < 70% on realistic workloads, causing
decode speed < 1 tok/s.
**Root cause**: Expert routing has low locality (tokens frequently need non-resident experts).
**Remediation**: (a) Reduce number of experts from 8 to 4 (fit more in VRAM).
(b) Increase expert sharing (shared FFN layers reduce expert-specific routing).
(c) Accept lower MoE benefit and rely more on retrieval.

### F3: Sub-Quadratic Attention Insufficient
**Signature**: GLA-based model cannot match standard attention quality even with
retrieval (CRS < 0.60 on attention-heavy tasks like code generation).
**Root cause**: Fixed-size state loses too much information.
**Remediation**: Hybrid approach: use GLA for first 30 layers, standard attention
for last 2 layers (limited context window of 512). This adds KV-cache cost but
only for 2 layers.

### F4: 2-bit Quantization Destroys Expert Quality
**Signature**: Per-expert accuracy at 2-bit < 80% of FP16 baseline on expert-specific
evaluation sets.
**Root cause**: 2-bit is too aggressive for this model architecture.
**Remediation**: (a) Move to 3-bit (increases VRAM by 50% for expert weights —
may require reducing to 1 active expert). (b) Use 2-bit with larger group size
(128 instead of 64) and higher-precision scales (FP32 instead of FP16).

### F5: Topological Memory Doesn't Improve CRS
**Signature**: Ablation A1 shows < 0.05 CRS improvement from retrieval.
**Root cause**: Retrieved evidence is irrelevant or embedding quality too low.
**Remediation**: (a) Fine-tune embedding model for domain-specific retrieval.
(b) Increase retrieval budget (more candidates, re-ranking step).
(c) If fundamentally broken: replace with simpler RAG (BM25 + vector hybrid).

### F6: Verifier Has No Discrimination
**Signature**: Verifier AUROC < 0.60 on claim verification benchmark.
**Root cause**: 150M-param verifier too small or training data insufficient.
**Remediation**: (a) Increase verifier to 300M params (costs ~200 MB more VRAM;
must reduce elsewhere). (b) Use main model as self-verifier (slower but more capable).
(c) Ensemble: use both dedicated verifier AND main model confidence.

## 9.5 Validation Schedule

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 0: Infrastructure | 2 weeks | Build Rust runtime, FFI layer, SQLite memory |
| Phase 1: Single expert | 2 weeks | Load 1 expert, verify VRAM budget, measure tok/s |
| Phase 2: MoE routing | 2 weeks | Add expert swapping, measure cache hit rate |
| Phase 3: Retrieval | 2 weeks | Build topological memory, measure retrieval latency |
| Phase 4: Verifier | 1 week | Integrate verifier, measure calibration |
| Phase 5: Integration | 2 weeks | Full system, run CRS benchmark |
| Phase 6: Stress tests | 1 week | ST-1 through ST-6 |
| Phase 7: Ablations | 1 week | A1 through A8 |
| Phase 8: Optimization | 2 weeks | Address failures, tune hyperparameters |

**Total estimated validation time: 14 weeks**

Go/no-go decision point after Phase 5: if CRS < 0.60, invoke failure analysis
before proceeding to stress tests.
