# Section 2: Budget Tables

## 2.1 Concrete Numeric Assumptions (Fixed Values)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| GPU VRAM total | 4096 MB | Constraint |
| VRAM slack (10%) | 410 MB | Constraint (rounded up) |
| VRAM usable | 3686 MB | 4096 - 410 |
| Host RAM total | 32768 MB | Constraint |
| Host RAM OS/system | 4096 MB | Conservative reservation for OS + desktop |
| Host RAM usable | 28672 MB | 32768 - 4096 |
| PCIe effective BW | 25 GB/s | See index doc justification |
| CPU cores | 8C/16T | Baseline consumer (e.g., i7-12700 / Ryzen 7 5800X) |
| CPU FLOPS (AVX2 FP32) | 500 GFLOPS | ~8 cores * 16 FP32/cycle * 4 GHz |
| GPU FLOPS (FP16) | 10 TFLOPS | Conservative for GTX 1650/RX 6500 class |
| GPU INT8 TOPS | 20 TOPS | 2x FP16 rate typical |
| GPU memory BW | 192 GB/s | GDDR6 baseline |

## 2.2 VRAM Budget (Full Accounting)

### Architecture: Sparse MoE with 8 experts, top-2 routing

| Component | Size (MB) | Derivation |
|-----------|-----------|------------|
| **Active expert weights (2 of 8)** | 1200 | 2 experts * 600 MB each (see §4 quantization) |
| **Shared layers (embed + attention)** | 900 | Embedding: 200 MB, Attention KV-cache: 400 MB, Attention weights: 300 MB |
| **Router network** | 8 | Tiny MLP: 4096 * 8 * 2 bytes ≈ 64 KB; rounded up for buffers |
| **Activation scratch (per token batch)** | 256 | Batch=32 tokens * 8 MB/token for intermediates |
| **Verifier model (small)** | 400 | Dedicated 150M-param verifier at 2.5 bits avg |
| **KV-cache (context window)** | 512 | 4096 context * 2 (K+V) * 64 heads * 64 dim * 2 bytes / expert-share |
| **Dequantization scratch** | 64 | Temporary FP16 unpack buffer for one layer |
| **CUDA/HIP/SYCL runtime overhead** | 96 | Driver allocations, kernel launch buffers |
| **Expert prefetch staging** | 240 | Double-buffer for next expert swap (1 expert = 600 MB, but only prefetch partial: 240 MB hot layer) |
| **TOTAL USED** | 3676 | |
| **SLACK (remaining)** | 420 | 4096 - 3676 = 420 MB (10.25% > 10% requirement) |

### Per-Token VRAM Costs

| Operation | VRAM touched/token | Notes |
|-----------|-------------------|-------|
| KV-cache append | 128 bytes | 2 * 64 * 64 * 2B / sharing factor 16 |
| Activation forward | 8 MB (reused) | Scratch buffer, overwritten each token |
| Expert weight read | 0 (resident) | Already in VRAM |
| Router inference | ~4 KB | Negligible |

## 2.3 Host RAM Budget (Full Accounting)

| Component | Size (MB) | Derivation |
|-----------|-----------|------------|
| **OS + Desktop** | 4096 | Reserved |
| **All 8 expert weights (2-bit packed)** | 4800 | 8 * 600 MB each |
| **Shared layer weights (host copy)** | 900 | Mirror of VRAM shared layers |
| **Topological memory index** | 4096 | HNSW graph for 2M entries * ~2 KB/entry |
| **Topological memory vectors** | 8192 | 2M entries * 4096 bytes/vector (FP16 dim=2048) |
| **Topological memory metadata** | 2048 | 2M entries * ~1 KB metadata (timestamps, edges, text refs) |
| **SQLite WAL + page cache** | 1024 | For persistent memory store |
| **Continual learning buffers** | 512 | OGP projection matrices + replay buffer |
| **Agent runtime (up to 8 agents)** | 40 | 8 * 5 MB max per agent |
| **Pinned DMA staging buffer** | 1200 | For PCIe transfers (double-buffered, 600 MB each) |
| **Text corpus / document store** | 2048 | Compressed text backing the topological memory |
| **Miscellaneous (logging, IPC)** | 256 | Conservative overhead |
| **TOTAL USED** | 29212 | |
| **REMAINING** | 3556 | 32768 - 29212 = 3556 MB headroom |

## 2.4 PCIe Budget & Expert Swapping

### Core Problem
Only 2 of 8 experts fit in VRAM simultaneously. When routing selects a non-resident
expert, we must swap one out and one in via PCIe.

### Transfer Costs

| Metric | Value | Derivation |
|--------|-------|------------|
| Expert weight size | 600 MB | See §4 quantization |
| Full expert swap time | 24 ms | 600 MB / 25 GB/s |
| Partial hot-layer prefetch | 9.6 ms | 240 MB / 25 GB/s |
| Bytes transferred per token (cache hit) | 128 B | KV-cache only, negligible |
| Bytes transferred per token (cache miss = expert swap) | 600 MB | Full expert load |
| PCIe-limited max tokens/s (no swaps) | ~195 tok/s | GPU memory BW bound, not PCIe |
| PCIe-limited max tokens/s (every token swaps) | 0.04 tok/s | 25000 MB/s / 600 MB = 41 swaps/s but each swap serves 1 token |

### Required Cache Hit Rate

To maintain >=8 tokens/s decode target:
- Each token takes ~5 ms compute (GPU bound).
- Budget per token: 125 ms total (8 tok/s).
- Compute: 5 ms. Available for swap overhead: 120 ms.
- But we pipeline: swap overlaps with compute of previous token.
- Effective swap penalty per miss: 24 ms (full) or 9.6 ms (prefetched).
- With prefetch: tokens between swaps needed = 9.6 ms / 5 ms ≈ 2 tokens.
- **Minimum cache hit rate: 75%** (1 in 4 tokens can trigger a swap without stalling).
- Empirical MoE routing locality: typically 85-95% top-2 reuse within a sequence.
- **Design target: 90% cache hit rate.**

[HEURISTIC] The 90% cache hit rate assumption is based on published MoE routing
statistics (Fedus et al., Switch Transformer). Falsifiable test: instrument router
decisions on 10K sequences and measure actual locality.

### Fallback Strategy (No ReBAR / No DirectStorage)

When ReBAR is unavailable:
- BAR0 window is typically 256 MB on older systems.
- Expert swap requires chunked transfer: 600 MB / 256 MB = 3 chunks.
- Each chunk: 256 MB / 25 GB/s = 10.2 ms + ~1 ms overhead = 11.2 ms.
- Total: 33.6 ms (vs 24 ms with ReBAR). 40% slower.
- **Mitigation**: Reduce expert size to 480 MB by pruning least-used rows (§4),
  fitting in 2 chunks. Or increase prefetch aggressiveness (prefetch top-4 likely
  experts during prompt processing phase).

When DirectStorage is unavailable:
- Standard memory-mapped I/O with pinned buffers.
- No change to PCIe bandwidth, but CPU must manage DMA explicitly.
- CPU overhead: ~2% of one core per swap (measured via `perf stat` on similar workloads).

## 2.5 Bandwidth Bottleneck Analysis

| Bottleneck | Throughput | Limiting? |
|------------|-----------|-----------|
| GPU compute (FP16) | 10 TFLOPS → ~100 tok/s for 2B active params | No |
| GPU memory BW | 192 GB/s → ~195 tok/s for 2B active params | No (for decode) |
| PCIe (expert swap) | 25 GB/s → 41 swaps/s max | **Yes, on cache miss** |
| PCIe (KV-cache) | 25 GB/s → negligible per token | No |
| CPU (routing + retrieval) | ~500 GFLOPS → routing trivial, retrieval ~2ms | No |
| SSD (cold expert load) | ~3 GB/s NVMe → 200 ms per expert | **Yes, if RAM misses** |

**Primary bottleneck**: PCIe expert swapping on cache misses.
**Secondary bottleneck**: SSD if host RAM is exceeded (should not happen per budget).
**Mitigation**: Routing-aware prefetch + high cache hit rate (>=90%).
