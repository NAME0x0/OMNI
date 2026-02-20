# § 05 — Layer-Streamed Ternary Execution

> No multiplication needed.  Each weight is add, subtract, or skip.
> Stream one layer at a time.  The GPU never waits.

---

## 1  Native Ternary Weights

### 1.1  The Format

Every expert parameter is one of three values: $\{-1, 0, +1\}$.

This is **not post-hoc quantisation** of a float model.  The weights are
trained natively in this format using Straight-Through Estimators (STE)
during back-propagation, following the BitNet b1.58 methodology:

```
Forward pass:  w_ternary = sign(w_latent) * round(|w_latent|)  ∈ {-1, 0, +1}
Backward pass: ∂L/∂w_latent = ∂L/∂w_ternary  (straight-through)
```

### 1.2  Information Content

Each ternary value carries $\log_2(3) = 1.585$ bits of information.
Packing: 5 trits fit in 8 bits ($3^5 = 243 \leq 256$).

| Precision | Bits/param | 1B params | 8.12B params (1 expert) |
|-----------|-----------|-----------|------------------------|
| FP16 | 16.0 | 2.00 GB | 16.24 GB |
| INT8 | 8.0 | 1.00 GB | 8.12 GB |
| 4-bit | 4.0 | 0.50 GB | 4.06 GB |
| 2-bit | 2.0 | 0.25 GB | 2.03 GB |
| **Ternary** | **1.585** | **0.198 GB** | **1.604 GB** |
| 1-bit binary | 1.0 | 0.125 GB | 1.015 GB |

Ternary is 10× denser than FP16 and 2.5× denser than 4-bit quantisation,
while retaining all three sign states (negative, zero, positive).

### 1.3  Packing Layout

```
Byte layout (5 trits per byte):
┌─────────────────────────┐
│ t₄ t₃ t₂ t₁ t₀ │ padding │    1 byte = 5 trits
│ val = t₀ + 3*t₁ + 9*t₂ + 27*t₃ + 81*t₄   (base-3 encoding, 0-242)
└─────────────────────────┘

File layout for one expert-layer:
┌─────────────┬─────────────┬─────────────┐
│ W_gate      │ W_up        │ W_down      │
│ 45.09M trits│ 45.09M trits│ 45.09M trits│
│ = 9.02 MB   │ = 9.02 MB   │ = 9.02 MB   │
└─────────────┴─────────────┴─────────────┘
Total per expert-layer: 135.27M trits = 27.05 MB
```

---

## 2  Ternary GEMM Kernels

### 2.1  The Key Insight

Ternary matrix-vector multiplication requires **zero multiplications**:

$$y_i = \sum_j W_{ij} \cdot x_j = \sum_{j: W_{ij}=+1} x_j - \sum_{j: W_{ij}=-1} x_j$$

Each weight contributes an addition, a subtraction, or nothing (if zero).
Since ~50% of ternary weights are zero after training, we only process
half the entries — plus each operation is an add/sub instead of a multiply.

### 2.2  CUDA Kernel (Primary Target)

```cuda
// Pseudocode — full kernel in src/kernels/ternary_gemm_cuda.cu
__global__ void ternary_matvec(
    const uint8_t* __restrict__ W_packed,  // packed ternary weights
    const half* __restrict__ x,            // input vector (FP16)
    half* __restrict__ y,                  // output vector (FP16)
    int M, int N                           // output dim, input dim
) {
    // Each warp processes one output row
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    
    float acc = 0.0f;
    
    // Process 5 trits per byte, 32 threads process 160 trits per iteration
    for (int col_byte = threadIdx.x; col_byte < (N + 4) / 5; col_byte += 32) {
        uint8_t packed = W_packed[row * ((N + 4) / 5) + col_byte];
        
        // Unpack 5 trits
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            int col = col_byte * 5 + k;
            if (col >= N) break;
            
            int trit = packed % 3;  // 0, 1, 2 → maps to 0, +1, -1
            packed /= 3;
            
            if (trit == 1)      acc += __half2float(x[col]);
            else if (trit == 2) acc -= __half2float(x[col]);
            // trit == 0: skip (50% of the time)
        }
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    
    if (threadIdx.x == 0)
        y[row] = __float2half(acc);
}
```

### 2.3  Performance Model

For a SwiGLU FFN layer (gate + up projection: 4096 → 11008, down: 11008 → 4096):

| Metric | FP16 dense | Ternary (ours) |
|--------|-----------|---------------|
| Total MAC ops | 135.3 M | 0 (add/sub only) |
| Effective ops (50% zero) | 135.3 M | ~67.6 M add/sub |
| Memory accessed (weights) | 270.5 MB | 27.1 MB |
| Arithmetic intensity | ~0.5 FLOP/byte | ~2.5 op/byte |
| Estimated GPU time (RTX 3060) | 8.1 ms | **0.14 ms** |

The 58× speedup comes from:
- 10× less data movement (ternary packing)
- ~2× less compute (zero skipping)
- ~3× per-op savings (add vs FMA)

### 2.4  Multi-Backend Support

| Backend | File | Status | Notes |
|---------|------|--------|-------|
| CUDA (NVIDIA) | `ternary_gemm_cuda.cu` | Primary | Uses warp shuffles, shared memory tiling |
| HIP (AMD) | `ternary_gemm_hip.cpp` | Ported | Near-identical to CUDA via HIP compatibility |
| SYCL (Intel) | `ternary_gemm_sycl.cpp` | Planned | For Intel Arc / Data Center GPUs |
| CPU (fallback) | `ternary_gemm_cpu.c` | Reference | AVX2/AVX-512 SIMD, no GPU required |

All backends expose the same FFI interface via `kernel_dispatch.rs`.

---

## 3  Layer Streaming Pipeline

### 3.1  Double-Buffered DMA

The core mechanism that makes 1.05T parameters fit in 4 GB VRAM:

```
Time →
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
DMA:    │ Load L1  │ │ Load L2  │ │ Load L3  │ │ Load L4  │ ...
        └─────────┘ └─────────┘ └─────────┘ └─────────┘
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
GPU:    │  (idle)  │ │Compute L1│ │Compute L2│ │Compute L3│ ...
        └─────────┘ └─────────┘ └─────────┘ └─────────┘
                          ↑
                     GPU starts as soon as L1 finishes loading
```

Two VRAM buffers (A and B), each 27 MB:
- While GPU computes from buffer A, DMA fills buffer B
- Swap at end of each layer
- Latency = max(DMA time, compute time) per layer

### 3.2  Pipeline Stages per Expert Layer

```
Stage 1: DMA Transfer (RAM → VRAM)
  - Source: hot expert cache in RAM (pinned memory)
  - Size: 27 MB (full) or 2.7 MB (delta)
  - Time: 3.86 ms (full) / 0.39 ms (delta) @ 7 GB/s sustained
  - Uses CUDA async memcpy on dedicated DMA stream

Stage 2: Trit Unpacking (GPU kernel)
  - Expand 5-trits-per-byte to FP16 lookup
  - Input: 27 MB packed → output: 270 MB FP16 in scratch
  - Actually: we DON'T fully unpack. The ternary GEMM kernel
    reads packed trits directly and accumulates in FP32.
  - Unpack time: 0 ms (fused with GEMM)

Stage 3: SwiGLU Computation (ternary GEMM)
  - gate = SiLU(TernaryMV(W_gate, h))   →  11008 elements
  - up   = TernaryMV(W_up, h)            →  11008 elements
  - out  = gate ⊙ up                    →  11008 elements
  - down = TernaryMV(W_down, out)        →  4096 elements
  - Time: ~0.14 ms (add/sub only, 50% sparsity)

Stage 4: Residual Add
  - x += down
  - Time: negligible
```

### 3.3  Expert Layer Caching

The pipeline maintains a **layer-expert cache** in VRAM:

```rust
struct LayerExpertCache {
    /// Which expert is currently in buffer A and B
    buffer_a_expert: Option<(u16, u16)>,  // (expert_id, layer_idx)
    buffer_b_expert: Option<(u16, u16)>,
    
    /// If the same expert is needed again, skip DMA entirely
    fn needs_transfer(&self, expert_id: u16, layer_idx: u16) -> TransferKind {
        if self.buffer_a_expert == Some((expert_id, layer_idx)) { Skip }
        else if self.is_neighbour(expert_id, layer_idx) { Delta }
        else { Full }
    }
}
```

---

## 4  NVMe → RAM Expert Loading

When a cold expert is needed (not in the hot-8 cache):

### 4.1  Load Pipeline

```
1. NVMe read:  1.604 GB at 6 GB/s = 267 ms
2. Decompress: ternary data is already in packed format (no decompression)
3. Pin memory:  Register the RAM buffer for DMA access
4. Evict LRU:   Free the least-recently-used hot expert from RAM
```

### 4.2  Prefetch Strategy

The Manifold Router runs at layer 30 (halfway through the model).
Its routing prediction for future layers provides a **prefetch hint**:

```
At layer 30:
  predicted_experts = manifold_route_predict(h_30, layers=[31..79])
  for each predicted expert not in hot cache:
    async_nvme_load(expert, priority=BACKGROUND)
```

This gives the NVMe ~35 ms to start fetching (while the GPU processes
layers 31–60), potentially hiding most of the 267 ms load time for
the least popular experts.

### 4.3  Expert Usage Statistics

The runtime tracks routing frequency per expert with exponential smoothing:

$$\text{freq}_i \leftarrow 0.999 \cdot \text{freq}_i + 0.001 \cdot \mathbb{1}[\text{selected}_i]$$

The top-8 by frequency are kept in the hot cache.  Cache turnover is
expected to be slow (experts change over conversation topic shifts, not
per-token).

---

## 5  Ternary Sparsity Patterns

### 5.1  Expected Distribution After Training

Based on BitNet b1.58 results scaled to our architecture:

| Weight value | Proportion | Meaning |
|-------------|-----------|---------|
| 0 | ~50% | Skip — no operation |
| +1 | ~25% | Add input to accumulator |
| -1 | ~25% | Subtract input from accumulator |

### 5.2  Structural Sparsity

The 50% zero rate is not uniform random — it has structure:

- Columns corresponding to rare features tend to be mostly zero
- Rows for less-active output dimensions are sparser
- The gate projection (W_gate) tends to be sparser than W_up or W_down

This structure is exploited in the CPU kernel using column-skip
optimisations (check if a column is all-zero before processing).

---

## 6  Precision Analysis

### 6.1  Accumulation

All ternary GEMM kernels accumulate in **FP32** and cast to FP16 only at
the final output.  This prevents catastrophic cancellation from the
large number of additions/subtractions.

### 6.2  Non-Ternary Components

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Expert FFN weights | Ternary (1.58-bit) | Massively compressed for streaming |
| Shared FFN weights | 2-bit (GPTQ) | Always in VRAM; denser but few layers |
| Attention weights (GQA) | 2-bit (GPTQ) | Always in VRAM |
| PDR weights | 2-bit (GPTQ) | Always in VRAM |
| Embedding table | INT8 | Embedding lookups are memory-bound |
| Activations | FP16 | Full precision for intermediate computation |
| PDR state | FP16 | Accumulated over many tokens, needs precision |
| LoRA adapters | FP16 | Tiny; precision matters for adaptation |
| RMSNorm scales | FP16 | Tiny but sensitive |

---

## 7  Hardware Compatibility

### 7.1  Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA 7.0+ (GTX 1060 6GB*) | RTX 3060+ |
| GPU VRAM | 4 GB | 6 GB+ |
| RAM | 32 GB | 64 GB |
| NVMe | 250 GB, ≥ 3 GB/s read | 500 GB, ≥ 5 GB/s |
| CPU | 4-core x86_64 with AVX2 | 8-core with AVX-512 |

*With a 6 GB GPU, the extra 2 GB allows caching more expert layers.

### 7.2  CPU-Only Mode

The CPU kernel (`ternary_gemm_cpu.c`) enables running without any GPU:
- Expert layers computed on CPU using AVX2/AVX-512
- All 32 GB RAM available for expert caching (no VRAM split)
- Expected throughput: ~1-2 tok/s (CPU-bound)
- Useful for testing, CI, and GPU-less servers

---

*Next: [§ 06 Holographic Memory](06_holographic_memory.md)*
