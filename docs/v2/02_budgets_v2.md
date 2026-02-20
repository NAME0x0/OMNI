# § 02 — Hardware Budgets

> Every byte accounted for.  4 GB VRAM.  32 GB RAM.  NVMe for the rest.

---

## 1  Design Constraints

| Resource | Available | Design target | Slack |
|----------|-----------|--------------|-------|
| GPU VRAM | 4 096 MB | ≤ 2 684 MB | 34.5 % |
| System RAM | 32 768 MB | ≤ 26 000 MB | 20.6 % |
| NVMe | 512+ GB | ~205 GB | — |
| PCIe bandwidth | 16 GB/s (Gen 4 ×16) | ~7 GB/s sustained | — |

Slack is intentional: memory allocators fragment, drivers reserve VRAM,
and the OS can spike.  A 34 % VRAM margin prevents OOM under load.

---

## 2  VRAM Budget (4 096 MB)

### 2.1  Itemised Allocation

| # | Component | Precision | Parameters | MB | Notes |
|---|-----------|-----------|-----------|-----|-------|
| 1 | PDR weights (W_p, W_k, W_v, W_q, W_o) ×60 | 2-bit | 3.15 B | 788 | 3 dense d×d + 2 low-rank d×r per layer |
| 2 | Windowed GQA weights (Q, K, V, O) ×20 | 2-bit | 838 M | 210 | 32 Q-heads, 8 KV-heads, d_h=128 |
| 3 | Shared FFN weights (W_g, W_u, W_d) ×20 | 2-bit | 2.71 B | 677 | SwiGLU 4096→11008→4096 |
| 4 | Embedding table (32K × 4096) | INT8 | 131 M | 131 | Tied with output head |
| 5 | PDR recurrent state (S_t) ×60 | FP16 | — | 120 | 60 × (4096 × 256) × 2 bytes |
| 6 | GQA KV-cache (512-window) ×20 | FP16 | — | 42 | 20 × 8 heads × 128 dim × 512 pos × 2 × 2B |
| 7 | Expert layer double-buffer | 1.58-bit | 135.3 M | 54 | 2 × one expert-layer (~27 MB each) |
| 8 | Ternary dequant scratch | FP16 | — | 100 | Unpacked expert layer for compute |
| 9 | Activation scratch | FP16 | — | 128 | Intermediate activations, GEMM workspace |
| 10 | RMSNorm scale vectors ×80 | FP16 | 0.66 M | 1.3 | 80 × 2 norms × 4096 × 2B |
| 11 | Router projection (all layers) | FP16 | 0.66 M | 1.3 | 80 × (4096 → 2) × 2B |
| 12 | GPU runtime (driver, cuBLAS, allocator) | — | — | 96 | Conservative estimate |
| | **TOTAL** | | | **2 348** | |
| | **Headroom** | | | **1 748** | 42.7 % free |

### 2.2  Derivations

**PDR weights (item 1):**
For `d=4096` and `r=256`, each PDR layer has:
- `W_p`: `d×d` = 16.78M
- `W_v`: `d×d` = 16.78M
- `W_o`: `d×d` = 16.78M
- `W_k`: `d×r` = 1.05M
- `W_q`: `d×r` = 1.05M

Per layer: `16.78 + 16.78 + 16.78 + 1.05 + 1.05 = 52.44M` params.

At 2-bit: `52.44M × 60 × 0.25 bytes = 786.6 MB ≈ 788 MB` ✓

**GQA weights (item 2):**
- Q: 4096 → 4096 (32 heads × 128) = 16.78M
- K: 4096 → 1024 (8 heads × 128) = 4.19M
- V: 4096 → 1024 (8 heads × 128) = 4.19M
- O: 4096 → 4096 = 16.78M
- Per layer: 41.94M. ×20 = 838.8M
- At 2-bit: 838.8M × 0.25 = 209.7 MB ≈ 210 MB ✓

**Shared FFN (item 3):**
- W_gate: 4096 × 11008 = 45.09M
- W_up:   4096 × 11008 = 45.09M
- W_down: 11008 × 4096 = 45.09M
- Per layer: 135.27M. ×20 = 2705.4M
- At 2-bit: 2705.4M × 0.25 = 676.4 MB ≈ 677 MB ✓

**Expert double-buffer (item 7):**
One expert-layer = 135.27M params at 1.58-bit = 135.27M × 0.1975 bytes = 26.72 MB ≈ 27 MB.
Double-buffer = 2 × 27 = 54 MB ✓

**PDR state (item 5):**
S_t ∈ R^{d×r} = 4096 × 256 = 1,048,576 values × 2 bytes (FP16) = 2.0 MB per layer.
60 layers × 2.0 = 120.0 MB.
Plus γ state (4096 × 2B × 60) = 0.48 MB. Total ≈ 120.5 MB.
Budgeted as **120 MB** in the table (rounded, with γ-state absorbed in slack).

**GQA KV-cache (item 6):**
Per layer: 8 KV-heads × 128 dim × 512 positions × 2 (K+V) × 2 bytes = 2.097 MB.
20 layers × 2.097 = 41.94 MB ≈ 42 MB ✓

---

## 3  RAM Budget (32 768 MB)

| # | Component | GB | Notes |
|---|-----------|-----|-------|
| 1 | Operating system + drivers | 4.0 | Conservative for Windows 11 |
| 2 | Hot expert cache (top-8) | 12.8 | 8 × 8.12B × 0.1975 B/param = 1.604 GB each |
| 3 | Shared weights (host mirror) | 1.8 | Full copy of VRAM-resident weights for reload |
| 4 | Holographic memory banks | 2.5 | 2000 banks × 1.25 KB superposition + metadata |
| 5 | HDM text store | 2.5 | ~50K entries × avg 50 bytes + indices |
| 6 | DMA staging buffers | 0.5 | PCIe ring buffer for expert streaming |
| 7 | LoRA adapters (FMEA) | 0.004 | 60 layers × rank 4 × 2 matrices × 4096 × 4 × 2B |
| 8 | Safety polytope (SPP) | 0.012 | 1000 anchors × 4096 dim × FP32 |
| 9 | Router state + manifold | 0.05 | 128 expert coordinates + distance cache |
| 10 | Application code + stack | 0.5 | Rust binary, thread stacks |
| 11 | Miscellaneous / fragmentation | 1.3 | Allocator overhead, page tables |
| | **TOTAL** | **~26.0** | |
| | **Headroom** | **~6.8** | 20.7 % free |

### 3.1  Hot Expert Selection

The 8 hottest experts (by recent routing frequency) are pinned in RAM.
When the router selects a hot expert, its layers stream directly from RAM
to VRAM at PCIe speed (~16 GB/s theoretical, ~7 GB/s sustained).

When a cold expert is needed:
1. Load from NVMe → RAM staging buffer (1.6 GB at ~6 GB/s = ~267 ms)
2. Evict the least-recently-used hot expert
3. Promote the new expert to hot cache
4. Stream layers to VRAM as usual

Expected cold-expert frequency: < 5 % of tokens (Zipf routing distribution).

---

## 4  NVMe Budget

| Component | Size | Format |
|-----------|------|--------|
| 128 expert weight files | 205 GB | Ternary-packed (1.58 bit/param) |
| Expert delta files (manifold neighbours) | ~40 GB | Sparse diff format |
| HDM overflow store | 5 GB | Binary vector + text |
| Model metadata + config | < 1 MB | JSON / MessagePack |
| **TOTAL** | **~250 GB** | |

Each expert file: 8.12B params × 1.58 bits ÷ 8 = **1.604 GB**.

Delta files between manifold-adjacent experts: ~10 % of full expert size
(most weights shared) = ~160 MB per pair.  128 experts × ~2 neighbours
each = ~256 delta files × 160 MB = ~40 GB.

---

## 5  PCIe Bandwidth Budget

### 5.1  Per-Token Transfer

| Transfer | Direction | Size | Time @ 7 GB/s |
|----------|-----------|------|---------------|
| Expert layer (full) | RAM → VRAM | 27 MB | 3.86 ms |
| Expert layer (delta) | RAM → VRAM | 2.8 MB | 0.40 ms |
| Expected (70% delta) | RAM → VRAM | ~10 MB avg | 1.43 ms |
| × 60 expert layers | | | **85.7 ms** |
| KV-cache update (GQA) | VRAM → VRAM | negligible | — |
| Activation readback (for MPD) | VRAM → RAM | 32 KB | < 0.01 ms |

With double-buffering, PCIe transfer of layer N+1 overlaps GPU compute of
layer N.  Since compute (~0.14 ms) ≪ transfer (~1.43 ms), the pipeline is
**PCIe-bound** and total time ≈ 60 × 1.43 ms = **85.7 ms** for expert layers.

Add GQA + shared FFN compute (~20 × 0.35 ms = 7 ms) + embedding + output
(~0.03 ms) = **~93 ms per token = ~10.8 tok/s**.

### 5.2  Best Case (fully cached, sequential expert reuse)

If the same expert is reused for a layer (common in continued text):
- Delta = 0 bytes → no transfer needed
- Compute only: 0.14 ms per expert layer
- Total: 60 × 0.14 + 7 = 15.4 ms → **65 tok/s** (GPU-bound)

### 5.3  Worst Case (all cold misses, full loads)

- Full expert layer: 27 MB / 7 GB/s = 3.86 ms
- 60 layers: 231.4 ms
- Plus GQA: +7 ms
- Total: ~238 ms → **4.2 tok/s**

Mitigation: prefetch next-token's experts based on routing prediction
from the PDR state at layer 30 (halfway point).

---

## 6  Thermal & Power Envelope

| Component | Typical power | Role |
|-----------|--------------|------|
| GPU (e.g., RTX 3060 4GB) | 170 W | Ternary GEMM, attention, norms |
| CPU (PCIe controller) | 20 W | DMA orchestration, HDM retrieval |
| RAM (DDR4/5) | 5 W | Expert cache, weights mirror |
| NVMe SSD | 7 W | Cold expert storage |
| **System total** | **~202 W** | |

This is well within the thermal envelope of a standard desktop or laptop
with a discrete GPU.

---

## 7  Scaling Considerations

| If you have… | Adjustment |
|--------------|-----------|
| 6 GB VRAM | Cache 4 expert layers in VRAM (instead of 2). Reduces PCIe transfers by ~50 %. ~16 tok/s. |
| 8 GB VRAM | Keep all 20 GQA layers + their KV-cache in VRAM permanently. ~20 tok/s. |
| 12+ GB VRAM | Pin entire hot experts in VRAM. Minimal PCIe. ~40+ tok/s. |
| 64 GB RAM | Cache all 128 experts in RAM. Zero NVMe reads. ~14 tok/s floor guaranteed. |
| 2× NVMe (RAID-0) | Double cold-expert load speed. Reduce worst case to ~6 tok/s. |
| PCIe Gen 5 | ~2× bandwidth → expert streaming at ~2.8 ms avg → ~20 tok/s. |

---

*Next: [§ 03 Perspective Decay Recurrence](03_perspective_decay.md)*
