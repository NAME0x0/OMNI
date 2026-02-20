# § 01 — Model Topology

> 80 layers.  60 recurrent + 20 windowed-attention.
> 128 experts, top-1 per token.  1.05 T total parameters.

---

## 1  High-Level Structure

The Perspective model is an **80-layer** transformer-variant organised into
two interleaved *clock speeds*:

| Clock | Layers | Count | Mechanism | Purpose |
|-------|--------|-------|-----------|---------|
| **Fast** | 0–79 (even) | 60 | Perspective Decay Recurrence (PDR) | Compress unbounded context into O(d·r) state |
| **Slow** | 4, 8, 12, …, 76 (every 4th) | 20 | Windowed Grouped-Query Attention (GQA) | Exact token-level recall over last 512 positions |

Every layer is followed by a feed-forward network (FFN):

- **Expert FFN** (60 layers): routed through the Manifold Router → one of
  128 ternary experts activated per token.
- **Shared FFN** (20 layers): dense, 2-bit quantised, always active.

```
Layer index:   0   1   2   3   4   5   6   7   8  ...  76  77  78  79
Sequence:     PDR PDR PDR PDR GQA PDR PDR PDR GQA ... GQA PDR PDR PDR
FFN type:     Exp Exp Exp Exp Shr Exp Exp Exp Shr ... Shr Exp Exp Exp
```

---

## 2  Per-Layer Data Flow

### 2.1  PDR + Expert FFN Layer (×60)

```
x_in ∈ R^{B×d}   (B = batch, d = 4096)
     │
     ├──▶ RMSNorm
     │       │
     │       ▼
     │   PDR(x)  →  o_t ∈ R^{B×d}       [§ 03 Perspective Decay]
     │       │
     │       ▼
     │   x + o_t  (residual)
     │       │
     ├──▶ RMSNorm
     │       │
     │       ▼
     │   ManifoldRoute(h)  → expert_id    [§ 04 Manifold Routing]
     │       │
     │       ▼
     │   StreamExpertFFN(h, expert_id)    [§ 05 Ternary Execution]
     │       │   ┌────────────────────┐
     │       │   │ gate = SiLU(W_g·h) │  SwiGLU FFN
     │       │   │ up   = W_u · h     │  d → 11008 → d
     │       │   │ down = W_d·(gate⊙up)│
     │       │   └────────────────────┘
     │       │
     │       ▼
     └──▶ x + ffn_out  (residual)
              │
              ▼
           x_out → next layer
```

**Key detail:** The expert FFN weights (`W_g`, `W_u`, `W_d`) are ternary
{-1, 0, +1} and streamed one layer at a time from RAM → VRAM via
double-buffered DMA.  Only ~27 MB occupies the GPU at any moment.

### 2.2  GQA + Shared FFN Layer (×20)

```
x_in ∈ R^{B×d}
     │
     ├──▶ RMSNorm
     │       │
     │       ▼
     │   WindowedGQA(x, window=512)      [§ 03.5]
     │       │
     │       │   32 query heads, 8 KV heads (4:1 GQA ratio)
     │       │   Head dim = 128
     │       │   Sliding window of 512 positions
     │       │   RoPE positional encoding
     │       │
     │       ▼
     │   x + attn_out  (residual)
     │       │
     ├──▶ RMSNorm
     │       │
     │       ▼
     │   SharedFFN(h)                     [2-bit quantised, always resident]
     │       │   Same SwiGLU structure: d → 11008 → d
     │       │   Weights are 2-bit GPTQ (NOT ternary)
     │       │
     │       ▼
     └──▶ x + ffn_out  (residual)
              │
              ▼
           x_out → next layer
```

---

## 3  Parameter Census

### 3.1  Shared Parameters (always in VRAM)

| Component | Per layer | Layers | Total params | Precision | Bytes |
|-----------|-----------|--------|-------------|-----------|-------|
| PDR projections (W_p, W_k, W_v, W_q) | 52.5 M | 60 | 3.15 B | 2-bit | 788 MB |
| PDR biases + norms | 0.02 M | 60 | 1.2 M | FP16 | 2.4 MB |
| GQA projections (Q, K, V, O) | 41.9 M | 20 | 838 M | 2-bit | 210 MB |
| GQA norms | 0.02 M | 20 | 0.4 M | FP16 | 0.8 MB |
| Shared FFN (W_g, W_u, W_d) | 135.3 M | 20 | 2.71 B | 2-bit | 677 MB |
| Embedding (32K × 4096) | — | 1 | 131 M | INT8 | 131 MB |
| Output head (tied w/ embed) | — | — | 0 | — | 0 |
| **Shared total** | | | **6.83 B** | | **~1 809 MB** |

### 3.2  Expert Parameters (on NVMe / RAM, streamed to VRAM)

| Component | Per expert-layer | Expert layers | Per expert | 128 experts |
|-----------|-----------------|---------------|-----------|-------------|
| W_gate (4096 → 11008) | 45.1 M | 60 | 2.71 B | 346.4 B |
| W_up (4096 → 11008) | 45.1 M | 60 | 2.71 B | 346.4 B |
| W_down (11008 → 4096) | 45.1 M | 60 | 2.71 B | 346.4 B |
| **Expert total** | **135.3 M** | **60** | **8.12 B** | **1 039 B** |

### 3.3  Grand Total

```
Shared parameters:       6.83 B
Expert parameters:    1 039.4  B   (128 × 8.12 B)
──────────────────────────────
TOTAL:                1 046.2  B   ≈ 1.05 T
Active per token:        14.95 B   (6.83 B shared + 8.12 B × 1 expert)
```

---

## 4  Layer Interleaving Pattern

The 80 layers follow a repeating motif of period 4:

```
Block k (k = 0, 1, … 19):
  Layer 4k+0:  PDR  + Expert FFN
  Layer 4k+1:  PDR  + Expert FFN
  Layer 4k+2:  PDR  + Expert FFN
  Layer 4k+3:  GQA  + Shared FFN
```

This gives exactly **60 PDR layers** and **20 GQA layers** per forward pass.

### Why this ratio?

- PDR is O(1) in sequence length per step and costs ~27 MB VRAM for the
  expert FFN via streaming.  It handles 75 % of depth cheaply.
- GQA provides *exact* token-level lookback within a 512-token sliding
  window.  At 25 % of layers, it keeps the KV-cache at only 42 MB.
- The 3:1 ratio was chosen so that every PDR segment accumulates at most
  3 layers of compression before a GQA layer "sharpens" attention.

---

## 5  Embedding and Output

### 5.1  Input Embedding

- Vocabulary: 32 768 tokens (BPE, trained jointly with the model)
- Embedding dimension: 4 096
- Precision: INT8 (1 byte per value) → 131 MB
- Positional encoding: **none at input**.  Position is injected via:
  - RoPE in the 20 GQA layers (relative position within window)
  - Implicit position from PDR state evolution (recurrence = position-aware)

### 5.2  Output Head

- Weight-tied with input embedding (standard practice)
- Final RMSNorm → linear projection → softmax / sampling
- SPP projection applied to the pre-softmax logit embedding (§ 09)

---

## 6  Normalisation

All normalisation is **RMSNorm** (no learnable bias):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

- $\gamma \in \mathbb{R}^d$ is a learnable scale vector (FP16)
- $\epsilon = 10^{-6}$
- Applied pre-layer (Pre-Norm architecture)

---

## 7  Residual Stream

The model uses a **single residual stream** of width 4 096 throughout all 80
layers.  Both PDR and GQA inject additively:

$$x^{(\ell+1)} = x^{(\ell)} + \text{SeqMech}^{(\ell)}(\text{Norm}(x^{(\ell)})) + \text{FFN}^{(\ell)}(\text{Norm}(\cdot))$$

No gating or scaling on the residual.  Gradient flow is unimpeded across
all 80 layers.

---

## 8  Token-Level Pipeline Summary

For a single token at decode time:

```
 1. Embed token                              →   0.01 ms
 2. For each layer ℓ = 0 … 79:
    a. RMSNorm                               →   0.002 ms
    b. If PDR layer:
       - PDR state update + readout          →   0.07 ms
       - Manifold route                      →   0.01 ms
       - Stream expert layer to VRAM         →   1.12 ms (overlapped)
       - Ternary FFN compute                 →   0.14 ms
    c. If GQA layer:
       - Windowed attention (512 window)     →   0.15 ms
       - Shared FFN (2-bit, in VRAM)         →   0.20 ms
    d. Residual add                          →   0.001 ms
 3. Final RMSNorm + output projection        →   0.02 ms
 4. SPP projection                           →   0.01 ms
 5. Sample token                             →   0.001 ms
────────────────────────────────────────────────────
 Total (PCIe-bound):  ~70–80 ms  →  12–14 tok/s
```

---

*Next: [§ 02 Hardware Budgets](02_budgets_v2.md)*
