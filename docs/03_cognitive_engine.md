# Section 3: Cognitive Engine (Sub-Quadratic)

## 3.1 Architecture Overview

The cognitive engine combines two sub-quadratic mechanisms:
1. **Linear Attention with Gated Recurrence** for sequence processing (O(n) time, O(d) state)
2. **Locality-Sensitive Hashing (LSH) Associative Retrieval** for long-range memory (O(n log n) time)

This avoids standard quadratic self-attention (O(n^2 d)) which at context length
n=4096 and d=2048 would require ~67 GB of attention computation workspace.

## 3.2 Sequence Processing: Gated Linear Attention (GLA)

### Update Equations

For each token position t, with input x_t ∈ R^d:

```
State update (recurrent):
  g_t = σ(W_g · x_t + b_g)           # gate vector, g_t ∈ R^(d_k)
  k_t = W_k · x_t                     # key projection, k_t ∈ R^(d_k)
  v_t = W_v · x_t                     # value projection, v_t ∈ R^(d_v)
  S_t = g_t ⊙ S_{t-1} + k_t ⊗ v_t   # state matrix update, S_t ∈ R^(d_k × d_v)

Query and output:
  q_t = W_q · x_t                     # query projection, q_t ∈ R^(d_k)
  o_t = q_t^T · S_t                   # output, o_t ∈ R^(d_v)
  y_t = W_o · o_t                     # final projection, y_t ∈ R^d
```

Where:
- σ is the sigmoid function
- ⊙ is element-wise (Hadamard) product (gate applied per-row of S)
- ⊗ is outer product
- d = 2048 (model dimension)
- d_k = d_v = 128 (per-head key/value dimension)
- Heads: 16 (total d_k * heads = 2048)

### Complexity Analysis

| Metric | Value | Derivation |
|--------|-------|------------|
| Time per token | O(d_k * d_v) = O(16384) per head | One outer product + matrix-vector |
| Total per token (all heads) | O(16 * 16384) = O(262144) | 16 heads |
| Total per token (FLOPs) | ~524 KFLOP | 2 * 262144 (mul + add) |
| Space (state matrix per head) | d_k * d_v * 2B = 32 KB | FP16 storage |
| Space (all heads) | 16 * 32 KB = 512 KB | Fits in L1/L2 cache |
| Space (all layers, 32 layers) | 32 * 512 KB = 16 MB | Fits in VRAM activation scratch |

**Sequence length independence**: State size is O(d_k * d_v * H * L) = 16 MB
regardless of sequence length. No KV-cache growth for the recurrent path.

### Failure Mode
- **Information loss**: The gated recurrence compresses all history into a fixed-size
  state. For long-range dependencies beyond ~2048 tokens, retrieval must compensate.
- **Fallback**: If retrieval latency spikes (>50ms), fall back to extended state
  (double d_k to 256, costing 64 MB state -- still fits in VRAM scratch).

## 3.3 Associative Retrieval: LSH Cross-Attention

For every R-th token (R=16, configurable), we perform retrieval from topological
memory using LSH to find the top-k most relevant stored vectors.

### Retrieval Equations

```
Every R tokens, at position t where t mod R == 0:

  # Compute retrieval query from current recurrent state
  r_t = W_r · flatten(S_t)            # r_t ∈ R^(d_r), d_r = 512

  # LSH lookup (see §6 for data structure details)
  H = {h_1(r_t), h_2(r_t), ..., h_L(r_t)}   # L=8 hash functions
  candidates = Union(Bucket(h_i(r_t)) for h_i in H)
  top_k = argmax_{c in candidates} cos_sim(r_t, c.vector)   # k=32

  # Cross-attention over retrieved vectors
  K_ret = stack([c.vector for c in top_k])    # K_ret ∈ R^(k × d_r)
  V_ret = stack([c.value for c in top_k])     # V_ret ∈ R^(k × d_v)
  α = softmax(r_t · K_ret^T / sqrt(d_r))     # α ∈ R^k
  m_t = α · V_ret                              # m_t ∈ R^(d_v)

  # Inject into recurrent state
  S_t = S_t + W_inject · m_t                  # additive injection
```

### Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| LSH hash computation | O(L * d_r) = O(4096) | O(L) = 64 B | L=8 hash functions |
| Bucket union | O(L * B) where B=avg bucket size | O(L*B) | B ≈ 500 for 2M entries |
| Top-k selection | O(L*B * d_r) = O(2M) | O(k) = 16 KB | Linear scan of candidates |
| Cross-attention | O(k * d_r) = O(16384) | O(k * d_v) = 8 KB | Small dense attention |
| **Total per retrieval** | **~2M FLOPs** | **~20 KB** | |
| **Amortized per token (R=16)** | **~125K FLOPs/token** | **~1.3 KB/token** | |

### Bandwidth Cost of Retrieval

- Each retrieval reads ~500 candidate vectors * 512 dimensions * 2 bytes = 512 KB from
  host RAM (topological memory lives in host RAM).
- At R=16 and 8 tok/s decode: 0.5 retrievals/s → 256 KB/s from host RAM. **Negligible**.
- Even at R=1 (every token): 8 * 512 KB = 4 MB/s. Still negligible vs RAM bandwidth.

### Latency Budget

| Step | Latency | Notes |
|------|---------|-------|
| Hash computation | 0.01 ms | CPU, trivial |
| Bucket lookup | 0.1 ms | Host RAM random access |
| Candidate scoring | 0.5 ms | 500 * 512 dot products on CPU |
| Cross-attention | 0.01 ms | 32 * 512, trivial |
| **Total retrieval** | **~0.62 ms** | Well within 5ms token budget |

[HEURISTIC] LSH bucket size B≈500 assumes uniform hashing over 2M entries with 8
hash functions and 256 buckets per function. Falsifiable test: insert 2M random
vectors, measure actual bucket size distribution. If max bucket > 2000, switch to
multi-probe LSH or increase number of hash tables.

## 3.4 Combined Forward Pass

For each token x_t:
```
1. Embed:          e_t = Embed(x_t) + PosEnc(t)          # O(d) = O(2048)
2. For each layer l in 1..32:
   a. GLA:         h_t^l = GLA_l(e_t, S_t^l)             # O(d_k * d_v * H)
   b. Expert FFN:  f_t^l = Expert_route(h_t^l)            # O(d * d_ff) with sparse expert
   c. Residual:    e_t = e_t + h_t^l + f_t^l
3. If t mod R == 0:
   Retrieval:      m_t = LSH_retrieve(S_t)                # O(L*B*d_r)
   Inject:         S_t += W_inject · m_t
4. Output:         logits = W_vocab · e_t                  # O(d * V), V=32000
```

### Total Per-Token Compute

| Component | FLOPs/token | % of total |
|-----------|-------------|------------|
| GLA (32 layers) | 32 * 524K = 16.8M | 0.6% |
| Expert FFN (2 active, 32 layers) | 32 * 2 * 2048 * 5460 = 714M | 24.8% |
| Shared FFN | 0 (replaced by experts) | 0% |
| Retrieval (amortized) | 0.125M | 0.004% |
| Embedding + output | 2 * 2048 * 32000 = 131M | 4.6% |
| Routing (32 layers) | 32 * 2048 * 8 = 0.5M | 0.02% |
| Verifier (amortized, 1 in 4 tokens) | ~500M / 4 = 125M | 4.3% |
| **Total** | **~987M FLOPs/token** | |

At 10 TFLOPS GPU: 987M / 10T = **0.099 ms/token compute**.
At 50% GPU utilization (realistic): **0.2 ms/token**.
Theoretical max throughput: 5000 tok/s (compute-bound).
**Actual bottleneck is PCIe expert swapping, not compute.**

## 3.5 Memory Residency Summary

| Component | Where | Size |
|-----------|-------|------|
| GLA state matrices | VRAM (activation scratch) | 16 MB |
| Active expert weights | VRAM | 1200 MB |
| Shared attention weights | VRAM | 300 MB |
| Embeddings | VRAM | 200 MB |
| LSH index | Host RAM | Part of topological memory |
| Retrieved vectors (temp) | CPU cache → GPU (if needed) | 512 KB per retrieval |

## 3.6 Failure Modes & Fallbacks

| Failure | Detection | Fallback |
|---------|-----------|----------|
| GLA state saturation (all gates near 1) | Monitor gate activation statistics | Reset state + re-process last 256 tokens |
| LSH hash collision (all candidates irrelevant) | cos_sim of top-k < 0.3 threshold | Fall back to exact nearest-neighbor on subset |
| Expert routing collapse (same expert always chosen) | Entropy of routing distribution < 0.5 bits | Add load-balancing loss; force uniform routing |
| Retrieval latency spike (>10ms) | Timer on retrieval path | Skip retrieval for this step; use state only |
