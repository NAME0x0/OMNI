# § 03 — Perspective Decay Recurrence (PDR)

> The model decides **how to look** before deciding **what to remember**.

---

## 1  Motivation

Current sequence mechanisms fall into three camps:

| Mechanism | Complexity | Drawback |
|-----------|-----------|----------|
| Full attention (GPT, LLaMA) | O(n² d) | Quadratic in context → VRAM explosion |
| Linear recurrence (RetNet, RWKV, Mamba) | O(n d²) | Fixed / input-gated decay — no perspective |
| Linear attention (Performer, cosFormer) | O(n d²) | Kernel approximation loses sharpness |

**PDR is different:** it introduces a *perspective projection* W_p that first
maps the input into a d-dimensional perspective space.  This perspective
vector p_t then generates the decay gate γ_t.  The conceptual shift:

> RetNet / GLA: "Given this input, how much do I forget?"
> Mamba: "Given this input, which dimensions do I select?"
> **PDR**: "Given this input, **what lens do I apply** — and that lens
> determines what I forget, remember, and retrieve?"

Each of the 60 PDR layers learns a different "lens."  Layer 3 might learn a
syntactic perspective; layer 40 a semantic one; layer 55 a reasoning one.

---

## 2  Equations

### 2.1  Single-Step Update (decode mode)

Given input $x_t \in \mathbb{R}^d$ at time step $t$:

$$
\begin{aligned}
p_t &= W_p \, x_t + b_p & \text{(perspective projection)} \\[4pt]
\gamma_t &= \sigma(p_t) & \text{(perspective-aware decay, element-wise sigmoid)} \\[4pt]
k_t &= W_k \, x_t & \text{(key, } k_t \in \mathbb{R}^r \text{)} \\[4pt]
v_t &= W_v \, x_t & \text{(value, } v_t \in \mathbb{R}^d \text{)} \\[4pt]
S_t &= \text{diag}(\gamma_t) \cdot S_{t-1} + v_t \, k_t^\top & \text{(state update, } S_t \in \mathbb{R}^{d \times r} \text{)} \\[4pt]
q_t &= W_q \, x_t & \text{(query, } q_t \in \mathbb{R}^r \text{)} \\[4pt]
\hat{o}_t &= S_t \, q_t & \text{(readout, } \hat{o}_t \in \mathbb{R}^d \text{)} \\[4pt]
o_t &= W_o \, \hat{o}_t & \text{(output projection)} \\[4pt]
\end{aligned}
$$

Dimensions:
- $d = 4096$ (model dimension)
- $r = 256$ (recurrent state rank)
- $W_p \in \mathbb{R}^{d \times d}$, $W_o \in \mathbb{R}^{d \times d}$
- $W_k, W_q \in \mathbb{R}^{r \times d}$
- $W_v \in \mathbb{R}^{d \times d}$

### 2.2  Parallel Scan (training / prefill mode)

For a chunk of $L$ tokens, we can parallelise the recurrence using a
log-depth scan.  Define the per-step "element" as:

$$
(A_t, B_t) = \bigl(\text{diag}(\gamma_t), \; v_t \, k_t^\top\bigr)
$$

The scan composition operator $\oplus$ is:

$$
(A_2, B_2) \oplus (A_1, B_1) = (A_2 \cdot A_1, \; A_2 \cdot B_1 + B_2)
$$

This is associative, so a parallel prefix scan computes all $S_1, \ldots, S_L$
in $O(\log L)$ sequential steps using $O(L \cdot d \cdot r)$ total work.

Chunk size $L = 256$ (tunable).  Within a chunk: parallel scan.  Across
chunks: sequential state passing.

---

## 3  What Makes PDR Novel

### 3.1  Perspective ≠ Gate

In existing gated recurrences:

| Model | Decay computation | Semantic |
|-------|------------------|----------|
| RetNet | $\gamma = \text{fixed constant}$ | No adaptivity |
| GLA | $\gamma_t = \sigma(W_g \, x_t)$ | Input-dependent gate |
| Mamba | $\Delta_t = \text{softplus}(W_\Delta \, x_t)$; discretised | Continuous-time selection |
| **PDR** | $p_t = W_p \, x_t$; $\gamma_t = \sigma(p_t)$ | Two-stage: perspective → decay |

The critical difference is that $W_p$ is a **full d×d matrix** (16.78M params),
not a small gate projection.  It transforms the input into a completely new
representation space before deriving the decay.  This is functionally
equivalent to applying a learned "lens" or "viewpoint" to the input.

### 3.2  Why the Extra Transformation Matters

Consider two inputs $x_a$ and $x_b$ that are similar in model space but should
trigger different memory behaviours (e.g., "the cat sat" vs "the cat died"):

- GLA computes $\sigma(W_g \, x)$ — a linear function of $x$.  Similar $x$
  get similar gates.
- PDR computes $\sigma(W_p \, x)$ — also linear, but the d×d matrix has
  16M parameters to learn a rotational view where the two inputs *separate*.
  The perspective projection can implement arbitrary rotations, scalings, and
  axis-aligned separations in d-dimensional space.

With a small gate vector (GLA uses ~$d$ parameters for the gate), the model
can only modulate decay along the same axes as the input.  With d×d, PDR can
modulate along *any* learned axis.

### 3.3  Multi-Layer Perspective Diversity

Each of the 60 PDR layers has its own $W_p$.  Through training, these
naturally specialise:

- Early layers: perspectives that track **syntactic structure** (open brackets,
  clause boundaries, verb tense).
- Middle layers: perspectives that track **entity state** (who/what is being
  discussed, relationships).
- Late layers: perspectives that track **reasoning chains** (premises,
  intermediate conclusions, confidence).

This multi-perspective decomposition is the architectural realisation of the
project's thesis: *Perspective Is All You Need*.

---

## 4  Complexity Analysis

### 4.1  FLOPs per Token per Layer

| Operation | Shape | FLOPs |
|-----------|-------|-------|
| $p_t = W_p \, x_t$ | (d, d) × (d,) | $2d^2 = 33.6\text{M}$ |
| $\gamma_t = \sigma(p_t)$ | (d,) | $d = 4\text{K}$ (negligible) |
| $k_t = W_k \, x_t$ | (r, d) × (d,) | $2rd = 2.1\text{M}$ |
| $v_t = W_v \, x_t$ | (d, d) × (d,) | $2d^2 = 33.6\text{M}$ |
| $\text{diag}(\gamma_t) \cdot S_{t-1}$ | (d,) ⊙ (d, r) | $dr = 1.05\text{M}$ |
| $v_t \, k_t^\top$ | (d,) × (r,) → (d, r) | $dr = 1.05\text{M}$ |
| $S_t$ (add) | (d, r) | $dr = 1.05\text{M}$ |
| $q_t = W_q \, x_t$ | (r, d) × (d,) | $2rd = 2.1\text{M}$ |
| $\hat{o}_t = S_t \, q_t$ | (d, r) × (r,) | $2dr = 2.1\text{M}$ |
| $o_t = W_o \, \hat{o}_t$ | (d, d) × (d,) | $2d^2 = 33.6\text{M}$ |
| **Total** | | **~110 M FLOPs** |

### 4.2  Memory per Layer

| Item | Size | Bytes |
|------|------|-------|
| Weights ($W_p, W_v, W_o$: 3×d², $W_k, W_q$: 2×r×d) | 52.5 M params | 13.1 MB @ 2-bit |
| State $S_t$ | d × r = 1.05 M | 2.0 MB @ FP16 |
| Activations (transient) | $p_t, k_t, v_t, q_t, \hat{o}_t, o_t$ | ~50 KB @ FP16 |

### 4.3  Comparison with Alternatives

| | PDR (ours) | Attention | RetNet | Mamba |
|---|---|---|---|---|
| Params/layer | 52.5 M | 67 M (MHA d=4096) | 33.6 M | 12 M |
| FLOPs/tok/layer | 110 M | 33.6M + O(n·d) | 67 M | 48 M |
| State memory | 2 MB (O(dr)) | O(n·d) KV-cache | 2 MB | 0.5 MB |
| Seq-length scaling | **O(1)** | O(n) | **O(1)** | **O(1)** |
| Adaptivity | **Full d×d** | Full (via QK) | Fixed scalar | d-dim selective |

PDR trades some per-token FLOPs (vs RetNet/Mamba) for much richer
adaptivity.  The extra cost is compensated by ternary execution in the FFN
layers, which saves far more compute than PDR adds.

---

## 5  State Management

### 5.1  Initialisation

At the start of generation, all 60 state matrices are zero-initialised:

$$S_0^{(\ell)} = \mathbf{0}_{d \times r} \quad \forall \ell \in [0, 59]$$

### 5.2  State Compression for Long Contexts

After processing $T$ tokens, the state naturally compresses information.
However, if $T$ is extremely large (> 100K), numerical drift can occur.
Mitigation: periodic state re-normalisation every 8192 tokens:

$$S_t \leftarrow S_t \cdot \frac{\alpha}{\| S_t \|_F}$$

where $\alpha = \sqrt{d \cdot r}$ is a scale constant.  This costs negligible
compute but prevents float overflow.

### 5.3  State Serialisation

The model's "context memory" is its 60 state matrices:
- Total: 60 × 4096 × 256 × 2 bytes = 120 MB
- Can be saved/loaded to resume a conversation
- Analogous to saving KV-cache in standard transformers, but O(1) size
  regardless of conversation length

---

## 6  Integration with Other Components

| Component | Interaction with PDR |
|-----------|---------------------|
| **Manifold Router** | Uses the PDR output $o_t$ at each layer to compute the routing query. The perspective already encodes what "kind" of processing is needed → routes to the most relevant expert. |
| **Expert FFN** | Follows every PDR layer. The expert transforms the PDR output through SwiGLU. |
| **GQA Layers** | Every 4th layer switches from PDR to windowed GQA. GQA provides "sharpening" — exact attention over recent tokens to correct any drift in PDR state. |
| **MPD** | During multi-perspective decoding, the PDR states diverge across the 4 perspectives (different expert routing → different state evolution). Disagreement in the state reflects genuine uncertainty. |
| **FMEA** | The perspective matrix $W_p$ is a prime target for LoRA adaptation: small rank-4 adjustments to W_p change the model's "viewpoint" without touching the much larger $W_v$ or $W_o$. |

---

## 7  Training Considerations

### 7.1  Initialisation of $W_p$

$W_p$ is initialised as a near-identity matrix with small random perturbations:

$$W_p = I_d + \epsilon \cdot N(0, 1), \quad \epsilon = 0.01$$

This ensures early training has minimal perspective distortion — the model
starts by looking at the input "as is" and gradually learns to rotate it.

### 7.2  Gradient Flow Through the Scan

The parallel scan preserves gradient flow through $O(\log L)$ steps.
Each step involves matrix multiplication by $\text{diag}(\gamma_t)$, whose
elements are in $(0, 1)$.  Over many time steps, this can cause gradient
decay — mitigated by:

1. The 3:1 PDR/GQA ratio: GQA layers provide skip-connection-like direct
   gradient paths every 4 layers.
2. Initialising $b_\gamma$ so that $\sigma(b_\gamma) \approx 0.95$ — slow
   initial decay, fast information flow.

---

*Next: [§ 04 Manifold Routing](04_manifold_routing.md)*
