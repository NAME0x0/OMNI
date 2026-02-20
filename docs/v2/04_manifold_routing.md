# § 04 — Manifold-Organised Expert Routing

> Experts are not a flat list.  They live on a torus —
> and moving between neighbours costs almost nothing.

---

## 1  Motivation

Standard MoE routing (Switch Transformer, Mixtral) treats the expert set as
an unstructured flat index:

```
h_t → W_route → logits ∈ R^{N_experts} → softmax → top-k
```

Problems with flat routing:
1. **No locality:** Expert 7 is as "far" from expert 8 as from expert 127.
   Switching experts always costs a full weight load.
2. **Load balancing is a loss term**, not a structural property.  Auxiliary
   losses fight the model's natural routing preferences.
3. **No interpolation:** You pick an expert or you don't.  There's no
   "between expert 7 and expert 8."
4. **No interpretability:** Expert indices are arbitrary.

### The Perspective Approach

Embed all 128 experts as points on a **2-D flat torus** $\mathbb{T}^2$.
The router maps the hidden state to a point on this surface, and the
nearest expert is activated.

Benefits:
- **Spatial locality →** nearby experts share most weights → delta streaming
- **Smooth routing →** the manifold coordinate is continuous even when the
  discrete selection jumps
- **Natural load balancing →** uniform expert placement on the torus gives
  each expert an equal Voronoi cell
- **Visualisable →** the 2-D torus can be unrolled into a rectangle for
  direct plotting of routing patterns

---

## 2  Torus Geometry

### 2.1  Definitions

The flat torus $\mathbb{T}^2 = [0, 1)^2$ with periodic boundaries.
A point $z = (z_1, z_2)$ wraps: $z_1 + 1 \equiv z_1$, $z_2 + 1 \equiv z_2$.

Geodesic (shortest-path) distance:

$$
d_{\mathbb{T}}(z, z') = \sqrt{\min(|z_1 - z'_1|, 1 - |z_1 - z'_1|)^2 + \min(|z_2 - z'_2|, 1 - |z_2 - z'_2|)^2}
$$

### 2.2  Expert Placement

128 experts are placed on an approximately uniform grid:

- $128 = 16 \times 8$ → a $16 \times 8$ grid on $[0, 1)^2$
- Expert $(i, j)$ is at position $\left(\frac{i}{16}, \frac{j}{8}\right)$ for $i \in [0, 15], j \in [0, 7]$

With slight jitter added during training to break symmetry:

$$
z_{\text{expert}}^{(i,j)} = \left(\frac{i}{16} + \epsilon_1, \; \frac{j}{8} + \epsilon_2\right) \mod 1
$$

where $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, 0.01)$ are learned offsets.

### 2.3  Voronoi Cells

Each expert owns a Voronoi cell on the torus — the set of all points closer
to it than to any other expert.  For a near-uniform grid, each cell has
area $\approx \frac{1}{128}$, giving natural load balance.

During training, a **manifold-consistency loss** nudges expert positions so
that Voronoi cells remain roughly equal:

$$
\mathcal{L}_{\text{balance}} = \text{Var}\left(\{A_i\}_{i=1}^{128}\right)
$$

where $A_i$ is the empirical routing frequency of expert $i$ over a batch.

---

## 3  Routing Algorithm

### 3.1  Route Computation

At each of the 60 expert FFN layers $\ell$:

$$
\begin{aligned}
z_{\text{query}}^{(\ell)} &= W_{\text{route}}^{(\ell)} \, h_t \mod 1
& \text{(project to torus, } W_{\text{route}} \in \mathbb{R}^{2 \times d}\text{)} \\[4pt]
d_i &= d_{\mathbb{T}}\!\left(z_{\text{query}}, z_{\text{expert}}^{(i)}\right)
& \text{(distance to each expert)} \\[4pt]
e^* &= \arg\min_i \; d_i
& \text{(select nearest expert)}
\end{aligned}
$$

The modular arithmetic ($\mod 1$) ensures the query always lands on the torus.

### 3.2  Routing Cost

- $W_{\text{route}}^{(\ell)}$ is $2 \times 4096 = 8192$ parameters per layer
- Distance computation: 128 experts × 6 ops = 768 FLOPs (negligible)
- Argmin: 128 comparisons
- **Total per layer: ~16K FLOPs** — essentially free

### 3.3  Per-Layer Independence

Each layer has its own $W_{\text{route}}^{(\ell)}$.  The same expert might be
selected at layer 12 but a completely different one at layer 13.  This
provides a combinatorial explosion of effective "expert paths":

$$
\text{Possible paths} = 128^{60} \approx 10^{126}
$$

This is the key to the model's effective capacity: 1.05T parameters span
an astronomically larger function space through combinatorial composition.

---

## 4  Delta Streaming

### 4.1  The Core Insight

If expert $j$ is a **manifold neighbour** of expert $i$ (one step on the
16×8 grid), their weights are trained to share structure:

$$
W_j^{(\ell)} = W_i^{(\ell)} + \Delta_{i \to j}^{(\ell)}
$$

where $\Delta_{i \to j}$ is a sparse difference matrix.  If we just loaded
expert $i$'s layer, switching to expert $j$ only requires streaming the
delta — not the full layer.

### 4.2  Delta Statistics

Through training with a **delta minimisation penalty**:

$$
\mathcal{L}_{\text{delta}} = \lambda \sum_{(i,j) \in \text{neighbours}} \left\| W_j - W_i \right\|_1
$$

We encourage adjacent experts to be similar.  Empirical expectation:

| Metric | Full layer | Delta (neighbour) | Delta (2-hop) |
|--------|-----------|-------------------|---------------|
| Non-zero ternary values | 67.6 M | ~6.8 M (10%) | ~20 M (30%) |
| Packed size | 27 MB | 2.7 MB | 8.1 MB |
| PCIe time @ 7 GB/s | 3.86 ms | 0.39 ms | 1.16 ms |

### 4.3  When Delta Streaming Applies

The pipeline tracks which expert was loaded at each layer for the previous
token.  For the current token:

1. **Same expert:** No transfer needed (0 ms).
2. **Neighbour expert:** Stream delta (0.39 ms) → 10× faster.
3. **2-hop expert:** Stream full layer but from a compressed delta (1.16 ms).
4. **Distant expert:** Full layer load (3.86 ms).

Expected distribution (from language modelling statistics):

| Case | Frequency | Transfer time |
|------|-----------|--------------|
| Same expert | ~30% | 0 ms |
| 1-hop neighbour | ~40% | 0.39 ms |
| 2-hop neighbour | ~15% | 1.16 ms |
| Distant | ~15% | 3.86 ms |
| **Weighted average** | | **~0.89 ms** |

This is ~4× faster than always loading full layers.

### 4.4  Delta Storage Format

```
struct DeltaLayer {
    source_expert: u16,       // base expert to apply delta to
    target_expert: u16,       // resulting expert
    layer_idx: u16,           // which layer
    num_changes: u32,         // number of changed trit positions
    positions: [u32],         // indices of changed values (varint compressed)
    new_values: [Trit],       // new ternary values at those positions (2-bit packed)
}
```

Stored in pre-computed files on NVMe alongside full expert files.

---

## 5  Manifold-Consistency Training Loss

The full routing-related training loss:

$$
\mathcal{L}_{\text{route}} = \mathcal{L}_{\text{task}} + \alpha \, \mathcal{L}_{\text{balance}} + \beta \, \mathcal{L}_{\text{delta}} + \gamma \, \mathcal{L}_{\text{smooth}}
$$

where:

- $\mathcal{L}_{\text{task}}$: standard cross-entropy language modelling loss
- $\mathcal{L}_{\text{balance}}$: variance of expert utilisation (see § 2.3)
- $\mathcal{L}_{\text{delta}}$: L1 norm of inter-expert weight differences (see § 4.2)
- $\mathcal{L}_{\text{smooth}}$: encourages routing queries to vary smoothly
  between adjacent tokens:

$$
\mathcal{L}_{\text{smooth}} = \frac{1}{T} \sum_{t=1}^{T-1} \left\| z_{\text{query}}^{(t+1)} - z_{\text{query}}^{(t)} \right\|_2
$$

Hyperparameters: $\alpha = 0.01$, $\beta = 0.001$, $\gamma = 0.005$.

---

## 6  Manifold Visualisation

The 2-D torus can be unrolled into a flat $[0, 1)^2$ plot:

```
z₂ ↑
  1 ┌──────────────────────────────────┐
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 7
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 6
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 5
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 4
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 3
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 2
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 1
    │  ●   ●   ●   ●   ●   ●   ●   ● │ row 0
  0 └──────────────────────────────────┘
    0                                  1 → z₁
         col 0   col 1  ...  col 15
```

During inference, routing queries $z_{\text{query}}^{(\ell)}(t)$ trace
paths across this surface.  These paths reveal:

- **Topic clusters:** mathematical text routes through one region,
  conversational text through another.
- **Layer specialisation:** early layers route to "syntactic" experts,
  late layers to "reasoning" experts.
- **Expert diversity:** multi-perspective decoding shows 4 different
  routing paths for the same input.

This is **directly interpretable** — a researcher can watch in real time
which expert region the model thinks is most relevant.

---

## 7  Comparison with Prior Routing

| Feature | Switch/Mixtral | Expert Choice | **Manifold (ours)** |
|---------|---------------|---------------|---------------------|
| Topology | Flat index | Flat index | 2-D torus |
| Locality | None | None | **Geodesic distance** |
| Delta streaming | N/A | N/A | **3-10× less transfer** |
| Load balance | Aux loss | Token dropping | **Geometric (Voronoi)** |
| Interpolation | None | Soft routing | **Smooth manifold** |
| Interpretability | ❌ | ❌ | **✅ 2-D visualisation** |
| Router params | N × d | N × d | **2 × d** (project to R²) |

---

*Next: [§ 05 Ternary Execution](05_ternary_execution.md)*
