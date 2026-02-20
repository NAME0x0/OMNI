# § 08 — Forward-Mode Evolutionary Adaptation (FMEA)

> Learn at inference time.  No backprop.  No activation storage.
> Just a forward pass, a perturbation, and natural selection.

---

## 1  Motivation

Standard continual learning approaches:

| Method | Memory | Problem |
|--------|--------|---------|
| Full fine-tuning | O(model) | Catastrophic forgetting, VRAM-prohibitive |
| LoRA via backprop | O(activations) | Still needs activation graph → 4-8× memory |
| EWC / SI | O(model) | Requires Fisher information, still backprop |

For a model that uses 2.7 GB of 4 GB VRAM just for weights and state,
there is **no room for activation storage**.  We cannot backpropagate.

### The FMEA Approach

Two complementary gradient-free learning mechanisms:

1. **Forward-mode differentiation (JVP)** for LoRA adapter weights —
   computes the directional derivative in one forward pass with no stored
   activations.

2. **Natural Evolution Strategies (NES)** for routing manifold positions —
   perturb expert coordinates, measure fitness, update via estimated
   gradient.

Together: O(1) memory overhead, ~1 forward-pass equivalent per update step.

---

## 2  LoRA Adapters via Forward-Mode Differentiation

### 2.1  Adapter Structure

Tiny rank-4 LoRA adapters on the perspective projection $W_p$ (the most
impactful weight for changing model behaviour):

$$
W_p^{\text{adapted}} = W_p + B \cdot A
$$

where $A \in \mathbb{R}^{4 \times d}$, $B \in \mathbb{R}^{d \times 4}$, $d = 4096$.

Parameters per layer: $4 \times 4096 + 4096 \times 4 = 32,768$
Total across 60 PDR layers: $60 \times 32,768 = 1,966,080 \approx 2\text{M params}$
Storage: 2M × 2 bytes (FP16) = **3.75 MB**

### 2.2  Forward-Mode Differentiation (JVP)

The **Jacobian-vector product** (JVP) computes:

$$
\frac{\partial \mathcal{L}}{\partial \theta} \cdot v = \lim_{\epsilon \to 0} \frac{\mathcal{L}(\theta + \epsilon v) - \mathcal{L}(\theta)}{\epsilon}
$$

In practice, using dual numbers:

$$
\mathcal{L}(\theta + \epsilon v) = \mathcal{L}(\theta) + \epsilon \, \nabla_v \mathcal{L}(\theta) + O(\epsilon^2)
$$

The JVP $\nabla_v \mathcal{L}$ can be computed in a **single forward pass**
by propagating "tangent vectors" alongside the primal computation.

### 2.3  Forward-Mode Update Algorithm

```
Input: current LoRA params θ = {A^(ℓ), B^(ℓ)}_{ℓ=1}^{60}
       training signal (token sequence + loss function)

1. Sample K random perturbation directions: v_1, ..., v_K ∈ R^{|θ|}
   (K = 8 typically, each v_k ~ N(0, I / |θ|))

2. For each v_k:
   a. Forward pass with dual numbers:
      - Primal: normal computation with θ
      - Tangent: propagate v_k through each operation
      - Output: L(θ) and ∇_{v_k} L(θ) simultaneously
   b. Store the directional derivative: g_k = ∇_{v_k} L(θ)

3. Estimate gradient via random projection:
   ĝ = (1/K) Σ_k  g_k · v_k

4. Update: θ ← θ - η · ĝ    (η = learning rate, default 1e-4)
```

### 2.4  Memory Analysis

| Component | Memory |
|-----------|--------|
| LoRA params θ | 3.75 MB |
| One tangent vector v_k | 3.75 MB |
| Directional derivatives (K=8 scalars) | 64 bytes |
| Gradient estimate ĝ | 3.75 MB |
| **Total** | **~11 MB** |

Compare with backprop LoRA:
- Activation storage for 80 layers: ~800 MB – 2 GB
- **FMEA is 72–180× more memory-efficient**

### 2.5  Convergence Properties

Forward-mode with K random directions gives an unbiased gradient estimate
with variance inversely proportional to K:

$$
\text{Var}[\hat{g}] = \frac{\|\nabla L\|^2}{K} \cdot \left(\frac{|\theta|}{K} - 1\right)
$$

With |θ| = 2M and K = 8, variance is high — but we're doing tiny
incremental adaptation, not full training.  The signal-to-noise ratio
is sufficient for:

- Learning user preferences (style, tone, format)
- Incorporating corrections ("actually, it's X not Y")
- Adapting to new domains over ~100 examples

**Not** sufficient for:
- Fundamental capability improvement (that requires pre-training)
- Large-scale knowledge injection (use HDM instead)

---

## 3  Routing Manifold Evolution via NES

### 3.1  What Evolves

The 128 expert positions on the torus: $\{z_i = (\theta_i, \phi_i, \psi_i)\}_{i=1}^{128}$.
Total parameters: 384 scalars (128 × 3D coordinates).

These positions determine which expert is "nearest" to any given routing
query.  By moving experts on the manifold, we change the model's routing
behaviour without touching any weights. Updates are applied as wrapped fold
steps on the existing coordinates (no append of new points).

### 3.2  Natural Evolution Strategies (NES)

NES estimates gradients through fitness-weighted perturbations:

```
Input: expert positions z = {z_1, ..., z_128}
       fitness function F(z) = -average_loss over recent N tokens

1. Sample M perturbations: ε_1, ..., ε_M ~ N(0, σ²I)
   (M = 16, σ = 0.01)

2. Evaluate fitness at each perturbation:
   F_m = F(z + ε_m)    for m = 1, ..., M
   (Each evaluation = 1 forward pass on a small batch)

3. Estimate gradient:
   ĝ = (1/(M·σ²)) Σ_m  F_m · ε_m

4. Update: z ← z + α · ĝ    (α = 0.001)
```

### 3.3  Memory and Compute Cost

| Resource | Cost |
|----------|------|
| Perturbation storage | 16 × 384 × 4 bytes = 24 KB |
| Forward passes per update | 16 (one per perturbation) |
| Wall time per update | 16 × 93 ms = ~1.5 s |
| Update frequency | Every 50 tokens |

This is lightweight enough to run asynchronously — while the user reads
the model's output, FMEA can complete several routing updates.

### 3.4  What Routing Evolution Achieves

- **Topic adaptation:** expert coverage shifts toward the current domain
- **User-specific routing:** frequently-needed expert regions get more
  fine-grained coverage (experts cluster toward high-usage areas)
- **Load balancing recovery:** if training left some experts underutilised,
  NES can redistribute them

---

## 4  Combined FMEA Pipeline

```
Every 50 tokens (background, between user turns):

1. Collect recent token buffer (last 50 tokens + losses)

2. LoRA update (JVP):
   - 8 random perturbation directions
   - 8 forward passes (each ~93 ms = 744 ms total)
   - Gradient estimate + update: < 1 ms
   - Total: ~0.75 s

3. Routing evolution (NES):
   - 16 perturbations of expert positions
   - 16 forward passes (~1.5 s total)
   - Gradient estimate + position update: < 1 ms
   - Total: ~1.5 s

4. Total adaptation time: ~2.25 s per 50 tokens
   (runs during user think time — no impact on generation speed)
```

---

## 5  Forgetting Prevention

### 5.1  LoRA Isolation

LoRA adapters are **separate** from base weights.  The base model is frozen.
Even if LoRA adapts destructively, the base model's knowledge is intact.

Reset strategy: if adaptation degrades quality (measured by MPD agreement
scores dropping), revert LoRA to the last checkpoint:

```
if avg_agreement(last_100_tokens) < 0.5 * avg_agreement(baseline):
    lora_params = lora_checkpoint  // rollback
    log("FMEA rollback: adaptation degraded quality")
```

### 5.2  Routing Manifold Anchoring

Expert positions have an **elastic potential** that resists large moves:

$$
\mathcal{L}_{\text{anchor}} = \lambda_a \sum_{i=1}^{128} \| z_i - z_i^{(0)} \|^2
$$

where $z_i^{(0)}$ is the original trained position.  This prevents experts
from collapsing into a small region of the torus.

$\lambda_a = 0.1$ — strong enough to prevent catastrophic drift, weak enough
to allow meaningful adaptation.

---

## 6  Comparison with Prior Adaptation Methods

| Method | Memory overhead | Requires backprop | Parameters updated | Forgetting risk |
|--------|-----------------|-------------------|--------------------|-----------------|
| Full fine-tuning | O(model) | Yes | All | High |
| LoRA (backprop) | O(activations) | Yes | Adapters | Low |
| Prompt tuning | O(prompt × d) | Yes | Virtual tokens | None |
| ICL (in-context) | O(examples × d) | No | None | None but ephemeral |
| **FMEA-LoRA** | **11 MB** | **No** | **2M adapter params** | **Low (isolated)** |
| **FMEA-NES** | **24 KB** | **No** | **384 routing coords** | **Low (anchored)** |

---

## 7  Theoretical Justification

### 7.1  JVP as Gradient Estimation

The Expected value of the random-projection gradient estimate:

$$
\mathbb{E}[\hat{g}] = \nabla_\theta \mathcal{L}
$$

i.e., the estimate is **unbiased**.  Convergence rate is $O(d/K)$ slower
than true gradient descent (where $d$ = param count, $K$ = perturbation
directions).  With d = 2M and K = 8, we converge ~250,000× slower per step.

But each step is also ~250,000× cheaper in memory and ~1× in compute
(one forward pass vs one forward + one backward).  For the small
incremental updates we need at inference time, this is more than sufficient.

### 7.2  NES for Non-Differentiable Routing

The argmin in manifold routing is non-differentiable.  Straight-through
estimators exist but are biased.  NES naturally handles non-differentiable
objectives because it evaluates fitness as a black-box function.

This is a genuine advantage over backprop-based routing updates, which
require approximate gradients through discrete selection.

---

*Next: [§ 09 Safety Polytope](09_safety_polytope.md)*
