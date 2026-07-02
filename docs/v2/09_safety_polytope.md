# § 09 — Safety Polytope Projection (SPP)

> Safety isn't a penalty to optimise against.
> A hard geometric constraint on where outputs can live.
> Membership is guaranteed; safety must still be earned.

---

## 1  Motivation

Current safety approaches:

| Method | Mechanism | Vulnerability |
|--------|-----------|---------------|
| RLHF | Reward model penalises harmful outputs | Reward hacking, jailbreaks via prompt engineering |
| Constitutional AI | LLM self-critique rules | Rules are prompts → manipulable |
| Content filters | Pattern matching on output text | Evasion via encoding, synonyms, multilingual |
| Activation steering | Move activations toward "safe" direction | Requires knowing the direction; reversible |

All of these are **soft constraints** — they add penalties or heuristics
that can be circumvented because the model's output space is unconstrained.

### The SPP Approach

Define a **convex polytope** in embedding space intended to contain safe outputs.
Project every output embedding onto this polytope.  If the embedding is
already inside → no change.  If outside → clamp to the nearest face.

Key properties:
- **Hard membership constraint:** the projected embedding cannot leave the
  half-space polytope when projection converges
- **Attack surface still exists:** half-space projection is piecewise-linear
  and differentiable almost everywhere; BPDA-style and gradient-free attacks
  still apply
- **Composable:** the polytope can be refined by adding/removing anchors
  without retraining

---

## 2  Polytope Construction

### 2.1  Anchor Embeddings

Start with 1,000 **anchor embeddings** — the output-head embeddings of
tokens/phrases known to be safe:

```
Safe anchors: embeddings of tokens like
  "The", "is", "because", "however", "approximately",
  "I think", "according to", "it depends on", ...
  
NOT included: embeddings of slurs, threats, instructions for harm, etc.
```

Each anchor: $a_i \in \mathbb{R}^{4096}$, stored in FP32.
Total: 1,000 × 4,096 × 4 bytes = **16 MB**.

### 2.2  Convex Hull

The safe polytope $\mathcal{P}$ is the **convex hull** of the 1,000 anchors:

$$
\mathcal{P} = \text{conv}(a_1, \ldots, a_{1000}) = \left\{ \sum_{i=1}^{1000} \lambda_i a_i \;\middle|\; \lambda_i \geq 0, \sum_i \lambda_i = 1 \right\}
$$

Any point inside $\mathcal{P}$ is a convex combination of the selected anchor
embeddings.  This constrains the geometry of the output embedding, but it does
**not** prove token-level safety: logits are still computed by dot products
against every token embedding, so a point in the hull can assign high logits to
tokens outside the anchor set.  The link between hull membership and safe text
is a design hypothesis that requires red-team validation.

### 2.3  Half-Space Representation

For efficient projection, pre-compute the half-space representation.
The polytope is the intersection of half-spaces:

$$
\mathcal{P} = \bigcap_{j=1}^{H} \left\{ x \in \mathbb{R}^{4096} \;\middle|\; n_j \cdot x \leq b_j \right\}
$$

where $(n_j, b_j)$ are the facet normals and offsets.

For 1,000 anchors in 4,096 dimensions, the polytope is full-dimensional
and the number of facets $H$ is bounded but potentially large.  In practice,
we use an **approximate** half-space representation with $H = 500$ facets
derived from the support vectors of the anchor set.

Storage: 500 × (4,096 + 1) × 4 bytes = **8 MB**.

---

## 3  Projection Algorithm

### 3.1  Fast Approximate Projection

Given an output embedding $e \in \mathbb{R}^{4096}$:

```
fn project_to_polytope(e: &[f32; 4096], normals: &[[f32; 4096]; 500], 
                        offsets: &[f32; 500]) -> [f32; 4096] {
    let mut x = e.clone();
    
    // Full Dykstra projection with correction terms.
    // Default: max 50 iterations, tolerance 1e-6, early exit on convergence.
    let mut p = [[0.0; 4096]; 500];
    for _iter in 0..50 {
        let x_start = x.clone();
        for j in 0..500 {
            let x_prev = x.clone();
            let mut y = x.clone();
            for d in 0..4096 {
                y[d] += p[j][d];
            }

            let dot = inner_product(&y, &normals[j]);
            if dot > offsets[j] {
                // Project onto half-space j
                let violation = dot - offsets[j];
                let norm_sq = inner_product(&normals[j], &normals[j]);
                for d in 0..4096 {
                    y[d] -= (violation / norm_sq) * normals[j][d];
                }
            }

            // p_j = (x_prev + p_j) - y
            for d in 0..4096 {
                p[j][d] = (x_prev[d] + p[j][d]) - y[d];
                x[d] = y[d];
            }
        }

        if l2_distance(&x, &x_start) < 1e-6 {
            break;
        }
    }
    
    x
}
```

### 3.2  Complexity

- Per half-space check: ~4,096 multiply-adds + 1 compare ≈ 8.2K FLOPs
- Active half-space projection: another ~8.2K FLOPs for the vector update
- Per iteration check cost: 500 half-spaces × ~8.2K ≈ 4.1M FLOPs,
  plus correction/projection work for active constraints
- Default iteration budget: up to **50 iterations** at tolerance `1e-6`
  (`crate::config::SPP_ITERATIONS`, `ProjectionConfig::default`)
- Worst-case check-only budget at 50 iterations: ~205M FLOPs; actual cost is
  adaptive because the implementation exits early on convergence
- Latency is a projection until benchmarked in the end-to-end runtime

### 3.3  Convergence

Dykstra's algorithm converges to the true projection onto the intersection
of convex sets.  The CPU reference implementation uses a maximum of 50
iterations and stops when the L2 change between iterations falls below
`1e-6`.

$$
\| x_t - x_{t-1} \|_2 < 10^{-6}
$$

Whether that residual is sufficient for token-level behaviour is an empirical
question; it should be checked in validation alongside red-team tests.

---

## 4  What SPP Does and Does Not Guarantee

### 4.1  The Problem with Differentiable Safety

In RLHF:
```
output = model(input)
safety_score = reward_model(output)
loss = -safety_score  (optimiser maximises safety)
```

An adversary can compute $\nabla_{\text{input}} \text{safety\_score}$ and
find inputs that minimise the safety score while maximising harmful content.
This is exactly how jailbreaks work.

### 4.2  Projection Is Not an Attack-Proof Barrier

SPP applies **after** the model's forward pass:

```
logit_embedding = model(input)           // differentiable up to here
safe_embedding = project(logit_embedding) // piecewise-linear projection
output_tokens = sample(safe_embedding)    // discrete sampling
```

Projection onto one half-space has the closed form:

$$
\Pi(x) = x - \max\left(0, \frac{a \cdot x - b}{\|a\|^2}\right)a
$$

This map is piecewise-linear and differentiable almost everywhere.  Its
Jacobian is $I$ when the constraint is inactive and
$I - aa^T / \|a\|^2$ when the constraint is active.  It is not a zero-gradient
wall.  Adaptive attacks can use BPDA-style approximations, gradients through
the differentiable upstream model, or gradient-free search.

### 4.3  Membership Guarantee

**Theorem (informal):** For any input $x$, the output of SPP satisfies:

$$
\text{project}(\text{model}(x)) \in \mathcal{P}
$$

regardless of $x$.  There is no input that can produce an output outside
$\mathcal{P}$.  This is an unconditional **membership** guarantee, not a
semantic safety guarantee.  The implementation also re-projects after its
optional hull-blend step to preserve half-space feasibility.

The only way to defeat SPP is to find a harmful output that lies *inside*
the enforced polytope.  That is possible: membership is a geometric predicate,
not a proof that the sampled text is harmless.  It is also easier to imagine
inside the $\epsilon$-inflated polytope used in § 7.3 than inside the exact
anchor hull, so the size of that inflated region must be red-team validated.

---

## 5  Anchor Selection Strategy

### 5.1  Positive Anchors (Included)

- All tokens/embeddings representing neutral vocabulary
- Common functional words, numbers, punctuation  
- Factual statement patterns
- Hedging language ("I think", "possibly", "according to")
- Refusal patterns ("I can't help with that")

### 5.2  Exclusion Zone

Anchors are specifically **not** placed near embeddings of:
- Slurs, hate speech vocabulary
- Violence-related content
- Personally identifiable information patterns
- Instructions for illegal activities

### 5.3  Refinement Protocol

```
Phase 1: Initial anchor set from safe vocabulary (automated)
Phase 2: Red-team testing → identify leaks → add blocking anchors
Phase 3: Shrink polytope around discovered harmful embeddings
Phase 4: Repeat until target red-team pass rate ≥ 98%
```

### 5.4  Dynamic Anchor Update

New anchors can be added at runtime without retraining:

```
fn add_safety_anchor(new_anchor: [f32; 4096]) {
    anchors.push(new_anchor);
    recompute_halfspaces();  // ~50 ms, done async
}
```

This allows the safety boundary to evolve with discovered threats.

---

## 6  Interaction with Other Components

| Component | Integration |
|-----------|-------------|
| **MPD** | SPP runs after MPD token selection. If MPD flags uncertainty AND SPP projects significantly, the token is blocked entirely. |
| **HDM** | SPP anchors can reference HDM entries: "this fact is safe to state." |
| **FMEA** | LoRA adaptation does not directly modify the SPP polytope. The intended separation reduces one coupling path, but safety under adaptation still requires validation. |
| **Manifold Router** | If a routing path consistently triggers SPP projections, the router learns to avoid that expert region (via NES fitness). |

---

## 7  Limitations and Mitigations

### 7.1  Over-Restriction

The polytope might block legitimate outputs (e.g., medical terminology
that's close to the violence exclusion zone).

Mitigation: **context-aware polytope switching**.  Different polytopes for:
- General conversation (tight safety)
- Medical/scientific context (wider polytope including clinical terms)
- Creative writing (wider polytope with literary terms)

Context detection: lightweight classifier on the PDR state at layer 20.

### 7.2  Indirect Harm

SPP constrains individual token embeddings, not multi-token semantics.
A harmful message could be constructed from individually safe tokens.

Mitigation: **sentence-level SPP** runs every 16 tokens, projecting the
mean embedding of the recent sentence window.  This catches semantic-level
harm at the cost of a few thousand extra FLOPs.

### 7.3  Polytope Dimensionality

In 4,096 dimensions, the convex hull of 1,000 points is very "thin" —
it doesn't cover the full safe output space.  Many safe outputs are outside
the hull and would be incorrectly projected.

Mitigation: **inflated polytope**.  Instead of the exact convex hull, use:

$$
\mathcal{P}_\epsilon = \left\{ x \;\middle|\; \forall j: n_j \cdot x \leq b_j + \epsilon \right\}
$$

where $\epsilon$ is an inflation parameter that expands each half-space.
$\epsilon = 0.5$ gives a comfortable margin that includes most safe outputs
while still aiming to exclude dangerous regions.  This larger region weakens
the simple "convex combination of safe anchors" intuition, so it must be tuned
against both false positives and harmful in-polytope outputs.

---

## 8  Performance Summary

| Metric | Value |
|--------|-------|
| FLOPs per token | Projected: ~4.1M check FLOPs per iteration; up to ~205M check FLOPs at 50 iterations, plus active projection work |
| Latency | Projected; unmeasured end-to-end |
| RAM for anchors + half-spaces | 24 MB |
| VRAM | 0 (runs on CPU, result copied to GPU for sampling) |
| Adversarial robustness | Membership guarantee; not attack-immune (see § 4) |
| False positive rate (safe content blocked) | Target: ≤ 2% with context-aware switching |
| Red-team block rate | Target: ≥ 98% |

---

*Next: [§ 10 Training Plan](10_training_plan.md)*
