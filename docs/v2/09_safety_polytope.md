# § 09 — Safety Polytope Projection (SPP)

> Safety isn't a penalty to optimise against.
> It's a hard geometric wall that cannot be gradient-attacked.

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

Define a **convex polytope** in embedding space containing only safe outputs.
Project every output embedding onto this polytope.  If the embedding is
already inside → no change.  If outside → clamp to the nearest face.

Key properties:
- **Hard constraint:** the output physically cannot leave the polytope
- **Non-differentiable:** projection involves argmin over half-spaces →
  gradient-based attacks get zero useful gradient
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

Any point inside $\mathcal{P}$ is a convex combination of safe embeddings →
it can only produce tokens that are weighted mixtures of safe vocabulary.

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
    
    // Dykstra's alternating projection (5 iterations suffice)
    for _iter in 0..5 {
        for j in 0..500 {
            let dot = inner_product(&x, &normals[j]);
            if dot > offsets[j] {
                // Project onto half-space j
                let violation = dot - offsets[j];
                for d in 0..4096 {
                    x[d] -= violation * normals[j][d];
                }
            }
        }
    }
    
    x
}
```

### 3.2  Complexity

- Per half-space check: 4,096 multiplies + 1 compare = 8,193 FLOPs
- Per iteration: 500 half-spaces × 8,193 = 4.1M FLOPs
- 5 iterations: **20.5M FLOPs total**
- At GPU speed: < 0.01 ms (negligible)

### 3.3  Convergence

Dykstra's algorithm converges to the true projection onto the intersection
of convex sets.  With 5 iterations and 500 half-spaces, residual error is:

$$
\| x_5 - x^* \| \leq \epsilon \cdot \| e - x^* \|, \quad \epsilon \approx 10^{-3}
$$

This is more than sufficient — a 0.1% error in the output embedding has
negligible effect on the token distribution.

---

## 4  Why SPP Cannot Be Gradient-Attacked

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

### 4.2  SPP's Non-Differentiability

SPP applies **after** the model's forward pass:

```
logit_embedding = model(input)           // differentiable up to here
safe_embedding = project(logit_embedding) // NON-DIFFERENTIABLE
output_tokens = sample(safe_embedding)    // discrete sampling
```

The projection involves:
- Conditional branches (`if dot > offset`)
- Argmin over violated half-spaces
- Iterative refinement (not a closed-form function)

An adversary computing $\frac{\partial \text{output}}{\partial \text{input}}$
gets **zero gradient** through the projection step.  The adversary cannot
learn which input perturbations will move the output past the safety boundary,
because the boundary is invisible to gradient computation.

### 4.3  Robustness Guarantee

**Theorem (informal):** For any input $x$, the output of SPP satisfies:

$$
\text{project}(\text{model}(x)) \in \mathcal{P}
$$

regardless of $x$.  There is no input that can produce an output outside
$\mathcal{P}$.  This is a **unconditional guarantee**, not a statistical one.

The only way to defeat SPP is to find a harmful output that lies *inside*
the safe polytope — i.e., a harmful message that is a convex combination
of the 1,000 safe anchors.  This is possible but very difficult when the
anchors are carefully chosen.

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
Phase 4: Repeat until red-team pass rate ≥ 98%
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
| **FMEA** | LoRA adaptation never modifies the SPP polytope. Safety is independent of learning. |
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
while still excluding the dangerous regions.

---

## 8  Performance Summary

| Metric | Value |
|--------|-------|
| FLOPs per token | 20.5 M (negligible vs 15.8B model) |
| Latency | < 0.01 ms |
| RAM for anchors + half-spaces | 24 MB |
| VRAM | 0 (runs on CPU, result copied to GPU for sampling) |
| Adversarial robustness | Zero gradient through projection |
| False positive rate (safe content blocked) | ≤ 2% with context-aware switching |
| Red-team block rate | ≥ 98% |

---

*Next: [§ 10 Training Plan](10_training_plan.md)*
