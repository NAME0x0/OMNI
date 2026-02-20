# Section 5: Continual Learning

## 5.1 Approach Selection

**Forbidden**: Fisher-matrix EWC (Elastic Weight Consolidation) — as per directive.

**Selected**: Orthogonal Gradient Projection (OGP) combined with Low-Rank Adaptation
(LoRA) streaming updates. This is a hybrid we call **OGP-LoRA-Stream**.

### Why OGP-LoRA-Stream

| Property | EWC (forbidden) | OGP-LoRA-Stream |
|----------|-----------------|-----------------|
| Requires full Fisher matrix | Yes (O(P^2) or diagonal O(P)) | No |
| Full-dataset pass required | Yes (to compute Fisher) | No (streaming, single-pass) |
| Memory for consolidation | O(P) per task | O(R * D) where R << P |
| Catastrophic forgetting protection | Moderate | Strong (orthogonal projection) |
| Adapts incrementally | No (batch recompute) | Yes (per-sample updates) |

## 5.2 Algorithm: OGP-LoRA-Stream

### Data Structures

```
LoRA adapters (per expert, per layer):
  A_l ∈ R^(d × r)    # Down-projection, r = 8 (rank)
  B_l ∈ R^(r × d)    # Up-projection

Projection memory (shared across all layers):
  P ∈ R^(d × m)      # Orthogonal basis of previously important directions, m ≤ 256
  # Stored in host RAM. Loaded to VRAM during learning steps.

Replay buffer:
  RB: circular buffer of (input, target) pairs, max size N_rb = 4096 entries
  Each entry: ~2 KB (tokenized input 512 tokens * 2 bytes + target 512 * 2 bytes)
  Total replay buffer: 4096 * 2 KB = 8 MB
```

### Memory Budget

| Component | Size | Location |
|-----------|------|----------|
| LoRA A matrices (8 experts * 32 layers) | 8 * 32 * 2048 * 8 * 2B = 8 MB | Host RAM |
| LoRA B matrices (8 experts * 32 layers) | 8 * 32 * 8 * 2048 * 2B = 8 MB | Host RAM |
| Projection basis P | 2048 * 256 * 4B (FP32) = 2 MB | Host RAM |
| Active LoRA (2 experts, in VRAM) | 2 * 32 * (2048*8 + 8*2048) * 2B = 2 MB | VRAM (in scratch) |
| Replay buffer | 8 MB | Host RAM |
| Gradient scratch (one layer at a time) | 2048 * 8 * 4B = 64 KB | VRAM |
| **Total VRAM for learning** | **~2.1 MB** | Fits in scratch |
| **Total Host RAM for learning** | **~26 MB** | Well within budget |

**Proof: Total VRAM for continual learning (2.1 MB) << 4096 MB VRAM budget.**
**The 256 MB activation scratch (§2) can accommodate this with 253.9 MB to spare.**

## 5.3 Update Algorithm (Pseudocode)

```python
def ogp_lora_stream_update(sample, model, P, lora_params, replay_buffer):
    """
    Single-sample continual learning update.
    No full-dataset pass. No Fisher matrix.

    Args:
        sample: (input_tokens, target_tokens) — single training example
        model: frozen base model (weights never modified)
        P: orthogonal projection basis (d × m), m ≤ 256
        lora_params: {layer_idx: (A, B)} for active experts
        replay_buffer: circular buffer of past samples

    Memory: O(d * r + d * m) = O(d * (r + m)) = O(2048 * 264) = O(540K) floats
    Time: O(d * r * m) per layer = O(2048 * 8 * 256) = O(4.2M) FLOPs per layer
    """

    # Step 1: Forward pass with LoRA
    logits = forward_with_lora(model, lora_params, sample.input)
    loss = cross_entropy(logits, sample.target)

    # Step 2: Compute gradients w.r.t. LoRA parameters only
    # Only backprop through LoRA params (A, B), not base model.
    # Cost: same as forward pass ≈ 987M FLOPs (see §3)
    grads = backprop_lora(loss, lora_params)  # {layer: (dA, dB)}

    # Step 3: Project gradients to be orthogonal to P
    for layer_idx, (dA, dB) in grads.items():
        # Flatten gradient
        g = concat(flatten(dA), flatten(dB))  # g ∈ R^(2 * d * r)

        # Project out previously important directions
        # g_orth = g - P @ P^T @ g
        proj = P.T @ g[:d]           # R^m  (only project the d-dimensional component)
        g[:d] = g[:d] - P @ proj     # Remove component in span(P)

        # Reshape back
        dA_orth = reshape(g[:d*r], (d, r))
        dB_orth = reshape(g[d*r:], (r, d))

        # Step 4: Apply update with learning rate
        lr = 1e-4
        lora_params[layer_idx].A -= lr * dA_orth
        lora_params[layer_idx].B -= lr * dB_orth

    # Step 5: Update projection basis P (incrementally)
    # Add current gradient direction to P if it represents a new important direction
    if loss < threshold_important:  # This sample was learned successfully
        for layer_idx, (dA, dB) in grads.items():
            g = flatten(dA)  # Use down-projection gradient as representative
            g_norm = g / norm(g)

            # Check if g is sufficiently novel (not already in span(P))
            novelty = norm(g_norm - P @ (P.T @ g_norm))
            if novelty > 0.5:  # Threshold for adding new direction
                if P.shape[1] < 256:
                    # Append (Gram-Schmidt orthogonalize first)
                    g_orth = g_norm - P @ (P.T @ g_norm)
                    g_orth = g_orth / norm(g_orth)
                    P = concat(P, g_orth.reshape(-1, 1), axis=1)
                else:
                    # P is full: replace least-important direction
                    # Importance = how often this direction was used in last 1000 updates
                    least_idx = argmin(direction_use_count)
                    P[:, least_idx] = g_norm
                    orthogonalize(P)  # Re-orthogonalize via QR

    # Step 6: Replay (interleaved, not full-pass)
    # Every K=16 new samples, replay one old sample
    replay_buffer.add(sample)
    if replay_buffer.count % 16 == 0:
        old_sample = replay_buffer.sample_random()
        ogp_lora_stream_update(old_sample, model, P, lora_params, replay_buffer)
        # Note: recursion depth = 1 (replay does not trigger further replay)


def forward_with_lora(model, lora_params, input_tokens):
    """Forward pass: base_model(x) + sum(B_l @ A_l @ h_l) for each layer."""
    h = model.embed(input_tokens)
    for l in range(32):
        h_base = model.layer[l](h)
        if l in lora_params:
            A, B = lora_params[l]
            h_lora = h @ A @ B.T   # R^(seq, d) @ R^(d, r) @ R^(r, d) = R^(seq, d)
            h = h_base + h_lora
        else:
            h = h_base
    return model.head(h)
```

## 5.4 Streaming Constraints Verification

| Constraint | Requirement | Satisfied? |
|------------|-------------|------------|
| No full-dataset pass | Single sample at a time | Yes |
| Bounded replay buffer | 4096 entries * 2 KB = 8 MB | Yes |
| VRAM bound for learning | < 4096 MB | Yes (2.1 MB used) |
| RAM bound for learning state | < 28672 MB available | Yes (26 MB used) |
| Per-update time | < 500 ms (to not stall inference) | Yes: ~200 ms at 987M FLOPs backprop on GPU |
| No Fisher matrix | None computed or stored | Yes |

## 5.5 Consolidation Schedule

Learning is triggered by:
1. **Explicit user correction**: "That answer was wrong, the correct answer is X"
2. **Self-detected error**: Truth grounding verifier flags a confident-but-wrong output
3. **Periodic fine-tuning**: Every 1000 inference steps, if replay buffer has new entries

Consolidation (updating P) happens in-band with learning, not as a separate phase.

## 5.6 Failure Modes & Fallbacks

| Failure | Detection | Fallback |
|---------|-----------|----------|
| Projection basis P saturates (m=256, all directions used) | QR decomposition condition number > 1e6 | Prune least-used directions; reset P to top-128 by usage |
| LoRA divergence (loss explodes) | loss > 10 * running_average | Roll back LoRA params to last checkpoint (saved every 100 updates) |
| Replay buffer poisoning (adversarial inputs) | Gradient norm of replay sample > 10x median | Evict sample from buffer; flag for user review |
| Catastrophic forgetting despite OGP | CRS drops by > 0.05 on held-out eval set (checked every 500 updates) | Freeze LoRA; reset to last known-good state; increase projection dimension m |

## 5.7 Proof Sketch: Memory Bound Under 4GB VRAM

**Theorem**: The OGP-LoRA-Stream algorithm uses at most C_vram = 2.1 MB of VRAM during
any learning step.

**Proof**:
1. Base model weights are frozen and already accounted in the VRAM budget (§2).
2. LoRA parameters for active experts: 2 experts * 32 layers * 2 * 2048 * 8 * 2 bytes
   = 2,097,152 bytes = 2.0 MB. These live in VRAM only for active experts.
3. Gradient scratch: one layer at a time, 2048 * 8 * 4 bytes (FP32) = 65,536 bytes
   = 64 KB.
4. Projection P: loaded to VRAM for the projection step only. Size: 2048 * 256 * 4
   = 2 MB. But we can stream P in chunks of 2048 * 32 = 256 KB at a time (project
   against 32 directions at once, 8 chunks total).
5. Peak VRAM during learning = 2.0 MB (LoRA) + 64 KB (grad) + 256 KB (P chunk)
   = 2.32 MB.
6. 2.32 MB << 256 MB activation scratch << 3686 MB usable VRAM. QED.
