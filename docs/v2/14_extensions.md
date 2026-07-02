# § 14 — Future Extensions (Adopted from v1)
> Some v1 subsystems were tied to OMNIS-SINGULARITY.  Some were not.
> This file preserves the orthogonal ones as candidates, not commitments.
---
## §0  Scope
This document catalogues v1 ideas that may be worth carrying forward into
PERSPECTIVE after the core v2 model works.  These are future extensions, not
current requirements.
This document is not an implementation claim.  Nothing below is implemented
for v2 unless explicitly marked as an existing substrate.  All items are gated
behind §10 Stage 0 small-scale architecture validation and the §12 validation
plan for the core PDR + ternary MoE + HDM + MPD + SPP loop.
---
## §1  Verifier Cascade
**Source:** legacy §07 truth grounding, plus the v1 §13 cascade repair.
**Status:** Design candidate — not implemented.
### 1.1  Summary
The v1 verifier design classified extracted factual claims against retrieved
evidence.  The later v1 repair made it a three-tier cascade:
| Tier | Mechanism | Decision |
|------|-----------|----------|
| 1 | 150M encoder-only verifier | SUPPORTED / REFUTED / INSUFFICIENT |
| 2 | Main-model self-check | Re-check uncertain claims against the same evidence |
| 3 | Evidence vote | Count supporting vs contradicting retrieved entries |
The action semantics were:
| Condition | Action |
|-----------|--------|
| Supported claim | Continue |
| Confidently refuted claim | Halt and branch |
| Branches exhausted | Abstain |
| Evidence insufficient | Branch if possible; otherwise continue with caveat |

### 1.2  Why It Complements MPD
MPD detects internal disagreement between decoding perspectives.  A verifier
cascade checks whether a claim is supported by retrieved evidence.  Those are
orthogonal signals: all MPD perspectives can agree on a false claim, while a
verifier can fail when evidence is missing, stale, or poisoned.
### 1.3  v2 Integration Sketch
If adopted, the verifier would sit after MPD and before final SPP handling:
```
PDR / GQA / ternary expert forward
  → MPD disagreement handling + HDM retrieval
  → claim extraction + verifier cascade
  → halt / branch / abstain decision
  → SPP projection and final output policy
```
Claim extraction should stay CPU-side and deterministic: sentence splitting,
factual-claim filtering, compound-claim decomposition, and HDM evidence lookup.
### 1.4  Cost
| Item | Cost |
|------|------|
| Verifier weights | 400 MB VRAM — estimate from v1 design (unvalidated) |
| Verifier activation scratch | 1.5 MB VRAM — estimate from v1 design (unvalidated) |
| Tier-1 verifier | ~2 ms per claim — estimate from v1 design (unvalidated) |
| Claim extraction | ~0.5 ms per output — estimate from v1 design (unvalidated) |
| Legacy post-hoc path for 10 claims | ~26.8 ms per output — estimate from v1 design (unvalidated) |
| Cascaded expected path | ~22 ms per output — estimate from v1 design (unvalidated) |

These numbers came from the old 2.5B GLA MoE design and are not evidence for
v2 runtime behavior.
---
## §2  Degradation Ladder & Circuit Breakers
**Source:** legacy §13 self-healing.
**Status:** Partial substrate exists (`src/runtime/health.rs`); ladder not
implemented.
### 2.1  Summary
The v1 design proposed circuit breakers plus a five-level degradation ladder:
| Level | Name | Runtime policy |
|-------|------|----------------|
| 0 | NOMINAL | All systems active |
| 1 | ELEVATED | Disable speculation; reduce lookahead |
| 2 | DEGRADED | Disable verification; reduce reasoning depth |
| 3 | FALLBACK | Use resident experts only; disable retrieval |
| 4 | EMERGENCY | CPU-only or smallest available inference path |
Circuit breakers would be per subsystem:
| Subsystem | Trigger | Fallback |
|-----------|---------|----------|
| Expert streaming | repeated load failure or stall | use current resident experts |
| HDM retrieval | repeated retrieval failure | generate without retrieval |
| Verification | verifier failure or timeout | skip verifier gate |
| Adaptation | unsafe update failure | freeze adaptation |
| Tool execution | repeated sandbox failure | disable tools for the conversation |

### 2.2  Why v2 Needs This More Than v1
PERSPECTIVE streams ternary experts from NVMe.  That adds failure modes beyond
the v1 GLA MoE: NVMe stalls, hot-cache misses, cache thrash, PCIe contention,
and delta-load failures that must fall back to full expert loads.  Without a
ladder, these become hangs or uncontrolled latency spikes.
### 2.3  Existing Substrate
`src/runtime/health.rs` already tracks rolling latency, VRAM, RAM, MPD
agreement, cache hit rate, and expert load stalls, with aggregate
`Healthy` / `Degraded` / `Critical` status.  This extension would add:
1. breaker state per subsystem
2. recovery timers and half-open probes
3. a concrete `DegradationLevel` enum
4. runtime policy switches for each level
5. §12 stress tests for escalation and recovery
### 2.4  Cost
| Item | Cost |
|------|------|
| Breaker state + metrics | ~10 KB RAM — estimate from v1 design (unvalidated) |
| Health polling decode-path latency | 0 ms — estimate from v1 design (unvalidated) |
| Health polling CPU | <0.1% of one core — estimate from v1 design (unvalidated) |
---
## §3  Multimodal Front-Ends
**Source:** legacy §12 multimodal.
**Status:** Design candidate — not implemented.
### 3.1  Summary
The v1 multimodal design kept encoders on CPU and passed projected embeddings
into the model as virtual tokens:
| Modality | Encoder | v2 input form |
|----------|---------|---------------|
| Vision | MobileCLIP-S2 on CPU | projected visual virtual tokens |
| Audio | Whisper-Tiny on CPU | transcribed text tokens |

For v2, the projection should inject into the residual stream using the same
virtual-token mechanism described for HDM in §06.6.3.  It should not alter PDR
recurrence.
### 3.2  Vision
```
Image → CPU MobileCLIP-S2 → projection MLP → residual-stream virtual tokens
```
| Item | Cost |
|------|------|
| MobileCLIP-S2 encoder | 35 MB RAM — estimate from v1 design (unvalidated) |
| Vision projector | 10 MB VRAM — estimate from v1 design (unvalidated) |
| 224px image latency | ~150 ms — estimate from v1 design (unvalidated) |
| 336px image latency | ~300 ms — estimate from v1 design (unvalidated) |

Quality caveat: the legacy doc was explicit that MobileCLIP-S2 is far below
frontier vision models.  This is a low-resource utility path, not a frontier
multimodal claim.
### 3.3  Audio
```
Audio → CPU Whisper-Tiny → transcript → normal tokenizer
```
| Item | Cost |
|------|------|
| Whisper-Tiny encoder | 39 MB RAM — estimate from v1 design (unvalidated) |
| Audio VRAM delta | 0 MB — estimate from v1 design (unvalidated) |
| 30-second audio chunk latency | ~500 ms — estimate from v1 design (unvalidated) |

This gives speech-to-text, not audio understanding.  Music, environmental
sound, speaker identity, emotion, and diarization remain out of scope.
---
## §4  Input Hardening & Retrieval Quarantine
**Source:** legacy §11 adversarial robustness.
**Status:** Design candidate — not implemented.  The v2 HDM `MemoryEntry`
sketch in §06 already carries `confidence` and `timestamp`; trust scoring
would extend that metadata.
### 4.1  Summary
The v1 hardening stack included Unicode normalization, zero-width removal,
homoglyph handling, injection heuristics, retrieval sanitization, provenance,
content hashes, quarantine, and periodic audit.
For v2, retrieval quarantine is the important carry-forward.  HDM can accept
writes from user corrections, documents, and potentially generation-time
consolidation.  Model-generated writes can be false or adversarially induced,
so poisoning risk is real.
### 4.2  Mapping to HDM
| HDM field / extension | Use |
|-----------------------|-----|
| `confidence` | initial write confidence or verifier confidence |
| `timestamp` | audit recency and stale-entry handling |
| `trust_score` | source reliability and promotion state |
| `provenance` | system, user-confirmed, model-generated, external ingest |
| `quarantine` | exclude from normal retrieval unless trusted results are insufficient |
| `content_hash` | detect tampering or stale serialized data |
Suggested write policy:
| Source | Initial handling |
|--------|------------------|
| System knowledge | trusted, read-only |
| User-confirmed correction | trusted after explicit user action |
| Model-generated summary | quarantined unless verifier-backed |
| External document | quarantined by default |
### 4.3  Cost
| Item | Cost |
|------|------|
| Trust score only | 8 MB RAM for 2M entries — estimate from v1 design (unvalidated) |
| Provenance + hash + flags | 96 MB RAM for 2M entries — estimate from v1 design (unvalidated) |
| Retrieval filtering | ~0.1 ms per retrieval — estimate from v1 design (unvalidated) |
| Periodic audit | ~200 ms per 100 entries in background — estimate from v1 design (unvalidated) |
This is high priority because it guards the core memory loop.
---
## §5  Sandboxed Tool Runtime
**Source:** legacy §08 agent runtime and §11 tool-abuse prevention.
**Status:** Design candidate — not implemented as a v2 tool runtime.  The repo
already has a ZeroClaw adapter (`src/runtime/zeroclaw.rs`) for external
agentic dispatch, which partially covers this need today.
### 5.1  Summary
The v1 runtime proposed sandboxed tool execution with allowlists, per-tool
argument validation, invocation limits, blocked argument patterns, user
confirmation for destructive operations, subprocess isolation, timeout
handling, and output truncation.
This is relevant only once PERSPECTIVE becomes an agent that can choose tools.
Until then, it is the lowest-priority extension here.
### 5.2  Policy Sketch
| Tool class | Required control |
|------------|------------------|
| Code execution | sandbox path, timeout, memory cap, no ambient network |
| File read | allowed directories only |
| File write | allowed directories and confirmation for destructive writes |
| Web fetch | domain allowlist and local-network denylist |
| Memory insert | quarantine path from §4 |
| External agent dispatch | validated task string and bounded workspace |
The current ZeroClaw adapter is a thin CLI wrapper.  It validates empty tasks
and delegates execution externally, but it is not a full allowlist,
argument-validation, or sandbox policy engine.
### 5.3  Cost
| Item | Cost |
|------|------|
| Tool policy tables | ~50 KB RAM — estimate from v1 design (unvalidated) |
| Tool-call validation | ~0.02 ms per tool call — estimate from v1 design (unvalidated) |
---
## §6  Priority
| Extension | Prerequisite | Priority | Rationale |
|-----------|--------------|----------|-----------|
| Degradation ladder & circuit breakers | Core inference loop emits reliable runtime metrics | HIGH | Guards the streamed-expert loop against stalls, OOM, and cache failure |
| Input hardening & retrieval quarantine | HDM write path and provenance metadata | HIGH | Prevents memory poisoning in a system that can write its own retrieved context |
| Verifier cascade | Working MPD + HDM + claim extraction benchmark | MEDIUM | Adds external grounding, but needs a functioning model first |
| Multimodal front-ends | Stable residual virtual-token injection | MEDIUM-LOW | Useful capability, but not required to validate PERSPECTIVE |
| Sandboxed tool runtime | Agentic tool-use product scope | LOW | Relevant only when the model is allowed to act through tools |
---
*Back to [§ 00 Index](00_V2_INDEX.md)*
