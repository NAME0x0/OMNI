# Section 6: Topological Memory

## 6.1 Overview

The topological memory stores 2M+ historical entries in a hybrid graph/manifold index
that supports:
- **Insertion**: Adding new memories from inference or user input
- **Retrieval**: Finding relevant memories via LSH + graph traversal
- **Consolidation**: Merging/compressing related memories over time
- **Garbage collection**: Evicting stale or redundant entries

All data structures reside in **host RAM** (not VRAM). Retrieval results are transferred
to GPU only as small tensors (32 vectors * 1 KB each = 32 KB per retrieval).

## 6.2 Data Structures

### 6.2.1 Memory Entry

```rust
struct MemoryEntry {
    id: u64,                        // 8 bytes
    vector: [f16; 2048],            // 4096 bytes — semantic embedding
    text_hash: u64,                 // 8 bytes — pointer to compressed text store
    timestamp: u64,                 // 8 bytes — insertion time (monotonic counter)
    access_count: u32,              // 4 bytes — retrieval frequency
    last_accessed: u64,             // 8 bytes — last retrieval time
    importance: f32,                // 4 bytes — computed importance score
    edges: SmallVec<[u32; 16]>,     // 64 bytes — graph edges (neighbor indices)
    cluster_id: u32,                // 4 bytes — manifold cluster assignment
    // Total: ~4204 bytes per entry
}
```

### 6.2.2 Graph Index: HNSW (Hierarchical Navigable Small World)

```
Parameters:
  M = 16          — max edges per node per layer
  M_0 = 32        — max edges on layer 0
  ef_construction = 200  — beam width during insertion
  ef_search = 64         — beam width during retrieval
  max_layers = 6         — ceil(ln(2M) / ln(M)) for 2M entries

Layer structure (approximate node counts):
  Layer 5: ~2 nodes          (entry points)
  Layer 4: ~30 nodes
  Layer 3: ~480 nodes
  Layer 2: ~7,700 nodes
  Layer 1: ~123,000 nodes
  Layer 0: ~2,000,000 nodes  (all entries)
```

### 6.2.3 Memory Layout in Host RAM

| Component | Size | Derivation |
|-----------|------|------------|
| Memory entries (2M) | 2M * 4204 B = 8,408 MB | See entry struct above |
| HNSW edge lists (layer 0) | 2M * 32 edges * 4 B = 256 MB | M_0=32 |
| HNSW edge lists (layers 1-5) | ~131K nodes * 16 * 4 B = 8.4 MB | Upper layers sparse |
| LSH hash tables (8 tables) | 8 * 256 buckets * 4 KB = 8 MB | Bucket = list of entry IDs |
| Cluster metadata (4096 clusters) | 4096 * 2048 * 2 B = 16 MB | Cluster centroids (FP16) |
| Compressed text store | 2048 MB | ~1 KB avg compressed text per entry |
| **Total** | **~10,745 MB** | |

**Wait — this exceeds our host RAM budget.**

### Budget Reconciliation

The §2 budget allocated:
- Topological memory index: 4096 MB
- Topological memory vectors: 8192 MB
- Topological memory metadata: 2048 MB
- Total: 14,336 MB

Our actual need: 10,745 MB. **Within budget.**

But the total host RAM used (§2) was 29,212 MB. Let me recheck with actual sizes:

| §2 line item | §2 estimate | Actual | Delta |
|--------------|-------------|--------|-------|
| Topo memory index | 4096 MB | 264 MB (HNSW edges) | -3832 MB |
| Topo memory vectors | 8192 MB | 8408 MB (entries w/ vectors) | +216 MB |
| Topo memory metadata | 2048 MB | 2072 MB (text + clusters) | +24 MB |
| **Subtotal topo** | **14,336** | **10,744** | **-3,592 MB** |

Revised total host RAM: 29,212 - 3,592 = **25,620 MB**. Leaves **7,148 MB** headroom.

This headroom allows scaling to **~3.4M entries** before hitting the 28,672 MB usable
RAM limit (each additional entry costs ~5.2 KB including HNSW edges).

## 6.3 Operations

### 6.3.1 Insertion

```
insert(entry):
    1. Compute embedding: vector = encode(entry.text)
       Time: ~5ms (CPU inference of small encoder, or use main model's embedding)
       [HEURISTIC] We assume a pre-computed embedding; if computed on GPU, add 2ms.

    2. HNSW insert:
       a. Assign random level l = floor(-ln(uniform()) * (1/ln(M)))
       b. Greedy search from top layer to layer l+1 (find nearest entry point)
       c. For each layer l..0:
          - Beam search with ef_construction=200 to find 200 nearest neighbors
          - Connect to top-M (or M_0 at layer 0) neighbors
          - If neighbor has too many edges, prune weakest
       Time complexity: O(M * ef_construction * log(N)) = O(16 * 200 * 21) = O(67,200)
       Wall time: ~2 ms (CPU, memory-bound on L3 cache)

    3. LSH index update:
       - Compute L=8 hash values for entry.vector
       - Append entry.id to each of the 8 bucket lists
       Time: O(L * d) = O(8 * 2048) = O(16,384) — negligible

    4. Cluster assignment:
       - Find nearest cluster centroid (4096 centroids, brute force)
       - Time: O(4096 * 2048) = O(8.4M) FLOPs — ~0.02 ms on CPU

    Total insertion time: ~7 ms
    Space per entry: ~5.2 KB (4204 B entry + ~1 KB HNSW overhead)
```

### 6.3.2 Retrieval

```
retrieve(query_vector, k=32):
    # Two-phase retrieval: LSH for coarse candidates, HNSW for precise ranking

    Phase 1: LSH (fast, approximate)
      - Compute 8 hash values for query_vector
      - Collect union of 8 bucket contents
      - Expected candidates: ~500 (for 2M entries, 256 buckets/table)
      Time: O(L * B) where B ≈ 500. ~0.1 ms

    Phase 2: HNSW graph traversal (precise)
      - Start from best LSH candidate
      - Beam search with ef_search=64 on HNSW layer 0
      - Compute exact cosine similarity for visited nodes
      - Return top-k=32
      Time: O(ef_search * M * d) = O(64 * 16 * 2048) = O(2.1M) FLOPs. ~0.5 ms

    Phase 3: Load associated data
      - Read text snippets for top-32 entries from compressed store
      - 32 * 1 KB = 32 KB sequential read — negligible

    Total retrieval time: ~0.62 ms
    Bandwidth: ~1 MB read from host RAM (candidates + graph traversal)
```

### 6.3.3 Consolidation

Consolidation merges related memories to prevent unbounded growth and improve
retrieval quality. Runs as a background task on CPU.

```
consolidate(trigger=every_10000_insertions):
    # Step 1: Identify clusters with > 1000 entries
    for cluster in clusters:
        if cluster.count > 1000:
            entries = get_cluster_entries(cluster.id)

            # Step 2: Within-cluster deduplication
            # Use MinHash on text content (not vectors) for exact dedup
            duplicates = minhash_dedup(entries, threshold=0.85)
            for dup_group in duplicates:
                # Keep highest-importance entry, merge access counts
                keeper = max(dup_group, key=lambda e: e.importance)
                keeper.access_count += sum(e.access_count for e in dup_group) - keeper.access_count
                for e in dup_group:
                    if e != keeper:
                        remove(e)  # Marks for GC

            # Step 3: Summarization of low-importance entries
            low_imp = [e for e in entries if e.importance < threshold_low]
            if len(low_imp) > 500:
                # Merge groups of 10 similar low-importance entries into 1 summary
                groups = kmeans_within_cluster(low_imp, k=len(low_imp)//10)
                for group in groups:
                    summary_vector = mean([e.vector for e in group])
                    summary_text = summarize([e.text for e in group])  # CPU text summarization
                    new_entry = MemoryEntry(vector=summary_vector, text=summary_text, ...)
                    insert(new_entry)
                    for e in group:
                        remove(e)

    # Step 4: Recompute cluster centroids
    for cluster in clusters:
        cluster.centroid = mean(get_cluster_entries(cluster.id).vectors)

    Time complexity: O(N_cluster * N_within^2 / N_clusters) for dedup
                   = O(2M * 500 / 4096) ≈ O(244K) comparisons — ~50 ms
    Frequency: Every 10K insertions (~once per hour of active use)
```

### 6.3.4 Garbage Collection

```
gc_policy:
    Trigger: when entry count > 2.5M OR host RAM usage > 26 GB

    Phase 1: Score all entries
      score(e) = importance * (1 + log(1 + access_count)) * decay(age)
      decay(age) = max(0.1, 1.0 - age / (365 * 24 * 3600))  # Linear decay over 1 year
      Time: O(N) = O(2M) — ~10 ms

    Phase 2: Evict lowest-scored entries until count < 2M
      Sort by score, remove bottom entries.
      For each removed entry:
        - Remove from HNSW: disconnect edges, repair neighbors' edge lists
        - Remove from LSH buckets
        - Free text store space (mark as free in allocator)
      Time: O(K_evict * M * log(N)) for HNSW repair, K_evict entries removed

    Phase 3: Compact memory (optional, during idle)
      - Defragment entry storage array (copy-compact)
      - Rebuild LSH bucket lists (full rehash)
      Time: O(N * d) — ~5 seconds for full compaction

    Frequency: Rare (only when approaching memory limit)
```

## 6.4 Worst-Case Degradation Behavior

| Scenario | Impact | Mitigation |
|----------|--------|------------|
| All 2M entries in same HNSW region (adversarial clustering) | Retrieval degrades to O(N) scan | LSH provides independent access path; if HNSW ef_search returns poor results, fall back to LSH-only retrieval |
| LSH hash collision (all entries in one bucket) | Bucket size = 2M, retrieval = O(N) | Use multi-probe LSH with perturbation; if bucket > 5000, trigger rehash with new hash functions |
| Memory fragmentation after many GC cycles | Allocation failures despite free space | Compaction phase (5s pause during idle); pre-allocate fixed-size arena |
| Concurrent insert during retrieval | Inconsistent reads | Lock-free via atomic pointer swaps for HNSW edges; retrievals see consistent snapshot |
| Entry count reaches 3.4M (RAM limit) | GC cannot keep up with insertions | Hard cap: reject insertions until GC completes; alert user |
| Cluster centroids drift after many consolidations | Retrieval quality degrades | Full re-clustering every 100K consolidation cycles (~annually) |

## 6.5 Complexity Summary

| Operation | Time Complexity | Wall Time | Space |
|-----------|----------------|-----------|-------|
| Insert | O(M * ef_c * log N) | ~7 ms | 5.2 KB/entry |
| Retrieve (top-32) | O(ef_s * M * d + L * B) | ~0.62 ms | 32 KB returned |
| Consolidate (batch) | O(N/C * K^2) | ~50 ms per batch | In-place |
| GC (evict K entries) | O(K * M * log N) | ~100 ms for K=500K | Frees K * 5.2 KB |
| Compact | O(N * d) | ~5 s | Temporary 2x space during copy |
