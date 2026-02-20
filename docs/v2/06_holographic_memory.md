# § 06 — Holographic Distributed Memory (HDM)

> Encode knowledge as interference patterns.  Retrieve in O(1).
> No index to maintain.  No vector database.  Just physics.

---

## 1  Motivation

Current LLM memory systems:

| Approach | Method | Drawback |
|----------|--------|----------|
| RAG + vector DB | Embed → HNSW / IVF → top-k retrieval | Index maintenance, O(log n) search, separate infrastructure |
| KV-cache | Store every past token's K, V | O(n) memory, no persistence across sessions |
| Fine-tuning | Bake knowledge into weights | Expensive, catastrophic forgetting |

**HDM is different:** it uses principles from **holographic / hyperdimensional
computing** (Kanerva, 2009; Plate, 2003) to store associations in
high-dimensional binary vectors.

Core properties:
- **O(1) storage:** adding a fact is one vector operation
- **O(1) retrieval:** querying is one correlation operation  
- **Self-organising:** no index structure to build or maintain
- **Graceful degradation:** capacity is soft — quality degrades smoothly
- **CPU-friendly:** all operations are XOR + popcount (SIMD-native)

---

## 2  Hyperdimensional Computing Primer

### 2.1  Representation

Every concept is a **10,000-bit binary vector** (called a *hypervector*).

```
"Paris"  = [1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1, ... ]  (10,000 bits = 1.25 KB)
"France" = [0,1,0,1,1,0,0,1,1,0,1,0,1,1,0,0, ... ]
"capital"= [1,1,0,0,1,1,0,1,0,0,1,1,0,1,0,0, ... ]
```

Random hypervectors are **approximately orthogonal** with high probability:

$$P\left[\text{cos\_sim}(a, b) > 0.1\right] < 10^{-10} \quad \text{for random } a, b \in \{0,1\}^{10000}$$

### 2.2  Three Operations

| Operation | Symbol | Definition | Purpose |
|-----------|--------|------------|---------|
| **Binding** | $a \circledast b$ | Circular convolution (or XOR for binary) | Create association between two concepts |
| **Bundling** | $a + b$ | Element-wise majority vote (or thresholded sum) | Superimpose multiple items into one vector |
| **Similarity** | $\delta(a, b)$ | Hamming distance (or normalised dot product) | Check if two vectors are related |

### 2.3  Key Properties

1. **Binding is invertible:** $a \circledast b \circledast b^{-1} \approx a$
   (where $b^{-1}$ is the "unbound" of $b$: bit-reversal for binary vectors)
2. **Bundling preserves components:** $\delta(a + b, a) > \text{threshold}$
3. **Binding distributes over bundling:** $a \circledast (b + c) = (a \circledast b) + (a \circledast c)$

---

## 3  HDM Architecture

### 3.1  Memory Banks

HDM organises memory into **2,000 banks**, each holding a superposition
of bound associations:

```
Bank_k = Σᵢ (key_vector_i ⊛ value_vector_i)   for all facts in bank k
```

Each bank is a single 10,000-bit vector (1.25 KB).  
Total index: 2,000 × 1.25 KB = **2.5 MB**.

### 3.2  Codebook

A **codebook** maps between the model's embedding space and hypervectors:

- 4,096 learned basis hypervectors (one per embedding dimension)
- To convert a model embedding $e \in \mathbb{R}^{4096}$ to a hypervector:

$$
\text{HV}(e) = \text{threshold}\!\left(\sum_{i=1}^{4096} e_i \cdot \text{basis}_i\right)
$$

where each basis$_i \in \{0, 1\}^{10000}$ and the threshold binarises at the
median.  This is a learned **locality-sensitive hash** from embedding space
to hypervector space.

Codebook size: 4,096 × 10,000 bits = **5 MB**.

### 3.3  Text Store

The actual text content of stored facts is kept alongside the holographic
index:

```
struct MemoryEntry {
    id: u64,
    key_text: String,       // "capital of France"
    value_text: String,     // "Paris"
    key_hv: BitVec<10000>,  // hypervector of key
    bank_id: u16,           // which bank this was bundled into
    timestamp: u64,
    confidence: f32,
}
```

Stored linearly in RAM: ~50,000 entries × ~100 bytes avg = **~5 MB**.

### 3.4  Total Memory Footprint

| Component | Size |
|-----------|------|
| Bank superpositions (2,000 × 1.25 KB) | 2.5 MB |
| Codebook (4,096 × 1.25 KB) | 5.0 MB |
| Entry metadata (50K × 100 B) | 5.0 MB |
| Working buffers | 0.5 MB |
| **Total** | **~13 MB** |

This fits trivially in RAM with massive headroom.

---

## 4  Operations

### 4.1  Store (Insert a Fact)

```
Input: key_embedding ∈ R^4096, value_text: &str

1. key_hv = codebook.encode(key_embedding)        // → {0,1}^10000
2. value_hv = codebook.encode(embed(value_text))   // → {0,1}^10000
3. bound = key_hv XOR value_hv                      // binding
4. bank_id = hash(key_hv) mod 2000                  // deterministic bank
5. banks[bank_id] = majority(banks[bank_id], bound) // bundle into bank
6. entries.push(MemoryEntry { ... })                 // store metadata
```

**Cost:** 3 × 10,000 bit-ops + hash = ~0.01 ms

### 4.2  Retrieve (Query a Fact)

```
Input: query_embedding ∈ R^4096

1. query_hv = codebook.encode(query_embedding)      // → {0,1}^10000
2. bank_id = hash(query_hv) mod 2000                 // same hash
3. unbound = banks[bank_id] XOR query_hv             // unbind query
4. // unbound ≈ value_hv if (query ⊛ value) was bundled in this bank
5. candidates = entries.filter(e => e.bank_id == bank_id)
6. best = argmin_{c ∈ candidates} hamming(unbound, c.value_hv)
7. return entries[best].value_text
```

**Cost:** XOR + popcount over ~25 candidates = ~0.03 ms

### 4.3  Multi-Bank Retrieval

If the fact might be in multiple banks (fuzzy key):

```
1. query_hv = codebook.encode(query_embedding)
2. // Check the top-k most similar banks
3. for bank_id in top_k_banks(query_hv, k=5):
4.     unbound = banks[bank_id] XOR query_hv
5.     score = min hamming distance among bank's entries
6. return entry with global best score
```

**Cost:** 5 × (XOR + popcount scan) = ~0.15 ms

---

## 5  Capacity Analysis

### 5.1  Per-Bank Capacity

A 10,000-bit bank can hold $\sim \sqrt{D}$ associations before
interference degrades retrieval below 90% accuracy:

$$
C_{\text{bank}} \approx \sqrt{10000} \approx 100 \text{ associations}
$$

With 2,000 banks: total capacity ≈ **200,000 associations** at ≥90% accuracy.

### 5.2  Graceful Degradation

| Associations per bank | Retrieval accuracy |
|----------------------|-------------------|
| 10 | 99.9% |
| 25 | 99.2% |
| 50 | 97.1% |
| 100 | 90.0% |
| 200 | 75.0% |
| 500 | 50.0% |

Above capacity, the bank becomes "noisy" — retrieval still works but
requires more candidate checking.

### 5.3  Scaling to 500K+ Entries

For larger knowledge bases, HDM uses **hierarchical banks on NVMe**:

```
Level 0 (RAM):   2,000 banks → 200K entries, 13 MB
Level 1 (NVMe):  20,000 banks → 2M entries, 130 MB
Level 2 (NVMe):  200,000 banks → 20M entries, 1.3 GB
```

Level 1+ banks are loaded on demand (single bank = 1.25 KB, negligible
NVMe read time).

---

## 6  Integration with the Model

### 6.1  When HDM is Consulted

HDM retrieval is triggered during **Multi-Perspective Decoding** (§ 07):

1. Four perspectives generate candidate tokens
2. If agreement < threshold → uncertainty detected
3. **HDM query:** the uncertain span's embedding is used as a retrieval key
4. Retrieved facts are injected as context for re-generation

### 6.2  When HDM is Updated

After every generation turn:

1. **User corrections:** if the user corrects a fact, store the correction
2. **Confident assertions:** high-agreement factual claims are stored as
   self-reinforcing associations
3. **External knowledge:** user-provided documents are chunked and encoded

### 6.3  Interaction with PDR State

The PDR state captures *how to process* information; HDM stores *what facts
are known*.  When HDM retrieves a fact:

- The fact's embedding is injected as a "virtual token" into the residual
  stream at the HDM integration layer (layer 40, midpoint)
- This modulates all subsequent PDR state updates without breaking the
  recurrence

---

## 7  Comparison with Prior Memory Systems

| Feature | RAG + HNSW | MemoryBank | **HDM (ours)** |
|---------|-----------|-----------|----------------|
| Index structure | Graph (O(log n)) | Flat scan | **None (O(1))** |
| Insert time | O(log n) + rebuild | O(1) | **O(1)** |
| Retrieval time | 0.5-2 ms | 1-5 ms | **0.03 ms** |
| Memory overhead | Index = 2-5× data | Minimal | **1.25 KB/bank** |
| Maintenance | Periodic re-index | None | **None** |
| Persistence | External DB | In-memory | **Binary file** |
| Capacity scaling | ∞ (disk) | RAM-limited | **Hierarchical** |
| Hardware | GPU-accelerated | CPU | **CPU (SIMD)** |
| Interpretability | Nearest-neighbour | Exact match | **Distributed** |

---

## 8  Implementation Notes

### 8.1  SIMD Acceleration

All HDM operations reduce to bit-vector XOR and popcount (Hamming distance).
These are natively supported on all modern CPUs:

| ISA | Instruction | Throughput |
|-----|------------|-----------|
| x86 AVX2 | `vpxor` + `vpopcntq` | 64 bits/cycle |
| x86 AVX-512 | `vpxorq` + `vpopcntq` (VPOPCNTDQ) | 512 bits/cycle |
| ARM NEON | `veorq` + `vcntq` | 128 bits/cycle |

A 10,000-bit XOR + popcount takes:
- AVX-512: ~20 cycles (~6 ns at 3 GHz)
- AVX2: ~160 cycles (~50 ns)

### 8.2  File Format

```
HDM save file (.hdm):
┌────────────────────────┐
│ Magic: "HDM1"  (4B)    │
│ Version: u32           │
│ Num banks: u32         │
│ Dim: u32 (10000)       │
│ Num entries: u64       │
├────────────────────────┤
│ Bank data (2000 × 1250B) │
├────────────────────────┤
│ Codebook (4096 × 1250B) │
├────────────────────────┤
│ Entry table (variable) │
└────────────────────────┘
```

---

*Next: [§ 07 Multi-Perspective Decoding](07_multi_perspective.md)*
