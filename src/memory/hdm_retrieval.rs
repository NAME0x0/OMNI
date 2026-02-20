//! HDM Retrieval — content-addressable memory search.
//!
//! Given a query hypervector, find the most similar stored traces.
//! Uses Hamming-space nearest-neighbour search with optional indexing.

use crate::memory::hdm::HyperVector;
use crate::memory::hdm_binding::MemoryTrace;

/// Result of a memory retrieval.
#[derive(Clone, Debug)]
pub struct RetrievalResult {
    /// Index into the memory store.
    pub index: usize,

    /// Cosine-like similarity (Hamming-based).
    pub similarity: f64,

    /// Label of the matching trace.
    pub label: String,

    /// Timestamp of the matching trace.
    pub timestamp: u64,
}

/// The episodic memory store — a flat collection of traces with search.
pub struct EpisodicMemory {
    /// All stored traces.
    pub traces: Vec<MemoryTrace>,

    /// Maximum capacity (oldest evicted when full).
    pub capacity: usize,

    /// Write pointer (ring buffer index).
    write_ptr: usize,

    /// Total traces ever written (for statistics).
    total_written: u64,
}

impl EpisodicMemory {
    /// Create a new episodic memory with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            traces: Vec::with_capacity(capacity),
            capacity,
            write_ptr: 0,
            total_written: 0,
        }
    }

    /// Store a trace. Evicts oldest if at capacity.
    pub fn store(&mut self, trace: MemoryTrace) {
        if self.traces.len() < self.capacity {
            self.traces.push(trace);
        } else {
            self.traces[self.write_ptr] = trace;
        }
        self.write_ptr = (self.write_ptr + 1) % self.capacity;
        self.total_written += 1;
    }

    /// Retrieve the top-k most similar traces to a query vector.
    pub fn retrieve(
        &self,
        query: &HyperVector,
        k: usize,
    ) -> Vec<RetrievalResult> {
        let mut scored: Vec<RetrievalResult> = self
            .traces
            .iter()
            .enumerate()
            .map(|(i, trace)| RetrievalResult {
                index: i,
                similarity: trace.vector.similarity(query),
                label: trace.label.clone(),
                timestamp: trace.timestamp,
            })
            .collect();

        // Sort by similarity descending.
        scored.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(k);
        scored
    }

    /// Retrieve traces above a similarity threshold.
    pub fn retrieve_threshold(
        &self,
        query: &HyperVector,
        threshold: f64,
    ) -> Vec<RetrievalResult> {
        self.traces
            .iter()
            .enumerate()
            .filter_map(|(i, trace)| {
                let sim = trace.vector.similarity(query);
                if sim >= threshold {
                    Some(RetrievalResult {
                        index: i,
                        similarity: sim,
                        label: trace.label.clone(),
                        timestamp: trace.timestamp,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Number of stored traces.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Is the memory empty?
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Total traces ever written.
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.traces.iter().map(|t| t.vector.size_bytes()).sum()
    }

    /// Clear all traces.
    pub fn clear(&mut self) {
        self.traces.clear();
        self.write_ptr = 0;
    }

    /// Get a trace by index.
    pub fn get(&self, index: usize) -> Option<&MemoryTrace> {
        self.traces.get(index)
    }
}

/// Working memory — small, high-priority buffer for recent context.
/// Distinct from episodic memory: smaller capacity, no eviction threshold.
pub struct WorkingMemory {
    /// Active traces in working memory.
    pub slots: Vec<Option<MemoryTrace>>,

    /// Number of occupied slots.
    pub occupied: usize,
}

impl WorkingMemory {
    /// Create working memory with given number of slots.
    pub fn new(num_slots: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
            occupied: 0,
        }
    }

    /// Insert a trace into the first available slot, or replace LRU.
    pub fn insert(&mut self, trace: MemoryTrace) {
        // Find first empty slot
        for slot in self.slots.iter_mut() {
            if slot.is_none() {
                *slot = Some(trace);
                self.occupied += 1;
                return;
            }
        }
        // All full — replace oldest (lowest timestamp)
        let oldest_idx = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|t| (i, t.timestamp)))
            .min_by_key(|&(_, ts)| ts)
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.slots[oldest_idx] = Some(trace);
    }

    /// Query all occupied slots for the best match.
    pub fn query_best(&self, query: &HyperVector) -> Option<(usize, f64)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref()
                    .map(|t| (i, t.vector.similarity(query)))
            })
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Clear a slot.
    pub fn clear_slot(&mut self, index: usize) {
        if index < self.slots.len() && self.slots[index].is_some() {
            self.slots[index] = None;
            self.occupied -= 1;
        }
    }

    /// Number of slots.
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::hdm::HyperVector;
    use crate::memory::hdm_binding::MemoryTrace;

    fn make_trace(seed: u64, label: &str, ts: u64) -> MemoryTrace {
        MemoryTrace::from_vector(HyperVector::random(seed), label, ts)
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = EpisodicMemory::new(100);
        let t1 = make_trace(1, "trace1", 0);
        let query = t1.vector.clone();
        mem.store(t1);

        let results = mem.retrieve(&query, 1);
        assert_eq!(results.len(), 1);
        assert!((results[0].similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_retrieve_top_k() {
        let mut mem = EpisodicMemory::new(100);
        for i in 0..10 {
            mem.store(make_trace(i, &format!("t{}", i), i));
        }

        let query = HyperVector::random(0); // same seed as trace 0
        let results = mem.retrieve(&query, 3);
        assert_eq!(results.len(), 3);
        // First result should be the exact match
        assert!((results[0].similarity - 1.0).abs() < 1e-10);
        assert_eq!(results[0].label, "t0");
    }

    #[test]
    fn test_capacity_eviction() {
        let mut mem = EpisodicMemory::new(3);
        mem.store(make_trace(1, "a", 0));
        mem.store(make_trace(2, "b", 1));
        mem.store(make_trace(3, "c", 2));
        assert_eq!(mem.len(), 3);

        // This should evict "a" (oldest, at write_ptr=0)
        mem.store(make_trace(4, "d", 3));
        assert_eq!(mem.len(), 3);
        assert_eq!(mem.total_written(), 4);
    }

    #[test]
    fn test_threshold_retrieval() {
        let mut mem = EpisodicMemory::new(100);
        let target = HyperVector::random(42);
        mem.store(MemoryTrace::from_vector(
            target.clone(),
            "target",
            0,
        ));
        // Add noise traces
        for i in 1..10 {
            mem.store(make_trace(i * 1000, &format!("noise{}", i), i));
        }

        let results = mem.retrieve_threshold(&target, 0.9);
        assert!(results.len() >= 1);
        assert_eq!(results[0].label, "target");
    }

    #[test]
    fn test_working_memory_insert() {
        let mut wm = WorkingMemory::new(3);
        wm.insert(make_trace(1, "a", 0));
        wm.insert(make_trace(2, "b", 1));
        assert_eq!(wm.occupied, 2);
    }

    #[test]
    fn test_working_memory_eviction() {
        let mut wm = WorkingMemory::new(2);
        wm.insert(make_trace(1, "a", 0));
        wm.insert(make_trace(2, "b", 1));
        // Full — should evict oldest (timestamp 0)
        wm.insert(make_trace(3, "c", 2));
        assert_eq!(wm.occupied, 2);
    }

    #[test]
    fn test_working_memory_query() {
        let mut wm = WorkingMemory::new(5);
        let target = make_trace(42, "target", 0);
        let query = target.vector.clone();
        wm.insert(target);
        wm.insert(make_trace(99, "other", 1));

        let best = wm.query_best(&query);
        assert!(best.is_some());
        let (_, sim) = best.unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_bytes() {
        let mut mem = EpisodicMemory::new(10);
        mem.store(make_trace(1, "a", 0));
        mem.store(make_trace(2, "b", 1));
        let bytes = mem.memory_bytes();
        assert_eq!(bytes, 2 * 1250); // 2 traces × 1250 bytes each
    }

    #[test]
    fn test_clear() {
        let mut mem = EpisodicMemory::new(10);
        mem.store(make_trace(1, "a", 0));
        mem.clear();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
    }
}
