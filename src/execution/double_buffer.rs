//! Double-buffered expert loading — overlap NVMe I/O with GPU computation.
//!
//! Two RAM buffers (A and B) alternate:
//! - While the GPU computes layer ℓ using buffer A,
//!   the NVMe→RAM DMA fills buffer B with layer ℓ+1's expert.
//!
//! This hides most of the PCIe transfer latency behind computation.


use crate::execution::ternary_pack::TernaryExpertFfn;

/// Buffer state.
#[derive(Clone, Debug, PartialEq)]
pub enum BufferState {
    /// Buffer is empty / available for loading.
    Empty,

    /// Buffer is being filled from NVMe (expert_id being loaded).
    Loading(usize),

    /// Buffer contains a ready expert.
    Ready(usize),

    /// Buffer is being consumed by the compute pipeline.
    InUse(usize),
}

/// A single expert buffer slot.
pub struct ExpertBuffer {
    /// Current state.
    pub state: BufferState,

    /// The expert weights (if loaded).
    pub expert: Option<TernaryExpertFfn>,

    /// Buffer identifier (A=0, B=1).
    pub id: usize,
}

impl ExpertBuffer {
    pub fn new(id: usize) -> Self {
        Self {
            state: BufferState::Empty,
            expert: None,
            id,
        }
    }

    /// Check if the buffer contains a specific expert and is ready.
    pub fn has_ready(&self, expert_id: usize) -> bool {
        self.state == BufferState::Ready(expert_id)
    }

    /// Mark as loading.
    pub fn start_loading(&mut self, expert_id: usize) {
        self.state = BufferState::Loading(expert_id);
        self.expert = None;
    }

    /// Complete loading — transition from Loading to Ready.
    pub fn finish_loading(&mut self, expert: TernaryExpertFfn) {
        if let BufferState::Loading(id) = self.state {
            self.state = BufferState::Ready(id);
            self.expert = Some(expert);
        }
    }

    /// Acquire for compute — transition from Ready to InUse.
    pub fn acquire(&mut self) -> Option<&TernaryExpertFfn> {
        if let BufferState::Ready(id) = self.state {
            self.state = BufferState::InUse(id);
            self.expert.as_ref()
        } else {
            None
        }
    }

    /// Release after compute — transition from InUse to Empty.
    pub fn release(&mut self) {
        if let BufferState::InUse(_) = self.state {
            self.state = BufferState::Empty;
            self.expert = None;
        }
    }

    /// Get the expert ID currently in this buffer (any state).
    pub fn current_expert(&self) -> Option<usize> {
        match self.state {
            BufferState::Empty => None,
            BufferState::Loading(id) | BufferState::Ready(id) | BufferState::InUse(id) => {
                Some(id)
            }
        }
    }
}

/// The double-buffer system with two alternating buffers.
pub struct DoubleBuffer {
    /// Buffer A.
    pub buffer_a: ExpertBuffer,

    /// Buffer B.
    pub buffer_b: ExpertBuffer,

    /// Which buffer is currently the "compute" buffer (0=A, 1=B).
    pub compute_idx: usize,

    /// Statistics.
    pub stats: DoubleBufferStats,
}

/// Performance statistics for the double buffer.
#[derive(Clone, Debug, Default)]
pub struct DoubleBufferStats {
    /// Number of times compute had to stall waiting for a load.
    pub stalls: u64,

    /// Number of successful overlapped loads (no stall).
    pub overlapped: u64,

    /// Number of cache hits (expert already in buffer).
    pub cache_hits: u64,

    /// Total layers processed.
    pub layers_processed: u64,
}

impl DoubleBufferStats {
    pub fn stall_rate(&self) -> f64 {
        let total = self.stalls + self.overlapped + self.cache_hits;
        if total == 0 {
            0.0
        } else {
            self.stalls as f64 / total as f64
        }
    }

    pub fn overlap_rate(&self) -> f64 {
        let total = self.stalls + self.overlapped + self.cache_hits;
        if total == 0 {
            0.0
        } else {
            self.overlapped as f64 / total as f64
        }
    }
}

impl DoubleBuffer {
    pub fn new() -> Self {
        Self {
            buffer_a: ExpertBuffer::new(0),
            buffer_b: ExpertBuffer::new(1),
            compute_idx: 0,
            stats: DoubleBufferStats::default(),
        }
    }

    /// Get the current compute buffer.
    pub fn compute_buffer(&self) -> &ExpertBuffer {
        if self.compute_idx == 0 {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }

    /// Get the current compute buffer mutably.
    pub fn compute_buffer_mut(&mut self) -> &mut ExpertBuffer {
        if self.compute_idx == 0 {
            &mut self.buffer_a
        } else {
            &mut self.buffer_b
        }
    }

    /// Get the loading (non-compute) buffer.
    pub fn loading_buffer(&self) -> &ExpertBuffer {
        if self.compute_idx == 0 {
            &self.buffer_b
        } else {
            &self.buffer_a
        }
    }

    /// Get the loading buffer mutably.
    pub fn loading_buffer_mut(&mut self) -> &mut ExpertBuffer {
        if self.compute_idx == 0 {
            &mut self.buffer_b
        } else {
            &mut self.buffer_a
        }
    }

    /// Swap compute and loading buffers.
    pub fn swap(&mut self) {
        self.compute_idx = 1 - self.compute_idx;
    }

    /// Check if the requested expert is already in one of the buffers.
    pub fn find_expert(&self, expert_id: usize) -> Option<usize> {
        if self.buffer_a.has_ready(expert_id) {
            Some(0)
        } else if self.buffer_b.has_ready(expert_id) {
            Some(1)
        } else {
            None
        }
    }

    /// Process one layer: ensure the expert is available, return a reference.
    /// Returns the expert FFN weights for computation.
    ///
    /// This orchestrates the double-buffer protocol:
    /// 1. If expert is already in compute buffer → use directly
    /// 2. If expert is in loading buffer → swap and use
    /// 3. Otherwise → stall and load synchronously
    pub fn get_expert_for_layer(
        &mut self,
        expert_id: usize,
        loader: &dyn Fn(usize) -> TernaryExpertFfn,
    ) -> &TernaryExpertFfn {
        self.stats.layers_processed += 1;

        // Check compute buffer
        if self.compute_buffer().has_ready(expert_id) {
            self.stats.cache_hits += 1;
            let buf = self.compute_buffer_mut();
            buf.acquire();
            return buf.expert.as_ref().unwrap();
        }

        // Check loading buffer
        if self.loading_buffer().has_ready(expert_id) {
            self.stats.overlapped += 1;
            self.swap();
            let buf = self.compute_buffer_mut();
            buf.acquire();
            return buf.expert.as_ref().unwrap();
        }

        // Stall: load synchronously into compute buffer
        self.stats.stalls += 1;
        let buf = self.compute_buffer_mut();
        buf.start_loading(expert_id);
        let expert = loader(expert_id);
        buf.finish_loading(expert);
        buf.acquire();
        buf.expert.as_ref().unwrap()
    }

    /// Start prefetching the next expert into the loading buffer.
    /// This should be called after acquiring the compute buffer.
    pub fn prefetch(
        &mut self,
        next_expert_id: usize,
        loader: &dyn Fn(usize) -> TernaryExpertFfn,
    ) {
        let buf = self.loading_buffer_mut();
        // Don't reload if already loaded
        if buf.has_ready(next_expert_id) {
            return;
        }
        buf.start_loading(next_expert_id);
        let expert = loader(next_expert_id);
        buf.finish_loading(expert);
    }

    /// Release the compute buffer after layer computation is done.
    pub fn release_compute(&mut self) {
        self.compute_buffer_mut().release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ternary_pack::TernaryMatrix;

    fn make_dummy_expert() -> TernaryExpertFfn {
        TernaryExpertFfn {
            w_gate: TernaryMatrix::zeros(8, 4),
            w_up: TernaryMatrix::zeros(8, 4),
            w_down: TernaryMatrix::zeros(4, 8),
            gate_scale: 1.0,
            up_scale: 1.0,
            down_scale: 1.0,
        }
    }

    #[test]
    fn test_buffer_lifecycle() {
        let mut buf = ExpertBuffer::new(0);
        assert_eq!(buf.state, BufferState::Empty);

        buf.start_loading(42);
        assert_eq!(buf.state, BufferState::Loading(42));

        buf.finish_loading(make_dummy_expert());
        assert_eq!(buf.state, BufferState::Ready(42));
        assert!(buf.has_ready(42));

        let _expert = buf.acquire();
        assert_eq!(buf.state, BufferState::InUse(42));

        buf.release();
        assert_eq!(buf.state, BufferState::Empty);
    }

    #[test]
    fn test_double_buffer_swap() {
        let mut db = DoubleBuffer::new();
        assert_eq!(db.compute_idx, 0);
        db.swap();
        assert_eq!(db.compute_idx, 1);
        db.swap();
        assert_eq!(db.compute_idx, 0);
    }

    #[test]
    fn test_get_expert_stall() {
        let mut db = DoubleBuffer::new();
        let loader = |_id: usize| make_dummy_expert();
        let _expert = db.get_expert_for_layer(5, &loader);
        assert_eq!(db.stats.stalls, 1);
        assert_eq!(db.stats.layers_processed, 1);
    }

    #[test]
    fn test_get_expert_cache_hit() {
        let mut db = DoubleBuffer::new();
        let loader = |_id: usize| make_dummy_expert();

        // First load: stall
        let _ = db.get_expert_for_layer(5, &loader);
        db.release_compute();

        // Re-load same expert into compute buffer
        db.compute_buffer_mut().start_loading(5);
        db.compute_buffer_mut().finish_loading(make_dummy_expert());

        // Second access: cache hit
        let _ = db.get_expert_for_layer(5, &loader);
        assert_eq!(db.stats.cache_hits, 1);
    }
}
