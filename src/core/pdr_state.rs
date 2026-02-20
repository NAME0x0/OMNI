//! PDR recurrent state management.
//!
//! Each PDR layer maintains a state matrix `s ∈ R^{rank}` that is a compressed
//! representation of the entire sequence history.  The state is updated via
//! decay-and-accumulate: `s' = λ ⊙ s + v`.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::config::PDR_RANK;

/// Recurrent state for a single PDR layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct PdrState {
    /// State vector: s ∈ R^{rank}
    pub state: Array1<f32>,

    /// Number of tokens processed through this state.
    pub tokens_seen: u64,
}

impl PdrState {
    /// Create a new zero-initialised state.
    pub fn new() -> Self {
        Self {
            state: Array1::zeros(PDR_RANK),
            tokens_seen: 0,
        }
    }

    /// Decay the current state by `λ` and accumulate `v`.
    ///
    /// `s' = λ ⊙ s + v`
    ///
    /// - `lambda`: decay factors ∈ (0, 1)^rank
    /// - `v`: new value to accumulate ∈ R^rank
    pub fn decay_and_accumulate(&mut self, lambda: &Array1<f32>, v: &Array1<f32>) {
        self.state = &self.state * lambda + v;
        self.tokens_seen += 1;
    }

    /// Get the mean state (for projection).  For now, returns the state
    /// directly.  In a multi-head variant, this would average across heads.
    pub fn mean_state(&self) -> Array1<f32> {
        self.state.clone()
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.tokens_seen = 0;
    }

    /// Check if the state has diverged (NaN or Inf).
    pub fn is_healthy(&self) -> bool {
        self.state.iter().all(|v| v.is_finite())
    }

    /// L2 norm of the state (for monitoring divergence).
    pub fn norm(&self) -> f32 {
        self.state.mapv(|v| v * v).sum().sqrt()
    }

    /// Serialise state to bytes (for persistence across sessions).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("PdrState serialisation should not fail")
    }

    /// Deserialise state from bytes.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }

    /// Memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        PDR_RANK * std::mem::size_of::<f32>() + std::mem::size_of::<u64>()
    }
}

impl Default for PdrState {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of PDR states for all PDR layers in the model.
#[derive(Clone, Serialize, Deserialize)]
pub struct PdrStateBank {
    pub states: Vec<PdrState>,
}

impl PdrStateBank {
    /// Create a new bank with `n` zero-initialised states.
    pub fn new(n: usize) -> Self {
        Self {
            states: (0..n).map(|_| PdrState::new()).collect(),
        }
    }

    /// Reset all states.
    pub fn reset_all(&mut self) {
        for s in &mut self.states {
            s.reset();
        }
    }

    /// Check all states are healthy.
    pub fn all_healthy(&self) -> bool {
        self.states.iter().all(|s| s.is_healthy())
    }

    /// Total memory footprint in bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.states.iter().map(|s| s.size_bytes()).sum()
    }

    /// Serialise all states.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("PdrStateBank serialisation should not fail")
    }

    /// Deserialise all states.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_init() {
        let state = PdrState::new();
        assert_eq!(state.state.len(), PDR_RANK);
        assert_eq!(state.tokens_seen, 0);
        assert!(state.is_healthy());
    }

    #[test]
    fn test_decay_and_accumulate() {
        let mut state = PdrState::new();
        let lambda = Array1::from_vec(vec![0.9; PDR_RANK]);
        let v = Array1::from_vec(vec![1.0; PDR_RANK]);

        state.decay_and_accumulate(&lambda, &v);
        assert_eq!(state.tokens_seen, 1);
        // s = 0.9 * 0 + 1.0 = 1.0
        assert!((state.state[0] - 1.0).abs() < 1e-6);

        state.decay_and_accumulate(&lambda, &v);
        assert_eq!(state.tokens_seen, 2);
        // s = 0.9 * 1.0 + 1.0 = 1.9
        assert!((state.state[0] - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_serialisation_roundtrip() {
        let mut state = PdrState::new();
        let lambda = Array1::from_vec(vec![0.5; PDR_RANK]);
        let v = Array1::from_vec(vec![0.3; PDR_RANK]);
        state.decay_and_accumulate(&lambda, &v);

        let bytes = state.to_bytes();
        let restored = PdrState::from_bytes(&bytes).unwrap();
        assert_eq!(state.tokens_seen, restored.tokens_seen);
        assert!((state.state[0] - restored.state[0]).abs() < 1e-8);
    }

    #[test]
    fn test_state_bank() {
        let bank = PdrStateBank::new(60);
        assert_eq!(bank.states.len(), 60);
        assert!(bank.all_healthy());
    }

    #[test]
    fn test_norm() {
        let mut state = PdrState::new();
        assert!((state.norm() - 0.0).abs() < 1e-8);

        let lambda = Array1::ones(PDR_RANK);
        let v = Array1::ones(PDR_RANK);
        state.decay_and_accumulate(&lambda, &v);
        let expected_norm = (PDR_RANK as f32).sqrt();
        assert!((state.norm() - expected_norm).abs() < 1e-4);
    }
}
