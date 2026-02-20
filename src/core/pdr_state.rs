//! PDR recurrent state management.
//!
//! Each PDR layer maintains a state matrix `S_t in R^{d_model x rank}`.
//! The state is updated as `S_t = diag(gamma_t) S_{t-1} + v_t k_t^T`.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::config::{D_MODEL, PDR_RANK};

/// Recurrent state for a single PDR layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct PdrState {
    /// State matrix: `S in R^{d_model x rank}`.
    pub state: Array2<f32>,

    /// Number of tokens processed through this state.
    pub tokens_seen: u64,
}

impl PdrState {
    /// Create a new zero-initialised state.
    pub fn new() -> Self {
        Self {
            state: Array2::zeros((D_MODEL, PDR_RANK)),
            tokens_seen: 0,
        }
    }

    /// Decay the current state and accumulate an outer-product update.
    ///
    /// `S' = diag(gamma) S + v k^T`
    ///
    /// If any vector has an unexpected shape, this is a no-op.
    pub fn decay_and_accumulate(&mut self, gamma: &Array1<f32>, v: &Array1<f32>, k: &Array1<f32>) {
        if gamma.len() != D_MODEL || v.len() != D_MODEL || k.len() != PDR_RANK {
            return;
        }

        let gamma_col = gamma.view().insert_axis(Axis(1));
        let v_col = v.view().insert_axis(Axis(1));
        let k_row = k.view().insert_axis(Axis(0));
        let update = v_col.dot(&k_row);

        self.state = &self.state * &gamma_col + &update;
        self.tokens_seen = self.tokens_seen.saturating_add(1);
    }

    /// Read out state with a query vector.
    ///
    /// `o_hat = S q`
    pub fn readout(&self, q: &Array1<f32>) -> Array1<f32> {
        if q.len() != PDR_RANK {
            return Array1::zeros(D_MODEL);
        }
        self.state.dot(q)
    }

    /// Mean over rank dimension for diagnostics.
    pub fn mean_state(&self) -> Array1<f32> {
        self.state
            .mean_axis(Axis(1))
            .unwrap_or_else(|| Array1::zeros(D_MODEL))
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

    /// Frobenius norm of the state (for monitoring divergence).
    pub fn norm(&self) -> f32 {
        self.state.mapv(|v| v * v).sum().sqrt()
    }

    /// Serialise state to bytes (for persistence across sessions).
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialise state from bytes.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }

    /// Memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        D_MODEL * PDR_RANK * std::mem::size_of::<f32>() + std::mem::size_of::<u64>()
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
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
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
        assert_eq!(state.state.shape(), &[D_MODEL, PDR_RANK]);
        assert_eq!(state.tokens_seen, 0);
        assert!(state.is_healthy());
    }

    #[test]
    fn test_decay_and_accumulate() {
        let mut state = PdrState::new();
        let gamma = Array1::from_vec(vec![0.9; D_MODEL]);
        let v = Array1::from_vec(vec![1.0; D_MODEL]);
        let k = Array1::from_vec(vec![1.0; PDR_RANK]);

        state.decay_and_accumulate(&gamma, &v, &k);
        assert_eq!(state.tokens_seen, 1);
        // S = diag(gamma) * 0 + v k^T = 1
        assert!((state.state[[0, 0]] - 1.0).abs() < 1e-6);

        state.decay_and_accumulate(&gamma, &v, &k);
        assert_eq!(state.tokens_seen, 2);
        // S = 0.9 * 1.0 + 1.0 = 1.9
        assert!((state.state[[0, 0]] - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_readout_shape() {
        let state = PdrState::new();
        let q = Array1::ones(PDR_RANK);
        let out = state.readout(&q);
        assert_eq!(out.len(), D_MODEL);
    }

    #[test]
    fn test_serialisation_roundtrip() {
        let mut state = PdrState::new();
        let gamma = Array1::from_vec(vec![0.5; D_MODEL]);
        let v = Array1::from_vec(vec![0.3; D_MODEL]);
        let k = Array1::from_vec(vec![0.4; PDR_RANK]);
        state.decay_and_accumulate(&gamma, &v, &k);

        let bytes = state.to_bytes().unwrap();
        let restored = PdrState::from_bytes(&bytes).unwrap();
        assert_eq!(state.tokens_seen, restored.tokens_seen);
        assert!((state.state[[0, 0]] - restored.state[[0, 0]]).abs() < 1e-8);
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

        let gamma = Array1::ones(D_MODEL);
        let v = Array1::ones(D_MODEL);
        let k = Array1::ones(PDR_RANK);
        state.decay_and_accumulate(&gamma, &v, &k);
        let expected_norm = ((D_MODEL * PDR_RANK) as f32).sqrt();
        assert!((state.norm() - expected_norm).abs() < 1e-4);
    }
}
