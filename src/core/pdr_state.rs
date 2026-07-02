//! PDR recurrent state management.
//!
//! Each PDR layer maintains a state matrix `S_t in R^{d_model x rank}`.
//! The state is updated as `S_t = diag(gamma_t) S_{t-1} + v_t k_t^T`.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Recurrent state for a single PDR layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct PdrState {
    /// State matrix: `S in R^{d_model x rank}`.
    pub state: Array2<f32>,

    /// Number of tokens processed through this state.
    pub tokens_seen: u64,
}

impl PdrState {
    /// Create a new zero-initialised state for the given `d_model`/`rank`.
    pub fn new(d_model: usize, rank: usize) -> Self {
        Self {
            state: Array2::zeros((d_model, rank)),
            tokens_seen: 0,
        }
    }

    /// Decay the current state and accumulate an outer-product update.
    ///
    /// `S' = diag(gamma) S + v k^T`
    ///
    /// If any vector has an unexpected shape (relative to this state's own
    /// `d_model`/`rank`, derived from `self.state`'s shape), this is a no-op.
    pub fn decay_and_accumulate(&mut self, gamma: &Array1<f32>, v: &Array1<f32>, k: &Array1<f32>) {
        let (d_model, rank) = self.state.dim();
        if gamma.len() != d_model || v.len() != d_model || k.len() != rank {
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
        let (d_model, rank) = self.state.dim();
        if q.len() != rank {
            return Array1::zeros(d_model);
        }
        self.state.dot(q)
    }

    /// Mean over rank dimension for diagnostics.
    pub fn mean_state(&self) -> Array1<f32> {
        let d_model = self.state.nrows();
        self.state
            .mean_axis(Axis(1))
            .unwrap_or_else(|| Array1::zeros(d_model))
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
        self.state.len() * std::mem::size_of::<f32>() + std::mem::size_of::<u64>()
    }
}

/// Collection of PDR states for all PDR layers in the model.
#[derive(Clone, Serialize, Deserialize)]
pub struct PdrStateBank {
    pub states: Vec<PdrState>,
}

impl PdrStateBank {
    /// Create a new bank with `n` zero-initialised states of the given
    /// `d_model`/`rank`.
    pub fn new(n: usize, d_model: usize, rank: usize) -> Self {
        Self {
            states: (0..n).map(|_| PdrState::new(d_model, rank)).collect(),
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
    use crate::core::ModelDims;

    #[test]
    fn test_state_init() {
        let dims = ModelDims::tiny();
        let state = PdrState::new(dims.d_model, dims.pdr_rank);
        assert_eq!(state.state.shape(), &[dims.d_model, dims.pdr_rank]);
        assert_eq!(state.tokens_seen, 0);
        assert!(state.is_healthy());
    }

    #[test]
    fn test_decay_and_accumulate() {
        let dims = ModelDims::tiny();
        let mut state = PdrState::new(dims.d_model, dims.pdr_rank);
        let gamma = Array1::from_vec(vec![0.9; dims.d_model]);
        let v = Array1::from_vec(vec![1.0; dims.d_model]);
        let k = Array1::from_vec(vec![1.0; dims.pdr_rank]);

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
        let dims = ModelDims::tiny();
        let state = PdrState::new(dims.d_model, dims.pdr_rank);
        let q = Array1::ones(dims.pdr_rank);
        let out = state.readout(&q);
        assert_eq!(out.len(), dims.d_model);
    }

    #[test]
    fn test_serialisation_roundtrip() {
        let dims = ModelDims::tiny();
        let mut state = PdrState::new(dims.d_model, dims.pdr_rank);
        let gamma = Array1::from_vec(vec![0.5; dims.d_model]);
        let v = Array1::from_vec(vec![0.3; dims.d_model]);
        let k = Array1::from_vec(vec![0.4; dims.pdr_rank]);
        state.decay_and_accumulate(&gamma, &v, &k);

        let bytes = state.to_bytes().unwrap();
        let restored = PdrState::from_bytes(&bytes).unwrap();
        assert_eq!(state.tokens_seen, restored.tokens_seen);
        assert!((state.state[[0, 0]] - restored.state[[0, 0]]).abs() < 1e-8);
    }

    #[test]
    fn test_state_bank() {
        let dims = ModelDims::tiny();
        let bank = PdrStateBank::new(dims.n_pdr_layers, dims.d_model, dims.pdr_rank);
        assert_eq!(bank.states.len(), dims.n_pdr_layers);
        assert!(bank.all_healthy());
    }

    #[test]
    fn test_norm() {
        let dims = ModelDims::tiny();
        let mut state = PdrState::new(dims.d_model, dims.pdr_rank);
        assert!((state.norm() - 0.0).abs() < 1e-8);

        let gamma = Array1::ones(dims.d_model);
        let v = Array1::ones(dims.d_model);
        let k = Array1::ones(dims.pdr_rank);
        state.decay_and_accumulate(&gamma, &v, &k);
        let expected_norm = ((dims.d_model * dims.pdr_rank) as f32).sqrt();
        assert!((state.norm() - expected_norm).abs() < 1e-4);
    }
}
