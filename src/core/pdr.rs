//! Perspective Decay Recurrence (PDR).
//!
//! This layer keeps a recurrent matrix state `S_t in R^{d_model x rank}`:
//!
//! ```text
//! p_t     = W_p h_t + b_p
//! gamma_t = sigmoid(p_t)
//! k_t     = W_k h_t
//! v_t     = W_v h_t
//! S_t     = diag(gamma_t) S_{t-1} + v_t k_t^T
//! q_t     = W_q h_t
//! o_hat_t = S_t q_t
//! o_t     = W_o o_hat_t
//! h'_t    = h_t + o_t
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::config::{D_MODEL, PDR_RANK};

use super::pdr_state::PdrState;

/// Weights for a single PDR layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct PdrLayer {
    /// Perspective projection: `W_p in R^{d_model x d_model}`.
    pub w_p: Array2<f32>,

    /// Perspective bias: `b_p in R^{d_model}`.
    pub b_p: Array1<f32>,

    /// Key projection: `W_k in R^{rank x d_model}`.
    pub w_k: Array2<f32>,

    /// Value projection: `W_v in R^{d_model x d_model}`.
    pub w_v: Array2<f32>,

    /// Query projection: `W_q in R^{rank x d_model}`.
    pub w_q: Array2<f32>,

    /// Output projection: `W_o in R^{d_model x d_model}`.
    pub w_o: Array2<f32>,

    /// RMSNorm scale for pre-norm.
    pub rms_scale: Array1<f32>,

    /// Layer index (for identification).
    pub layer_idx: usize,
}

impl PdrLayer {
    /// Create a new PDR layer with zero-initialised weights.
    pub fn zeros(layer_idx: usize) -> Self {
        Self {
            w_p: Array2::zeros((D_MODEL, D_MODEL)),
            b_p: Array1::zeros(D_MODEL),
            w_k: Array2::zeros((PDR_RANK, D_MODEL)),
            w_v: Array2::zeros((D_MODEL, D_MODEL)),
            w_q: Array2::zeros((PDR_RANK, D_MODEL)),
            w_o: Array2::zeros((D_MODEL, D_MODEL)),
            rms_scale: Array1::ones(D_MODEL),
            layer_idx,
        }
    }

    /// Single-step forward pass (autoregressive decode).
    ///
    /// Updates `state` in-place and returns the output hidden state.
    pub fn forward_step(&self, h: &Array1<f32>, state: &mut PdrState) -> Array1<f32> {
        if h.len() != D_MODEL {
            return h.clone();
        }

        // Pre-norm (RMSNorm)
        let h_norm = rms_norm(h, &self.rms_scale);

        // Perspective -> decay gate
        let p = self.w_p.dot(&h_norm) + &self.b_p;
        let gamma = sigmoid(&p);

        // Key/value projections for recurrent update
        let k = self.w_k.dot(&h_norm);
        let v = self.w_v.dot(&h_norm);

        // S_t = diag(gamma_t) S_{t-1} + v_t k_t^T
        state.decay_and_accumulate(&gamma, &v, &k);

        // Query and readout
        let q = self.w_q.dot(&h_norm);
        let readout = state.readout(&q);

        // Output projection + residual connection
        let output = self.w_o.dot(&readout);
        h + &output
    }

    /// Parallel scan for prompt prefill (processes all tokens at once).
    ///
    /// Returns the final hidden states and the updated PDR state.
    pub fn forward_prefill(
        &self,
        h_seq: &Array2<f32>, // [seq_len, d_model]
        state: &mut PdrState,
    ) -> Array2<f32> {
        let seq_len = h_seq.nrows();
        let mut outputs = Array2::zeros((seq_len, D_MODEL));

        // Sequential reference path (parallel scan can replace this later).
        for t in 0..seq_len {
            let h_t = h_seq.row(t).to_owned();
            let out_t = self.forward_step(&h_t, state);
            outputs.row_mut(t).assign(&out_t);
        }

        outputs
    }

    /// Parameter count for this layer.
    pub fn param_count(&self) -> usize {
        let w_p = D_MODEL * D_MODEL;
        let b_p = D_MODEL;
        let w_k = PDR_RANK * D_MODEL;
        let w_v = D_MODEL * D_MODEL;
        let w_q = PDR_RANK * D_MODEL;
        let w_o = D_MODEL * D_MODEL;
        let rms = D_MODEL;
        w_p + b_p + w_k + w_v + w_q + w_o + rms
    }
}

/// RMSNorm: `x * scale / sqrt(mean(x^2) + eps)`.
fn rms_norm(x: &Array1<f32>, scale: &Array1<f32>) -> Array1<f32> {
    let eps = 1e-6_f32;
    let mean_sq = x.mapv(|v| v * v).mean().unwrap_or(1.0);
    let rms = (mean_sq + eps).sqrt();
    x / rms * scale
}

/// Element-wise sigmoid: `1 / (1 + exp(-x))`.
fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_pdr_step_shape() {
        let layer = PdrLayer::zeros(0);
        let mut state = PdrState::new();
        let h = Array1::ones(D_MODEL);
        let out = layer.forward_step(&h, &mut state);
        assert_eq!(out.len(), D_MODEL);
    }

    #[test]
    fn test_pdr_residual() {
        // With zero weights, output should equal input (residual only)
        let layer = PdrLayer::zeros(0);
        let mut state = PdrState::new();
        let h = Array1::from_vec(vec![1.0; D_MODEL]);
        let out = layer.forward_step(&h, &mut state);
        for i in 0..D_MODEL {
            assert!((out[i] - h[i]).abs() < 1e-5, "Residual broken at {}", i);
        }
    }

    #[test]
    fn test_sigmoid_bounds() {
        let x = Array1::from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0]);
        let s = sigmoid(&x);
        for &v in &s {
            assert!(v > 0.0 && v < 1.0, "sigmoid out of bounds: {}", v);
        }
        // Check symmetry: sigmoid(0) ~= 0.5
        assert!((s[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_scale() {
        let x = Array1::from_vec(vec![3.0, 4.0]);
        let scale = Array1::ones(2);
        let normed = rms_norm(&x, &scale);
        // RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ~= 3.536
        let expected_rms = (12.5_f32 + 1e-6).sqrt();
        assert!((normed[0] - 3.0 / expected_rms).abs() < 1e-4);
    }

    #[test]
    fn test_pdr_param_count() {
        let layer = PdrLayer::zeros(0);
        let expected = 3 * D_MODEL * D_MODEL + 2 * PDR_RANK * D_MODEL + 2 * D_MODEL;
        assert_eq!(layer.param_count(), expected);
    }
}
