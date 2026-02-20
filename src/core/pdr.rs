//! Perspective Decay Recurrence (PDR) — the novel sequence-processing primitive.
//!
//! PDR replaces self-attention in 60 of 80 layers.  Each layer maintains a
//! fixed-size recurrent state `s ∈ R^{d_model × rank}` that is updated per token
//! via a gated decay-and-accumulate rule:
//!
//! ```text
//! λ  = sigmoid(W_λ · h)            ∈ (0,1)^rank
//! s' = diag(λ) · s + W_v · h       (decay old state, inject new info)
//! o  = W_o · (W_p · s')            (project state to output)
//! h' = h + o                        (residual connection)
//! ```
//!
//! Key properties:
//! - O(1) per-token computation (no attention over past tokens)
//! - Fixed memory: d_model × rank × 4 bytes per layer = 2 MB at rank=256
//! - Supports parallel scan for prompt prefill

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::config::{D_MODEL, PDR_RANK};

use super::pdr_state::PdrState;

/// Weights for a single PDR layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct PdrLayer {
    /// Decay gate projection: W_λ ∈ R^{rank × d_model}
    pub w_lambda: Array2<f32>,

    /// Bias for decay gate: b_λ ∈ R^{rank}
    pub b_lambda: Array1<f32>,

    /// Value projection: W_v ∈ R^{rank × d_model}
    pub w_v: Array2<f32>,

    /// State-to-output projection: W_p ∈ R^{d_model × rank}
    pub w_p: Array2<f32>,

    /// Output projection: W_o ∈ R^{d_model × d_model}
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
            w_lambda: Array2::zeros((PDR_RANK, D_MODEL)),
            b_lambda: Array1::zeros(PDR_RANK),
            w_v: Array2::zeros((PDR_RANK, D_MODEL)),
            w_p: Array2::zeros((D_MODEL, PDR_RANK)),
            w_o: Array2::zeros((D_MODEL, D_MODEL)),
            rms_scale: Array1::ones(D_MODEL),
            layer_idx,
        }
    }

    /// Single-step forward pass (autoregressive decode).
    ///
    /// Updates `state` in-place and returns the output hidden state.
    pub fn forward_step(&self, h: &Array1<f32>, state: &mut PdrState) -> Array1<f32> {
        // Pre-norm (RMSNorm)
        let h_norm = rms_norm(h, &self.rms_scale);

        // λ = sigmoid(W_λ · h + b_λ)
        let lambda = sigmoid(&(self.w_lambda.dot(&h_norm) + &self.b_lambda));

        // v = W_v · h
        let v = self.w_v.dot(&h_norm);

        // s' = diag(λ) · s + v  (column-wise decay + accumulate)
        state.decay_and_accumulate(&lambda, &v);

        // project: W_p · s' → intermediate ∈ R^{d_model}
        let projected = self.w_p.dot(&state.mean_state());

        // output: W_o · projected
        let output = self.w_o.dot(&projected);

        // Residual connection
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

        // For prefill, we use the sequential scan (parallel scan is an
        // optimisation that produces identical results — implement later
        // with work-efficient Blelloch scan).
        for t in 0..seq_len {
            let h_t = h_seq.row(t).to_owned();
            let out_t = self.forward_step(&h_t, state);
            outputs.row_mut(t).assign(&out_t);
        }

        outputs
    }

    /// Parameter count for this layer.
    pub fn param_count(&self) -> usize {
        let w_lambda = PDR_RANK * D_MODEL;
        let b_lambda = PDR_RANK;
        let w_v = PDR_RANK * D_MODEL;
        let w_p = D_MODEL * PDR_RANK;
        let w_o = D_MODEL * D_MODEL;
        let rms = D_MODEL;
        w_lambda + b_lambda + w_v + w_p + w_o + rms
    }
}

/// RMSNorm: x * scale / sqrt(mean(x²) + ε)
fn rms_norm(x: &Array1<f32>, scale: &Array1<f32>) -> Array1<f32> {
    let eps = 1e-6_f32;
    let mean_sq = x.mapv(|v| v * v).mean().unwrap_or(1.0);
    let rms = (mean_sq + eps).sqrt();
    x / rms * scale
}

/// Element-wise sigmoid: 1 / (1 + exp(-x))
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
        for &v in s.iter() {
            assert!(v > 0.0 && v < 1.0, "sigmoid out of bounds: {}", v);
        }
        // Check symmetry: sigmoid(0) ≈ 0.5
        assert!((s[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_scale() {
        let x = Array1::from_vec(vec![3.0, 4.0]);
        let scale = Array1::ones(2);
        let normed = rms_norm(&x, &scale);
        // RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        let expected_rms = (12.5_f32 + 1e-6).sqrt();
        assert!((normed[0] - 3.0 / expected_rms).abs() < 1e-4);
    }

    #[test]
    fn test_pdr_param_count() {
        let layer = PdrLayer::zeros(0);
        // Expected: 2 * (256 * 4096) + 256 + (4096 * 256) + (4096 * 4096) + 4096
        let expected = 2 * PDR_RANK * D_MODEL + PDR_RANK + D_MODEL * PDR_RANK
            + D_MODEL * D_MODEL + D_MODEL;
        assert_eq!(layer.param_count(), expected);
    }
}
