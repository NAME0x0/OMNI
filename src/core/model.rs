//! Top-level model skeleton — wires PDR, GQA, routing, expert FFN, and all
//! novel components into the full 80-layer PERSPECTIVE architecture.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::config;
use crate::core::pdr::PdrLayer;
use crate::core::pdr_state::PdrStateBank;
use crate::core::windowed_gqa::{GqaLayer, KVCacheBank};
use crate::core::ModelDims;

/// SwiGLU Feed-Forward Network weights (shared/non-expert layers use this directly;
/// expert layers use the ternary-packed variant via the execution module).
#[derive(Clone, Serialize, Deserialize)]
pub struct FfnWeights {
    /// Gate projection: W_gate ∈ R^{ffn_intermediate × d_model}
    pub w_gate: Array2<f32>,

    /// Up projection: W_up ∈ R^{ffn_intermediate × d_model}
    pub w_up: Array2<f32>,

    /// Down projection: W_down ∈ R^{d_model × ffn_intermediate}
    pub w_down: Array2<f32>,

    /// RMSNorm scale for pre-FFN norm.
    pub rms_scale: Array1<f32>,
}

impl FfnWeights {
    /// Create zero-initialised weights for the given `d_model`/`ffn_intermediate`.
    pub fn zeros(d_model: usize, ffn_intermediate: usize) -> Self {
        Self {
            w_gate: Array2::zeros((ffn_intermediate, d_model)),
            w_up: Array2::zeros((ffn_intermediate, d_model)),
            w_down: Array2::zeros((d_model, ffn_intermediate)),
            rms_scale: Array1::ones(d_model),
        }
    }

    /// SwiGLU forward: down(silu(gate(x)) ⊙ up(x))
    pub fn forward(&self, h: &Array1<f32>) -> Array1<f32> {
        let h_norm = rms_norm(h, &self.rms_scale);
        let gate = self.w_gate.dot(&h_norm).mapv(silu);
        let up = self.w_up.dot(&h_norm);
        let intermediate = gate * up;
        self.w_down.dot(&intermediate)
    }

    pub fn param_count(&self) -> usize {
        self.w_gate.len() + self.w_up.len() + self.w_down.len() + self.rms_scale.len()
    }
}

/// Embedding layer (shared between input and output).
#[derive(Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Embedding matrix: [vocab_size, d_model]
    pub weight: Array2<f32>,
}

impl Embedding {
    /// Create a zero-initialised embedding table for the given
    /// `vocab_size`/`d_model`.
    pub fn zeros(vocab_size: usize, d_model: usize) -> Self {
        Self {
            weight: Array2::zeros((vocab_size, d_model)),
        }
    }

    /// Look up embedding for a single token.
    pub fn embed(&self, token_id: u32) -> Array1<f32> {
        self.weight.row(token_id as usize).to_owned()
    }

    /// Project hidden state to logits (tied weights).
    pub fn to_logits(&self, h: &Array1<f32>) -> Array1<f32> {
        self.weight.dot(h)
    }
}

/// A single transformer layer — either PDR or GQA, plus expert FFN.
pub enum LayerKind {
    Pdr(PdrLayer),
    Gqa(GqaLayer),
}

/// The complete PERSPECTIVE model.
pub struct PerspectiveModel {
    /// Token embedding (shared with LM head).
    pub embedding: Embedding,

    /// Layer kinds (80 layers: PDR or GQA).
    pub layers: Vec<LayerKind>,

    /// Shared FFN weights for non-expert path (fallback / shared residual).
    pub shared_ffn: Vec<FfnWeights>,

    /// Final RMSNorm before LM head.
    pub final_norm_scale: Array1<f32>,
}

impl PerspectiveModel {
    /// Create an uninitialised model (zero weights) sized per `dims`.
    pub fn zeros(dims: &ModelDims) -> Self {
        let n_layers = dims.n_layers();
        let mut layers = Vec::with_capacity(n_layers);
        let mut shared_ffn = Vec::with_capacity(n_layers);
        let mut pdr_idx = 0;
        let mut gqa_idx = 0;

        for layer in 0..n_layers {
            if config::is_pdr_layer(layer) {
                layers.push(LayerKind::Pdr(PdrLayer::zeros(
                    pdr_idx,
                    dims.d_model,
                    dims.pdr_rank,
                )));
                pdr_idx += 1;
            } else {
                layers.push(LayerKind::Gqa(GqaLayer::zeros(
                    gqa_idx,
                    dims.d_model,
                    dims.gqa_q_heads,
                    dims.gqa_kv_heads,
                    dims.head_dim(),
                )));
                gqa_idx += 1;
            }
            shared_ffn.push(FfnWeights::zeros(dims.d_model, dims.ffn_intermediate));
        }

        assert_eq!(pdr_idx, dims.n_pdr_layers);
        assert_eq!(gqa_idx, dims.n_gqa_layers);

        Self {
            embedding: Embedding::zeros(dims.vocab_size, dims.d_model),
            layers,
            shared_ffn,
            final_norm_scale: Array1::ones(dims.d_model),
        }
    }

    /// Single-token forward pass through all layers.
    ///
    /// This is the core inference path.  Expert loading is handled externally
    /// by the runtime pipeline — this function accepts an optional expert FFN
    /// callback for each layer.
    pub fn forward_token(
        &self,
        token_id: u32,
        pdr_states: &mut PdrStateBank,
        kv_caches: &mut KVCacheBank,
        expert_ffn: Option<&dyn Fn(usize, &Array1<f32>) -> Array1<f32>>,
    ) -> Array1<f32> {
        let mut h = self.embedding.embed(token_id);
        let mut pdr_idx = 0;
        let mut gqa_idx = 0;

        for (layer_idx, (layer, ffn)) in self.layers.iter().zip(self.shared_ffn.iter()).enumerate()
        {
            // Attention / recurrence
            h = match layer {
                LayerKind::Pdr(pdr) => {
                    let out = pdr.forward_step(&h, &mut pdr_states.states[pdr_idx]);
                    pdr_idx += 1;
                    out
                }
                LayerKind::Gqa(gqa) => {
                    let out = gqa.forward_step(&h, &mut kv_caches.caches[gqa_idx]);
                    gqa_idx += 1;
                    out
                }
            };

            // FFN (expert or shared fallback)
            let ffn_out = if let Some(expert_fn) = &expert_ffn {
                expert_fn(layer_idx, &h)
            } else {
                ffn.forward(&h)
            };
            h = &h + &ffn_out;
        }

        // Final norm
        h = rms_norm(&h, &self.final_norm_scale);

        // LM head (tied embedding)
        self.embedding.to_logits(&h)
    }

    /// Total shared parameter count (excluding experts).
    pub fn shared_param_count(&self) -> usize {
        let embed = self.embedding.weight.len();
        let layer_params: usize = self
            .layers
            .iter()
            .map(|l| match l {
                LayerKind::Pdr(p) => p.param_count(),
                LayerKind::Gqa(g) => g.param_count(),
            })
            .sum();
        let ffn_params: usize = self.shared_ffn.iter().map(|f| f.param_count()).sum();
        let final_norm = self.final_norm_scale.len();
        embed + layer_params + ffn_params + final_norm
    }
}

// ---- Utility ----

fn rms_norm(x: &Array1<f32>, scale: &Array1<f32>) -> Array1<f32> {
    let eps = 1e-6_f32;
    let mean_sq = x.mapv(|v| v * v).mean().unwrap_or(1.0);
    let rms = (mean_sq + eps).sqrt();
    x / rms * scale
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_layer_count() {
        let dims = ModelDims::tiny();
        let model = PerspectiveModel::zeros(&dims);
        assert_eq!(model.layers.len(), dims.n_layers());
        assert_eq!(model.shared_ffn.len(), dims.n_layers());
    }

    #[test]
    fn test_model_layer_types() {
        let dims = ModelDims::tiny();
        let model = PerspectiveModel::zeros(&dims);
        let pdr_count = model
            .layers
            .iter()
            .filter(|l| matches!(l, LayerKind::Pdr(_)))
            .count();
        let gqa_count = model
            .layers
            .iter()
            .filter(|l| matches!(l, LayerKind::Gqa(_)))
            .count();
        assert_eq!(pdr_count, dims.n_pdr_layers);
        assert_eq!(gqa_count, dims.n_gqa_layers);
    }

    #[test]
    fn test_embedding() {
        let dims = ModelDims::tiny();
        let emb = Embedding::zeros(dims.vocab_size, dims.d_model);
        let h = emb.embed(0);
        assert_eq!(h.len(), dims.d_model);
    }

    #[test]
    fn test_ffn_zeros_passthrough() {
        let dims = ModelDims::tiny();
        let ffn = FfnWeights::zeros(dims.d_model, dims.ffn_intermediate);
        let h = Array1::from_vec(vec![1.0; dims.d_model]);
        let out = ffn.forward(&h);
        // With zero weights, SwiGLU output should be zero
        for &v in out.iter() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.5);
        assert!(silu(-1.0) < 0.0);
    }
}
