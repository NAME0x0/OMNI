//! Core module — PDR recurrence, GQA attention, and model skeleton.

pub mod model;
pub mod pdr;
pub mod pdr_state;
pub mod windowed_gqa;

/// Explicit model dimensions, threaded through `core` constructors so callers
/// (production code and tests alike) can build instances at any size instead
/// of the hardcoded full-scale `crate::config` constants.
///
/// `crate::config` remains the canonical source of truth for the full
/// 1.05T-parameter model; [`ModelDims::full`] simply mirrors those constants.
/// [`ModelDims::tiny`] provides a small configuration suitable for fast,
/// low-memory unit tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelDims {
    /// Model hidden dimension.
    pub d_model: usize,

    /// PDR recurrence rank.
    pub pdr_rank: usize,

    /// FFN intermediate dimension (SwiGLU).
    pub ffn_intermediate: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// GQA query heads.
    pub gqa_q_heads: usize,

    /// GQA key/value heads.
    pub gqa_kv_heads: usize,

    /// GQA window size.
    pub gqa_window: usize,

    /// Number of PDR layers.
    pub n_pdr_layers: usize,

    /// Number of windowed GQA layers.
    pub n_gqa_layers: usize,
}

impl ModelDims {
    /// Head dimension, derived as `d_model / gqa_q_heads`.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.gqa_q_heads
    }

    /// Total layer count (PDR + GQA).
    pub fn n_layers(&self) -> usize {
        self.n_pdr_layers + self.n_gqa_layers
    }

    /// Full-scale PERSPECTIVE model dimensions — mirrors `crate::config`.
    pub fn full() -> Self {
        Self {
            d_model: crate::config::D_MODEL,
            pdr_rank: crate::config::PDR_RANK,
            ffn_intermediate: crate::config::FFN_INTERMEDIATE,
            vocab_size: crate::config::VOCAB_SIZE,
            gqa_q_heads: crate::config::GQA_Q_HEADS,
            gqa_kv_heads: crate::config::GQA_KV_HEADS,
            gqa_window: crate::config::GQA_WINDOW,
            n_pdr_layers: crate::config::N_PDR_LAYERS,
            n_gqa_layers: crate::config::N_GQA_LAYERS,
        }
    }

    /// Small dimensions for fast, low-memory unit tests.
    ///
    /// Keeps the period-4 PDR/GQA interleave valid: 8 layers = two full
    /// periods of 3 PDR + 1 GQA, matching `crate::config::is_pdr_layer`.
    pub fn tiny() -> Self {
        Self {
            d_model: 32,
            pdr_rank: 8,
            ffn_intermediate: 64,
            vocab_size: 128,
            gqa_q_heads: 4,
            gqa_kv_heads: 2,
            gqa_window: 16,
            n_pdr_layers: 6,
            n_gqa_layers: 2,
        }
    }
}
