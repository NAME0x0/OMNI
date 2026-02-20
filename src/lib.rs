//! # PERSPECTIVE
//!
//! **Perspective Is All You Need** — A 1.05T parameter sparse Mixture-of-Experts
//! architecture designed to run on consumer hardware (4 GB VRAM + 32 GB RAM).
//!
//! ## Novel Components
//!
//! 1. **PDR** — Perspective Decay Recurrence (O(1) sequence processing)
//! 2. **Manifold Routing** — Expert selection on a 2D flat torus
//! 3. **Ternary Execution** — Natively {-1, 0, +1} weights at 1.58 bits/param
//! 4. **HDM** — Holographic Distributed Memory (hyperdimensional long-term memory)
//! 5. **MPD** — Multi-Perspective Decoding (4-perspective calibrated generation)
//! 6. **FMEA** — Forward-Mode Evolutionary Adaptation (online learning, 11 MB)
//! 7. **SPP** — Safety Polytope Projection (non-differentiable safety constraints)
//!
//! ## Architecture
//!
//! - 80 layers: 60 PDR + 20 windowed GQA (period-4 interleave)
//! - 128 experts (8.12B params each), top-1 routing
//! - 14.95B active parameters per token
//! - Total: 1.05T parameters in ~230 GB on NVMe

pub mod core;
pub mod decoding;
pub mod execution;
pub mod kernels;
pub mod learning;
pub mod memory;
pub mod routing;
pub mod runtime;
pub mod safety;

/// Model-wide constants.
pub mod config {
    /// Model hidden dimension.
    pub const D_MODEL: usize = 4096;

    /// PDR recurrence rank.
    pub const PDR_RANK: usize = 256;

    /// Number of PDR layers.
    pub const N_PDR_LAYERS: usize = 60;

    /// Number of windowed GQA layers.
    pub const N_GQA_LAYERS: usize = 20;

    /// Total layers.
    pub const N_LAYERS: usize = N_PDR_LAYERS + N_GQA_LAYERS;

    /// GQA query heads.
    pub const GQA_Q_HEADS: usize = 32;

    /// GQA key/value heads.
    pub const GQA_KV_HEADS: usize = 8;

    /// Head dimension (d_model / q_heads).
    pub const HEAD_DIM: usize = D_MODEL / GQA_Q_HEADS; // 128

    /// GQA window size.
    pub const GQA_WINDOW: usize = 512;

    /// FFN intermediate dimension (SwiGLU).
    pub const FFN_INTERMEDIATE: usize = 11008;

    /// Vocabulary size.
    pub const VOCAB_SIZE: usize = 32768;

    /// Number of experts.
    pub const N_EXPERTS: usize = 128;

    /// Experts selected per token (top-k).
    pub const TOP_K: usize = 1;

    /// Manifold grid dimensions (N_EXPERTS = GRID_ROWS × GRID_COLS).
    pub const GRID_ROWS: usize = 16;
    pub const GRID_COLS: usize = 8;

    /// HDM hypervector dimension.
    pub const HDM_DIM: usize = 10000;

    /// HDM number of banks.
    pub const HDM_BANKS: usize = 2000;

    /// HDM codebook size.
    pub const HDM_CODEBOOK_SIZE: usize = 4096;

    /// MPD number of perspectives.
    pub const MPD_PERSPECTIVES: usize = 4;

    /// MPD activation threshold (max prob).
    pub const MPD_PROB_THRESHOLD: f32 = 0.6;

    /// MPD activation threshold (entropy).
    pub const MPD_ENTROPY_THRESHOLD: f32 = 3.0;

    /// MPD JSD agreement threshold.
    pub const MPD_JSD_THRESHOLD: f32 = 0.15;

    /// FMEA LoRA rank.
    pub const FMEA_LORA_RANK: usize = 4;

    /// FMEA JVP perturbation count.
    pub const FMEA_JVP_K: usize = 8;

    /// FMEA NES perturbation count.
    pub const FMEA_NES_K: usize = 16;

    /// FMEA time budget (seconds).
    pub const FMEA_TIME_BUDGET: f64 = 2.25;

    /// SPP anchor count.
    pub const SPP_ANCHORS: usize = 1000;

    /// SPP half-space facet count.
    pub const SPP_FACETS: usize = 500;

    /// SPP Dykstra iterations.
    pub const SPP_ITERATIONS: usize = 5;

    /// SPP epsilon (polytope inflation).
    pub const SPP_EPSILON: f32 = 0.5;

    /// Hot expert cache size (number of experts in RAM).
    pub const HOT_CACHE_SIZE: usize = 8;

    /// Maximum context length.
    pub const MAX_CONTEXT: usize = 4096;

    /// Returns whether a given layer index is a PDR layer.
    /// Period-4 interleave: PDR-PDR-PDR-GQA repeated.
    pub fn is_pdr_layer(layer: usize) -> bool {
        (layer % 4) != 3
    }

    /// Returns the GQA layer index (0..19) for a GQA layer, or None.
    pub fn gqa_index(layer: usize) -> Option<usize> {
        if is_pdr_layer(layer) {
            None
        } else {
            Some(layer / 4)
        }
    }
}
