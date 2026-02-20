//! Multi-Perspective Decoding (MPD).
//!
//! Instead of sampling from a single forward pass, MPD runs 4 "perspectives"
//! — controlled perturbations of the same model — and accepts a token only
//! when they agree. This dramatically reduces hallucination.

pub mod mpd;
pub mod perspective_config;
pub mod agreement;
pub mod calibration;
