//! Runtime module: ties all components into a complete inference pipeline.
//!
//! This module provides:
//! - Model weight provider (NVMe → mmap → memory)
//! - Full inference pipeline (token in → token out)
//! - Health monitoring and diagnostics
//! - CLI binary entry point

pub mod provider;
pub mod pipeline;
pub mod health;
