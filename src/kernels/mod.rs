//! Kernel backend dispatch and native implementations.
//!
//! This module abstracts over four backends:
//! - **CPU**: Pure Rust fallback (always available)
//! - **CUDA**: NVIDIA GPU kernels (feature `cuda`)
//! - **HIP**: AMD GPU kernels (feature `hip`)
//! - **SYCL**: Intel GPU kernels (feature `sycl`)

pub mod kernel_dispatch;
#[cfg(feature = "native-manifold")]
pub mod manifold_native;
pub mod ternary_gemm_cpu;
