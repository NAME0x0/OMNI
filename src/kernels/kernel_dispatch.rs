//! Kernel dispatch — routes ternary GEMM calls to the correct backend.
//!
//! At compile time, exactly one GPU backend is selected via feature flags.
//! At runtime, dispatch checks capabilities and falls back gracefully:
//!   CUDA → HIP → SYCL → CPU (always available).

use ndarray::Array1;

use crate::execution::ternary_pack::TernaryMatrix;
use crate::kernels::ternary_gemm_cpu;

/// Backend selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "hip")]
    Hip,
    #[cfg(feature = "sycl")]
    Sycl,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu => write!(f, "CPU"),
            #[cfg(feature = "cuda")]
            Backend::Cuda => write!(f, "CUDA"),
            #[cfg(feature = "hip")]
            Backend::Hip => write!(f, "HIP"),
            #[cfg(feature = "sycl")]
            Backend::Sycl => write!(f, "SYCL"),
        }
    }
}

/// Detect the best available backend at runtime.
pub fn detect_backend() -> Backend {
    #[cfg(feature = "cuda")]
    {
        if cuda_available() {
            return Backend::Cuda;
        }
    }
    #[cfg(feature = "hip")]
    {
        if hip_available() {
            return Backend::Hip;
        }
    }
    #[cfg(feature = "sycl")]
    {
        if sycl_available() {
            return Backend::Sycl;
        }
    }
    Backend::Cpu
}

/// Check CUDA availability (stub — real impl would query driver).
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    // TODO: Call cuInit / cuDeviceGetCount via FFI
    false
}

/// Check HIP availability (stub).
#[cfg(feature = "hip")]
fn hip_available() -> bool {
    // TODO: Call hipDeviceGetCount via FFI
    false
}

/// Check SYCL availability (stub).
#[cfg(feature = "sycl")]
fn sycl_available() -> bool {
    // TODO: Query SYCL platforms
    false
}

/// Dispatch a ternary matrix-vector multiply to the best backend.
pub fn dispatch_matvec(
    mat: &TernaryMatrix,
    x: &Array1<f32>,
    scale: f32,
    backend: Backend,
) -> Array1<f32> {
    match backend {
        Backend::Cpu => {
            let x_slice = x.as_slice().unwrap();
            let out = ternary_gemm_cpu::ternary_matvec_packed(
                &mat.data,
                mat.rows,
                mat.cols,
                x_slice,
                scale,
            );
            Array1::from_vec(out)
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            // TODO: launch CUDA kernel via FFI
            // Fallback to CPU for now
            dispatch_matvec(mat, x, scale, Backend::Cpu)
        }
        #[cfg(feature = "hip")]
        Backend::Hip => {
            dispatch_matvec(mat, x, scale, Backend::Cpu)
        }
        #[cfg(feature = "sycl")]
        Backend::Sycl => {
            dispatch_matvec(mat, x, scale, Backend::Cpu)
        }
    }
}

/// Dispatch a parallel ternary matvec.
pub fn dispatch_matvec_parallel(
    mat: &TernaryMatrix,
    x: &Array1<f32>,
    scale: f32,
    backend: Backend,
) -> Array1<f32> {
    match backend {
        Backend::Cpu => {
            let x_slice = x.as_slice().unwrap();
            let out = ternary_gemm_cpu::ternary_matvec_parallel(
                &mat.data,
                mat.rows,
                mat.cols,
                x_slice,
                scale,
            );
            Array1::from_vec(out)
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => dispatch_matvec_parallel(mat, x, scale, Backend::Cpu),
        #[cfg(feature = "hip")]
        Backend::Hip => dispatch_matvec_parallel(mat, x, scale, Backend::Cpu),
        #[cfg(feature = "sycl")]
        Backend::Sycl => dispatch_matvec_parallel(mat, x, scale, Backend::Cpu),
    }
}

/// Full SwiGLU FFN dispatch.
pub fn dispatch_swiglu_ffn(
    w_gate: &TernaryMatrix,
    w_up: &TernaryMatrix,
    w_down: &TernaryMatrix,
    gate_scale: f32,
    up_scale: f32,
    down_scale: f32,
    x: &Array1<f32>,
    backend: Backend,
) -> Array1<f32> {
    match backend {
        Backend::Cpu => {
            ternary_gemm_cpu::ternary_swiglu_ffn(
                w_gate, w_up, w_down, gate_scale, up_scale, down_scale, x,
            )
        }
        // GPU backends fall back to CPU for now
        #[cfg(feature = "cuda")]
        Backend::Cuda => dispatch_swiglu_ffn(
            w_gate, w_up, w_down, gate_scale, up_scale, down_scale, x, Backend::Cpu,
        ),
        #[cfg(feature = "hip")]
        Backend::Hip => dispatch_swiglu_ffn(
            w_gate, w_up, w_down, gate_scale, up_scale, down_scale, x, Backend::Cpu,
        ),
        #[cfg(feature = "sycl")]
        Backend::Sycl => dispatch_swiglu_ffn(
            w_gate, w_up, w_down, gate_scale, up_scale, down_scale, x, Backend::Cpu,
        ),
    }
}

/// Kernel performance characteristics for each backend.
#[derive(Clone, Debug)]
pub struct KernelProfile {
    pub backend: Backend,
    /// Theoretical peak throughput (GFLOPS-equivalent for add/sub ops).
    pub peak_gops: f64,
    /// Memory bandwidth (GB/s).
    pub mem_bandwidth_gbps: f64,
    /// Kernel launch overhead (microseconds).
    pub launch_overhead_us: f64,
}

impl KernelProfile {
    pub fn cpu_default() -> Self {
        Self {
            backend: Backend::Cpu,
            peak_gops: 50.0,       // ~50 GOPS on modern CPU with AVX512
            mem_bandwidth_gbps: 40.0, // DDR5 dual-channel
            launch_overhead_us: 0.0,  // no launch overhead
        }
    }

    /// Estimate time (ms) for a ternary matvec of given dimensions.
    pub fn estimate_matvec_ms(&self, rows: usize, cols: usize, sparsity: f64) -> f64 {
        let ops = (rows * cols) as f64 * (1.0 - sparsity);
        let compute_ms = ops / (self.peak_gops * 1e6); // ms

        let bytes = ((rows * cols + 4) / 5) as f64 + (cols * 4) as f64; // packed weights + input
        let mem_ms = bytes / (self.mem_bandwidth_gbps * 1e6);

        self.launch_overhead_us / 1000.0 + compute_ms.max(mem_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ternary_pack::TernaryMatrix;

    #[test]
    fn test_detect_backend_cpu() {
        // Without GPU features, should always be CPU
        let b = detect_backend();
        assert_eq!(b, Backend::Cpu);
    }

    #[test]
    fn test_dispatch_matvec_identity() {
        // 3×3 identity in ternary
        let trits: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let mat = TernaryMatrix::from_trits(3, 3, &trits);
        let x = Array1::from_vec(vec![2.0, 3.0, 5.0]);

        let out = dispatch_matvec(&mat, &x, 1.0, Backend::Cpu);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
        assert!((out[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dispatch_parallel_matches() {
        let trits: Vec<i8> = vec![1, -1, 0, 1, 0, -1, -1, 1, 1];
        let mat = TernaryMatrix::from_trits(3, 3, &trits);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let seq = dispatch_matvec(&mat, &x, 0.5, Backend::Cpu);
        let par = dispatch_matvec_parallel(&mat, &x, 0.5, Backend::Cpu);

        for i in 0..3 {
            assert!((seq[i] - par[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", Backend::Cpu), "CPU");
    }

    #[test]
    fn test_kernel_profile_estimate() {
        let prof = KernelProfile::cpu_default();
        let ms = prof.estimate_matvec_ms(4096, 4096, 0.33);
        assert!(ms > 0.0);
        assert!(ms < 100.0); // sanity
    }

    #[test]
    fn test_dispatch_swiglu_zeros() {
        let d = 8;
        let ffn = 16;
        let w_gate = TernaryMatrix::zeros(ffn, d);
        let w_up = TernaryMatrix::zeros(ffn, d);
        let w_down = TernaryMatrix::zeros(d, ffn);
        let x = Array1::from_vec(vec![1.0; d]);

        let out = dispatch_swiglu_ffn(&w_gate, &w_up, &w_down, 1.0, 1.0, 1.0, &x, Backend::Cpu);
        assert_eq!(out.len(), d);
        for &v in out.iter() {
            assert!(v.abs() < 1e-6);
        }
    }
}
