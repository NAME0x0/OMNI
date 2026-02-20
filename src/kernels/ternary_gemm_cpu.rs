//! CPU reference implementation of ternary GEMM (zero-multiply).
//!
//! This is the always-available fallback that uses pure Rust.
//! The key insight: ternary weights ∈ {-1, 0, +1} mean we never
//! need a single floating-point multiply — only add, subtract, or skip.
//!
//! Three implementations with increasing optimization level:
//! 1. `ternary_matvec_naive` — scalar, unpacked, educational
//! 2. `ternary_matvec_packed` — operates on packed base-3 bytes
//! 3. `ternary_matvec_parallel` — uses rayon for row parallelism

use ndarray::Array1;
use rayon::prelude::*;

use crate::execution::ternary_pack::{unpack5, TernaryMatrix};

// ──────────────────────────────────────────────────────────────
// 1. Naive scalar implementation (reference / testing)
// ──────────────────────────────────────────────────────────────

/// Naive ternary matrix-vector multiply.
/// `trits` is row-major [rows × cols], values in {-1, 0, 1}.
/// Returns output vector of length `rows`.
pub fn ternary_matvec_naive(
    trits: &[i8],
    rows: usize,
    cols: usize,
    x: &[f32],
    scale: f32,
) -> Vec<f32> {
    assert_eq!(trits.len(), rows * cols);
    assert_eq!(x.len(), cols);

    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            let w = trits[r * cols + c];
            match w {
                1 => acc += x[c],
                -1 => acc -= x[c],
                0 => {} // skip — the zero-multiply innovation
                _ => unreachable!("ternary weight must be -1, 0, or 1"),
            }
        }
        out[r] = acc * scale;
    }
    out
}

// ──────────────────────────────────────────────────────────────
// 2. Packed implementation (operates on base-3 encoded bytes)
// ──────────────────────────────────────────────────────────────

/// Packed ternary matvec — reads 5 trits per byte.
/// `packed` is the base-3 encoded weight matrix (row-major).
/// `packed_cols` = ceil(cols / 5) — number of packed bytes per row.
pub fn ternary_matvec_packed(
    packed: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    scale: f32,
) -> Vec<f32> {
    let packed_cols = (cols + 4) / 5;
    assert!(packed.len() >= rows * packed_cols);
    assert_eq!(x.len(), cols);

    let mut out = vec![0.0f32; rows];

    for r in 0..rows {
        let row_start = r * packed_cols;
        let mut acc = 0.0f32;
        let mut col = 0usize;

        for p in 0..packed_cols {
            let trits = unpack5(packed[row_start + p]);
            for &t in &trits {
                if col >= cols {
                    break;
                }
                match t {
                    1 => acc += x[col],
                    -1 => acc -= x[col],
                    0 => {}
                    _ => {}
                }
                col += 1;
            }
        }

        out[r] = acc * scale;
    }
    out
}

// ──────────────────────────────────────────────────────────────
// 3. Parallel implementation (rayon)
// ──────────────────────────────────────────────────────────────

/// Parallel packed ternary matvec — distributes rows across threads.
pub fn ternary_matvec_parallel(
    packed: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    scale: f32,
) -> Vec<f32> {
    let packed_cols = (cols + 4) / 5;
    assert!(packed.len() >= rows * packed_cols);
    assert_eq!(x.len(), cols);

    (0..rows)
        .into_par_iter()
        .map(|r| {
            let row_start = r * packed_cols;
            let mut acc = 0.0f32;
            let mut col = 0usize;

            for p in 0..packed_cols {
                let trits = unpack5(packed[row_start + p]);
                for &t in &trits {
                    if col >= cols {
                        break;
                    }
                    match t {
                        1 => acc += x[col],
                        -1 => acc -= x[col],
                        _ => {}
                    }
                    col += 1;
                }
            }

            acc * scale
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────
// 4. SwiGLU FFN (ternary) — complete expert compute
// ──────────────────────────────────────────────────────────────

/// Compute a SwiGLU FFN using ternary packed weights.
/// gate = silu(W_gate · x · gate_scale)
/// up   = W_up · x · up_scale  
/// h    = gate ⊙ up
/// out  = W_down · h · down_scale
pub fn ternary_swiglu_ffn(
    w_gate: &TernaryMatrix,
    w_up: &TernaryMatrix,
    w_down: &TernaryMatrix,
    gate_scale: f32,
    up_scale: f32,
    down_scale: f32,
    x: &Array1<f32>,
) -> Array1<f32> {
    let x_slice = x.as_slice().unwrap();

    // Gate path
    let gate_raw = ternary_matvec_packed(
        &w_gate.data,
        w_gate.rows,
        w_gate.cols,
        x_slice,
        gate_scale,
    );

    // Up path
    let up_raw = ternary_matvec_packed(
        &w_up.data,
        w_up.rows,
        w_up.cols,
        x_slice,
        up_scale,
    );

    // SiLU(gate) ⊙ up
    let intermediate: Vec<f32> = gate_raw
        .iter()
        .zip(up_raw.iter())
        .map(|(&g, &u)| {
            let silu = g / (1.0 + (-g).exp()); // silu(x) = x · σ(x)
            silu * u
        })
        .collect();

    // Down projection
    let out = ternary_matvec_packed(
        &w_down.data,
        w_down.rows,
        w_down.cols,
        &intermediate,
        down_scale,
    );

    Array1::from_vec(out)
}

// ──────────────────────────────────────────────────────────────
// 5. Batch utilities
// ──────────────────────────────────────────────────────────────

/// Count operations for a ternary matvec (non-zero weight count).
/// This gives the actual FLOP count since zero weights are skipped.
pub fn count_ops(trits: &TernaryMatrix) -> u64 {
    let all = trits.to_trits();
    all.iter().filter(|&&t| t != 0).count() as u64
}

/// Effective GFLOPS given operation count and elapsed time.
pub fn effective_gflops(ops: u64, elapsed_secs: f64) -> f64 {
    ops as f64 / elapsed_secs / 1e9
}

/// Memory bandwidth utilisation: bytes_read / elapsed / peak_bandwidth.
pub fn bandwidth_utilisation(
    matrix_bytes: u64,
    vector_bytes: u64,
    elapsed_secs: f64,
    peak_gbps: f64,
) -> f64 {
    let total_bytes = matrix_bytes + vector_bytes;
    let actual_gbps = total_bytes as f64 / elapsed_secs / 1e9;
    actual_gbps / peak_gbps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ternary_pack::TernaryMatrix;

    #[test]
    fn test_naive_identity() {
        // 3×3 identity-like ternary matrix: diag = 1, rest = 0
        let trits: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let x = vec![2.0, 3.0, 5.0];
        let out = ternary_matvec_naive(&trits, 3, 3, &x, 1.0);
        assert_eq!(out, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_naive_negation() {
        let trits: Vec<i8> = vec![-1, 0, 0, 0, -1, 0, 0, 0, -1];
        let x = vec![2.0, 3.0, 5.0];
        let out = ternary_matvec_naive(&trits, 3, 3, &x, 1.0);
        assert_eq!(out, vec![-2.0, -3.0, -5.0]);
    }

    #[test]
    fn test_naive_scale() {
        let trits: Vec<i8> = vec![1, 1, 1];
        let x = vec![1.0, 2.0, 3.0];
        let out = ternary_matvec_naive(&trits, 1, 3, &x, 0.5);
        assert!((out[0] - 3.0).abs() < 1e-6); // (1+2+3) * 0.5 = 3.0
    }

    #[test]
    fn test_packed_matches_naive() {
        // Create a small matrix with known trits
        let trits_raw: Vec<i8> = vec![
            1, -1, 0, 1, -1, // row 0
            0, 1, 1, -1, 0,  // row 1
        ];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let naive = ternary_matvec_naive(&trits_raw, 2, 5, &x, 1.0);

        // Pack
        let mat = TernaryMatrix::from_trits(2, 5, &trits_raw);
        let packed = ternary_matvec_packed(&mat.data, 2, 5, &x, 1.0);

        for i in 0..2 {
            assert!(
                (naive[i] - packed[i]).abs() < 1e-6,
                "row {}: naive={} packed={}",
                i,
                naive[i],
                packed[i]
            );
        }
    }

    #[test]
    fn test_parallel_matches_packed() {
        let trits_raw: Vec<i8> = vec![
            1, 0, -1, 1, 0, -1, 1, 0, -1, 1, // row 0 (10 cols)
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, // row 1
            0, 0, 1, 1, -1, -1, 0, 0, 1, 1,   // row 2
        ];
        let x: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();

        let mat = TernaryMatrix::from_trits(3, 10, &trits_raw);
        let packed = ternary_matvec_packed(&mat.data, 3, 10, &x, 0.7);
        let par = ternary_matvec_parallel(&mat.data, 3, 10, &x, 0.7);

        for i in 0..3 {
            assert!(
                (packed[i] - par[i]).abs() < 1e-6,
                "row {}: packed={} parallel={}",
                i,
                packed[i],
                par[i]
            );
        }
    }

    #[test]
    fn test_count_ops() {
        let trits: Vec<i8> = vec![1, 0, -1, 0, 1, 0]; // 3 nonzero out of 6
        let mat = TernaryMatrix::from_trits(2, 3, &trits);
        assert_eq!(count_ops(&mat), 3);
    }

    #[test]
    fn test_effective_gflops() {
        let gf = effective_gflops(1_000_000_000, 1.0);
        assert!((gf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bandwidth_utilisation() {
        // 1 GB in 1 second with 10 GB/s peak = 10%
        let util = bandwidth_utilisation(1_000_000_000, 0, 1.0, 10.0);
        assert!((util - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_swiglu_zeros() {
        // With all-zero weights, output should be zero
        let d = 8;
        let ffn = 16;
        let w_gate = TernaryMatrix::zeros(ffn, d);
        let w_up = TernaryMatrix::zeros(ffn, d);
        let w_down = TernaryMatrix::zeros(d, ffn);

        let x = Array1::from_vec(vec![1.0; d]);
        let out = ternary_swiglu_ffn(&w_gate, &w_up, &w_down, 1.0, 1.0, 1.0, &x);
        assert_eq!(out.len(), d);
        for &v in out.iter() {
            assert!((v).abs() < 1e-6);
        }
    }
}
