//! Ternary weight packing — store {-1, 0, +1} at 1.58 bits per parameter.
//!
//! Format: 5 trits per byte using base-3 encoding.
//!
//! ```text
//! byte_value = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
//! where ti ∈ {0, 1, 2} representing {-1, 0, +1}
//! ```
//!
//! This maps each trit to {0 → -1, 1 → 0, 2 → +1}, and 3^5 = 243 ≤ 255
//! so 5 trits fit in one byte exactly.
//!
//! Storage: 1.6 bits/param (5 trits / 8 bits per byte).

use serde::{Deserialize, Serialize};

/// Encode a trit value (-1, 0, +1) to the internal representation (0, 1, 2).
#[inline]
fn trit_encode(v: i8) -> u8 {
    match v {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => panic!("Invalid trit value: {}", v),
    }
}

/// Decode the internal representation (0, 1, 2) back to a trit (-1, 0, +1).
#[inline]
fn trit_decode(v: u8) -> i8 {
    match v {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => panic!("Invalid encoded trit: {}", v),
    }
}

/// Pack 5 trits into a single byte.
#[inline]
pub fn pack5(trits: [i8; 5]) -> u8 {
    let t: [u8; 5] = [
        trit_encode(trits[0]),
        trit_encode(trits[1]),
        trit_encode(trits[2]),
        trit_encode(trits[3]),
        trit_encode(trits[4]),
    ];
    t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4]
}

/// Unpack a byte into 5 trits.
#[inline]
pub fn unpack5(byte: u8) -> [i8; 5] {
    let mut val = byte;
    let mut trits = [0i8; 5];
    for i in 0..5 {
        trits[i] = trit_decode(val % 3);
        val /= 3;
    }
    trits
}

/// A ternary-packed matrix in row-major order.
///
/// Logical shape: [rows, cols] where each element ∈ {-1, 0, +1}.
/// Physical storage: ceil(rows * cols / 5) bytes.
#[derive(Clone, Serialize, Deserialize)]
pub struct TernaryMatrix {
    /// Packed bytes (5 trits per byte).
    pub data: Vec<u8>,

    /// Number of rows.
    pub rows: usize,

    /// Number of columns.
    pub cols: usize,
}

impl TernaryMatrix {
    /// Bytes per packed row.
    #[inline]
    pub fn packed_cols(cols: usize) -> usize {
        (cols + 4) / 5
    }

    /// Create from a flat slice of trit values.
    /// Packs each row independently (row-aligned layout).
    pub fn from_trits(rows: usize, cols: usize, trits: &[i8]) -> Self {
        assert_eq!(trits.len(), rows * cols);
        let pc = Self::packed_cols(cols);
        let mut data = Vec::with_capacity(rows * pc);

        for r in 0..rows {
            let row_start = r * cols;
            let row_slice = &trits[row_start..row_start + cols];
            let mut i = 0;
            while i + 5 <= cols {
                data.push(pack5([
                    row_slice[i],
                    row_slice[i + 1],
                    row_slice[i + 2],
                    row_slice[i + 3],
                    row_slice[i + 4],
                ]));
                i += 5;
            }
            // Handle remainder (pad with zeros/neutral)
            if i < cols {
                let mut chunk = [0i8; 5];
                for (j, &t) in row_slice[i..].iter().enumerate() {
                    chunk[j] = t;
                }
                data.push(pack5(chunk));
            }
        }

        Self { data, rows, cols }
    }

    /// Create a zero (all-neutral) matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let pc = Self::packed_cols(cols);
        // All-zero trits encode to: 1 + 3 + 9 + 27 + 81 = 121
        let neutral_byte = pack5([0, 0, 0, 0, 0]);
        Self {
            data: vec![neutral_byte; rows * pc],
            rows,
            cols,
        }
    }

    /// Unpack all trits to a flat vector.
    pub fn to_trits(&self) -> Vec<i8> {
        let pc = Self::packed_cols(self.cols);
        let mut trits = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            let row_start = r * pc;
            let mut col = 0;
            for b in 0..pc {
                let t = unpack5(self.data[row_start + b]);
                for &v in &t {
                    if col >= self.cols {
                        break;
                    }
                    trits.push(v);
                    col += 1;
                }
            }
        }
        trits
    }

    /// Get a single element.
    pub fn get(&self, row: usize, col: usize) -> i8 {
        let pc = Self::packed_cols(self.cols);
        let byte_idx = row * pc + col / 5;
        let trit_idx = col % 5;
        let trits = unpack5(self.data[byte_idx]);
        trits[trit_idx]
    }

    /// Ternary matrix-vector multiply: y = W · x
    ///
    /// This is the core operation — no multiplications needed!
    /// For each element: -1 → subtract, 0 → skip, +1 → add.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0f32; self.rows];

        let trits = self.to_trits();
        for row in 0..self.rows {
            let mut sum = 0.0f32;
            let base = row * self.cols;
            for col in 0..self.cols {
                match trits[base + col] {
                    1 => sum += x[col],
                    -1 => sum -= x[col],
                    0 => {} // skip — the key efficiency win
                    _ => unreachable!(),
                }
            }
            y[row] = sum;
        }

        y
    }

    /// Optimised matvec that operates directly on packed bytes.
    /// Avoids full unpack — processes 5 elements at a time.
    /// Uses row-aligned layout: each row starts at row * packed_cols.
    pub fn matvec_packed(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let pc = Self::packed_cols(self.cols);
        let mut y = vec![0.0f32; self.rows];

        for row in 0..self.rows {
            let mut sum = 0.0f32;
            let row_start = row * pc;
            let mut col = 0;

            for b in 0..pc {
                let trits = unpack5(self.data[row_start + b]);
                for &t in &trits {
                    if col >= self.cols {
                        break;
                    }
                    match t {
                        1 => sum += x[col],
                        -1 => sum -= x[col],
                        _ => {}
                    }
                    col += 1;
                }
            }
            y[row] = sum;
        }

        y
    }

    /// Storage size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Number of zero-valued trits (sparsity).
    pub fn count_zeros(&self) -> usize {
        self.to_trits().iter().filter(|&&t| t == 0).count()
    }

    /// Sparsity ratio (fraction of zeros).
    pub fn sparsity(&self) -> f32 {
        self.count_zeros() as f32 / (self.rows * self.cols) as f32
    }

    /// Bits per parameter (includes packing overhead for remainder).
    pub fn bits_per_param(&self) -> f32 {
        (self.data.len() * 8) as f32 / (self.rows * self.cols) as f32
    }

    /// Compute the delta between two ternary matrices.
    /// Returns a sparse representation of positions that differ.
    pub fn delta(&self, other: &TernaryMatrix) -> TernaryDelta {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let self_trits = self.to_trits();
        let other_trits = other.to_trits();

        let mut diffs = Vec::new();
        for (i, (&a, &b)) in self_trits.iter().zip(other_trits.iter()).enumerate() {
            if a != b {
                diffs.push((i as u32, b)); // store index + new value
            }
        }

        TernaryDelta {
            rows: self.rows,
            cols: self.cols,
            diffs,
        }
    }

    /// Apply a delta to produce a new matrix.
    pub fn apply_delta(&self, delta: &TernaryDelta) -> TernaryMatrix {
        let mut trits = self.to_trits();
        for &(idx, val) in &delta.diffs {
            trits[idx as usize] = val;
        }
        TernaryMatrix::from_trits(self.rows, self.cols, &trits)
    }
}

/// Sparse delta between two ternary matrices.
#[derive(Clone, Serialize, Deserialize)]
pub struct TernaryDelta {
    pub rows: usize,
    pub cols: usize,
    /// (flat_index, new_value) pairs.
    pub diffs: Vec<(u32, i8)>,
}

impl TernaryDelta {
    /// Fraction of elements that differ.
    pub fn diff_ratio(&self) -> f32 {
        self.diffs.len() as f32 / (self.rows * self.cols) as f32
    }

    /// Approximate storage size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.diffs.len() * 5 // u32 index + i8 value
    }
}

/// Expert FFN in ternary format (SwiGLU architecture).
#[derive(Clone, Serialize, Deserialize)]
pub struct TernaryExpertFfn {
    /// Gate projection (ternary): [ffn_intermediate, d_model]
    pub w_gate: TernaryMatrix,

    /// Up projection (ternary): [ffn_intermediate, d_model]
    pub w_up: TernaryMatrix,

    /// Down projection (ternary): [d_model, ffn_intermediate]
    pub w_down: TernaryMatrix,

    /// Per-tensor scale factors (FP32) for dequantisation.
    pub gate_scale: f32,
    pub up_scale: f32,
    pub down_scale: f32,
}

impl TernaryExpertFfn {
    /// SwiGLU forward: scale_d * W_down · (silu(scale_g * W_gate · x) ⊙ (scale_u * W_up · x))
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Gate path
        let gate_raw = self.w_gate.matvec(x);
        let gate: Vec<f32> = gate_raw
            .iter()
            .map(|&v| silu(v * self.gate_scale))
            .collect();

        // Up path
        let up: Vec<f32> = self
            .w_up
            .matvec(x)
            .iter()
            .map(|&v| v * self.up_scale)
            .collect();

        // Element-wise multiply
        let intermediate: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();

        // Down projection
        let out_raw = self.w_down.matvec(&intermediate);
        out_raw.iter().map(|&v| v * self.down_scale).collect()
    }

    /// Total storage size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.w_gate.size_bytes() + self.w_up.size_bytes() + self.w_down.size_bytes() + 12
        // 3 scales
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let gate = self.w_gate.rows * self.w_gate.cols;
        let up = self.w_up.rows * self.w_up.cols;
        let down = self.w_down.rows * self.w_down.cols;
        gate + up + down
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        for t0 in [-1i8, 0, 1] {
            for t1 in [-1i8, 0, 1] {
                for t2 in [-1i8, 0, 1] {
                    for t3 in [-1i8, 0, 1] {
                        for t4 in [-1i8, 0, 1] {
                            let original = [t0, t1, t2, t3, t4];
                            let packed = pack5(original);
                            let unpacked = unpack5(packed);
                            assert_eq!(original, unpacked, "Pack/unpack failed for {:?}", original);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_byte_range() {
        // All possible 5-trit combos should fit in 0..243
        for t0 in [-1i8, 0, 1] {
            for t1 in [-1i8, 0, 1] {
                for t2 in [-1i8, 0, 1] {
                    for t3 in [-1i8, 0, 1] {
                        for t4 in [-1i8, 0, 1] {
                            let packed = pack5([t0, t1, t2, t3, t4]);
                            assert!(packed < 243, "Packed value {} >= 243", packed);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_matrix_zeros() {
        let m = TernaryMatrix::zeros(4, 4);
        let trits = m.to_trits();
        assert!(trits.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_matrix_from_trits() {
        let trits = vec![1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0];
        let m = TernaryMatrix::from_trits(3, 4, &trits);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        let recovered = m.to_trits();
        assert_eq!(trits, recovered);
    }

    #[test]
    fn test_matrix_get() {
        let trits = vec![1, -1, 0, 1, -1, 0];
        let m = TernaryMatrix::from_trits(2, 3, &trits);
        assert_eq!(m.get(0, 0), 1);
        assert_eq!(m.get(0, 1), -1);
        assert_eq!(m.get(0, 2), 0);
        assert_eq!(m.get(1, 0), 1);
    }

    #[test]
    fn test_matvec_identity_like() {
        // 3×3 matrix: [[1,0,0], [0,1,0], [0,0,1]]
        let trits = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let m = TernaryMatrix::from_trits(3, 3, &trits);
        let x = vec![3.0, 5.0, 7.0];
        let y = m.matvec(&x);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 5.0).abs() < 1e-6);
        assert!((y[2] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_negation() {
        // All -1 matrix
        let trits = vec![-1; 9];
        let m = TernaryMatrix::from_trits(3, 3, &trits);
        let x = vec![1.0, 2.0, 3.0];
        let y = m.matvec(&x);
        // Each row: -(1+2+3) = -6
        for &v in &y {
            assert!((v - (-6.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sparsity() {
        let trits = vec![0, 0, 0, 1, -1, 0]; // 4/6 = 67% zeros
        let m = TernaryMatrix::from_trits(2, 3, &trits);
        assert!((m.sparsity() - 4.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta() {
        let a_trits = vec![1, 0, -1, 1, 0, -1];
        let b_trits = vec![1, 1, -1, -1, 0, -1]; // differs at pos 1 and 3
        let a = TernaryMatrix::from_trits(2, 3, &a_trits);
        let b = TernaryMatrix::from_trits(2, 3, &b_trits);
        let delta = a.delta(&b);
        assert_eq!(delta.diffs.len(), 2);

        // Apply delta to a should produce b
        let reconstructed = a.apply_delta(&delta);
        assert_eq!(reconstructed.to_trits(), b_trits);
    }

    #[test]
    fn test_bits_per_param() {
        let m = TernaryMatrix::zeros(100, 100);
        let bpp = m.bits_per_param();
        // 10000 params, ceil(10000/5) = 2000 bytes, 16000 bits / 10000 = 1.6
        assert!((bpp - 1.6).abs() < 0.01);
    }
}
