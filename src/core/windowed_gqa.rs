//! Windowed Grouped-Query Attention (GQA).
//!
//! Optimisations:
//! 1. Windowed attention over last `GQA_WINDOW` tokens.
//! 2. Grouped queries: `GQA_Q_HEADS` query heads share `GQA_KV_HEADS` KV heads.
//! 3. RoPE positional encoding on q/k before attention scoring.

use ndarray::{s, Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::config::{D_MODEL, GQA_KV_HEADS, GQA_Q_HEADS, GQA_WINDOW, HEAD_DIM};

const ROPE_BASE: f32 = 10_000.0;

/// KV cache for a single GQA layer (ring buffer).
#[derive(Clone)]
pub struct KVCache {
    /// Key cache: [window, kv_heads, head_dim]
    pub keys: Array3<f32>,

    /// Value cache: [window, kv_heads, head_dim]
    pub values: Array3<f32>,

    /// Current write position (wraps around).
    pub pos: usize,

    /// Number of valid entries (up to WINDOW).
    pub len: usize,

    /// Number of KV entries ever inserted (absolute token position tracker).
    pub tokens_seen: u64,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: Array3::zeros((GQA_WINDOW, GQA_KV_HEADS, HEAD_DIM)),
            values: Array3::zeros((GQA_WINDOW, GQA_KV_HEADS, HEAD_DIM)),
            pos: 0,
            len: 0,
            tokens_seen: 0,
        }
    }

    /// Absolute position for the next token that will be inserted.
    pub fn next_position(&self) -> usize {
        self.tokens_seen.min(usize::MAX as u64) as usize
    }

    /// Insert a new KV pair.
    pub fn insert(&mut self, k: &Array2<f32>, v: &Array2<f32>) {
        // k, v: [kv_heads, head_dim]
        self.keys.slice_mut(s![self.pos, .., ..]).assign(k);
        self.values.slice_mut(s![self.pos, .., ..]).assign(v);
        self.pos = (self.pos + 1) % GQA_WINDOW;
        if self.len < GQA_WINDOW {
            self.len += 1;
        }
        self.tokens_seen = self.tokens_seen.saturating_add(1);
    }

    /// Get valid keys: [len, kv_heads, head_dim]
    pub fn get_keys(&self) -> Array3<f32> {
        if self.len < GQA_WINDOW {
            self.keys.slice(s![..self.len, .., ..]).to_owned()
        } else {
            // Ring buffer: reorder so oldest is first
            let start = self.pos; // oldest entry
            let mut out = Array3::zeros((GQA_WINDOW, GQA_KV_HEADS, HEAD_DIM));
            for i in 0..GQA_WINDOW {
                let src = (start + i) % GQA_WINDOW;
                out.slice_mut(s![i, .., ..])
                    .assign(&self.keys.slice(s![src, .., ..]));
            }
            out
        }
    }

    /// Get valid values: [len, kv_heads, head_dim]
    pub fn get_values(&self) -> Array3<f32> {
        if self.len < GQA_WINDOW {
            self.values.slice(s![..self.len, .., ..]).to_owned()
        } else {
            let start = self.pos;
            let mut out = Array3::zeros((GQA_WINDOW, GQA_KV_HEADS, HEAD_DIM));
            for i in 0..GQA_WINDOW {
                let src = (start + i) % GQA_WINDOW;
                out.slice_mut(s![i, .., ..])
                    .assign(&self.values.slice(s![src, .., ..]));
            }
            out
        }
    }

    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.pos = 0;
        self.len = 0;
        self.tokens_seen = 0;
    }

    /// Memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        2 * GQA_WINDOW * GQA_KV_HEADS * HEAD_DIM * std::mem::size_of::<f32>()
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Weights for a single GQA attention layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct GqaLayer {
    /// Query projection: `W_q in R^{(q_heads * head_dim) x d_model}`
    pub w_q: Array2<f32>,

    /// Key projection: `W_k in R^{(kv_heads * head_dim) x d_model}`
    pub w_k: Array2<f32>,

    /// Value projection: `W_v in R^{(kv_heads * head_dim) x d_model}`
    pub w_v: Array2<f32>,

    /// Output projection: `W_o in R^{d_model x (q_heads * head_dim)}`
    pub w_o: Array2<f32>,

    /// RMSNorm scale for pre-norm.
    pub rms_scale: Array1<f32>,

    /// Layer index.
    pub layer_idx: usize,
}

impl GqaLayer {
    /// Create with zero weights.
    pub fn zeros(layer_idx: usize) -> Self {
        let q_dim = GQA_Q_HEADS * HEAD_DIM;
        let kv_dim = GQA_KV_HEADS * HEAD_DIM;
        Self {
            w_q: Array2::zeros((q_dim, D_MODEL)),
            w_k: Array2::zeros((kv_dim, D_MODEL)),
            w_v: Array2::zeros((kv_dim, D_MODEL)),
            w_o: Array2::zeros((D_MODEL, q_dim)),
            rms_scale: Array1::ones(D_MODEL),
            layer_idx,
        }
    }

    /// Single-token forward pass with KV cache update.
    pub fn forward_step(&self, h: &Array1<f32>, cache: &mut KVCache) -> Array1<f32> {
        if h.len() != D_MODEL {
            return h.clone();
        }

        if GQA_KV_HEADS == 0 || GQA_Q_HEADS % GQA_KV_HEADS != 0 {
            return h.clone();
        }
        let group_size = GQA_Q_HEADS / GQA_KV_HEADS;

        let h_norm = rms_norm(h, &self.rms_scale);

        // Project Q, K, V
        let q_flat = self.w_q.dot(&h_norm); // [q_heads * head_dim]
        let k_flat = self.w_k.dot(&h_norm); // [kv_heads * head_dim]
        let v_flat = self.w_v.dot(&h_norm); // [kv_heads * head_dim]

        // Checked reshapes; on failure, fallback to residual path.
        let mut q_2d = match reshape_heads(q_flat, GQA_Q_HEADS, HEAD_DIM) {
            Some(v) => v,
            None => return h.clone(),
        };
        let mut k_2d = match reshape_heads(k_flat, GQA_KV_HEADS, HEAD_DIM) {
            Some(v) => v,
            None => return h.clone(),
        };
        let v_2d = match reshape_heads(v_flat, GQA_KV_HEADS, HEAD_DIM) {
            Some(v) => v,
            None => return h.clone(),
        };

        // RoPE on current-token q/k before cache insert / scoring.
        let token_pos = cache.next_position();
        apply_rope(&mut q_2d, token_pos);
        apply_rope(&mut k_2d, token_pos);

        // Insert rope-transformed key and value into cache.
        cache.insert(&k_2d, &v_2d);

        // Get cached K, V: [cache_len, kv_heads, head_dim]
        let cached_k = cache.get_keys();
        let cached_v = cache.get_values();
        let cache_len = cached_k.shape()[0];

        // Compute attention for each query head.
        let mut attn_out = Array1::zeros(GQA_Q_HEADS * HEAD_DIM);
        let scale = (HEAD_DIM as f32).sqrt();

        for qh in 0..GQA_Q_HEADS {
            let kv_idx = qh / group_size; // Which KV head this Q head uses
            let q = q_2d.row(qh); // [head_dim]

            // Scores: q dot k^T / sqrt(d)
            let mut scores = Array1::zeros(cache_len);
            for t in 0..cache_len {
                let k_t = cached_k.slice(s![t, kv_idx, ..]);
                scores[t] = q.dot(&k_t) / scale;
            }

            // Softmax + weighted value sum.
            let scores = softmax(&scores);
            let mut head_out = Array1::zeros(HEAD_DIM);
            for t in 0..cache_len {
                let v_t = cached_v.slice(s![t, kv_idx, ..]);
                head_out = head_out + &(v_t.to_owned() * scores[t]);
            }

            // Write head output.
            let start = qh * HEAD_DIM;
            for d in 0..HEAD_DIM {
                attn_out[start + d] = head_out[d];
            }
        }

        // Output projection + residual.
        let output = self.w_o.dot(&attn_out);
        h + &output
    }

    /// Parameter count for this layer.
    pub fn param_count(&self) -> usize {
        let q = GQA_Q_HEADS * HEAD_DIM * D_MODEL;
        let k = GQA_KV_HEADS * HEAD_DIM * D_MODEL;
        let v = GQA_KV_HEADS * HEAD_DIM * D_MODEL;
        let o = D_MODEL * GQA_Q_HEADS * HEAD_DIM;
        let rms = D_MODEL;
        q + k + v + o + rms
    }
}

/// Checked reshape of a flat `[rows * cols]` vector into `[rows, cols]`.
fn reshape_heads(x: Array1<f32>, rows: usize, cols: usize) -> Option<Array2<f32>> {
    if x.len() != rows * cols {
        return None;
    }
    x.into_shape_with_order((rows, cols)).ok()
}

/// Apply RoPE in-place to `[heads, head_dim]`.
fn apply_rope(x: &mut Array2<f32>, position: usize) {
    let pairs = HEAD_DIM / 2;
    if pairs == 0 {
        return;
    }

    let pos = position as f32;
    let mut cos_cache = vec![0.0; pairs];
    let mut sin_cache = vec![0.0; pairs];

    for i in 0..pairs {
        let exponent = (2 * i) as f32 / HEAD_DIM as f32;
        let theta = pos / ROPE_BASE.powf(exponent);
        cos_cache[i] = theta.cos();
        sin_cache[i] = theta.sin();
    }

    for h in 0..x.nrows() {
        for i in 0..pairs {
            let i0 = 2 * i;
            let i1 = i0 + 1;
            let a = x[[h, i0]];
            let b = x[[h, i1]];
            let cos_theta = cos_cache[i];
            let sin_theta = sin_cache[i];
            x[[h, i0]] = a * cos_theta - b * sin_theta;
            x[[h, i1]] = a * sin_theta + b * cos_theta;
        }
    }
}

/// RMSNorm.
fn rms_norm(x: &Array1<f32>, scale: &Array1<f32>) -> Array1<f32> {
    let eps = 1e-6_f32;
    let mean_sq = x.mapv(|v| v * v).mean().unwrap_or(1.0);
    let rms = (mean_sq + eps).sqrt();
    x / rms * scale
}

/// Softmax over a 1D array.
fn softmax(x: &Array1<f32>) -> Array1<f32> {
    if x.is_empty() {
        return x.clone();
    }

    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = x.mapv(|v| (v - max_val).exp());
    let sum = exp.sum();

    if sum > 0.0 {
        exp / sum
    } else {
        Array1::from_vec(vec![1.0 / x.len() as f32; x.len()])
    }
}

/// Bank of KV caches for all GQA layers.
pub struct KVCacheBank {
    pub caches: Vec<KVCache>,
}

impl KVCacheBank {
    pub fn new(n: usize) -> Self {
        Self {
            caches: (0..n).map(|_| KVCache::new()).collect(),
        }
    }

    pub fn reset_all(&mut self) {
        for c in &mut self.caches {
            c.reset();
        }
    }

    pub fn total_size_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.size_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_insert() {
        let mut cache = KVCache::new();
        let k = Array2::ones((GQA_KV_HEADS, HEAD_DIM));
        let v = Array2::ones((GQA_KV_HEADS, HEAD_DIM));
        cache.insert(&k, &v);
        assert_eq!(cache.len, 1);
        assert_eq!(cache.pos, 1);
        assert_eq!(cache.tokens_seen, 1);
    }

    #[test]
    fn test_kv_cache_wrap() {
        let mut cache = KVCache::new();
        let k = Array2::ones((GQA_KV_HEADS, HEAD_DIM));
        let v = Array2::ones((GQA_KV_HEADS, HEAD_DIM));
        for _ in 0..GQA_WINDOW + 5 {
            cache.insert(&k, &v);
        }
        assert_eq!(cache.len, GQA_WINDOW);
        assert_eq!(cache.pos, 5);
        assert_eq!(cache.tokens_seen, (GQA_WINDOW + 5) as u64);
    }

    #[test]
    fn test_rope_position_zero_identity() {
        let mut q = Array2::ones((GQA_Q_HEADS, HEAD_DIM));
        let baseline = q.clone();
        apply_rope(&mut q, 0);
        assert_eq!(q, baseline);
    }

    #[test]
    fn test_gqa_residual() {
        // With zero weights, output = input (residual only)
        let layer = GqaLayer::zeros(0);
        let mut cache = KVCache::new();
        // Insert a dummy entry so cache is non-empty
        let k = Array2::zeros((GQA_KV_HEADS, HEAD_DIM));
        let v = Array2::zeros((GQA_KV_HEADS, HEAD_DIM));
        cache.insert(&k, &v);

        let h = Array1::from_vec(vec![1.0; D_MODEL]);
        let out = layer.forward_step(&h, &mut cache);
        assert_eq!(out.len(), D_MODEL);
        for i in 0..D_MODEL {
            assert!(
                (out[i] - h[i]).abs() < 1e-4,
                "Residual broken at {}: got {}",
                i,
                out[i]
            );
        }
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&x);
        assert!((s.sum() - 1.0).abs() < 1e-5);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let x = Array1::zeros(0);
        let s = softmax(&x);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_cache_size() {
        let cache = KVCache::new();
        let expected = 2 * GQA_WINDOW * GQA_KV_HEADS * HEAD_DIM * 4;
        assert_eq!(cache.size_bytes(), expected);
    }
}
