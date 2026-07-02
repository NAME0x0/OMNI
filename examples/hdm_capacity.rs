//! HDM per-bank capacity curve — measures real retrieval accuracy of the
//! Holographic Distributed Memory (HDM) implementation as a function of how
//! many key/value associations are superposed into a single bank.
//!
//! This exists to replace the FABRICATED table in docs/v2/06_holographic_memory.md
//! §5.2 with numbers actually produced by `perspective::memory::hdm`.
//!
//! Mechanism (matches the codebase's actual bundling behaviour):
//! `MemoryTrace::add_binding` stores every role⊕filler binding it has ever
//! seen in `self.bindings` and, on each call, recomputes
//! `self.vector = majority_vote(&all_bindings)` from scratch (see
//! `hdm_binding.rs::MemoryTrace::recompute_vector`). That means the final
//! bank vector after N bindings is *exactly* `majority_vote` applied once to
//! all N bound vectors — NOT an iterated pairwise majority
//! (majority(majority(a, b), c), ...), which would compound rounding bias
//! differently. We therefore build each bank with a single call to the
//! project's real `hdm::majority_vote` over all N bound pairs, which is
//! mathematically identical to what `MemoryTrace` produces, and avoids
//! O(N^2) redundant recomputation so the sweep finishes quickly.
//!
//! For each load N, a bank is built as:
//!     bank = majority_vote([key_1 XOR value_1, ..., key_N XOR value_N])
//! and each key_i is queried by unbinding:
//!     probe_i = bank XOR key_i
//! then finding the nearest stored value (by Hamming distance) among the N
//! candidate values. Accuracy = fraction of the N keys whose nearest value
//! is the correct value_i.

use perspective::memory::hdm::{majority_vote, HyperVector};

/// Bank loads to sweep, per docs/v2/06_holographic_memory.md §5.2.
const LOADS: &[usize] = &[10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

/// Number of independent banks (seeds) averaged per load.
const REPEATS: usize = 8;

/// Build one bank of N key/value bindings and return (keys, values, bank).
fn build_bank(n: usize, seed: u64) -> (Vec<HyperVector>, Vec<HyperVector>, HyperVector) {
    // Distinct, non-overlapping seed ranges for keys and values so that no
    // key vector ever coincides with a value vector.
    let keys: Vec<HyperVector> = (0..n)
        .map(|i| HyperVector::random(seed.wrapping_mul(2).wrapping_add(2 * i as u64)))
        .collect();
    let values: Vec<HyperVector> = (0..n)
        .map(|i| HyperVector::random(seed.wrapping_mul(2).wrapping_add(2 * i as u64 + 1)))
        .collect();

    let bound: Vec<HyperVector> = keys.iter().zip(values.iter()).map(|(k, v)| k.xor(v)).collect();
    let bound_refs: Vec<&HyperVector> = bound.iter().collect();
    let bank = majority_vote(&bound_refs);

    (keys, values, bank)
}

/// Query every key in the bank and return the fraction correctly recovered.
/// Maximum number of keys queried per bank. The candidate scan is O(N) per
/// query, so querying all N keys is O(N²); at N = 20 000 that is ~4×10⁸
/// Hamming comparisons per bank. Sampling ≤ 500 keys per bank keeps the
/// estimate unbiased (keys are exchangeable) while bounding runtime.
const MAX_QUERIES: usize = 500;

fn measure_accuracy(keys: &[HyperVector], values: &[HyperVector], bank: &HyperVector) -> f64 {
    let n = keys.len();
    let step = n.div_ceil(MAX_QUERIES).max(1);
    let mut queried = 0usize;
    let mut correct = 0usize;

    for i in (0..n).step_by(step) {
        queried += 1;
        let probe = bank.xor(&keys[i]);

        // Nearest-neighbour search over the N candidate values by Hamming distance.
        let mut best_idx = 0usize;
        let mut best_dist = u32::MAX;
        for (j, v) in values.iter().enumerate() {
            let d = probe.hamming_distance(v);
            if d < best_dist {
                best_dist = d;
                best_idx = j;
            }
        }

        if best_idx == i {
            correct += 1;
        }
    }

    correct as f64 / queried as f64
}

fn mean_stddev(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn main() {
    println!("# HDM per-bank capacity curve — measured\n");
    println!(
        "10,000-bit hypervectors, XOR binding + majority-vote bundling, {} banks/seeds per N.\n",
        REPEATS
    );

    let mut rows = Vec::new();

    for &n in LOADS {
        let mut accuracies = Vec::with_capacity(REPEATS);
        for r in 0..REPEATS {
            // Unique, reproducible seed per (N, repeat).
            let seed = 0x1234_5678u64
                .wrapping_add((n as u64) * 1_000_003)
                .wrapping_add(r as u64 * 97);
            let (keys, values, bank) = build_bank(n, seed);
            let acc = measure_accuracy(&keys, &values, &bank);
            accuracies.push(acc);
        }
        let (mean, stddev) = mean_stddev(&accuracies);
        eprintln!(
            "N={:>4}  mean={:.4}  stddev={:.4}  raw={:?}",
            n, mean, stddev, accuracies
        );
        rows.push((n, mean, stddev));
    }

    println!("| N per bank | measured accuracy | stddev |");
    println!("|-----------:|-------------------:|-------:|");
    for (n, mean, stddev) in &rows {
        println!("| {} | {:.1}% | {:.2}% |", n, mean * 100.0, stddev * 100.0);
    }
}
