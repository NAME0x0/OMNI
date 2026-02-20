//! Delta streaming — exploit manifold locality for efficient expert loading.
//!
//! When consecutive tokens route to neighbouring experts on the torus,
//! we load a compact delta (difference) instead of the full expert weights.
//! Delta files are ~2.7 MB vs ~27 MB for full experts → 10× smaller.
//!
//! Delta streaming is the key to achieving acceptable throughput on PCIe 3.0.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::config::N_EXPERTS;
use crate::routing::manifold::ExpertManifold;

/// Metadata about a delta file between two experts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeltaFileInfo {
    /// Source expert (the one we already have in the buffer).
    pub source: usize,

    /// Target expert (the one we want).
    pub target: usize,

    /// File path on disk.
    pub path: PathBuf,

    /// File size in bytes.
    pub size_bytes: u64,

    /// Number of non-zero trit differences.
    pub diff_count: u64,

    /// Fraction of weights that differ (0.0 to 1.0).
    pub diff_ratio: f32,
}

/// Manages delta file lookup and loading decisions.
pub struct DeltaStreamManager {
    /// Available delta files: (source, target) → info.
    pub deltas: HashMap<(usize, usize), DeltaFileInfo>,

    /// Full expert file sizes (for comparison).
    pub full_sizes: Vec<u64>,

    /// Threshold: only use delta if diff_ratio < this value.
    pub delta_threshold: f32,

    /// Statistics tracking.
    pub stats: DeltaStats,
}

/// Statistics for delta streaming performance.
#[derive(Clone, Debug, Default)]
pub struct DeltaStats {
    /// Number of delta loads.
    pub delta_loads: u64,

    /// Number of full loads.
    pub full_loads: u64,

    /// Total bytes saved by delta loading.
    pub bytes_saved: u64,

    /// Total bytes loaded (delta + full).
    pub bytes_loaded: u64,
}

impl DeltaStats {
    /// Fraction of loads that used delta.
    pub fn delta_ratio(&self) -> f64 {
        let total = self.delta_loads + self.full_loads;
        if total == 0 {
            0.0
        } else {
            self.delta_loads as f64 / total as f64
        }
    }

    /// Bandwidth savings ratio.
    pub fn bandwidth_savings(&self) -> f64 {
        let total_possible = self.bytes_loaded + self.bytes_saved;
        if total_possible == 0 {
            0.0
        } else {
            self.bytes_saved as f64 / total_possible as f64
        }
    }
}

/// The decision of how to load an expert.
#[derive(Clone, Debug)]
pub enum LoadStrategy {
    /// Load the full expert from NVMe.
    Full { expert_id: usize, size_bytes: u64 },

    /// Load a delta from the currently buffered expert.
    Delta {
        source: usize,
        target: usize,
        delta_info: DeltaFileInfo,
    },

    /// Expert is already in the buffer — no load needed.
    Cached { expert_id: usize },
}

impl DeltaStreamManager {
    /// Create a new manager (no deltas loaded yet).
    pub fn new() -> Self {
        Self {
            deltas: HashMap::new(),
            full_sizes: vec![0; N_EXPERTS],
            delta_threshold: 0.5, // Only use delta if < 50% of weights differ
            stats: DeltaStats::default(),
        }
    }

    /// Register a delta file.
    pub fn register_delta(&mut self, info: DeltaFileInfo) {
        self.deltas.insert((info.source, info.target), info);
    }

    /// Register expert full file size.
    pub fn register_expert_size(&mut self, expert_id: usize, size: u64) {
        self.full_sizes[expert_id] = size;
    }

    /// Decide how to load an expert given the currently buffered expert.
    pub fn decide(
        &self,
        target: usize,
        buffered: Option<usize>,
        manifold: &ExpertManifold,
    ) -> LoadStrategy {
        // Already buffered?
        if buffered == Some(target) {
            return LoadStrategy::Cached { expert_id: target };
        }

        // Do we have a delta from the buffered expert?
        if let Some(source) = buffered {
            let are_neighbours =
                manifold.are_neighbours(source, target) || manifold.are_neighbours(target, source);

            if let Some(delta_info) = self.deltas.get(&(source, target)) {
                if are_neighbours || delta_info.diff_ratio < self.delta_threshold {
                    return LoadStrategy::Delta {
                        source,
                        target,
                        delta_info: delta_info.clone(),
                    };
                }
            }

            // Check reverse delta (if A→B exists, we can reverse it)
            if let Some(delta_info) = self.deltas.get(&(target, source)) {
                if are_neighbours || delta_info.diff_ratio < self.delta_threshold {
                    // Create a reversed version
                    let reversed = DeltaFileInfo {
                        source,
                        target,
                        path: delta_info.path.clone(), // Same file, applied in reverse
                        size_bytes: delta_info.size_bytes,
                        diff_count: delta_info.diff_count,
                        diff_ratio: delta_info.diff_ratio,
                    };
                    return LoadStrategy::Delta {
                        source,
                        target,
                        delta_info: reversed,
                    };
                }
            }
        }

        // Fallback: full load
        LoadStrategy::Full {
            expert_id: target,
            size_bytes: self.full_sizes[target],
        }
    }

    /// Record a load event for statistics.
    pub fn record_load(&mut self, strategy: &LoadStrategy) {
        match strategy {
            LoadStrategy::Full { size_bytes, .. } => {
                self.stats.full_loads += 1;
                self.stats.bytes_loaded += size_bytes;
            }
            LoadStrategy::Delta { delta_info, .. } => {
                self.stats.delta_loads += 1;
                self.stats.bytes_loaded += delta_info.size_bytes;
                // Savings = full size - delta size
                let full_size = self.full_sizes[delta_info.target];
                if full_size > delta_info.size_bytes {
                    self.stats.bytes_saved += full_size - delta_info.size_bytes;
                }
            }
            LoadStrategy::Cached { .. } => {
                // No load needed
            }
        }
    }

    /// Estimate the expected load time for a strategy (milliseconds).
    pub fn estimated_load_time_ms(&self, strategy: &LoadStrategy, bandwidth_gbps: f64) -> f64 {
        let bytes = match strategy {
            LoadStrategy::Full { size_bytes, .. } => *size_bytes as f64,
            LoadStrategy::Delta { delta_info, .. } => delta_info.size_bytes as f64,
            LoadStrategy::Cached { .. } => 0.0,
        };
        let bandwidth_bytes = bandwidth_gbps * 1e9;
        (bytes / bandwidth_bytes) * 1000.0
    }

    /// Scan a directory for delta files and register them.
    pub fn scan_delta_directory(&mut self, dir: &Path) -> anyhow::Result<usize> {
        let mut count = 0;
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read delta directory '{}'", dir.display()))?;
        for entry in entries {
            let entry =
                entry.with_context(|| format!("failed to read an entry in '{}'", dir.display()))?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Parse filename: delta_SSS_TTT.trd
                if let Some(info) = parse_delta_filename(name, &path).with_context(|| {
                    format!("failed to parse delta metadata for '{}'", path.display())
                })? {
                    self.register_delta(info);
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

/// Parse a delta filename like "delta_042_043.trd" into a DeltaFileInfo.
fn parse_delta_filename(name: &str, path: &Path) -> anyhow::Result<Option<DeltaFileInfo>> {
    let Some(stripped) = name
        .strip_prefix("delta_")
        .and_then(|n| n.strip_suffix(".trd"))
    else {
        return Ok(None);
    };
    let Some((source_s, target_s)) = stripped.split_once('_') else {
        return Ok(None);
    };
    if source_s.is_empty() || target_s.is_empty() || target_s.contains('_') {
        return Ok(None);
    }

    let source: usize = match source_s.parse() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let target: usize = match target_s.parse() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    if source >= N_EXPERTS || target >= N_EXPERTS {
        return Ok(None);
    }

    let size_bytes = std::fs::metadata(path)
        .with_context(|| format!("failed to read metadata for '{}'", path.display()))?
        .len();

    Ok(Some(DeltaFileInfo {
        source,
        target,
        path: path.to_path_buf(),
        size_bytes,
        diff_count: 0,   // Unknown until loaded
        diff_ratio: 0.1, // Default estimate: 10% diff for neighbours
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_cached_strategy() {
        let mgr = DeltaStreamManager::new();
        let manifold = ExpertManifold::default_grid();
        let strategy = mgr.decide(5, Some(5), &manifold);
        assert!(matches!(strategy, LoadStrategy::Cached { expert_id: 5 }));
    }

    #[test]
    fn test_full_load_when_no_delta() {
        let mut mgr = DeltaStreamManager::new();
        mgr.register_expert_size(10, 27_000_000);
        let manifold = ExpertManifold::default_grid();
        let strategy = mgr.decide(10, Some(50), &manifold);
        assert!(matches!(strategy, LoadStrategy::Full { expert_id: 10, .. }));
    }

    #[test]
    fn test_delta_load_when_available() {
        let mut mgr = DeltaStreamManager::new();
        mgr.register_expert_size(1, 27_000_000);
        mgr.register_delta(DeltaFileInfo {
            source: 0,
            target: 1,
            path: PathBuf::from("delta_000_001.trd"),
            size_bytes: 2_700_000,
            diff_count: 1_000_000,
            diff_ratio: 0.1,
        });
        let manifold = ExpertManifold::default_grid();
        let strategy = mgr.decide(1, Some(0), &manifold);
        assert!(matches!(strategy, LoadStrategy::Delta { .. }));
    }

    #[test]
    fn test_delta_stats() {
        let mut stats = DeltaStats::default();
        stats.delta_loads = 70;
        stats.full_loads = 30;
        assert!((stats.delta_ratio() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_load_time_estimate() {
        let mgr = DeltaStreamManager::new();
        let strategy = LoadStrategy::Full {
            expert_id: 0,
            size_bytes: 27_000_000,
        };
        let time_ms = mgr.estimated_load_time_ms(&strategy, 12.4);
        // 27MB / 12.4 GB/s ≈ 2.18 ms
        assert!(time_ms > 2.0 && time_ms < 2.5);
    }

    #[test]
    fn test_parse_delta_filename() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("delta_042_043.trd");
        fs::write(&path, [1_u8, 2, 3, 4]).unwrap();

        let info = parse_delta_filename("delta_042_043.trd", &path)
            .unwrap()
            .unwrap();
        assert_eq!(info.source, 42);
        assert_eq!(info.target, 43);
        assert_eq!(info.size_bytes, 4);
    }

    #[test]
    fn test_parse_invalid_filename() {
        assert!(parse_delta_filename("expert_042.trit", Path::new("x"))
            .unwrap()
            .is_none());
        assert!(parse_delta_filename("delta_999_001.trd", Path::new("x"))
            .unwrap()
            .is_none());
        assert!(
            parse_delta_filename("delta_001_002_003.trd", Path::new("x"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_parse_delta_filename_propagates_metadata_errors() {
        let err = parse_delta_filename("delta_001_002.trd", Path::new("missing_delta.trd"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("failed to read metadata"));
    }

    #[test]
    fn test_scan_delta_directory_propagates_read_dir_errors() {
        let mut mgr = DeltaStreamManager::new();
        let tmp = tempdir().unwrap();
        let missing_dir = tmp.path().join("missing");
        let err = mgr
            .scan_delta_directory(&missing_dir)
            .unwrap_err()
            .to_string();
        assert!(err.contains("failed to read delta directory"));
    }

    #[test]
    fn test_scan_delta_directory_registers_valid_delta_files() {
        let mut mgr = DeltaStreamManager::new();
        let tmp = tempdir().unwrap();
        fs::write(tmp.path().join("delta_000_001.trd"), [9_u8, 9]).unwrap();
        fs::write(tmp.path().join("not_delta.txt"), [1_u8]).unwrap();

        let count = mgr.scan_delta_directory(tmp.path()).unwrap();
        assert_eq!(count, 1);
        assert!(mgr.deltas.contains_key(&(0, 1)));
    }

    #[test]
    fn test_decide_prefers_delta_for_neighbours() {
        let mut mgr = DeltaStreamManager::new();
        mgr.delta_threshold = 0.01;
        mgr.register_expert_size(1, 27_000_000);
        mgr.register_delta(DeltaFileInfo {
            source: 0,
            target: 1,
            path: PathBuf::from("delta_000_001.trd"),
            size_bytes: 10_000_000,
            diff_count: 50_000_000,
            diff_ratio: 0.9,
        });

        let manifold = ExpertManifold::default_grid();
        assert!(manifold.are_neighbours(0, 1) || manifold.are_neighbours(1, 0));

        let strategy = mgr.decide(1, Some(0), &manifold);
        assert!(matches!(
            strategy,
            LoadStrategy::Delta {
                source: 0,
                target: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_decide_falls_back_to_full_for_non_neighbour_high_diff() {
        let mut mgr = DeltaStreamManager::new();
        mgr.delta_threshold = 0.01;
        mgr.register_expert_size(68, 27_000_000);
        mgr.register_delta(DeltaFileInfo {
            source: 0,
            target: 68,
            path: PathBuf::from("delta_000_068.trd"),
            size_bytes: 20_000_000,
            diff_count: 60_000_000,
            diff_ratio: 0.9,
        });

        let manifold = ExpertManifold::default_grid();
        assert!(!manifold.are_neighbours(0, 68));
        assert!(!manifold.are_neighbours(68, 0));

        let strategy = mgr.decide(68, Some(0), &manifold);
        assert!(matches!(strategy, LoadStrategy::Full { expert_id: 68, .. }));
    }
}
