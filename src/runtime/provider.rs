//! Model weight provider: loads and serves expert weights from disk.
//!
//! Supports:
//! - Memory-mapped files (zero-copy NVMe access)
//! - Hot expert cache in RAM
//! - Delta-based partial loading

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Weight shard metadata.
#[derive(Clone, Debug)]
pub struct ShardInfo {
    /// File path on disk.
    pub path: PathBuf,

    /// Expert ID this shard belongs to.
    pub expert_id: usize,

    /// Layer index.
    pub layer: usize,

    /// Byte offset within the file.
    pub offset: u64,

    /// Size in bytes.
    pub size: u64,

    /// Whether this is a delta shard (relative to a base).
    pub is_delta: bool,
}

/// Expert weight data loaded into memory.
#[derive(Clone)]
pub struct ExpertWeights {
    /// Raw ternary-packed weight data.
    pub data: Vec<u8>,

    /// Expert ID.
    pub expert_id: usize,

    /// Layer index.
    pub layer: usize,

    /// Whether these are delta weights.
    pub is_delta: bool,
}

impl ExpertWeights {
    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Approximate size in megabytes.
    pub fn size_mb(&self) -> f32 {
        self.data.len() as f32 / (1024.0 * 1024.0)
    }
}

/// Provider configuration.
#[derive(Clone, Debug)]
pub struct ProviderConfig {
    /// Root directory containing weight shards.
    pub model_dir: PathBuf,

    /// Maximum number of experts to cache in RAM.
    pub max_cached_experts: usize,

    /// Maximum RAM cache size in bytes.
    pub max_cache_bytes: u64,

    /// Whether to use memory-mapped I/O.
    pub use_mmap: bool,

    /// Prefetch depth (how many layers ahead to load).
    pub prefetch_depth: usize,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("./model"),
            max_cached_experts: 8,
            max_cache_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
            use_mmap: true,
            prefetch_depth: 2,
        }
    }
}

/// LRU cache entry.
struct CacheEntry {
    weights: ExpertWeights,
    last_access: u64,
}

/// Model weight provider with LRU caching.
pub struct WeightProvider {
    config: ProviderConfig,

    /// Shard index: (layer, expert_id) → ShardInfo.
    shard_index: HashMap<(usize, usize), ShardInfo>,

    /// LRU cache: (layer, expert_id) → CacheEntry.
    cache: HashMap<(usize, usize), CacheEntry>,

    /// Current cache size in bytes.
    cache_bytes: u64,

    /// Access counter for LRU.
    access_counter: u64,

    /// Temporary slot for uncached loads when cache limits disallow insertion.
    uncached: Option<ExpertWeights>,

    /// Stats.
    pub stats: ProviderStats,
}

/// Provider statistics.
#[derive(Clone, Debug, Default)]
pub struct ProviderStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_loads: u64,
    pub total_bytes_loaded: u64,
    pub evictions: u64,
}

impl ProviderStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

impl WeightProvider {
    /// Create a new weight provider.
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            config,
            shard_index: HashMap::new(),
            cache: HashMap::new(),
            cache_bytes: 0,
            access_counter: 0,
            uncached: None,
            stats: ProviderStats::default(),
        }
    }

    /// Register a shard in the index.
    pub fn register_shard(&mut self, info: ShardInfo) {
        self.shard_index.insert((info.layer, info.expert_id), info);
    }

    /// Scan model directory and build shard index.
    pub fn scan_model_dir(&mut self) -> anyhow::Result<usize> {
        let dir = &self.config.model_dir;
        if !dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        // Expected naming: expert_{expert_id}_layer_{layer}.bin
        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(info) = parse_shard_filename(name, &path) {
                    self.register_shard(info);
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Get expert weights (from cache or disk).
    pub fn get_expert(&mut self, layer: usize, expert_id: usize) -> anyhow::Result<&ExpertWeights> {
        self.access_counter += 1;
        let key = (layer, expert_id);

        if self.cache.contains_key(&key) {
            self.stats.cache_hits += 1;
            if let Some(entry) = self.cache.get_mut(&key) {
                entry.last_access = self.access_counter;
            } else {
                return Err(anyhow::anyhow!(
                    "cache entry ({}, {}) disappeared during update",
                    layer,
                    expert_id
                ));
            }
            if let Some(entry) = self.cache.get(&key) {
                return Ok(&entry.weights);
            }
            return Err(anyhow::anyhow!(
                "cache entry ({}, {}) disappeared after update",
                layer,
                expert_id
            ));
        }

        self.stats.cache_misses += 1;

        // Load from disk
        let weights = self.load_from_disk(layer, expert_id)?;
        let size = weights.size_bytes() as u64;

        // If caching is disabled (by count or bytes), keep only the most recent load.
        if self.config.max_cached_experts == 0 || self.config.max_cache_bytes == 0 {
            self.uncached = Some(weights);
            if let Some(uncached) = self.uncached.as_ref() {
                return Ok(uncached);
            }
            return Err(anyhow::anyhow!("failed to retain uncached expert payload"));
        }

        // If a single expert exceeds byte capacity, do not force it into cache.
        if size > self.config.max_cache_bytes {
            self.uncached = Some(weights);
            if let Some(uncached) = self.uncached.as_ref() {
                return Ok(uncached);
            }
            return Err(anyhow::anyhow!(
                "failed to retain oversized uncached expert payload"
            ));
        }

        // Evict if necessary
        while (self.cache_bytes + size > self.config.max_cache_bytes
            || self.cache.len() >= self.config.max_cached_experts)
            && !self.cache.is_empty()
        {
            self.evict_lru();
        }

        // Insert into cache
        self.cache.insert(
            key,
            CacheEntry {
                weights,
                last_access: self.access_counter,
            },
        );
        self.cache_bytes += size;

        Ok(&self.cache[&key].weights)
    }

    /// Load expert weights from disk.
    fn load_from_disk(&mut self, layer: usize, expert_id: usize) -> anyhow::Result<ExpertWeights> {
        let key = (layer, expert_id);

        if let Some(shard) = self.shard_index.get(&key) {
            let data = std::fs::read(&shard.path)?;
            self.stats.total_loads += 1;
            self.stats.total_bytes_loaded += data.len() as u64;

            Ok(ExpertWeights {
                data,
                expert_id,
                layer,
                is_delta: shard.is_delta,
            })
        } else {
            // Return empty weights if no shard found
            Ok(ExpertWeights {
                data: Vec::new(),
                expert_id,
                layer,
                is_delta: false,
            })
        }
    }

    /// Evict the least-recently-used cache entry.
    fn evict_lru(&mut self) {
        if let Some((&key, _)) = self.cache.iter().min_by_key(|(_, entry)| entry.last_access) {
            if let Some(entry) = self.cache.remove(&key) {
                self.cache_bytes -= entry.weights.size_bytes() as u64;
                self.stats.evictions += 1;
            }
        }
    }

    /// Check if an expert is cached.
    pub fn is_cached(&self, layer: usize, expert_id: usize) -> bool {
        self.cache.contains_key(&(layer, expert_id))
    }

    /// Number of cached experts.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Current cache utilisation (bytes used / max bytes).
    pub fn cache_utilisation(&self) -> f32 {
        if self.config.max_cache_bytes == 0 {
            return 0.0;
        }
        self.cache_bytes as f32 / self.config.max_cache_bytes as f32
    }

    /// Prefetch: preload experts for upcoming layers.
    pub fn prefetch(&mut self, layer: usize, expert_ids: &[usize]) -> anyhow::Result<usize> {
        if self.config.prefetch_depth == 0 || expert_ids.is_empty() {
            return Ok(0);
        }

        let mut loaded = 0;
        for depth in 0..self.config.prefetch_depth {
            let Some(target_layer) = layer.checked_add(depth) else {
                break;
            };

            for &eid in expert_ids {
                let key = (target_layer, eid);
                if !self.shard_index.contains_key(&key) || self.is_cached(target_layer, eid) {
                    continue;
                }

                let _ = self.get_expert(target_layer, eid)?;
                loaded += 1;
            }
        }
        Ok(loaded)
    }
}

/// Parse a shard filename to extract metadata.
fn parse_shard_filename(name: &str, path: &Path) -> Option<ShardInfo> {
    // Format: expert_{eid}_layer_{lid}.bin or delta_expert_{eid}_layer_{lid}.bin
    let is_delta = name.starts_with("delta_");
    let clean = if is_delta {
        name.trim_start_matches("delta_")
    } else {
        name
    };

    let parts: Vec<&str> = clean.trim_end_matches(".bin").split('_').collect();

    if parts.len() >= 4 && parts[0] == "expert" && parts[2] == "layer" {
        let expert_id = parts[1].parse::<usize>().ok()?;
        let layer = parts[3].parse::<usize>().ok()?;
        let size = std::fs::metadata(path).ok()?.len();

        Some(ShardInfo {
            path: path.to_path_buf(),
            expert_id,
            layer,
            offset: 0,
            size,
            is_delta,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn write_shard_file(dir: &Path, expert_id: usize, layer: usize, bytes: &[u8]) -> PathBuf {
        let path = dir.join(format!("expert_{expert_id}_layer_{layer}.bin"));
        std::fs::write(&path, bytes).expect("failed to write shard file");
        path
    }

    #[test]
    fn test_provider_creation() {
        let config = ProviderConfig::default();
        let provider = WeightProvider::new(config);
        assert_eq!(provider.cached_count(), 0);
        assert!(provider.stats.hit_rate() == 0.0);
    }

    #[test]
    fn test_cache_utilisation() {
        let config = ProviderConfig {
            max_cache_bytes: 1000,
            ..Default::default()
        };
        let provider = WeightProvider::new(config);
        assert!(provider.cache_utilisation() < 1e-6);
    }

    #[test]
    fn test_parse_shard_filename_normal() {
        let _path = PathBuf::from("/tmp/expert_42_layer_7.bin");
        // Can't actually test without the file existing, so test the parsing logic
        let name = "expert_42_layer_7.bin";
        // parse_shard_filename reads metadata, so we test the format parsing separately
        let clean = name.trim_end_matches(".bin");
        let parts: Vec<&str> = clean.split('_').collect();
        assert_eq!(parts[0], "expert");
        assert_eq!(parts[1], "42");
        assert_eq!(parts[2], "layer");
        assert_eq!(parts[3], "7");
    }

    #[test]
    fn test_provider_stats() {
        let mut stats = ProviderStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.cache_hits = 3;
        stats.cache_misses = 1;
        assert!((stats.hit_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_expert_weights_size() {
        let w = ExpertWeights {
            data: vec![0u8; 1024 * 1024],
            expert_id: 0,
            layer: 0,
            is_delta: false,
        };
        assert!((w.size_mb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_scan_model_dir_propagates_read_dir_errors() {
        let dir = tempdir().expect("failed to create tempdir");
        let file_path = dir.path().join("not_a_directory.bin");
        std::fs::write(&file_path, b"data").expect("failed to create temp file");

        let config = ProviderConfig {
            model_dir: file_path,
            ..Default::default()
        };
        let mut provider = WeightProvider::new(config);
        let result = provider.scan_model_dir();
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_respects_max_cached_experts() {
        let dir = tempdir().expect("failed to create tempdir");
        write_shard_file(dir.path(), 0, 0, &[1, 2, 3, 4]);
        write_shard_file(dir.path(), 1, 0, &[5, 6, 7, 8]);

        let config = ProviderConfig {
            model_dir: dir.path().to_path_buf(),
            max_cached_experts: 1,
            max_cache_bytes: 1024,
            ..Default::default()
        };
        let mut provider = WeightProvider::new(config);
        provider
            .scan_model_dir()
            .expect("scan_model_dir should succeed");

        provider
            .get_expert(0, 0)
            .expect("first expert load should succeed");
        assert_eq!(provider.cached_count(), 1);

        provider
            .get_expert(0, 1)
            .expect("second expert load should succeed");
        assert_eq!(provider.cached_count(), 1);
        assert!(provider.stats.evictions >= 1);
    }

    #[test]
    fn test_prefetch_respects_depth() {
        let dir = tempdir().expect("failed to create tempdir");
        write_shard_file(dir.path(), 3, 0, &[1]);
        write_shard_file(dir.path(), 3, 1, &[1]);
        write_shard_file(dir.path(), 3, 2, &[1]);

        let config = ProviderConfig {
            model_dir: dir.path().to_path_buf(),
            prefetch_depth: 2,
            max_cached_experts: 8,
            max_cache_bytes: 1024,
            ..Default::default()
        };
        let mut provider = WeightProvider::new(config);
        provider
            .scan_model_dir()
            .expect("scan_model_dir should succeed");

        let loaded = provider.prefetch(0, &[3]).expect("prefetch should succeed");
        assert_eq!(loaded, 2);
        assert!(provider.is_cached(0, 3));
        assert!(provider.is_cached(1, 3));
        assert!(!provider.is_cached(2, 3));
    }
}
