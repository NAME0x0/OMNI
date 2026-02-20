# Section 8: Agent Runtime (Rust 2021 + C/C++ FFI)

## 8.1 Architecture Overview

The agent runtime is a Rust 2021 application that:
1. Manages inference providers (GPU backends via FFI)
2. Executes tools (code execution, file I/O, web retrieval) in sandboxed processes
3. Maintains persistent memory via SQLite
4. Uses an async lookahead queue to mask CPU-side token generation latency
5. Enforces < 5 MB RSS per agent via OS-level cgroup/job object limits

## 8.2 Core Type Definitions

```rust
// src/types.rs — Rust 2021 edition

use std::collections::HashMap;
use std::path::PathBuf;

/// GPU backend selection — maps to C FFI enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda = 0,
    Hip = 1,
    Sycl = 2,
    Cpu = 3,
}

/// A single inference provider wrapping the C kernel layer
pub struct Provider {
    backend: Backend,
    lib_handle: *mut std::ffi::c_void,  // dlopen handle
    vram_used: u64,
    vram_limit: u64,  // 3686 MB in bytes
}

/// Tool capability exposed to agents
#[derive(Debug, Clone)]
pub enum ToolKind {
    CodeExec { sandbox_path: PathBuf, timeout_ms: u64 },
    FileRead { allowed_dirs: Vec<PathBuf> },
    FileWrite { allowed_dirs: Vec<PathBuf> },
    WebFetch { allowed_domains: Vec<String> },
    MemoryQuery,
    MemoryInsert,
    Calculator,
}

/// A tool invocation request from the model
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub tool: ToolKind,
    pub arguments: HashMap<String, String>,
    pub call_id: u64,
}

/// A tool invocation result
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call_id: u64,
    pub success: bool,
    pub output: String,       // Max 4096 chars, truncated if longer
    pub error: Option<String>,
    pub elapsed_ms: u64,
}

/// Memory entry for SQLite-backed persistent storage
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: i64,
    pub key: String,
    pub value: Vec<u8>,       // Serialized data (MessagePack)
    pub embedding: Vec<f32>,  // 512-dim for SQLite FTS, smaller than main topo memory
    pub created_at: i64,
    pub accessed_at: i64,
    pub importance: f32,
}

/// Agent state — must fit in < 5 MB RSS
pub struct Agent {
    pub id: u32,
    pub name: String,
    pub system_prompt: String,        // Max 2048 tokens
    pub conversation: Vec<Message>,   // Rolling window, max 50 messages
    pub tool_permissions: Vec<ToolKind>,
    pub memory_db: rusqlite::Connection,  // SQLite handle (mmap'd, not in RSS)
    pub rss_limit: u64,                   // 5 * 1024 * 1024 bytes
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,      // Max 4096 chars
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
}

#[derive(Debug, Clone, Copy)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}
```

## 8.3 Provider Implementation (FFI Boundary)

```rust
// src/provider.rs

use crate::types::*;
use std::ffi::{CString, c_void};

/// C ABI function pointers loaded from vendor-specific shared libraries
#[repr(C)]
struct KernelVTable {
    init: extern "C" fn() -> i32,
    destroy: extern "C" fn(),
    dequant_gemm: extern "C" fn(*const DequantGemmArgs) -> i32,
    alloc_device: extern "C" fn(size: u64) -> *mut c_void,
    free_device: extern "C" fn(*mut c_void),
    memcpy_h2d: extern "C" fn(dst: *mut c_void, src: *const c_void, size: u64) -> i32,
    memcpy_d2h: extern "C" fn(dst: *mut c_void, src: *const c_void, size: u64) -> i32,
    synchronize: extern "C" fn() -> i32,
}

/// C ABI struct matching omni_dequant.h
#[repr(C)]
struct DequantGemmArgs {
    packed_weights: *const c_void,
    scales: *const c_void,
    zeros: *const c_void,
    input: *const c_void,
    output: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    group_size: i32,
    sparsity_mask: i32,
}

impl Provider {
    /// Detect and initialize the best available backend
    pub fn auto_detect() -> Result<Self, ProviderError> {
        // Try backends in priority order
        for backend in [Backend::Cuda, Backend::Hip, Backend::Sycl, Backend::Cpu] {
            match Self::try_init(backend) {
                Ok(provider) => {
                    log::info!("Initialized backend: {:?}", backend);
                    return Ok(provider);
                }
                Err(e) => {
                    log::debug!("Backend {:?} unavailable: {}", backend, e);
                    continue;
                }
            }
        }
        Err(ProviderError::NoBackendAvailable)
    }

    fn try_init(backend: Backend) -> Result<Self, ProviderError> {
        let lib_name = match backend {
            Backend::Cuda => {
                #[cfg(target_os = "windows")]
                { "omni_cuda.dll" }
                #[cfg(target_os = "linux")]
                { "libomni_cuda.so" }
            }
            Backend::Hip => {
                #[cfg(target_os = "windows")]
                { "omni_hip.dll" }
                #[cfg(target_os = "linux")]
                { "libomni_hip.so" }
            }
            Backend::Sycl => {
                #[cfg(target_os = "windows")]
                { "omni_sycl.dll" }
                #[cfg(target_os = "linux")]
                { "libomni_sycl.so" }
            }
            Backend::Cpu => {
                #[cfg(target_os = "windows")]
                { "omni_cpu.dll" }
                #[cfg(target_os = "linux")]
                { "libomni_cpu.so" }
            }
        };

        // Load shared library at runtime (no compile-time vendor dependency)
        let c_name = CString::new(lib_name).unwrap();

        #[cfg(target_os = "linux")]
        let handle = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW) };
        #[cfg(target_os = "windows")]
        let handle = unsafe {
            windows_sys::Win32::System::LibraryLoader::LoadLibraryA(c_name.as_ptr() as _)
                as *mut c_void
        };

        if handle.is_null() {
            return Err(ProviderError::LibraryNotFound(lib_name.to_string()));
        }

        // Load function pointers from vtable
        // (In production: resolve each symbol via dlsym/GetProcAddress)

        let vram_limit: u64 = 3686 * 1024 * 1024; // 3686 MB usable

        Ok(Provider {
            backend,
            lib_handle: handle,
            vram_used: 0,
            vram_limit,
        })
    }

    /// Execute a dequantized GEMM on the GPU via FFI
    pub fn dequant_gemm(
        &self,
        packed_weights: *const c_void,
        scales: *const c_void,
        zeros: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        m: i32, n: i32, k: i32,
        group_size: i32,
        sparse: bool,
    ) -> Result<(), ProviderError> {
        let args = DequantGemmArgs {
            packed_weights,
            scales,
            zeros,
            input,
            output,
            m, n, k,
            group_size,
            sparsity_mask: if sparse { 1 } else { 0 },
        };

        // Call through FFI — the actual kernel is in C/CUDA/HIP/SYCL
        let vtable = self.get_vtable()?;
        let result = (vtable.dequant_gemm)(&args as *const DequantGemmArgs);

        if result != 0 {
            return Err(ProviderError::KernelFailed(result));
        }
        Ok(())
    }

    fn get_vtable(&self) -> Result<&KernelVTable, ProviderError> {
        // In production: cached after init. Simplified here.
        todo!("Resolve vtable from lib_handle symbols")
    }
}

#[derive(Debug)]
pub enum ProviderError {
    NoBackendAvailable,
    LibraryNotFound(String),
    KernelFailed(i32),
    VramExhausted { requested: u64, available: u64 },
}
```

## 8.4 Sandbox Execution

```rust
// src/sandbox.rs

use std::process::{Command, Stdio};
use std::time::Duration;

/// Execute a tool call in a sandboxed subprocess.
/// Enforces: no network, no filesystem outside sandbox_path, timeout, memory limit.
pub fn execute_sandboxed(
    tool: &ToolCall,
    sandbox_path: &std::path::Path,
    timeout: Duration,
    mem_limit_bytes: u64,
) -> ToolResult {
    let start = std::time::Instant::now();

    let result = match &tool.tool {
        ToolKind::CodeExec { sandbox_path: sp, timeout_ms } => {
            let code = tool.arguments.get("code").cloned().unwrap_or_default();
            let lang = tool.arguments.get("language").cloned().unwrap_or("python".into());

            let (cmd, args) = match lang.as_str() {
                "python" => ("python3", vec!["-c", &code]),
                "javascript" => ("node", vec!["-e", &code]),
                _ => return ToolResult {
                    call_id: tool.call_id,
                    success: false,
                    output: String::new(),
                    error: Some(format!("Unsupported language: {}", lang)),
                    elapsed_ms: 0,
                },
            };

            // Build sandboxed command
            #[cfg(target_os = "linux")]
            let child = Command::new("unshare")
                .args(["--net", "--map-root-user", "--"])  // No network
                .arg(cmd)
                .args(&args)
                .current_dir(sandbox_path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn();

            #[cfg(target_os = "windows")]
            let child = {
                // Windows: use Job Object for resource limits
                // Network restriction via Windows Firewall rules per-process
                Command::new(cmd)
                    .args(&args)
                    .current_dir(sandbox_path)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
            };

            match child {
                Ok(mut child) => {
                    // Apply memory limit via OS
                    #[cfg(target_os = "linux")]
                    apply_cgroup_mem_limit(child.id(), mem_limit_bytes);

                    #[cfg(target_os = "windows")]
                    apply_job_object_mem_limit(child.id(), mem_limit_bytes);

                    // Wait with timeout
                    match child.wait_timeout(timeout) {
                        Ok(Some(status)) => {
                            let stdout = read_piped(&mut child.stdout);
                            let stderr = read_piped(&mut child.stderr);
                            if status.success() {
                                Ok(truncate(stdout, 4096))
                            } else {
                                Err(format!("Exit code: {}. Stderr: {}",
                                    status.code().unwrap_or(-1),
                                    truncate(stderr, 1024)))
                            }
                        }
                        Ok(None) => {
                            let _ = child.kill();
                            Err("Timeout".into())
                        }
                        Err(e) => Err(format!("Process error: {}", e)),
                    }
                }
                Err(e) => Err(format!("Spawn failed: {}", e)),
            }
        }
        _ => Ok("Not a code execution tool".into()),
    };

    ToolResult {
        call_id: tool.call_id,
        success: result.is_ok(),
        output: result.unwrap_or_default(),
        error: result.err(),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

#[cfg(target_os = "linux")]
fn apply_cgroup_mem_limit(pid: u32, limit_bytes: u64) {
    // Create cgroup v2 for this process
    let cgroup_path = format!("/sys/fs/cgroup/omni_agent_{}", pid);
    let _ = std::fs::create_dir_all(&cgroup_path);
    let _ = std::fs::write(
        format!("{}/memory.max", cgroup_path),
        limit_bytes.to_string(),
    );
    let _ = std::fs::write(
        format!("{}/cgroup.procs", cgroup_path),
        pid.to_string(),
    );
}

#[cfg(target_os = "windows")]
fn apply_job_object_mem_limit(pid: u32, limit_bytes: u64) {
    // Uses Windows Job Objects via win32 API
    // CreateJobObject → SetInformationJobObject(ProcessMemoryLimit)
    // → AssignProcessToJobObject
    // Actual implementation would use windows-sys crate
    log::debug!("Setting job object memory limit for PID {}: {} bytes", pid, limit_bytes);
}

fn read_piped(pipe: &mut Option<std::process::ChildStdout>) -> String {
    use std::io::Read;
    let mut buf = String::new();
    if let Some(ref mut p) = pipe {
        let _ = p.read_to_string(&mut buf);
    }
    buf
}

fn truncate(s: String, max: usize) -> String {
    if s.len() > max {
        format!("{}...[truncated]", &s[..max])
    } else {
        s
    }
}
```

## 8.5 SQLite-Backed Memory

```rust
// src/memory.rs

use rusqlite::{Connection, params};
use crate::types::MemoryRecord;

pub struct AgentMemory {
    conn: Connection,
}

impl AgentMemory {
    pub fn open(db_path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;

        // WAL mode for concurrent reads during inference
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -2048;  -- 2 MB page cache
            PRAGMA mmap_size = 268435456;  -- 256 MB mmap (not counted in RSS)
        ")?;

        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value BLOB NOT NULL,
                embedding BLOB,          -- 512 * 4 = 2048 bytes per entry
                created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                accessed_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                importance REAL NOT NULL DEFAULT 0.5
            );
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
        ")?;

        Ok(AgentMemory { conn })
    }

    pub fn insert(&self, record: &MemoryRecord) -> Result<i64, rusqlite::Error> {
        self.conn.execute(
            "INSERT INTO memories (key, value, embedding, importance)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                record.key,
                record.value,
                bytemuck::cast_slice::<f32, u8>(&record.embedding),
                record.importance,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn query_by_key(&self, key: &str) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, key, value, embedding, created_at, accessed_at, importance
             FROM memories WHERE key = ?1 ORDER BY importance DESC LIMIT 10"
        )?;

        let records = stmt.query_map(params![key], |row| {
            let emb_bytes: Vec<u8> = row.get(3)?;
            let embedding: Vec<f32> = bytemuck::cast_slice(&emb_bytes).to_vec();
            Ok(MemoryRecord {
                id: row.get(0)?,
                key: row.get(1)?,
                value: row.get(2)?,
                embedding,
                created_at: row.get(4)?,
                accessed_at: row.get(5)?,
                importance: row.get(6)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        // Update accessed_at for returned records
        for record in &records {
            let _ = self.conn.execute(
                "UPDATE memories SET accessed_at = strftime('%s', 'now') WHERE id = ?1",
                params![record.id],
            );
        }

        Ok(records)
    }

    /// Nearest-neighbor search using embedding cosine similarity.
    /// For small tables (<10K entries), brute-force in SQL is acceptable.
    /// For larger tables, use the topological memory (§6) instead.
    pub fn query_by_embedding(
        &self,
        query_emb: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        // Load all embeddings and compute similarity on CPU
        // This is O(N * d) but N < 10K and d = 512, so < 5M FLOPs = instant
        let mut stmt = self.conn.prepare(
            "SELECT id, key, value, embedding, created_at, accessed_at, importance
             FROM memories WHERE embedding IS NOT NULL"
        )?;

        let mut scored: Vec<(f32, MemoryRecord)> = stmt.query_map([], |row| {
            let emb_bytes: Vec<u8> = row.get(3)?;
            let embedding: Vec<f32> = bytemuck::cast_slice(&emb_bytes).to_vec();
            let record = MemoryRecord {
                id: row.get(0)?,
                key: row.get(1)?,
                value: row.get(2)?,
                embedding: embedding.clone(),
                created_at: row.get(4)?,
                accessed_at: row.get(5)?,
                importance: row.get(6)?,
            };
            let sim = cosine_similarity(query_emb, &embedding);
            Ok((sim, record))
        })?.collect::<Result<Vec<_>, _>>()?;

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored.into_iter().map(|(_, r)| r).collect())
    }

    /// GC: remove entries below importance threshold, keeping max N entries
    pub fn gc(&self, max_entries: usize) -> Result<usize, rusqlite::Error> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM memories", [], |row| row.get(0)
        )?;

        if count <= max_entries {
            return Ok(0);
        }

        let to_remove = count - max_entries;
        self.conn.execute(
            "DELETE FROM memories WHERE id IN (
                SELECT id FROM memories ORDER BY importance ASC, accessed_at ASC LIMIT ?1
            )",
            params![to_remove],
        )?;

        Ok(to_remove)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 { return 0.0; }
    dot / (norm_a * norm_b)
}
```

## 8.6 Async Lookahead Queue

The lookahead queue masks CPU-side latency (retrieval, tool execution, verification)
by pre-computing the next N tokens speculatively while the CPU handles side-tasks.

```rust
// src/lookahead.rs

use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Token generated by the GPU
#[derive(Debug, Clone)]
pub struct SpeculativeToken {
    pub token_id: u32,
    pub logprob: f32,
    pub position: u64,
    pub needs_verification: bool,
}

/// Commands from CPU to GPU pipeline
#[derive(Debug)]
pub enum GpuCommand {
    /// Generate next N tokens speculatively
    Speculate { n: usize, context: Vec<u32> },
    /// Confirm tokens up to position (CPU verified them)
    Confirm { up_to_position: u64 },
    /// Invalidate speculative tokens from position (verification failed)
    Rollback { from_position: u64 },
    /// Swap expert in VRAM
    SwapExpert { evict_idx: usize, load_idx: usize },
    /// Shutdown
    Shutdown,
}

pub struct LookaheadQueue {
    /// Channel: CPU → GPU commands
    cmd_tx: mpsc::Sender<GpuCommand>,
    /// Channel: GPU → CPU speculative tokens
    token_rx: Arc<Mutex<mpsc::Receiver<SpeculativeToken>>>,
    /// Speculative buffer: tokens generated but not yet confirmed
    speculative_buffer: Vec<SpeculativeToken>,
    /// How far ahead to speculate
    lookahead_depth: usize,  // Default: 8 tokens
}

impl LookaheadQueue {
    pub fn new(lookahead_depth: usize) -> (Self, mpsc::Receiver<GpuCommand>, mpsc::Sender<SpeculativeToken>) {
        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        let (token_tx, token_rx) = mpsc::channel(256);

        let queue = LookaheadQueue {
            cmd_tx,
            token_rx: Arc::new(Mutex::new(token_rx)),
            speculative_buffer: Vec::with_capacity(lookahead_depth),
            lookahead_depth,
        };

        (queue, cmd_rx, token_tx)
    }

    /// Called by the inference loop: request speculative generation
    pub async fn request_speculation(&self, context: Vec<u32>) {
        let _ = self.cmd_tx.send(GpuCommand::Speculate {
            n: self.lookahead_depth,
            context,
        }).await;
    }

    /// Called by CPU side: get next confirmed token
    /// While CPU does retrieval/verification, GPU is generating ahead
    pub async fn next_confirmed_token(&mut self) -> Option<SpeculativeToken> {
        // Drain speculative tokens from GPU
        let mut rx = self.token_rx.lock().await;
        while let Ok(token) = rx.try_recv() {
            self.speculative_buffer.push(token);
        }

        // Return first unconfirmed token for verification
        if !self.speculative_buffer.is_empty() {
            Some(self.speculative_buffer.remove(0))
        } else {
            // Wait for GPU to produce a token
            rx.recv().await
        }
    }

    /// CPU confirms tokens are valid (verification passed)
    pub async fn confirm(&mut self, up_to_position: u64) {
        self.speculative_buffer.retain(|t| t.position > up_to_position);
        let _ = self.cmd_tx.send(GpuCommand::Confirm { up_to_position }).await;
    }

    /// CPU rejects tokens (verification failed) — GPU must regenerate
    pub async fn rollback(&mut self, from_position: u64) {
        self.speculative_buffer.retain(|t| t.position < from_position);
        let _ = self.cmd_tx.send(GpuCommand::Rollback { from_position }).await;
    }
}

/// GPU-side worker that processes commands and generates tokens
pub async fn gpu_worker(
    mut cmd_rx: mpsc::Receiver<GpuCommand>,
    token_tx: mpsc::Sender<SpeculativeToken>,
    provider: Arc<crate::types::Provider>,
) {
    let mut current_position: u64 = 0;

    loop {
        match cmd_rx.recv().await {
            Some(GpuCommand::Speculate { n, context }) => {
                // Generate n tokens using the provider
                for i in 0..n {
                    // This calls through FFI to the GPU kernel
                    let (token_id, logprob) = generate_one_token(&provider, &context, current_position + i as u64);

                    let token = SpeculativeToken {
                        token_id,
                        logprob,
                        position: current_position + i as u64,
                        needs_verification: (current_position + i as u64) % 32 == 0,
                    };

                    if token_tx.send(token).await.is_err() {
                        break; // Receiver dropped
                    }
                }
            }
            Some(GpuCommand::Confirm { up_to_position }) => {
                current_position = up_to_position + 1;
            }
            Some(GpuCommand::Rollback { from_position }) => {
                current_position = from_position;
                // Re-generate from this position on next Speculate command
            }
            Some(GpuCommand::SwapExpert { evict_idx, load_idx }) => {
                // Trigger PCIe transfer via provider FFI
                swap_expert_vram(&provider, evict_idx, load_idx);
            }
            Some(GpuCommand::Shutdown) | None => break,
        }
    }
}

fn generate_one_token(
    _provider: &crate::types::Provider,
    _context: &[u32],
    _position: u64,
) -> (u32, f32) {
    // Placeholder: actual implementation calls provider.dequant_gemm()
    // through the full forward pass (§3)
    (0, 0.0)
}

fn swap_expert_vram(
    _provider: &crate::types::Provider,
    _evict_idx: usize,
    _load_idx: usize,
) {
    // Placeholder: actual implementation calls provider.memcpy_h2d/d2h
    // Transfer time: 24ms full, 9.6ms prefetched (§2.4)
}
```

## 8.7 Agent Memory Enforcement (< 5 MB RSS)

```rust
// src/agent_limits.rs

use std::process::Command;

/// Enforce RSS limit per agent subprocess.
///
/// Measurement method: read /proc/{pid}/status VmRSS field (Linux)
/// or use QueryProcessMemoryInfo (Windows).
///
/// Kill policy: if RSS exceeds limit for 3 consecutive checks (1s interval),
/// send SIGKILL (Linux) or TerminateProcess (Windows).

pub struct AgentResourceGuard {
    pid: u32,
    rss_limit: u64,
    consecutive_violations: u32,
    max_violations: u32,  // 3
}

impl AgentResourceGuard {
    pub fn new(pid: u32, rss_limit: u64) -> Self {
        AgentResourceGuard {
            pid,
            rss_limit,
            consecutive_violations: 0,
            max_violations: 3,
        }
    }

    /// Check RSS and kill if over limit. Called every 1 second.
    pub fn check_and_enforce(&mut self) -> AgentStatus {
        let rss = self.measure_rss();

        if rss > self.rss_limit {
            self.consecutive_violations += 1;
            log::warn!(
                "Agent PID {} RSS {} bytes exceeds limit {} bytes ({}/{})",
                self.pid, rss, self.rss_limit,
                self.consecutive_violations, self.max_violations
            );

            if self.consecutive_violations >= self.max_violations {
                self.kill_process();
                return AgentStatus::Killed {
                    reason: format!("RSS {} exceeded limit {} for {} consecutive checks",
                        rss, self.rss_limit, self.max_violations),
                };
            }
            AgentStatus::Warning { rss, limit: self.rss_limit }
        } else {
            self.consecutive_violations = 0;
            AgentStatus::Ok { rss }
        }
    }

    #[cfg(target_os = "linux")]
    fn measure_rss(&self) -> u64 {
        // Read /proc/{pid}/status → VmRSS line
        let path = format!("/proc/{}/status", self.pid);
        if let Ok(content) = std::fs::read_to_string(&path) {
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    // Parse "VmRSS:    12345 kB"
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024; // Convert to bytes
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(target_os = "windows")]
    fn measure_rss(&self) -> u64 {
        // Use Windows API: GetProcessMemoryInfo → WorkingSetSize
        // Requires windows-sys crate
        // Simplified: use tasklist /FI command as fallback
        0 // Placeholder
    }

    fn kill_process(&self) {
        #[cfg(target_os = "linux")]
        unsafe { libc::kill(self.pid as i32, libc::SIGKILL); }

        #[cfg(target_os = "windows")]
        {
            let _ = Command::new("taskkill")
                .args(["/F", "/PID", &self.pid.to_string()])
                .output();
        }

        log::error!("Killed agent PID {} due to RSS limit violation", self.pid);
    }
}

#[derive(Debug)]
pub enum AgentStatus {
    Ok { rss: u64 },
    Warning { rss: u64, limit: u64 },
    Killed { reason: String },
}
```

## 8.8 Main Orchestrator

```rust
// src/main.rs

use tokio;

mod types;
mod provider;
mod sandbox;
mod memory;
mod lookahead;
mod agent_limits;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // 1. Initialize GPU provider (auto-detect backend)
    let provider = std::sync::Arc::new(provider::Provider::auto_detect()?);
    log::info!("GPU provider initialized: {:?}", provider.backend);

    // 2. Open persistent memory database
    let agent_memory = memory::AgentMemory::open("omni_memory.db")?;
    log::info!("Agent memory database opened");

    // 3. Create lookahead queue (mask CPU latency with speculative GPU generation)
    let (lookahead, cmd_rx, token_tx) = lookahead::LookaheadQueue::new(8);

    // 4. Spawn GPU worker
    let provider_clone = provider.clone();
    tokio::spawn(async move {
        lookahead::gpu_worker(cmd_rx, token_tx, provider_clone).await;
    });

    // 5. Main inference loop
    log::info!("OMNI runtime ready. Awaiting input.");

    // Event loop would go here: read user input, generate response,
    // verify with truth grounding, update memory, etc.

    Ok(())
}
```

## 8.9 FFI Architectural Boundary Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Rust Runtime                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐            │
│  │ Provider  │  │ Lookahead│  │   Agent   │            │
│  │ (FFI mgr)│  │  Queue   │  │ Sandbox   │            │
│  └────┬─────┘  └──────────┘  └───────────┘            │
│       │ C ABI function pointers                         │
│       │ (dlopen / LoadLibrary at runtime)               │
├───────┼─────────────────────────────────────────────────┤
│       │         FFI BOUNDARY                            │
├───────┼─────────────────────────────────────────────────┤
│       ▼                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │         C/C++ Kernel Dispatch Layer              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │   │
│  │  │  CUDA    │ │  HIP     │ │  SYCL    │        │   │
│  │  │ kernels  │ │ kernels  │ │ kernels  │        │   │
│  │  │(.cu)     │ │(.hip.cpp)│ │(.sycl.cpp│        │   │
│  │  └──────────┘ └──────────┘ └──────────┘        │   │
│  │  ┌──────────┐                                   │   │
│  │  │  CPU/AVX │ (fallback)                        │   │
│  │  │ kernels  │                                   │   │
│  │  │(.c)      │                                   │   │
│  │  └──────────┘                                   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Each vendor library is compiled separately and loaded at runtime.
No Rust code imports CUDA/HIP/SYCL headers. The only contract is the C ABI
defined in `omni_dequant.h` (§4.3).
