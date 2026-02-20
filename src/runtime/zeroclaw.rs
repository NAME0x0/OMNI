//! ZeroClaw integration for agentic swarm orchestration.
//!
//! This adapter shells out to a local `zeroclaw` installation so PERSPECTIVE
//! can delegate swarm-style task execution without embedding a second runtime.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use anyhow::{bail, Context, Result};

/// ZeroClaw CLI configuration.
#[derive(Clone, Debug)]
pub struct ZeroclawConfig {
    /// Executable name or absolute path.
    pub executable: PathBuf,
    /// Working directory passed to the CLI.
    pub workspace: PathBuf,
    /// Daemon host.
    pub host: String,
    /// Daemon port.
    pub port: u16,
}

impl Default for ZeroclawConfig {
    fn default() -> Self {
        Self {
            executable: PathBuf::from("zeroclaw"),
            workspace: PathBuf::from("."),
            host: "127.0.0.1".to_string(),
            port: 3000,
        }
    }
}

/// Result from one ZeroClaw CLI invocation.
#[derive(Clone, Debug)]
pub struct ZeroclawRunResult {
    pub status_code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Thin wrapper around the ZeroClaw CLI.
pub struct ZeroclawClient {
    config: ZeroclawConfig,
}

impl ZeroclawClient {
    pub fn new(config: ZeroclawConfig) -> Self {
        Self { config }
    }

    /// Check whether `zeroclaw` is available.
    pub fn check_available(&self) -> Result<()> {
        let output = Command::new(&self.config.executable)
            .arg("--help")
            .current_dir(&self.config.workspace)
            .output()
            .with_context(|| {
                format!(
                    "failed to execute '{}' for availability check",
                    self.config.executable.display()
                )
            })?;

        if output.status.success() {
            Ok(())
        } else {
            bail!(
                "zeroclaw availability check failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    /// Run one swarm task through the ZeroClaw CLI.
    pub fn run_task(&self, task: &str) -> Result<ZeroclawRunResult> {
        if task.trim().is_empty() {
            bail!("zeroclaw task must not be empty");
        }

        let output = Command::new(&self.config.executable)
            .arg("run")
            .arg(task)
            .current_dir(&self.config.workspace)
            .output()
            .with_context(|| {
                format!(
                    "failed to execute '{} run ...'",
                    self.config.executable.display()
                )
            })?;

        let status_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(ZeroclawRunResult {
                status_code,
                stdout,
                stderr,
            })
        } else {
            bail!(
                "zeroclaw run failed (code {}): {}",
                status_code,
                stderr.trim()
            )
        }
    }

    /// Run multiple tasks sequentially via ZeroClaw.
    pub fn run_batch(&self, tasks: &[String]) -> Result<Vec<ZeroclawRunResult>> {
        let mut results = Vec::with_capacity(tasks.len());
        for task in tasks {
            results.push(self.run_task(task)?);
        }
        Ok(results)
    }

    /// Launch a background ZeroClaw daemon process.
    pub fn spawn_daemon(&self) -> Result<Child> {
        let child = Command::new(&self.config.executable)
            .arg("daemon")
            .arg("--host")
            .arg(&self.config.host)
            .arg("--port")
            .arg(self.config.port.to_string())
            .current_dir(&self.config.workspace)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn '{} daemon'",
                    self.config.executable.display()
                )
            })?;
        Ok(child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let cfg = ZeroclawConfig::default();
        assert_eq!(cfg.executable, PathBuf::from("zeroclaw"));
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 3000);
    }

    #[test]
    fn test_run_task_rejects_empty_task() {
        let client = ZeroclawClient::new(ZeroclawConfig::default());
        let err = client.run_task("   ").expect_err("empty task should error");
        assert!(err.to_string().contains("must not be empty"));
    }
}
