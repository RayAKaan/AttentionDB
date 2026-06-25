use std::process::{Command, Stdio};
use std::time::Duration;

pub struct ServerManager;

impl ServerManager {
    pub fn kill_existing() -> anyhow::Result<()> {
        #[cfg(target_os = "windows")]
        {
            let output = Command::new("taskkill")
                .args(["/F", "/IM", "attentiondb-server.exe"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            match output {
                Ok(status) if status.success() => {
                    tracing::info!("Killed existing server process");
                }
                _ => {
                    tracing::debug!("No existing server to kill");
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let output = Command::new("pkill")
                .args(["-f", "attentiondb-server"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            match output {
                Ok(status) if status.success() => {
                    tracing::info!("Killed existing server process");
                }
                _ => {
                    tracing::debug!("No existing server to kill");
                }
            }
        }
        Ok(())
    }

    pub fn clean_data_dir(data_dir: &str) -> anyhow::Result<()> {
        let path = std::path::Path::new(data_dir);
        if !path.exists() {
            return Ok(());
        }

        let wal_path = path.join("engine.wal");
        if wal_path.exists() {
            std::fs::remove_file(&wal_path)?;
            tracing::info!("Deleted WAL: {}", wal_path.display());
        }

        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.extension().map_or(false, |ext| ext == "sst") {
                    std::fs::remove_file(&entry_path)?;
                    tracing::info!("Deleted SST: {}", entry_path.display());
                }
            }
        }
        Ok(())
    }

    fn find_server_exe() -> Option<String> {
        let candidates = if cfg!(target_os = "windows") {
            vec![
                "target/release/attentiondb-server.exe",
                "../target/release/attentiondb-server.exe",
            ]
        } else {
            vec![
                "target/release/attentiondb-server",
                "../target/release/attentiondb-server",
            ]
        };
        candidates
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .map(|s| s.to_string())
    }

    pub fn start_server(port: u16) -> anyhow::Result<()> {
        let server_exe = Self::find_server_exe()
            .ok_or_else(|| anyhow::anyhow!(
                "Cannot find attentiondb-server executable. Build it first with: cargo build --bin attentiondb-server --release -p attentiondb-api"
            ))?;

        Command::new(&server_exe)
            .env("ATTENTIONDB_REST_PORT", port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to start server at {}: {}", server_exe, e))?;

        tracing::info!("Server process started: {} (port {})", server_exe, port);
        Ok(())
    }

    pub async fn wait_for_health(port: u16, timeout: Duration) -> anyhow::Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?;
        let deadline = std::time::Instant::now() + timeout;

        while std::time::Instant::now() < deadline {
            match client
                .get(&format!("http://localhost:{}/health", port))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Server health check passed on port {}", port);
                    return Ok(());
                }
                _ => {
                    tracing::debug!("Waiting for server to become healthy on port {}...", port);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }

        anyhow::bail!(
            "Server did not become healthy on port {} within {:?}",
            port,
            timeout
        )
    }

    pub async fn clean_start(port: u16) -> anyhow::Result<()> {
        tracing::info!("=== Clean server start on port {} ===", port);

        Self::kill_existing()?;

        let data_dir =
            std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into());
        Self::clean_data_dir(&data_dir)?;

        // Brief pause to ensure OS releases file handles
        tokio::time::sleep(Duration::from_secs(2)).await;

        Self::start_server(port)?;
        Self::wait_for_health(port, Duration::from_secs(120)).await?;

        tracing::info!("=== Server ready on port {} ===", port);
        Ok(())
    }
}
