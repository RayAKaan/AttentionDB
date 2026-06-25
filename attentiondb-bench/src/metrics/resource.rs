use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMeasurement {
    pub peak_memory_rss_mb: f64,
    pub disk_usage_gb: f64,
    pub index_build_time_s: f64,
    pub memory_baseline_mb: f64,
    pub memory_after_index_mb: f64,
    pub memory_after_warmup_mb: f64,
}

pub struct ResourceMonitor;

impl ResourceMonitor {
    #[cfg(target_os = "linux")]
    pub fn read_process_rss_mb(pid: u32) -> anyhow::Result<f64> {
        let content = std::fs::read_to_string(format!("/proc/{}/status", pid))?;
        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                let kb: f64 = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| anyhow::anyhow!("VmRSS parse error"))?
                    .parse()?;
                return Ok(kb / 1024.0);
            }
        }
        anyhow::bail!("VmRSS not found in /proc/{}/status", pid)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn read_process_rss_mb(_pid: u32) -> anyhow::Result<f64> {
        anyhow::bail!("RSS measurement only supported on Linux")
    }

    pub async fn read_container_rss_mb(container_name: &str) -> anyhow::Result<f64> {
        let output = tokio::process::Command::new("docker")
            .args(["stats", "--no-stream", "--format",
                   "{{.MemUsage}}", container_name])
            .output()
            .await?;

        let stdout = String::from_utf8(output.stdout)?;
        let usage_str = stdout.trim().split('/').next()
            .ok_or_else(|| anyhow::anyhow!("Docker stats parse error"))?
            .trim();

        if usage_str.ends_with("GiB") {
            let gb: f64 = usage_str.trim_end_matches("GiB").trim().parse()?;
            Ok(gb * 1024.0)
        } else if usage_str.ends_with("MiB") {
            let mb: f64 = usage_str.trim_end_matches("MiB").trim().parse()?;
            Ok(mb)
        } else if usage_str.ends_with("KiB") {
            let kb: f64 = usage_str.trim_end_matches("KiB").trim().parse()?;
            Ok(kb / 1024.0)
        } else {
            anyhow::bail!("Unknown memory unit: {}", usage_str)
        }
    }

    pub fn measure_disk_usage(path: &str) -> anyhow::Result<f64> {
        let output = std::process::Command::new("du")
            .args(["-sb", path])
            .output()?;
        let stdout = String::from_utf8(output.stdout)?;
        let bytes: f64 = stdout
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow::anyhow!("du parse error"))?
            .parse()?;
        Ok(bytes / (1024.0 * 1024.0 * 1024.0))
    }
}
