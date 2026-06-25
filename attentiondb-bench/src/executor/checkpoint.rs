use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub completed_experiments: HashSet<String>,
    pub partial_results: Vec<serde_json::Value>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub total_experiments: usize,
}

pub struct CheckpointManager {
    path: PathBuf,
}

impl CheckpointManager {
    pub fn new(output_dir: &Path) -> Self {
        std::fs::create_dir_all(output_dir).ok();
        Self {
            path: output_dir.join("checkpoint.json"),
        }
    }

    pub fn save(&self, checkpoint: &Checkpoint) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(checkpoint)?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }

    pub fn load(&self) -> anyhow::Result<Option<Checkpoint>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let json = std::fs::read_to_string(&self.path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&json)?;
        Ok(Some(checkpoint))
    }

    pub fn experiment_id(
        database: &str,
        dataset: &str,
        scale: usize,
        difficulty: &str,
    ) -> String {
        format!("{}/{}/{}@{}", database, dataset, scale, difficulty)
    }

    pub fn clear(&self) -> anyhow::Result<()> {
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}
