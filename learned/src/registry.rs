//! Phase 2B — minimal model registry (§17): per-collection directory of
//! versioned gating model files plus an `active` pointer. Deliberately NOT an
//! ML platform: save / activate / deactivate / inspect / load / remove, with
//! the hard rule that an ACTIVE model file cannot be removed (§17).
//!
//! Hot-swap (§18): activation is an atomic rename of the `active` pointer
//! file; readers who already loaded a card keep the old weights until they
//! reload, so a running query sees either the old or the new model — never a
//! partially-written file (writes also land via tmp+rename).

use crate::gating_v2::ModelCard;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    NotFound(String),
    ActiveModelProtected,
    NoActiveModel,
    Io(String),
    Model(crate::gating_v2::ModelError),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::NotFound(p) => write!(f, "model not found: {p}"),
            RegistryError::ActiveModelProtected => {
                write!(f, "cannot remove the ACTIVE model — deactivate first (§17)")
            }
            RegistryError::NoActiveModel => write!(f, "no active model"),
            RegistryError::Io(e) => write!(f, "registry io error: {e}"),
            RegistryError::Model(e) => write!(f, "model invalid: {e}"),
        }
    }
}

impl std::error::Error for RegistryError {}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActivePointer {
    pub version: u32,
    pub model_id: String,
}

pub struct ModelRegistry {
    gating_dir: PathBuf,
}

impl ModelRegistry {
    /// `root` is the registry directory (per collection: `<db>/models/<coll>`).
    pub fn new(root: &Path) -> Self {
        let gating_dir = root.join("gating");
        let _ = std::fs::create_dir_all(&gating_dir);
        ModelRegistry { gating_dir }
    }

    fn model_path(&self, version: u32) -> PathBuf {
        self.gating_dir.join(format!("model-v{version}.json"))
    }

    fn active_path(&self) -> PathBuf {
        self.gating_dir.join("active")
    }

    fn next_version(&self) -> u32 {
        let mut max = 0;
        if let Ok(entries) = std::fs::read_dir(&self.gating_dir) {
            for e in entries.flatten() {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if let Some(v) = name
                    .strip_prefix("model-v")
                    .and_then(|s| s.strip_suffix(".json"))
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    max = max.max(v);
                }
            }
        }
        max + 1
    }

    /// Persist a new version; returns its version number.
    pub fn save_new(&self, card: &ModelCard) -> Result<u32, RegistryError> {
        card.validate().map_err(RegistryError::Model)?;
        let v = self.next_version();
        let path = self.model_path(v);
        // tmp + rename: readers never see a partial file
        let tmp = self.gating_dir.join(format!(".tmp-model-v{v}"));
        card.save(&tmp).map_err(RegistryError::Model)?;
        std::fs::rename(&tmp, &path).map_err(|e| RegistryError::Io(e.to_string()))?;
        Ok(v)
    }

    /// Atomically point `active` at a version (§18).
    pub fn activate(&self, version: u32) -> Result<(), RegistryError> {
        let path = self.model_path(version);
        if !path.exists() {
            return Err(RegistryError::NotFound(path.display().to_string()));
        }
        // validate BEFORE activation — a broken model is never active
        ModelCard::load(&path).map_err(RegistryError::Model)?;
        let card = ModelCard::load(&path).map_err(RegistryError::Model)?;
        let ptr = ActivePointer {
            version,
            model_id: card.model_id,
        };
        let tmp = self.gating_dir.join(".tmp-active");
        std::fs::write(
            &tmp,
            serde_json::to_vec(&ptr).map_err(|e| RegistryError::Io(e.to_string()))?,
        )
        .map_err(|e| RegistryError::Io(e.to_string()))?;
        std::fs::rename(&tmp, self.active_path()).map_err(|e| RegistryError::Io(e.to_string()))
    }

    pub fn deactivate(&self) -> Result<(), RegistryError> {
        match std::fs::remove_file(self.active_path()) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(RegistryError::Io(e.to_string())),
        }
    }

    pub fn active_version(&self) -> Result<Option<u32>, RegistryError> {
        match std::fs::read(self.active_path()) {
            Ok(bytes) => {
                let ptr: ActivePointer =
                    serde_json::from_slice(&bytes).map_err(|e| RegistryError::Io(e.to_string()))?;
                Ok(Some(ptr.version))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(RegistryError::Io(e.to_string())),
        }
    }

    pub fn load_active(&self) -> Result<Option<ModelCard>, RegistryError> {
        match self.active_version()? {
            None => Ok(None),
            Some(v) => Ok(Some(self.load_version(v)?)),
        }
    }

    pub fn load_version(&self, version: u32) -> Result<ModelCard, RegistryError> {
        let path = self.model_path(version);
        ModelCard::load(&path).map_err(RegistryError::Model)
    }

    /// Inspect metadata without loading weights into a runtime.
    pub fn inspect(&self, version: u32) -> Result<ModelCard, RegistryError> {
        self.load_version(version)
    }

    pub fn list(&self) -> Vec<u32> {
        let mut vs: Vec<u32> = std::fs::read_dir(&self.gating_dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|e| {
                        let name = e.file_name();
                        let name = name.to_string_lossy();
                        name.strip_prefix("model-v")
                            .and_then(|s| s.strip_suffix(".json"))
                            .and_then(|s| s.parse::<u32>().ok())
                    })
                    .collect()
            })
            .unwrap_or_default();
        vs.sort_unstable();
        vs
    }

    /// Remove a version. An ACTIVE model is protected (§17).
    pub fn remove(&self, version: u32) -> Result<(), RegistryError> {
        if self.active_version()? == Some(version) {
            return Err(RegistryError::ActiveModelProtected);
        }
        match std::fs::remove_file(self.model_path(version)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(RegistryError::NotFound(
                self.model_path(version).display().to_string(),
            )),
            Err(e) => Err(RegistryError::Io(e.to_string())),
        }
    }
}
