use crate::alignment::AlignmentProjection;
use crate::errors::Result;
use crate::projection::QkvProjection;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Versioned model card for C7 learned attention models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C7ModelCard {
    /// Format identifier.
    pub format: String,
    /// Format version.
    pub format_version: u32,
    /// Unique model identifier.
    pub model_id: String,
    /// Architecture name.
    pub arch: String,
    /// Attention dimension d_a.
    pub attention_dim: usize,
    /// Key dimension d_k.
    pub key_dim: usize,
    /// Value dimension d_v.
    pub value_dim: usize,
    /// Retrieval head names in order (must match alignment order).
    pub head_names: Vec<String>,
    /// Per-head alignment projections P_h.
    pub head_alignments: Vec<AlignmentProjection>,
    /// Query alignment projection P_q.
    pub query_alignment: AlignmentProjection,
    /// Q/K/V projections.
    pub qkv_projection: QkvProjection,
    /// Scorer weights.
    pub scorer: crate::scorer::AttentionScorer,
    /// Training metadata.
    pub training: TrainingMeta,
    /// FNV-1a hash of the training dataset (reproducibility).
    pub dataset_hash: u64,
}

impl C7ModelCard {
    pub const FORMAT: &'static str = "attentiondb-c7-attention";
    pub const VERSION: u32 = 1;

    #[allow(clippy::too_many_arguments)]
    /// Create a new model card from components.
    pub fn new(
        model_id: &str,
        arch: &str,
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        head_names: Vec<String>,
        head_alignments: Vec<crate::alignment::AlignmentProjection>,
        query_alignment: crate::alignment::AlignmentProjection,
        qkv_projection: QkvProjection,
        scorer: crate::scorer::AttentionScorer,
        training: TrainingMeta,
        dataset_hash: u64,
    ) -> Result<Self> {
        let card = Self {
            format: Self::FORMAT.to_string(),
            format_version: Self::VERSION,
            model_id: model_id.to_string(),
            arch: arch.to_string(),
            attention_dim,
            key_dim,
            value_dim,
            head_names,
            head_alignments,
            query_alignment,
            qkv_projection,
            scorer,
            training,
            dataset_hash,
        };
        card.validate()?;
        Ok(card)
    }

    /// Validate the model card (dimensions, finiteness, consistency).
    pub fn validate(&self) -> Result<()> {
        if self.format != Self::FORMAT {
            return Err(crate::errors::AttentionError::Config("bad format".into()));
        }
        if self.format_version != Self::VERSION {
            return Err(crate::errors::AttentionError::Config(
                "unsupported version".into(),
            ));
        }
        if self.attention_dim == 0 || self.key_dim == 0 || self.value_dim == 0 {
            return Err(crate::errors::AttentionError::Config(
                "dimensions must be > 0".into(),
            ));
        }
        if self.head_names.len() != self.head_alignments.len() {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.head_names.len(),
                found: self.head_alignments.len(),
            });
        }
        for pa in &self.head_alignments {
            if pa.output_dim != self.attention_dim {
                return Err(crate::errors::AttentionError::DimensionMismatch {
                    expected: self.attention_dim,
                    found: pa.output_dim,
                });
            }
        }
        let qa = &self.query_alignment;
        if qa.output_dim != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: qa.output_dim,
            });
        }
        let qkv = &self.qkv_projection;
        if qkv.attention_dim != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: qkv.attention_dim,
            });
        }
        // Check finiteness
        for pa in &self.head_alignments {
            for &w in &pa.weights {
                if !w.is_finite() {
                    return Err(crate::errors::AttentionError::NonFinite(
                        "head alignment weight".into(),
                    ));
                }
            }
            for &b in &pa.bias {
                if !b.is_finite() {
                    return Err(crate::errors::AttentionError::NonFinite(
                        "head alignment bias".into(),
                    ));
                }
            }
        }
        for &w in &self.query_alignment.weights {
            if !w.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite(
                    "query alignment weight".into(),
                ));
            }
        }
        for &b in &self.query_alignment.bias {
            if !b.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite(
                    "query alignment bias".into(),
                ));
            }
        }
        for &w in &self.qkv_projection.w_q {
            if !w.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite("w_q".into()));
            }
        }
        for &w in &self.qkv_projection.w_k {
            if !w.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite("w_k".into()));
            }
        }
        for &w in &self.qkv_projection.w_v {
            if !w.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite("w_v".into()));
            }
        }
        Ok(())
    }

    /// Save to JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let txt = std::fs::read_to_string(path)?;
        let card: Self = serde_json::from_str(&txt)?;
        card.validate()?;
        Ok(card)
    }

    /// Convert to AttentionConfig for runtime use.
    pub fn to_config(&self, use_evidence: bool) -> crate::config::AttentionConfig {
        crate::config::AttentionConfig::learned(
            self.head_alignments.len(),
            self.attention_dim,
            self.key_dim,
            self.value_dim,
            self.head_alignments.clone(),
            self.query_alignment.clone(),
            self.qkv_projection.clone(),
            use_evidence,
            self.scorer.clone(),
        )
    }
}

/// Training metadata for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMeta {
    pub seed: u64,
    pub dataset_hash: u64,
    pub objective: String,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs_run: usize,
    pub best_val_loss: Option<f32>,
    pub l2: f32,
    pub timestamp_unix: u64,
    pub code_commit: String,
    pub hardware: String,
}

impl Default for TrainingMeta {
    fn default() -> Self {
        Self {
            seed: 20260925,
            dataset_hash: 0,
            objective: "contrastive".into(),
            learning_rate: 0.01,
            batch_size: 32,
            epochs_run: 0,
            best_val_loss: None,
            l2: 1e-4,
            timestamp_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            code_commit: String::new(),
            hardware: String::new(),
        }
    }
}
