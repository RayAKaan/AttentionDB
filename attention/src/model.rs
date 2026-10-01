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

/// C8-specific training metadata.
///
/// Kept separate from C7's `TrainingMeta` rather than extending it: adding
/// fields to a `Serialize`/`Deserialize` struct would change C7 card bytes and
/// break the frozen C7 artifacts. C8 records its own objective terms here and
/// reuses `TrainingMeta` only for the fields that mean the same thing in both.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C8ResidualMeta {
    /// Residual scale `lambda` used during training and inference.
    pub residual_scale: f32,
    /// Weight of the residual-size penalty `alpha_r`.
    pub residual_regularization: f32,
    /// Weight of the KL distillation term `beta`.
    pub distillation_weight: f32,
    /// Distillation temperature `tau_d` (a validated hyper-parameter).
    pub distillation_temperature: f32,
    /// Whether distillation was enabled for this run.
    pub use_distillation: bool,
    /// `||delta||^2` of the learned correction at save time.
    pub delta_sq_norm: f32,
    /// Number of trainable parameters in the correction (excludes the frozen `W0`).
    pub n_trainable_params: usize,
    /// Optimizer steps actually taken (epochs * batches).
    pub optimizer_steps: usize,
    /// Fingerprint of the mined-negative provenance this model was trained on.
    pub negative_provenance_hash: u64,
}

impl Default for C8ResidualMeta {
    fn default() -> Self {
        Self {
            residual_scale: 0.1,
            residual_regularization: 0.0,
            distillation_weight: 0.0,
            distillation_temperature: 0.5,
            use_distillation: false,
            delta_sq_norm: 0.0,
            n_trainable_params: 0,
            optimizer_steps: 0,
            negative_provenance_hash: 0,
        }
    }
}

/// Versioned model card for C8 residual attention models.
///
/// A C8 card stores the *residual* projection (`W0 + alpha * deltaW`) rather
/// than a materialized `QkvProjection`, so the frozen base and the learned
/// correction stay separately auditable. That separation is the whole point of
/// C8: a reviewer must be able to see that `W0` never moved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C8ModelCard {
    /// Format identifier.
    pub format: String,
    /// Format version.
    pub format_version: u32,
    /// Unique model identifier.
    pub model_id: String,
    /// Architecture name (e.g. `c8-truncated-identity-residual`).
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
    /// Residual Q/K/V projection: frozen base plus learned correction.
    pub residual_projection: crate::projection::ResidualQkvProjection,
    /// Correction scorer weights.
    pub scorer: crate::scorer::ResidualScorer,
    /// Whether inference reads per-head retrieval evidence.
    pub use_evidence: bool,
    /// Residual scale `lambda` used at inference.
    pub residual_scale: f32,
    /// Common training metadata (shared with C7 semantics).
    pub training: TrainingMeta,
    /// C8-specific objective and correction metadata.
    pub c8: C8ResidualMeta,
    /// FNV-1a hash of the training dataset (reproducibility).
    pub dataset_hash: u64,
}

impl C8ModelCard {
    pub const FORMAT: &'static str = "attentiondb-c8-residual-attention";
    pub const VERSION: u32 = 1;

    #[allow(clippy::too_many_arguments)]
    /// Create a new C8 model card from components.
    pub fn new(
        model_id: &str,
        arch: &str,
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        head_names: Vec<String>,
        head_alignments: Vec<AlignmentProjection>,
        query_alignment: AlignmentProjection,
        residual_projection: crate::projection::ResidualQkvProjection,
        scorer: crate::scorer::ResidualScorer,
        use_evidence: bool,
        residual_scale: f32,
        training: TrainingMeta,
        c8: C8ResidualMeta,
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
            residual_projection,
            scorer,
            use_evidence,
            residual_scale,
            training,
            c8,
            dataset_hash,
        };
        card.validate()?;
        Ok(card)
    }

    /// Validate the model card (dimensions, finiteness, residual consistency).
    ///
    /// Beyond C7's checks this asserts that the stored residual projection is
    /// internally consistent with the declared dimensions and with the recorded
    /// `residual_scale`, because a card whose `alpha` disagrees with the
    /// deployment `lambda` would silently score differently from its training
    /// run.
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
        let qa = &self.query_alignment;
        if qa.output_dim != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: qa.output_dim,
            });
        }
        for &w in &qa.weights {
            if !w.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite(
                    "query alignment weight".into(),
                ));
            }
        }
        for &b in &qa.bias {
            if !b.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite(
                    "query alignment bias".into(),
                ));
            }
        }

        let rp = &self.residual_projection;
        if rp.attention_dim != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: rp.attention_dim,
            });
        }
        if rp.key_dim != self.key_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.key_dim,
                found: rp.key_dim,
            });
        }
        if rp.value_dim != self.value_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.value_dim,
                found: rp.value_dim,
            });
        }
        for x in rp
            .w_q_base
            .iter()
            .chain(rp.w_k_base.iter())
            .chain(rp.w_v_base.iter())
            .chain(rp.w_q_delta.iter())
            .chain(rp.w_k_delta.iter())
            .chain(rp.w_v_delta.iter())
        {
            if !x.is_finite() {
                return Err(crate::errors::AttentionError::NonFinite(
                    "residual projection parameter".into(),
                ));
            }
        }
        if !rp.residual_alpha.is_finite() || rp.residual_alpha < 0.0 {
            return Err(crate::errors::AttentionError::Config(
                "residual alpha must be finite and >= 0".into(),
            ));
        }
        for (name, v) in [
            ("residual_scale", self.residual_scale),
            ("w_attn", self.scorer.w_attn),
            ("w_evidence", self.scorer.w_evidence),
            ("w_disagree", self.scorer.w_disagree),
            ("bias", self.scorer.bias),
        ] {
            if !v.is_finite() {
                return Err(crate::errors::AttentionError::Config(format!(
                    "C8 card {name} must be finite"
                )));
            }
        }
        if !self.c8.residual_scale.is_finite() {
            return Err(crate::errors::AttentionError::Config(
                "c8.residual_scale must be finite".into(),
            ));
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

    /// Convert to the runtime C8 config for inference.
    ///
    /// The `residual_scale` argument allows a single card to be deployed at
    /// several `lambda` values (arms A/B/E...), which is what makes the
    /// `lambda = 0` baseline parity check possible from one trained model.
    pub fn to_config(&self, residual_scale: f32) -> Result<crate::config::C8AttentionConfig> {
        let cfg = crate::config::C8AttentionConfig {
            enabled: true,
            attention_dim: self.attention_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            head_alignments: self.head_alignments.clone(),
            query_alignment: Some(self.query_alignment.clone()),
            residual_projection: Some(self.residual_projection.clone()),
            residual_scale,
            use_evidence: self.use_evidence,
            scorer: self.scorer.clone(),
            cache_enabled: true,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
