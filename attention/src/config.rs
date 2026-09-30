use crate::alignment::AlignmentProjection;
use crate::errors::Result;
use crate::projection::QkvProjection;
use serde::{Deserialize, Serialize};

/// Configuration for the C7 attention subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Whether attention is enabled.
    pub enabled: bool,
    /// Alignment dimension d_a.
    pub attention_dim: usize,
    /// Key dimension d_k.
    pub key_dim: usize,
    /// Value dimension d_v.
    pub value_dim: usize,
    /// Per-head alignment projections (one per retrieval head).
    #[serde(default)]
    pub head_alignments: Vec<AlignmentProjection>,
    /// Query alignment projection.
    #[serde(default)]
    pub query_alignment: Option<AlignmentProjection>,
    /// Q/K/V projections (if None, will use fixed/identity).
    #[serde(default)]
    pub qkv_projection: Option<QkvProjection>,
    /// Whether to use retrieval evidence in the scorer (C7-F).
    pub use_evidence: bool,
    /// Scorer weights for combining attention output and evidence.
    #[serde(default)]
    pub scorer: crate::scorer::AttentionScorer,
}

/// FNV-1a fingerprint of a serialized config. Used by callers (e.g. the core
/// retrieval path) to cache a built subsystem per distinct config: identical
/// config <-> identical fingerprint <-> reusable subsystem.
pub fn config_fingerprint(cfg: &AttentionConfig) -> u64 {
    let json = match serde_json::to_string(cfg) {
        Ok(j) => j,
        Err(_) => "".to_string(),
    };
    let mut hash = 0xcbf29ce484222325u64;
    for b in json.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl AttentionConfig {
    /// Create a config with fixed/identity projections for C7-D diagnostic.
    pub fn fixed_identity(
        num_heads: usize,
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Self {
        let head_alignments = (0..num_heads)
            .map(|_| AlignmentProjection::identity(attention_dim))
            .collect();
        let query_alignment = Some(AlignmentProjection::identity(attention_dim));
        let qkv_projection = Some(QkvProjection::identity(attention_dim));
        Self {
            enabled: true,
            attention_dim,
            key_dim,
            value_dim,
            head_alignments,
            query_alignment,
            qkv_projection,
            use_evidence: false,
            scorer: crate::scorer::AttentionScorer::default(),
        }
    }

    /// Create a config with learned projections loaded from model card (C7-E/F).
    #[allow(clippy::too_many_arguments)]
    pub fn learned(
        _num_heads: usize,
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        head_alignments: Vec<AlignmentProjection>,
        query_alignment: AlignmentProjection,
        qkv_projection: QkvProjection,
        use_evidence: bool,
        scorer: crate::scorer::AttentionScorer,
    ) -> Self {
        Self {
            enabled: true,
            attention_dim,
            key_dim,
            value_dim,
            head_alignments,
            query_alignment: Some(query_alignment),
            qkv_projection: Some(qkv_projection),
            use_evidence,
            scorer,
        }
    }

    /// Create a disabled config (backward compatible).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            attention_dim: 0,
            key_dim: 0,
            value_dim: 0,
            head_alignments: Vec::new(),
            query_alignment: None,
            qkv_projection: None,
            use_evidence: false,
            scorer: crate::scorer::AttentionScorer::default(),
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.attention_dim == 0 || self.key_dim == 0 || self.value_dim == 0 {
            return Err(crate::errors::AttentionError::Config(
                "dimensions must be > 0 when enabled".into(),
            ));
        }
        if self.head_alignments.is_empty() {
            return Err(crate::errors::AttentionError::Config(
                "head_alignments cannot be empty when enabled".into(),
            ));
        }
        if self.query_alignment.is_none() {
            return Err(crate::errors::AttentionError::Config(
                "query_alignment required when enabled".into(),
            ));
        }
        if self.qkv_projection.is_none() {
            return Err(crate::errors::AttentionError::Config(
                "qkv_projection required when enabled".into(),
            ));
        }
        let qa = self.query_alignment.as_ref().unwrap();
        if qa.input_dim != self.attention_dim || qa.output_dim != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: qa.input_dim,
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
        let qkv = self.qkv_projection.as_ref().unwrap();
        if qkv.attention_dim != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: qkv.attention_dim,
            });
        }
        if qkv.key_dim != self.key_dim || qkv.value_dim != self.value_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.key_dim,
                found: qkv.key_dim,
            });
        }
        Ok(())
    }
}
