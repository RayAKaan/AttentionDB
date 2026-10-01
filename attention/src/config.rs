use crate::alignment::AlignmentProjection;
use crate::errors::Result;
use crate::projection::{QkvProjection, ResidualQkvProjection};
use crate::scorer::ResidualScorer;
use serde::{Deserialize, Serialize};

/// FNV-1a fingerprint helper shared by the C7 and C8 configs.
fn fnv1a(json: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for b in json.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

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
    fnv1a(&json)
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

/// Configuration for the C8 residual QKV attention subsystem.
///
/// C8 is defined by a single rule:
/// ```text
/// S_final = S_base + residual_scale * dS_attention
/// ```
/// where `S_base` is produced by the retrieval pipeline and is never touched by
/// this config. `residual_scale` (`lambda`) lives *here*, deliberately outside
/// the general retrieval fusion system, for two reasons:
///
/// 1. the correction is an isolated experimental component - putting `lambda`
///    into [`crate::core`-side fusion weights] would let it interact with the
///    attention/mhs/bm25 mix and confound the C8-B vs C8-E/H comparison;
/// 2. `lambda = 0` must reproduce the baseline bit-for-bit, which is only a
///    meaningful claim if the whole attention subsystem is switched off rather
///    than folded into a weight vector that still normalizes.
///
/// `lambda` is selected on validation data only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C8AttentionConfig {
    /// Whether the C8 residual correction is active.
    pub enabled: bool,
    /// Alignment dimension d_a.
    pub attention_dim: usize,
    /// Key dimension d_k (reduced: 32/64/128/384).
    pub key_dim: usize,
    /// Value dimension d_v (reduced: 32/64/128/384).
    pub value_dim: usize,
    /// Per-head alignment projections (one per retrieval head), mapping each
    /// head's native dimension to d_a.
    #[serde(default)]
    pub head_alignments: Vec<AlignmentProjection>,
    /// Query alignment projection (d_h -> d_a for the concatenated query).
    #[serde(default)]
    pub query_alignment: Option<AlignmentProjection>,
    /// Residual Q/K/V projection (`W = W0 + alpha*dW`, or `W = dW` when
    /// unrestricted). `None` means "use the deterministic truncated-identity
    /// base with a zero correction" (arm C).
    #[serde(default)]
    pub residual_projection: Option<ResidualQkvProjection>,
    /// Residual scale `lambda`. `0.0` reproduces the baseline exactly.
    pub residual_scale: f32,
    /// Whether the correction may read retrieval evidence (arm F).
    pub use_evidence: bool,
    /// Scorer weights for the correction.
    #[serde(default)]
    pub scorer: ResidualScorer,
    /// Whether the in-memory document-side K/V cache is used (arm I).
    pub cache_enabled: bool,
}

impl Default for C8AttentionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            attention_dim: 0,
            key_dim: 0,
            value_dim: 0,
            head_alignments: Vec::new(),
            query_alignment: None,
            residual_projection: None,
            residual_scale: 0.0,
            use_evidence: false,
            scorer: ResidualScorer::default(),
            cache_enabled: false,
        }
    }
}

impl C8AttentionConfig {
    /// A disabled config (backward compatible; correction is a no-op).
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Arm C: deterministic truncated-identity attention with `lambda = 0`.
    /// The correction is computed but multiplied out, so this is the identity
    /// control on the C8 machinery itself.
    pub fn truncated_identity_control(
        num_heads: usize,
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Self {
        let head_alignments = (0..num_heads)
            .map(|_| AlignmentProjection::identity(attention_dim))
            .collect();
        Self {
            enabled: true,
            attention_dim,
            key_dim,
            value_dim,
            head_alignments,
            query_alignment: Some(AlignmentProjection::identity(attention_dim)),
            residual_projection: None,
            residual_scale: 0.0,
            use_evidence: false,
            scorer: ResidualScorer::default(),
            cache_enabled: false,
        }
    }

    /// Arm E/F/G/H: a learned residual projection at scale `lambda`.
    #[allow(clippy::too_many_arguments)]
    pub fn residual(
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        head_alignments: Vec<AlignmentProjection>,
        query_alignment: AlignmentProjection,
        residual_projection: ResidualQkvProjection,
        residual_scale: f32,
        use_evidence: bool,
        scorer: ResidualScorer,
    ) -> Self {
        Self {
            enabled: true,
            attention_dim,
            key_dim,
            value_dim,
            head_alignments,
            query_alignment: Some(query_alignment),
            residual_projection: Some(residual_projection),
            residual_scale,
            use_evidence,
            scorer,
            cache_enabled: false,
        }
    }

    /// Arm I: enable the in-memory document-side K/V cache.
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Set the residual scale `lambda`. Lets one trained projection be swept
    /// across the lambda arms without rebuilding the projection itself.
    pub fn with_residual_scale(mut self, scale: f32) -> Self {
        self.residual_scale = scale;
        self
    }

    /// The residual projection actually in force, materializing the default
    /// deterministic base when none is stored.
    pub fn effective_projection(&self) -> Result<ResidualQkvProjection> {
        match &self.residual_projection {
            Some(p) => Ok(p.clone()),
            None => ResidualQkvProjection::truncated_identity_residual(
                self.attention_dim,
                self.key_dim,
                self.value_dim,
                0.0,
            ),
        }
    }

    /// FNV-1a fingerprint of the serialized config. Identical config implies
    /// identical fingerprint implies a reusable subsystem and a reusable cache.
    pub fn fingerprint(&self) -> u64 {
        let json = match serde_json::to_string(self) {
            Ok(j) => j,
            Err(_) => "".to_string(),
        };
        fnv1a(&json)
    }

    /// Validate geometry, scale, and scorer consistency.
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.attention_dim == 0 || self.key_dim == 0 || self.value_dim == 0 {
            return Err(crate::errors::AttentionError::Config(
                "C8 dimensions must be > 0 when enabled".into(),
            ));
        }
        if self.key_dim > self.attention_dim || self.value_dim > self.attention_dim {
            return Err(crate::errors::AttentionError::Config(
                "reduced key/value dims must not exceed attention_dim".into(),
            ));
        }
        if self.head_alignments.is_empty() {
            return Err(crate::errors::AttentionError::Config(
                "C8 head_alignments cannot be empty when enabled".into(),
            ));
        }
        let qa = self.query_alignment.as_ref().ok_or_else(|| {
            crate::errors::AttentionError::Config("C8 query_alignment required when enabled".into())
        })?;
        if qa.output_dim != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: qa.output_dim,
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
        if !self.residual_scale.is_finite() {
            return Err(crate::errors::AttentionError::Config(
                "C8 residual_scale must be finite".into(),
            ));
        }
        for (name, w) in [
            ("w_attn", self.scorer.w_attn),
            ("w_evidence", self.scorer.w_evidence),
            ("w_disagree", self.scorer.w_disagree),
            ("bias", self.scorer.bias),
        ] {
            if !w.is_finite() {
                return Err(crate::errors::AttentionError::Config(format!(
                    "C8 scorer.{name} must be finite"
                )));
            }
        }
        if let Some(p) = &self.residual_projection {
            if p.attention_dim != self.attention_dim
                || p.key_dim != self.key_dim
                || p.value_dim != self.value_dim
            {
                return Err(crate::errors::AttentionError::DimensionMismatch {
                    expected: self.key_dim,
                    found: p.key_dim,
                });
            }
        }
        Ok(())
    }
}
