//! Phase 2B — trainable query-conditioned head gating (§2–7, §15–17, §24).
//!
//! Design contracts:
//! - **Deterministic**: seeded xorshift64* init and shuffling; identical
//!   (data, config, seed) ⇒ bit-identical model. No `thread_rng`, no threads
//!   in the training loop.
//! - **Dataset-cached** (§4): training reads a serializable JSON dataset of
//!   per-query per-head candidates/scores/quality; HNSW never runs during
//!   training.
//! - **Targets from retrieval ground truth** (§2): per-head Recall@K /
//!   NDCG@K / MRR, stored per example; target construction happens at training
//!   time from these (soft distribution by default).
//! - **Three objectives** (§6): quality regression, soft-target cross-entropy,
//!   pairwise logistic. Selection is by VALIDATION loss — never test.
//! - **Safe persistence** (§15–16): JSON model cards with format version and
//!   architecture metadata; loads validate dims, head count, and finiteness
//!   and return typed errors — never silent fallbacks.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64*, same family as the benchmark harness)
// ---------------------------------------------------------------------------

/// Deterministic RNG: xorshift64*. Reproducibility contract of this module.
pub struct DetRng(u64);

impl DetRng {
    pub fn new(seed: u64) -> Self {
        DetRng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Standard normal via Box–Muller.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    /// Seeded Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_u64() % (i + 1) as u64) as usize;
            xs.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset (§3, §4)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Split {
    Train,
    Val,
    Test,
}

/// Per-head observations for one query, produced ONCE by candidate generation
/// (§4) and reused for every training iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadExample {
    /// Candidate doc ids in head-rank order.
    pub candidates: Vec<u64>,
    /// Head's raw similarity scores (same order as candidates).
    pub raw_scores: Vec<f32>,
    /// Per-head MinMax-normalized scores (matches pipeline stage 3).
    pub norm_scores: Vec<f32>,
    /// Exact (non-approximate) similarity vs the query, same order (§32).
    pub exact_scores: Vec<f32>,
    /// Head quality vs ground truth: recall@K of this candidate list.
    pub recall_at_k: f32,
    pub ndcg_at_k: f32,
    pub mrr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExample {
    pub query_id: u64,
    pub query: Vec<f32>,
    /// Query group label — dataset metadata for controlled benchmarks and
    /// ORACLE baselines only (§36: never a model input feature).
    pub query_group: Option<u32>,
    pub split: Split,
    /// Ground-truth relevant ids, best first (ordered).
    pub ground_truth: Vec<u64>,
    pub heads: Vec<HeadExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingDataset {
    /// Discriminator + version for safe parsing (§16).
    pub format: String,
    pub version: u32,
    pub num_heads: usize,
    pub input_dim: usize,
    /// K used for the stored per-head quality metrics.
    pub top_k: usize,
    pub corpus_desc: String,
    pub seed: u64,
    pub queries: Vec<QueryExample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetError {
    BadFormat,
    BadVersion,
    DimMismatch { expected: usize, found: usize },
    HeadCountMismatch { expected: usize, found: usize },
    Corrupt(String),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::BadFormat => write!(f, "not a gating dataset (bad format tag)"),
            DatasetError::BadVersion => write!(f, "unsupported dataset version"),
            DatasetError::DimMismatch { expected, found } => {
                write!(f, "query dim mismatch: expected {expected}, found {found}")
            }
            DatasetError::HeadCountMismatch { expected, found } => {
                write!(f, "head count mismatch: expected {expected}, found {found}")
            }
            DatasetError::Corrupt(e) => write!(f, "corrupt dataset: {e}"),
        }
    }
}

impl GatingDataset {
    pub fn parse(json: &str) -> Result<Self, DatasetError> {
        let ds: GatingDataset =
            serde_json::from_str(json).map_err(|e| DatasetError::Corrupt(e.to_string()))?;
        ds.validate()?;
        Ok(ds)
    }

    pub fn validate(&self) -> Result<(), DatasetError> {
        if self.format != "attentiondb-gating-dataset" {
            return Err(DatasetError::BadFormat);
        }
        if self.version != 1 {
            return Err(DatasetError::BadVersion);
        }
        for q in &self.queries {
            if q.query.len() != self.input_dim {
                return Err(DatasetError::DimMismatch {
                    expected: self.input_dim,
                    found: q.query.len(),
                });
            }
            if q.heads.len() != self.num_heads {
                return Err(DatasetError::HeadCountMismatch {
                    expected: self.num_heads,
                    found: q.heads.len(),
                });
            }
            for h in &q.heads {
                if h.candidates.len() != h.norm_scores.len()
                    || h.raw_scores.len() != h.candidates.len()
                    || h.exact_scores.len() != h.candidates.len()
                {
                    return Err(DatasetError::Corrupt(
                        "head arrays must have equal lengths".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn split_counts(&self) -> (usize, usize, usize) {
        let mut c = (0usize, 0usize, 0usize);
        for q in &self.queries {
            match q.split {
                Split::Train => c.0 += 1,
                Split::Val => c.1 += 1,
                Split::Test => c.2 += 1,
            }
        }
        c
    }

    /// FNV-1a hash of the canonical JSON — dataset identity for run metadata.
    pub fn content_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Model (§5, §15, §16)
// ---------------------------------------------------------------------------

/// Seeded MLP: input → ReLU(hidden) → head logits → softmax weights.
#[derive(Debug, Clone)]
pub struct GatingMlp {
    pub input_dim: usize,
    pub hidden: usize,
    pub num_heads: usize,
    /// hidden × input, row-major
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// heads × hidden, row-major
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl GatingMlp {
    /// Seeded init: N(0, 1/√fan_in). Deterministic for a given seed.
    pub fn new(input_dim: usize, hidden: usize, num_heads: usize, seed: u64) -> Self {
        let mut rng = DetRng::new(seed);
        let s1 = 1.0 / (input_dim as f32).sqrt();
        let s2 = 1.0 / (hidden as f32).sqrt();
        let mut w1 = vec![0.0f32; hidden * input_dim];
        for w in w1.iter_mut() {
            *w = rng.gauss() * s1;
        }
        let mut w2 = vec![0.0f32; num_heads * hidden];
        for w in w2.iter_mut() {
            *w = rng.gauss() * s2;
        }
        // zero biases: symmetry broken by weights alone
        GatingMlp {
            input_dim,
            hidden,
            num_heads,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: vec![0.0; num_heads],
        }
    }

    /// Logits = W2·relu(W1·x + b1) + b2.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0f32; self.hidden];
        for (i, hi) in h.iter_mut().enumerate() {
            let mut s = self.b1[i];
            for (j, &xj) in x.iter().enumerate() {
                s += self.w1[i * self.input_dim + j] * xj;
            }
            *hi = if s > 0.0 { s } else { 0.0 };
        }
        let mut out = vec![0.0f32; self.num_heads];
        for (k, ok) in out.iter_mut().enumerate() {
            let mut s = self.b2[k];
            for (i, &hi) in h.iter().enumerate() {
                s += self.w2[k * self.hidden + i] * hi;
            }
            *ok = s;
        }
        out
    }

    /// Stable softmax over logits.
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Head weights for a query (inference path; must be cheap).
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        Self::softmax(&self.logits(x))
    }
}

/// Persisted model with metadata (§15). JSON, self-describing, no executable
/// content (§16): weights are data, validated on load.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCard {
    pub format: String,
    pub format_version: u32,
    pub model_id: String,
    pub arch: String,
    pub input_dim: usize,
    pub hidden: usize,
    pub num_heads: usize,
    pub activation: String,
    pub loss: String,
    /// Head names in model-index order (compat check + name→weight mapping).
    /// Optional for backward compatibility with earlier cards.
    #[serde(default)]
    pub head_names: Option<Vec<String>>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub training: TrainingMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingMeta {
    pub seed: u64,
    /// FNV-1a content hash of the training dataset (reproducibility, §24).
    pub dataset_hash: u64,
    pub objective: String,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs_run: usize,
    pub best_val_loss: f32,
    pub l2: f32,
    pub timestamp_unix: u64,
    pub code_commit: String,
    pub hardware: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    BadFormat,
    BadVersion,
    ArchUnsupported,
    DimMismatch { expected: usize, found: usize },
    HeadCountMismatch { expected: usize, found: usize },
    NonFiniteWeight,
    Corrupt(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::BadFormat => write!(f, "not a gating model (bad format tag)"),
            ModelError::BadVersion => write!(f, "unsupported model format version"),
            ModelError::ArchUnsupported => write!(f, "unsupported model architecture"),
            ModelError::DimMismatch { expected, found } => {
                write!(f, "input dim mismatch: expected {expected}, found {found}")
            }
            ModelError::HeadCountMismatch { expected, found } => {
                write!(f, "head count mismatch: expected {expected}, found {found}")
            }
            ModelError::NonFiniteWeight => write!(f, "model contains a non-finite weight"),
            ModelError::Corrupt(e) => write!(f, "corrupt model file: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl ModelCard {
    pub const FORMAT: &'static str = "attentiondb-gating-model";
    pub const VERSION: u32 = 1;

    pub fn from_mlp(m: &GatingMlp, meta: TrainingMeta, model_id: &str, loss: &str) -> Self {
        ModelCard {
            format: Self::FORMAT.to_string(),
            format_version: Self::VERSION,
            model_id: model_id.to_string(),
            arch: "mlp_relu_softmax".to_string(),
            input_dim: m.input_dim,
            hidden: m.hidden,
            num_heads: m.num_heads,
            activation: "relu".to_string(),
            loss: loss.to_string(),
            head_names: None,
            w1: m.w1.clone(),
            b1: m.b1.clone(),
            w2: m.w2.clone(),
            b2: m.b2.clone(),
            training: meta,
        }
    }

    pub fn to_mlp(&self) -> GatingMlp {
        GatingMlp {
            input_dim: self.input_dim,
            hidden: self.hidden,
            num_heads: self.num_heads,
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
        }
    }

    /// Full §16 validation — dims, sizes, finiteness. Typed errors, no
    /// silent fallback.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.format != Self::FORMAT {
            return Err(ModelError::BadFormat);
        }
        if self.format_version != Self::VERSION {
            return Err(ModelError::BadVersion);
        }
        if self.arch != "mlp_relu_softmax" {
            return Err(ModelError::ArchUnsupported);
        }
        let check = |xs: &[f32]| {
            if xs.iter().any(|v| !v.is_finite()) {
                return Err(ModelError::NonFiniteWeight);
            }
            Ok(())
        };
        if self.w1.len() != self.hidden * self.input_dim {
            return Err(ModelError::DimMismatch {
                expected: self.hidden * self.input_dim,
                found: self.w1.len(),
            });
        }
        check(&self.w1)?;
        if self.b1.len() != self.hidden {
            return Err(ModelError::DimMismatch {
                expected: self.hidden,
                found: self.b1.len(),
            });
        }
        check(&self.b1)?;
        if self.w2.len() != self.num_heads * self.hidden {
            return Err(ModelError::HeadCountMismatch {
                expected: self.num_heads * self.hidden,
                found: self.w2.len(),
            });
        }
        check(&self.w2)?;
        if self.b2.len() != self.num_heads {
            return Err(ModelError::HeadCountMismatch {
                expected: self.num_heads,
                found: self.b2.len(),
            });
        }
        check(&self.b2)?;
        Ok(())
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), ModelError> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| ModelError::Corrupt(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| ModelError::Corrupt(e.to_string()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, ModelError> {
        let txt = std::fs::read_to_string(path).map_err(|e| ModelError::Corrupt(e.to_string()))?;
        let card: ModelCard =
            serde_json::from_str(&txt).map_err(|e| ModelError::Corrupt(e.to_string()))?;
        card.validate()?;
        Ok(card)
    }
}

// ---------------------------------------------------------------------------
// Objectives (§6)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// A: MSE(logits, per-head quality).
    QualityRegression,
    /// B: cross-entropy against softmax(quality/τ) — the default.
    SoftTarget,
    /// C: pairwise logistic on logit gaps for quality gaps ≥ margin.
    Pairwise,
}

impl Objective {
    pub fn name(&self) -> &'static str {
        match self {
            Objective::QualityRegression => "quality_regression",
            Objective::SoftTarget => "soft_target",
            Objective::Pairwise => "pairwise",
        }
    }

    fn dlogits_and_loss(
        &self,
        logits: &[f32],
        quality: &[f32],
        tau: f32,
        margin: f32,
    ) -> (Vec<f32>, f32) {
        match self {
            Objective::QualityRegression => {
                let n = logits.len() as f32;
                let mut g = vec![0.0f32; logits.len()];
                let mut loss = 0.0f32;
                for i in 0..logits.len() {
                    let d = logits[i] - quality[i];
                    loss += d * d;
                    g[i] = 2.0 * d / n;
                }
                (g, loss / n)
            }
            Objective::SoftTarget => {
                // t = softmax(quality/τ); CE(t, softmax(logits))
                let scaled: Vec<f32> = quality.iter().map(|&q| q / tau).collect();
                let t = GatingMlp::softmax(&scaled);
                let p = GatingMlp::softmax(logits);
                let loss = -t
                    .iter()
                    .zip(p.iter())
                    .map(|(&ti, &pi)| ti * pi.ln())
                    .sum::<f32>();
                let g: Vec<f32> = p.iter().zip(t.iter()).map(|(&pi, &ti)| pi - ti).collect();
                (g, loss)
            }
            Objective::Pairwise => {
                let n = logits.len();
                let mut g = vec![0.0f32; n];
                let mut loss = 0.0f32;
                for i in 0..n {
                    for j in 0..n {
                        if i == j || quality[i] - quality[j] < margin {
                            continue;
                        }
                        let z = logits[i] - logits[j];
                        // dL/dz = -sigmoid(-z); L = softplus(-z)
                        let s = 1.0 / (1.0 + z.exp());
                        g[i] += -s;
                        g[j] += s;
                        loss += (-z).exp().ln_1p();
                    }
                }
                (g, loss / (n * (n - 1)) as f32)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer (§5, §6, §24, §25, §26)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub objective: Objective,
    pub tau: f32,
    pub margin: f32,
    pub lr: f32,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub min_delta: f32,
    pub l2: f32,
    pub hidden: usize,
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            objective: Objective::SoftTarget,
            tau: 0.1,
            margin: 0.05,
            lr: 0.01,
            batch_size: 32,
            max_epochs: 200,
            patience: 15,
            min_delta: 1e-4,
            l2: 1e-4,
            hidden: 32,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EpochPoint {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    /// Pearson corr(predicted weight, head quality) on validation.
    pub val_weight_quality_corr: f32,
}

#[derive(Debug, Clone)]
pub struct TrainOutcome {
    pub model: GatingMlp,
    pub curves: Vec<EpochPoint>,
    pub best_epoch: usize,
    pub stopped_early: bool,
    pub final_train_loss: f32,
    pub best_val_loss: f32,
}

/// Adam optimizer: per-tensor state (deterministic, no threads).
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

struct Adam {
    lr: f32,
    l2: f32,
    t: u64,
    tensors: Vec<AdamState>,
}

impl Adam {
    fn new(sizes: &[usize], lr: f32, l2: f32) -> Self {
        Adam {
            lr,
            l2,
            t: 0,
            tensors: sizes
                .iter()
                .map(|&n| AdamState {
                    m: vec![0.0; n],
                    v: vec![0.0; n],
                })
                .collect(),
        }
    }

    /// One optimizer step over all tensors. Weight decay (L2) applies to
    /// tensors flagged as weight matrices (not biases).
    fn step(&mut self, model: &mut GatingMlp, grads: &[Vec<f32>], decay: &[bool]) {
        self.t += 1;
        let c1 = 1.0 - 0.9f32.powi(self.t as i32);
        let c2 = 1.0 - 0.999f32.powi(self.t as i32);
        let all: Vec<&mut [f32]> = vec![
            &mut model.w1[..],
            &mut model.b1[..],
            &mut model.w2[..],
            &mut model.b2[..],
        ];
        for (ti, tensor) in all.into_iter().enumerate() {
            let st = &mut self.tensors[ti];
            for i in 0..tensor.len() {
                let mut g = grads[ti][i];
                if decay[ti] {
                    g += self.l2 * tensor[i];
                }
                st.m[i] = 0.9 * st.m[i] + 0.1 * g;
                st.v[i] = 0.999 * st.v[i] + 0.001 * g * g;
                let mhat = st.m[i] / c1;
                let vhat = st.v[i] / c2;
                tensor[i] -= self.lr * mhat / (vhat.sqrt() + 1e-8);
            }
        }
    }
}

/// Forward + backward for one example: returns loss, accumulates gradients.
fn forward_backward(
    m: &GatingMlp,
    x: &[f32],
    quality: &[f32],
    cfg: &TrainingConfig,
    grads: &mut [Vec<f32>],
) -> f32 {
    let mut h_pre = vec![0.0f32; m.hidden];
    for (i, hp) in h_pre.iter_mut().enumerate() {
        let mut s = m.b1[i];
        for (j, &xj) in x.iter().enumerate() {
            s += m.w1[i * m.input_dim + j] * xj;
        }
        *hp = s;
    }
    let h: Vec<f32> = h_pre
        .iter()
        .map(|&s| if s > 0.0 { s } else { 0.0 })
        .collect();
    let mut logits = vec![0.0f32; m.num_heads];
    for (k, lk) in logits.iter_mut().enumerate() {
        let mut s = m.b2[k];
        for (i, &hi) in h.iter().enumerate() {
            s += m.w2[k * m.hidden + i] * hi;
        }
        *lk = s;
    }
    let (dlogits, loss) = cfg
        .objective
        .dlogits_and_loss(&logits, quality, cfg.tau, cfg.margin);
    let mut dh = vec![0.0f32; m.hidden];
    for k in 0..m.num_heads {
        let dl = dlogits[k];
        grads[3][k] += dl;
        for i in 0..m.hidden {
            grads[2][k * m.hidden + i] += dl * h[i];
            dh[i] += dl * m.w2[k * m.hidden + i];
        }
    }
    for i in 0..m.hidden {
        if h_pre[i] <= 0.0 {
            continue;
        }
        grads[1][i] += dh[i];
        for j in 0..m.input_dim {
            grads[0][i * m.input_dim + j] += dh[i] * x[j];
        }
    }
    loss
}

fn zero_grads(m: &GatingMlp) -> Vec<Vec<f32>> {
    vec![
        vec![0.0; m.w1.len()],
        vec![0.0; m.b1.len()],
        vec![0.0; m.w2.len()],
        vec![0.0; m.b2.len()],
    ]
}

/// Which stored metric defines head quality (§2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTarget {
    Recall,
    Ndcg,
    Mrr,
}

fn quality_vector(q: &QueryExample, target: QualityTarget) -> Vec<f32> {
    q.heads
        .iter()
        .map(|h| match target {
            QualityTarget::Recall => h.recall_at_k,
            QualityTarget::Ndcg => h.ndcg_at_k,
            QualityTarget::Mrr => h.mrr,
        })
        .collect()
}

/// Train with early stopping on VALIDATION loss (§26: never on test).
pub fn train_gating(
    ds: &GatingDataset,
    cfg: &TrainingConfig,
    target: QualityTarget,
) -> TrainOutcome {
    let mut model = GatingMlp::new(ds.input_dim, cfg.hidden, ds.num_heads, cfg.seed);
    let mut rng = DetRng::new(cfg.seed ^ 0x7D2B);
    let mut opt = Adam::new(
        &[
            model.w1.len(),
            model.b1.len(),
            model.w2.len(),
            model.b2.len(),
        ],
        cfg.lr,
        cfg.l2,
    );

    let train: Vec<usize> = (0..ds.queries.len())
        .filter(|&i| ds.queries[i].split == Split::Train)
        .collect();
    let val: Vec<usize> = (0..ds.queries.len())
        .filter(|&i| ds.queries[i].split == Split::Val)
        .collect();

    let mut curves = Vec::new();
    let mut best_val = f32::INFINITY;
    let mut best_snapshot = model.clone();
    let mut best_epoch = 0usize;
    let mut stall = 0usize;
    let mut stopped_early = false;
    let mut final_train_loss = 0.0f32;

    for epoch in 0..cfg.max_epochs {
        // ---- train epoch: seeded shuffle + minibatch Adam steps ----
        let mut order = train.clone();
        rng.shuffle(&mut order);
        let mut grads = zero_grads(&model);
        let mut batch_loss = 0.0f32;
        let mut in_batch = 0usize;
        for &qi in &order {
            let q = &ds.queries[qi];
            let y = quality_vector(q, target);
            let l = forward_backward(&model, &q.query, &y, cfg, &mut grads);
            batch_loss += l;
            in_batch += 1;
            if in_batch == cfg.batch_size {
                let scaled: Vec<Vec<f32>> = grads
                    .iter()
                    .map(|g| g.iter().map(|&x| x / in_batch as f32).collect())
                    .collect();
                for (g, sc) in grads.iter_mut().zip(scaled) {
                    *g = sc;
                }
                opt.step(&mut model, &grads, &[true, false, true, false]);
                grads = zero_grads(&model);
                batch_loss = 0.0;
                in_batch = 0;
            }
        }
        if in_batch > 0 {
            let scaled: Vec<Vec<f32>> = grads
                .iter()
                .map(|g| g.iter().map(|&x| x / in_batch as f32).collect())
                .collect();
            for (g, sc) in grads.iter_mut().zip(scaled) {
                *g = sc;
            }
            opt.step(&mut model, &grads, &[true, false, true, false]);
        }
        final_train_loss = batch_loss / in_batch.max(1) as f32;

        // ---- validation loss (§26: early stopping NEVER sees test) ----
        let mut val_loss = 0.0f32;
        for &vi in &val {
            let q = &ds.queries[vi];
            let y = quality_vector(q, target);
            let logits = model.logits(&q.query);
            let (_, l) = cfg
                .objective
                .dlogits_and_loss(&logits, &y, cfg.tau, cfg.margin);
            val_loss += l;
        }
        let val_loss = if val.is_empty() {
            0.0
        } else {
            val_loss / val.len() as f32
        };
        let corr = weight_quality_correlation(ds, &model, Split::Val, target);

        curves.push(EpochPoint {
            epoch,
            train_loss: final_train_loss,
            val_loss,
            val_weight_quality_corr: corr,
        });

        if best_val - val_loss > cfg.min_delta {
            best_val = val_loss;
            best_snapshot = model.clone();
            best_epoch = epoch;
            stall = 0;
        } else {
            stall += 1;
            if stall >= cfg.patience {
                stopped_early = true;
                break;
            }
        }
    }

    TrainOutcome {
        model: best_snapshot,
        curves,
        best_epoch,
        stopped_early,
        final_train_loss,
        best_val_loss: if best_val.is_finite() { best_val } else { 0.0 },
    }
}

// ---------------------------------------------------------------------------
// Diagnostics + evaluation (§7, §19, §28)
// ---------------------------------------------------------------------------

/// Pearson correlation between predicted weight and actual head quality,
/// pooled over all queries of a split. NaN if no variance (reported as NaN —
/// callers must surface it, not hide it).
pub fn weight_quality_correlation(
    ds: &GatingDataset,
    m: &GatingMlp,
    split: Split,
    target: QualityTarget,
) -> f32 {
    let mut ws = Vec::new();
    let mut qs = Vec::new();
    for q in &ds.queries {
        if q.split != split {
            continue;
        }
        let w = m.predict(&q.query);
        for (h, wi) in q.heads.iter().zip(w) {
            ws.push(wi as f64);
            qs.push(match target {
                QualityTarget::Recall => h.recall_at_k as f64,
                QualityTarget::Ndcg => h.ndcg_at_k as f64,
                QualityTarget::Mrr => h.mrr as f64,
            });
        }
    }
    pearson(&ws, &qs) as f32
}

pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return f64::NAN;
    }
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1e-30 || vb <= 1e-30 {
        f64::NAN
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

/// Mean normalized entropy of the weight distribution (1 = uniform, 0 = one-hot).
pub fn mean_normalized_entropy(weights_per_query: &[Vec<f32>]) -> f64 {
    if weights_per_query.is_empty() {
        return f64::NAN;
    }
    let mut total = 0.0f64;
    for w in weights_per_query {
        let h = w.len() as f64;
        if h <= 1.0 {
            continue;
        }
        let ent: f64 = -w
            .iter()
            .map(|&p| {
                if p > 0.0 {
                    (p as f64) * (p as f64).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        total += ent / h.ln();
    }
    total / weights_per_query.len() as f64
}

/// Aggregate diagnostics over a set of weight vectors (§7).
#[derive(Debug, Clone, Serialize)]
pub struct GatingDiagnostics {
    pub avg_weights: Vec<f64>,
    pub mean_normalized_entropy: f64,
    /// fraction of queries where each head has the max weight
    pub selection_frequency: Vec<f64>,
    pub weight_quality_pearson: f64,
}

pub fn diagnostics(
    weights_per_query: &[Vec<f32>],
    qualities_per_query: &[Vec<f32>],
) -> GatingDiagnostics {
    let n_heads = weights_per_query.first().map(|w| w.len()).unwrap_or(0);
    let n = weights_per_query.len() as f64;
    let mut avg = vec![0.0; n_heads];
    let mut sel = vec![0.0; n_heads];
    for w in weights_per_query {
        let mut best = 0usize;
        for (i, &wi) in w.iter().enumerate() {
            avg[i] += wi as f64;
            if wi > w[best] {
                best = i;
            }
        }
        sel[best] += 1.0;
    }
    for a in avg.iter_mut() {
        *a /= n;
    }
    for s in sel.iter_mut() {
        *s /= n;
    }
    let flat_w: Vec<f64> = weights_per_query
        .iter()
        .flat_map(|w| w.iter().map(|&x| x as f64))
        .collect();
    let flat_q: Vec<f64> = qualities_per_query
        .iter()
        .flat_map(|q| q.iter().map(|&x| x as f64))
        .collect();
    GatingDiagnostics {
        avg_weights: avg,
        mean_normalized_entropy: mean_normalized_entropy(weights_per_query),
        selection_frequency: sel,
        weight_quality_pearson: pearson(&flat_w, &flat_q),
    }
}
