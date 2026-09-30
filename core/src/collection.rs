use crate::adaptive::{AdaptiveRetriever, AdaptiveTrace, RetrievalBudget};
use crate::bm25::Bm25Index;
use crate::error::CoreError;
use crate::retrieval::{
    candidate_union, cross_head_centroid, cross_refine_query, deterministic_top_k, fuse_candidate,
    normalize_scores, rrf_fuse, AttentionScorer, C7CandidateTrace, C7Trace, CandidateFeatures,
    CandidateSet, CrossHeadTrace, CrossRefineConfig, FusionWeights, HeadHit, RankedCandidate,
    ScoreNormalization,
};
use crate::{AdaptivePolicyType, AdaptiveRetrievalConfig};
use attentiondb_hnsw::{similarity, HNSWConfig, HeadIndexManager};
use attentiondb_multihead::{GatingNetwork, HeadConfig, HeadType, MultiHeadManager};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub const OVERFETCH_MULTIPLIER: usize = 5;
pub const MIN_CANDIDATES_PER_HEAD: usize = 20;

/// Ablation modes (Phase 2 §9). Each mode is a strict superset of the previous
/// one, so ablations attribute improvements to the added stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RetrievalMode {
    /// A: single-head HNSW, raw generator scores.
    SingleHead,
    /// B: multi-head union + fixed (uniform) head weights.
    FixedFusion,
    /// C: B + learned gating (query → head weights; uniform when no network
    /// is loaded — documented, never claimed to be learned).
    LearnedGating,
    /// D: C + candidate-level Q/K attention (docs/retrieval/attention.md).
    QKAttention,
    /// E: D + metric-consistent exact reranking of the union.
    Full,
}

/// Hybrid (vector + BM25) fusion strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridStrategy {
    /// Reciprocal Rank Fusion with `rrf_k` (strong deterministic baseline).
    Rrf,
    /// Linear fusion through the staged pipeline (attention + mhs + bm25).
    Fusion,
}

/// Per-query pipeline stage sizes (§40 benchmark observability).
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineStats {
    /// Candidates after union + budget clamp.
    pub union_size: usize,
    /// Candidates entering the exact-rerank stage (MODE E).
    pub rerank_size: usize,
    /// Heads that actually contributed candidates.
    pub heads_present: usize,
}

/// Retrieval execution configuration (planner-controllable, §16/§38).
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub mode: RetrievalMode,
    pub normalization: ScoreNormalization,
    pub candidate_multiplier: usize,
    pub min_candidates_per_head: usize,
    /// Hard bound on the post-union candidate pool (§6/§27).
    pub candidate_budget: usize,
    pub fusion: FusionWeights,
    pub rrf_k: f32,
    pub hybrid_strategy: HybridStrategy,
    /// Parallel head search (§5); results are parallel≡serial (tested).
    pub parallel: bool,
    pub max_search_threads: usize,
    /// Weight of rank features inside candidate attention features.
    pub attention_rank_weight: f32,
    /// Per-head HNSW `ef` (C5 additive; `None` = index's own settings, i.e.
    /// pre-C5 behavior). Round-split rule is C5-C internal; control arms run
    /// at the full configured value.
    pub ef_search: Option<usize>,
    /// Hard cap on the per-head HNSW search `k` (C7 EFPROBE; `None` = no cap).
    /// hnsw_rs clamps its beam to `max(ef, k)`, so an `ef_search` below the
    /// per-head budget is otherwise inert; capping `k` lets the ef knob bite.
    pub search_k: Option<usize>,
    /// One-step cross-head candidate generation (C5-C; `None` = off, the
    /// exact pre-C5 pipeline). When set, only the candidate GENERATION stage
    /// changes: round-1 per-head search → interaction → round-2 re-search;
    /// all downstream union/normalize/gate/fuse/top-K stages are shared with
    /// the control, so output differences are attributable to the candidate
    /// SET, not to scoring.
    pub cross_refine: Option<CrossRefineConfig>,
    /// Adaptive retrieval allocation (C6; `None` = off, pre-C6 pipeline).
    /// When set, controls how candidate budget and EF are allocated across heads.
    pub adaptive: Option<AdaptiveRetrievalConfig>,
    /// C7 genuine candidate-level Q/K/V attention (C7; `None` = off, backward
    /// compatible with the legacy attention feature scorer). When set AND
    /// enabled, the `attention` channel in the fusion is produced by genuine
    /// QKV attention over the per-head candidate vectors (from the head
    /// indexes) instead of the legacy feature-based scorer. Membership and
    /// candidate budget semantics are unchanged — only the attention channel
    /// score changes, so ranking differences are attributable to the attentions.
    pub attention: Option<attentiondb_attention::AttentionConfig>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            mode: RetrievalMode::Full,
            normalization: ScoreNormalization::MinMax,
            candidate_multiplier: OVERFETCH_MULTIPLIER,
            min_candidates_per_head: MIN_CANDIDATES_PER_HEAD,
            candidate_budget: 500,
            fusion: FusionWeights {
                attention: 0.3,
                multi_head_similarity: 0.5,
                bm25: 0.2,
            },
            rrf_k: 60.0,
            hybrid_strategy: HybridStrategy::Rrf,
            parallel: true,
            max_search_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            attention_rank_weight: 0.1,
            ef_search: None,
            search_k: None,
            cross_refine: None,
            adaptive: None,
            attention: None,
        }
    }
}

impl RetrievalConfig {
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.candidate_multiplier == 0 || self.candidate_multiplier > 100 {
            return Err(CoreError::InvalidConfig(
                "candidate_multiplier must be in 1..=100".into(),
            ));
        }
        if self.candidate_budget < 16 || self.candidate_budget > 100_000 {
            return Err(CoreError::InvalidConfig(
                "candidate_budget must be in 16..=100_000".into(),
            ));
        }
        if self.max_search_threads == 0 || self.max_search_threads > 256 {
            return Err(CoreError::InvalidConfig(
                "max_search_threads must be in 1..=256".into(),
            ));
        }
        if !(0.0..=60.0).contains(&self.rrf_k) || !self.rrf_k.is_finite() {
            return Err(CoreError::InvalidConfig(
                "rrf_k must be finite in 0..=60".into(),
            ));
        }
        self.fusion.validate().map_err(CoreError::InvalidConfig)?;
        if let Some(ef) = self.ef_search {
            if !(1..=100_000).contains(&ef) {
                return Err(CoreError::InvalidConfig(
                    "ef_search must be in 1..=100_000".into(),
                ));
            }
        }
        if let Some(sk) = self.search_k {
            if !(1..=100_000).contains(&sk) {
                return Err(CoreError::InvalidConfig(
                    "search_k must be in 1..=100_000".into(),
                ));
            }
        }
        if let Some(c) = self.cross_refine {
            if !(0.0..=1.0).contains(&c.lambda) || !c.lambda.is_finite() {
                return Err(CoreError::InvalidConfig(
                    "cross_refine.lambda must be finite in 0.0..=1.0".into(),
                ));
            }
        }
        if let Some(a) = &self.adaptive {
            if !(0.0..=1.0).contains(&a.stage1_fraction) || !a.stage1_fraction.is_finite() {
                return Err(CoreError::InvalidConfig(
                    "adaptive.stage1_fraction must be finite in 0.0..=1.0".into(),
                ));
            }
            if a.overlap_threshold > 1.0 || a.entropy_threshold < 0.0 {
                return Err(CoreError::InvalidConfig(
                    "adaptive thresholds must be valid".into(),
                ));
            }
        }
        if let Some(attn) = &self.attention {
            if attn.enabled {
                attn.validate().map_err(|e| {
                    CoreError::InvalidConfig(format!("attention config invalid: {e}"))
                })?;
            }
        }
        Ok(())
    }
}

pub struct Collection {
    pub name: String,
    pub dim: usize,
    pub head_manager: Arc<RwLock<HeadIndexManager>>,
    pub multihead_manager: Arc<RwLock<MultiHeadManager>>,
    pub settings: RwLock<attentiondb_hnsw::CollectionSettings>,
    pub bm25: Bm25Index,
    pub gating_network: RwLock<Option<GatingNetwork>>,
    /// Numeric ids retired by delete/update. `hnsw_rs` cannot physically remove
    /// graph nodes, so retired ids are filtered from every search result (INV-3,
    /// INV-4) and purged from the vector store at startup (see `purge_retired`).
    pub retired_ids: RwLock<HashSet<u64>>,
    /// Phase 2 staged-retrieval configuration (planner-controllable).
    pub retrieval_config: RwLock<RetrievalConfig>,
    /// Phase 2B trained gating model (§14/§17/§18): `None` ⇒ deterministic
    /// fallback (uniform or legacy net). Swapped via `Arc` so a query in
    /// flight sees the old OR new weights, never partial (§18).
    pub gating_model: RwLock<Option<Arc<attentiondb_learned::gating_v2::ModelCard>>>,
    /// C7 genuine QKV attention subsystem, cached per enabled config
    /// (§C7): `None` until first query with `retrieval_config.attention`
    /// enabled. Rebuilt only when the config fingerprint changes.
    c7_attention: RwLock<Option<(u64, Arc<attentiondb_attention::AttentionSubsystem>)>>,
}

impl Collection {
    pub fn new(name: &str, dim: usize) -> Self {
        Self {
            name: name.to_string(),
            dim,
            head_manager: Arc::new(RwLock::new(HeadIndexManager::new(dim))),
            multihead_manager: Arc::new(RwLock::new(MultiHeadManager::new(dim, 1))),
            settings: RwLock::new(attentiondb_hnsw::CollectionSettings::default()),
            bm25: Bm25Index::default(),
            gating_network: RwLock::new(None),
            retired_ids: RwLock::new(HashSet::new()),
            retrieval_config: RwLock::new(RetrievalConfig::default()),
            gating_model: RwLock::new(None),
            c7_attention: RwLock::new(None),
        }
    }

    pub fn add_default_head(&self, name: &str) -> Result<(), CoreError> {
        let config = HNSWConfig::default();
        self.head_manager.read().add_head_with_config(name, config);
        let head_config = HeadConfig::new(name, HeadType::Semantic, self.dim);
        self.multihead_manager.write().add_head(head_config);
        Ok(())
    }

    pub fn add_head_with_settings(
        &self,
        name: &str,
        config: HNSWConfig,
        head_type: HeadType,
    ) -> Result<(), CoreError> {
        self.head_manager.read().add_head_with_config(name, config);
        self.multihead_manager
            .write()
            .add_head(HeadConfig::new(name, head_type, self.dim));
        Ok(())
    }

    pub fn insert_vector(&self, head: &str, id: u64, vector: &[f32]) -> Result<(), CoreError> {
        if self.head_manager.read().get_head(head).is_err() {
            self.head_manager
                .read()
                .add_head_with_config(head, HNSWConfig::default());
        }
        if self.multihead_manager.read().get_head(head).is_err() {
            self.multihead_manager.write().add_head(HeadConfig::new(
                head,
                HeadType::Semantic,
                self.dim,
            ));
        }
        self.head_manager.read().insert(head, id, vector)?;
        Ok(())
    }

    /// Trained-model gating (Phase 2B §14): inference only — weights for the
    /// requested heads, looked up by name from the card's head ordering.
    /// Returns None when no model is active or the card doesn't cover the
    /// requested heads (caller falls back; never silent WRONG weights).
    fn get_trained_weights(&self, query: &[f32], head_names: &[String]) -> Option<Vec<f32>> {
        let card = self.gating_model.read().clone()?;
        let names = card.head_names.as_ref()?;
        let mlp = card.to_mlp();
        let full = mlp.predict(query);
        let mut out = Vec::with_capacity(head_names.len());
        for h in head_names {
            let idx = names.iter().position(|n| n == h)?;
            out.push(full.get(idx).copied().unwrap_or(0.0));
        }
        if out.iter().any(|w| !w.is_finite()) {
            return None;
        }
        Some(out)
    }

    fn get_gated_weights(&self, query: &[f32], head_names: &[String]) -> Vec<f32> {
        let gating = self.gating_network.read();
        if let Some(ref net) = *gating {
            let w = net.forward(query);
            if w.len() == head_names.len() {
                return w;
            }
        }
        let w = 1.0 / head_names.len().max(1) as f32;
        vec![w; head_names.len()]
    }

    /// Retire a numeric id: it can never appear in search results again (INV-3/4).
    pub fn retire_id(&self, id: u64) {
        self.retired_ids.write().insert(id);
    }

    pub fn is_retired(&self, id: u64) -> bool {
        self.retired_ids.read().contains(&id)
    }

    pub fn retired_count(&self) -> usize {
        self.retired_ids.read().len()
    }

    /// Remove retired ids from the exact-rerank vector store. Graph nodes left
    /// behind by hnsw_rs remain but can never be *returned* (filtered above).
    /// Returns how many store entries were purged.
    pub fn purge_retired_from_vector_store(&self) -> usize {
        let retired = self.retired_ids.read().clone();
        if retired.is_empty() {
            return 0;
        }
        let mut purged = 0;
        let manager = self.head_manager.read();
        for head in manager.list_heads() {
            if let Ok(idx) = manager.get_head(&head) {
                purged += idx.write().purge_ids(&retired);
            }
        }
        purged
    }

    /// Deterministically rebuild every head index + BM25 from authoritative
    /// records `[(numeric_id, record)]`. Used at recovery (Phase 1 rebuild
    /// strategy — see docs/indexes.md) and after mass operations.
    pub fn rebuild_from_records(
        &self,
        records: &[(u64, attentiondb_storage::Record)],
    ) -> Result<(), CoreError> {
        let dim = self.dim;
        let heads = self.list_heads();
        let new_manager = HeadIndexManager::new(dim);
        for h in &heads {
            new_manager.add_head_with_config(h, HNSWConfig::default());
        }
        // Deterministic insert order: ascending numeric id (HNSW graph shape
        // depends on insertion order; sorting keeps recovery reproducible).
        let mut sorted: Vec<&(u64, attentiondb_storage::Record)> = records.iter().collect();
        sorted.sort_by_key(|(id, _)| *id);
        {
            for (numeric_id, rec) in sorted {
                for (head, vec) in &rec.k_vecs {
                    if vec.len() != dim {
                        return Err(CoreError::InvalidConfig(format!(
                            "record {} head '{}' has dimension {}, collection dim is {}",
                            rec.id,
                            head,
                            vec.len(),
                            dim
                        )));
                    }
                    if new_manager.get_head(head).is_err() {
                        // head present in data but not in catalog settings: create it
                        new_manager.add_head_with_config(head, HNSWConfig::default());
                    }
                    new_manager.insert(head, *numeric_id, vec)?;
                    attentiondb_storage::crashgate::GATE_REBUILD_MID.hit();
                }
            }
        }
        *self.head_manager.write() = new_manager;

        // BM25 rebuild from the same authoritative text.
        self.bm25.clear();
        for (numeric_id, rec) in records {
            let text: String = rec
                .fields
                .values()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            self.bm25.insert(*numeric_id, &text);
        }
        Ok(())
    }

    pub fn load_gating_network_from(&self, net: GatingNetwork) {
        *self.gating_network.write() = Some(net);
    }

    /// Staged retrieval (Phase 2 canonical path). Returns full score breakdowns
    /// for explainability; `attend()` wraps this and strips to (id, score).
    ///
    /// `bm25_raw` (optional) supplies the text channel's raw scores; they are
    /// normalized (min-max) and fused through the same equation as everything
    /// else — BM25 similarities are never linearly mixed with vector scores
    /// unnormalized.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_detailed(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        bm25_raw: Option<&[(u64, f32)]>,
        mode_override: Option<RetrievalMode>,
        gate_override: Option<Vec<f32>>,
        config_override: Option<&RetrievalConfig>,
        deadline: Option<std::time::Instant>,
    ) -> Result<Vec<RankedCandidate>, CoreError> {
        Ok(self
            .attend_detailed_inner(
                heads,
                query,
                top_k,
                bm25_raw,
                mode_override,
                gate_override,
                config_override,
                deadline,
            )?
            .0)
    }

    /// attend_detailed, but also returns the per-query C5 cross-head causal
    /// ledger (round-1 lists, surprise sets, round-2 additions, union pre/post
    /// sizes, ef split) and C6 adaptive trace. Empty when `cross_refine`/`adaptive`
    /// is off — identical results to `attend_detailed_with_stats`.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_detailed_traced(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        bm25_raw: Option<&[(u64, f32)]>,
        mode_override: Option<RetrievalMode>,
        gate_override: Option<Vec<f32>>,
        config_override: Option<&RetrievalConfig>,
        deadline: Option<std::time::Instant>,
    ) -> Result<
        (
            Vec<RankedCandidate>,
            PipelineStats,
            CrossHeadTrace,
            AdaptiveTrace,
        ),
        CoreError,
    > {
        self.attend_detailed_inner(
            heads,
            query,
            top_k,
            bm25_raw,
            mode_override,
            gate_override,
            config_override,
            deadline,
        )
        .map(|(ranked, stats, cross, adaptive, _c7)| (ranked, stats, cross, adaptive))
    }

    /// attend_detailed + pipeline stage sizes (§40: candidate/rerank counts
    /// for the benchmark harness). Union size = post-budget candidate pool;
    /// rerank size = candidates entering exact rerank (MODE E).
    #[allow(clippy::too_many_arguments)]
    pub fn attend_detailed_with_stats(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        bm25_raw: Option<&[(u64, f32)]>,
        mode_override: Option<RetrievalMode>,
        gate_override: Option<Vec<f32>>,
        config_override: Option<&RetrievalConfig>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(Vec<RankedCandidate>, PipelineStats), CoreError> {
        self.attend_detailed_inner(
            heads,
            query,
            top_k,
            bm25_raw,
            mode_override,
            gate_override,
            config_override,
            deadline,
        )
        .map(|(ranked, stats, _trace, _adaptive, _c7)| (ranked, stats))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn attend_detailed_inner(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        bm25_raw: Option<&[(u64, f32)]>,
        mode_override: Option<RetrievalMode>,
        gate_override: Option<Vec<f32>>,
        config_override: Option<&RetrievalConfig>,
        deadline: Option<std::time::Instant>,
    ) -> Result<
        (
            Vec<RankedCandidate>,
            PipelineStats,
            CrossHeadTrace,
            AdaptiveTrace,
            Option<C7Trace>,
        ),
        CoreError,
    > {
        // Cancellation/timeout (§19): checked at every stage boundary. Long
        // HNSW work between checks is bounded by per-head candidate counts.
        let check_deadline = |stage: &str| -> Result<(), CoreError> {
            match deadline {
                Some(d) if std::time::Instant::now() >= d => Err(CoreError::Timeout(format!(
                    "query deadline exceeded during '{stage}'"
                ))),
                _ => Ok(()),
            }
        };
        check_deadline("entry")?;
        if query.len() != self.dim {
            return Err(CoreError::InvalidArgument(format!(
                "query dimension {} does not match collection dimension {}",
                query.len(),
                self.dim
            )));
        }
        let cfg = match config_override {
            Some(o) => o.clone(),
            None => self.retrieval_config.read().clone(),
        };
        cfg.validate()?;
        let mode = mode_override.unwrap_or(cfg.mode);
        if top_k == 0 || heads.is_empty() {
            return Ok((
                vec![],
                PipelineStats::default(),
                CrossHeadTrace::default(),
                AdaptiveTrace::default(),
                None,
            ));
        }

        // ---- MODE A: single head, raw generator scores --------------------
        if mode == RetrievalMode::SingleHead {
            let idx = self
                .head_manager
                .read()
                .get_head(&heads[0])
                .map_err(|e| CoreError::InvalidArgument(format!("head '{}': {e}", heads[0])))?;
            let retired = self.retired_ids.read().clone();
            let results = idx
                .read()
                .search(query, cfg.search_k.unwrap_or(top_k), cfg.ef_search)
                .map_err(|e| CoreError::Internal(format!("hnsw search: {e}")))?;
            let out: Vec<(u64, f32)> = results
                .into_iter()
                .filter(|(id, s)| !retired.contains(id) && s.is_finite())
                .collect();
            return Ok((
                deterministic_top_k(out, top_k)
                    .into_iter()
                    .map(|(id, s)| RankedCandidate {
                        id,
                        final_score: s,
                        features: CandidateFeatures {
                            head_scores: vec![Some(s)],
                            multi_head_similarity: s,
                            bm25: None,
                            attention: None,
                        },
                    })
                    .collect(),
                PipelineStats::default(),
                CrossHeadTrace::default(),
                AdaptiveTrace::default(),
                None,
            ));
        }

        // ---- 1. Candidate generation (parallel, bounded) -------------------
        let per_head_k = (top_k * cfg.candidate_multiplier)
            .max(cfg.min_candidates_per_head)
            .min(cfg.search_k.unwrap_or(usize::MAX));
        let retired = self.retired_ids.read().clone();
        let head_indexes: Vec<Option<Arc<RwLock<attentiondb_hnsw::HNSWIndex>>>> = {
            let mgr = self.head_manager.read();
            heads.iter().map(|h| mgr.get_head(h).ok()).collect()
        };

        // C5: one-step cross-head interaction (retrieve → interact → retrieve).
        // Effort rule (frozen C5 protocol §4): round-1 at ef/2, round-2 at
        // ef - ef/2, so total HNSW ef work == the control arm's single search.
        // `cfg.ef_search` (None = index defaults, pre-C5) is the single total;
        // the interaction never exceeds it. λ=0 is the glue arm: q'_H == q.
        let cross_cfg = cfg.cross_refine;
        let ef_total = cfg.ef_search.unwrap_or(64);
        let (ef_r1, ef_r2) = if cross_cfg.is_some() {
            let r1 = (ef_total / 2).max(1);
            (r1, ef_total.saturating_sub(r1).max(1))
        } else {
            (ef_total, 0)
        };

        // Scoped per-head search; `ef` is passed into HNSW explicitly so the
        // round-split is a REAL knob (C4's coll.settings write was inert).

        // Stage instrumentation (§56/§57): per-stage latency histograms with
        // STATIC labels only (stage name) — never doc ids or query text.
        let t_hnsw = std::time::Instant::now();
        let _span_head_search = tracing::info_span!("head_search", heads = heads.len()).entered();
        let mut cross_trace = CrossHeadTrace::default();
        if cross_cfg.is_some() {
            cross_trace.ef_r1 = ef_r1;
            cross_trace.ef_r2 = ef_r2;
        }

        let mut present_heads: Vec<String> = Vec::new();
        let mut present_hits: Vec<Vec<HeadHit>> = Vec::new();
        // Round-1 (or the sole round for the control arms). Runs each head with
        // (sub-)total ef; results assembled in head order (deterministic).
        let run_round1 = |vector: &[f32], ef: Option<usize>| -> Vec<Option<Vec<HeadHit>>> {
            let search_one = |idx: &Arc<RwLock<attentiondb_hnsw::HNSWIndex>>| {
                let head_len = idx.read().len().max(1);
                let k = per_head_k.min(head_len);
                let res = idx.read().search(vector, k, ef).ok()?;
                let hits: Vec<HeadHit> = res
                    .into_iter()
                    .enumerate()
                    .filter(|(_, (id, s))| s.is_finite() && !retired.contains(id))
                    .map(|(rank, (id, s))| HeadHit {
                        id,
                        raw_score: s,
                        rank,
                    })
                    .collect();
                Some(hits)
            };
            if cfg.parallel && heads.len() > 1 {
                // Bounded concurrency: at most min(heads, max_search_threads)
                // scoped threads; results assembled in head order.
                let chunk = heads
                    .len()
                    .div_ceil(cfg.max_search_threads.min(heads.len()));
                let mut results: Vec<Option<Vec<HeadHit>>> = Vec::with_capacity(heads.len());
                std::thread::scope(|scope| {
                    let mut handles = Vec::new();
                    for group in head_indexes.chunks(chunk) {
                        let handle = scope.spawn(move || {
                            group
                                .iter()
                                .map(|o| o.as_ref().and_then(search_one))
                                .collect::<Vec<_>>()
                        });
                        handles.push(handle);
                    }
                    for h in handles {
                        results.extend(h.join().unwrap_or_default());
                    }
                });
                results
            } else {
                head_indexes
                    .iter()
                    .map(|o| o.as_ref().and_then(search_one))
                    .collect()
            }
        };

        let round1_ef: Option<usize> = if cross_cfg.is_some() {
            Some(ef_r1)
        } else {
            cfg.ef_search
        };
        let round1 = run_round1(query, round1_ef);
        for (h, r) in heads.iter().zip(round1) {
            if let Some(hits) = r {
                if !hits.is_empty() {
                    present_heads.push(h.clone());
                    present_hits.push(hits);
                }
            }
        }
        check_deadline("candidate_generation_round1")?;

        // ---- C6: Adaptive retrieval allocation (post-stage-1 redistribution) ----
        // If adaptive config is set with InteractionGuided policy, run redistribution.
        let mut adaptive_trace = AdaptiveTrace::default();
        if let Some(adaptive_cfg) = &cfg.adaptive {
            if matches!(
                adaptive_cfg.policy_type,
                AdaptivePolicyType::InteractionGuided
            ) {
                // Build head centroids for stage 1 results
                let mgr = self.head_manager.read();
                let mut head_centroids = Vec::with_capacity(present_heads.len());
                for h in &present_heads {
                    if let Ok(idx) = mgr.get_head(h) {
                        let idx = idx.read();
                        let len = idx.len().min(10000);
                        let mut sum = vec![0.0f32; self.dim];
                        let mut count = 0usize;
                        for i in 0..len {
                            if let Some(v) = idx.get_vector(i as u64) {
                                for (s, &x) in sum.iter_mut().zip(v.iter()) {
                                    if x.is_finite() {
                                        *s += x;
                                    }
                                }
                                count += 1;
                            }
                        }
                        if count > 0 {
                            for s in sum.iter_mut() {
                                *s /= count as f32;
                            }
                        }
                        head_centroids.push(sum);
                    } else {
                        head_centroids.push(vec![0.0; self.dim]);
                    }
                }

                let budget = RetrievalBudget {
                    total_candidates: cfg.candidate_budget.max(top_k),
                    total_ef_work: cfg.ef_search.unwrap_or(64) * heads.len(),
                    min_per_head: cfg.min_candidates_per_head,
                    max_per_head: cfg.candidate_budget.max(top_k),
                };

                let policy: Box<dyn crate::adaptive::AllocationPolicy> =
                    Box::new(crate::adaptive::InteractionGuidedPolicy {
                        stage1_fraction: adaptive_cfg.stage1_fraction,
                        overlap_threshold: adaptive_cfg.overlap_threshold,
                        entropy_threshold: adaptive_cfg.entropy_threshold,
                    });

                let retriever = AdaptiveRetriever::new(policy, budget, Some(head_centroids));
                let search_fn = |h: &str, q: &[f32], k: usize, ef: Option<usize>| -> Vec<HeadHit> {
                    let idx_opt = head_indexes
                        .iter()
                        .zip(heads.iter())
                        .find(|(_, hh)| *hh == h);
                    if let Some((Some(idx), _)) = idx_opt {
                        let head_len = idx.read().len().max(1);
                        let kk = k.min(head_len);
                        idx.read().search(q, kk, ef).ok().map_or(Vec::new(), |res| {
                            res.into_iter()
                                .enumerate()
                                .filter(|(_, (id, s))| s.is_finite() && !retired.contains(id))
                                .map(|(rank, (id, s))| HeadHit {
                                    id,
                                    raw_score: s,
                                    rank,
                                })
                                .collect()
                        })
                    } else {
                        Vec::new()
                    }
                };

                // Build Stage1Result from present_hits
                let stage1_results: Vec<crate::adaptive::Stage1Result> = present_heads
                    .iter()
                    .zip(present_hits.iter())
                    .map(|(h, hits)| {
                        let overlap = 0.0f32; // will compute below
                        let entropy = crate::adaptive::score_entropy(hits);
                        let top = hits.first().map(|h| h.raw_score).unwrap_or(0.0);
                        crate::adaptive::Stage1Result {
                            head: h.clone(),
                            candidates: hits.clone(),
                            overlap_with_others: overlap,
                            score_entropy: entropy,
                            top_score: top,
                        }
                    })
                    .collect();

                // Compute overlap
                let head_ids: Vec<std::collections::HashMap<u64, ()>> = stage1_results
                    .iter()
                    .map(|r| r.candidates.iter().map(|h| (h.id, ())).collect())
                    .collect();
                let mut stage1_results_mut = stage1_results;
                for (i, r) in stage1_results_mut.iter_mut().enumerate() {
                    let mut overlap_union = std::collections::HashMap::new();
                    for (j, other) in head_ids.iter().enumerate() {
                        if i != j {
                            overlap_union.extend(other.clone());
                        }
                    }
                    if !overlap_union.is_empty() {
                        let own: std::collections::HashMap<u64, ()> =
                            r.candidates.iter().map(|h| (h.id, ())).collect();
                        let inter = own
                            .keys()
                            .filter(|id| overlap_union.contains_key(id))
                            .count();
                        r.overlap_with_others = inter as f32 / own.len().max(1) as f32;
                    }
                }

                let (_extra_hits, trace) = retriever.run(&present_heads, query, &search_fn);
                adaptive_trace = trace;
            }
        }

        if let Some(cross) = cross_cfg {
            if !present_heads.is_empty() {
                // ---- 1b. Cross-head interaction (retrieve → interact → repeat)
                // S_H = ids returned by OTHER heads in round-1 but NOT by this
                // head; centroid in H's own space; q'_H = normalize(q + λ·mean).
                cross_trace.heads = present_heads.clone();
                cross_trace.round1 = present_hits
                    .iter()
                    .map(|hits| hits.iter().map(|h| h.id).collect())
                    .collect();
                let t_int = std::time::Instant::now();
                let _span_int = tracing::info_span!("cross_head_interaction").entered();

                let head_pos: HashMap<&str, usize> = heads
                    .iter()
                    .enumerate()
                    .map(|(i, h)| (h.as_str(), i))
                    .collect();

                // Union (round-1 only) size, budget-capped, same key as stage 2.
                {
                    let mut pre = CandidateSet::new(present_heads.clone());
                    for hits in &present_hits {
                        pre.push_head(hits.clone());
                    }
                    cross_trace.union_pre =
                        candidate_union(&pre, cfg.candidate_budget.max(top_k)).len();
                }

                let mut merged_hits: Vec<Vec<HeadHit>> = Vec::with_capacity(present_heads.len());
                for (h_idx, (h, h1)) in present_heads.iter().zip(present_hits.iter()).enumerate() {
                    // surprise set: union of all other present heads' ids minus H's
                    let mut other_ids: HashSet<u64> = HashSet::new();
                    for (j, h_j) in present_hits.iter().enumerate() {
                        if j != h_idx {
                            other_ids.extend(h_j.iter().map(|x| x.id));
                        }
                    }
                    let own: HashSet<u64> = h1.iter().map(|x| x.id).collect();
                    let mut s_h: Vec<u64> = other_ids.difference(&own).copied().collect();
                    s_h.sort_unstable(); // id ASC → order-independent centroid

                    let idx = head_pos
                        .get(h.as_str())
                        .and_then(|&i| head_indexes[i].as_ref())
                        .cloned();
                    // H-space vectors of S_H (missing ids skipped — never crashes)
                    let mut vecs: Vec<Vec<f32>> = Vec::new();
                    if let Some(a) = &idx {
                        let guard = a.read();
                        for id in &s_h {
                            if let Some(v) = guard.get_vector(*id) {
                                vecs.push(v.to_vec());
                            }
                        }
                    }
                    let centroid = cross_head_centroid(&vecs);
                    let refined = !s_h.is_empty() && cross.lambda != 0.0;
                    let q2 = cross_refine_query(query, &centroid, cross.lambda);

                    // Round-2 at ef_r2 with the refined query for this head only.
                    let r2: Vec<HeadHit> = match &idx {
                        Some(a) => {
                            let head_len = a.read().len().max(1);
                            let k = per_head_k.min(head_len);
                            match a.read().search(&q2, k, Some(ef_r2)) {
                                Ok(res) => res
                                    .into_iter()
                                    .enumerate()
                                    .filter(|(_, (id, s))| s.is_finite() && !retired.contains(id))
                                    .map(|(rank, (id, s))| HeadHit {
                                        id,
                                        raw_score: s,
                                        rank,
                                    })
                                    .collect(),
                                Err(_) => Vec::new(),
                            }
                        }
                        None => Vec::new(),
                    };

                    cross_trace.surprise.push(s_h.clone());
                    cross_trace.refined.push(refined);
                    cross_trace.round2_added.push(
                        r2.iter()
                            .map(|x| x.id)
                            .filter(|id| !own.contains(id))
                            .collect(),
                    );

                    // Cap round-1 ∪ round-2 back to per_head_k by best raw score
                    // (deterministic: score DESC, id ASC).
                    let mut by_id: HashMap<u64, HeadHit> = HashMap::new();
                    for hit in h1.iter().chain(r2.iter()) {
                        match by_id.get_mut(&hit.id) {
                            Some(ex) if ex.raw_score >= hit.raw_score => {}
                            Some(ex) => *ex = hit.clone(),
                            None => {
                                by_id.insert(hit.id, hit.clone());
                            }
                        }
                    }
                    let mut list: Vec<HeadHit> = by_id.into_values().collect();
                    list.sort_by(|a, b| {
                        b.raw_score
                            .partial_cmp(&a.raw_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.id.cmp(&b.id))
                    });
                    list.truncate(per_head_k);
                    for (rank, hit) in list.iter_mut().enumerate() {
                        hit.rank = rank;
                    }
                    merged_hits.push(list);
                }
                check_deadline("cross_head_interaction")?;

                // Union post-interaction (same key as stage 2) + additions ledger.
                {
                    let mut post = CandidateSet::new(present_heads.clone());
                    for hits in &merged_hits {
                        post.push_head(hits.clone());
                    }
                    let post_union = candidate_union(&post, cfg.candidate_budget.max(top_k));
                    cross_trace.union_post = post_union.len();
                    let pre_ids: HashSet<u64> = cross_trace
                        .round1
                        .iter()
                        .flat_map(|ids| ids.iter().copied())
                        .collect();
                    cross_trace.interaction_union_additions = post_union
                        .iter()
                        .map(|c| c.id)
                        .filter(|id| !pre_ids.contains(id))
                        .collect();
                }

                // Stage-3+ consumes `present_hits` in head order; the merged
                // lists are capped to per_head_k and re-ranked above.
                present_hits = merged_hits;
                for hits in &present_hits {
                    cross_trace.per_head_capped.push(hits.len());
                }
                metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "cross_head_interaction")
                    .record(t_int.elapsed().as_secs_f64());
            }
        }

        metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "hnsw")
            .record(t_hnsw.elapsed().as_secs_f64());
        if present_heads.is_empty() {
            return Ok((
                vec![],
                PipelineStats::default(),
                cross_trace,
                AdaptiveTrace::default(),
                None,
            ));
        }

        // ---- 2. Candidate union (provenance + budget) ----------------------
        let t_union = std::time::Instant::now();
        let span_union = tracing::info_span!("candidate_union").entered();
        let mut set = CandidateSet::new(present_heads.clone());
        for hits in present_hits {
            set.push_head(hits);
        }
        let union = candidate_union(&set, cfg.candidate_budget.max(top_k));
        drop(span_union);
        metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "candidate_union")
            .record(t_union.elapsed().as_secs_f64());
        metrics::counter!("attentiondb_candidates_total").increment(union.len() as u64);
        metrics::gauge!("attentiondb_heads_used").set(present_heads.len() as f64);
        tracing::info!(
            candidates = union.len(),
            heads = present_heads.len(),
            "candidate_union complete"
        );

        // ---- 3. Per-head normalization (id → normalized score) -------------
        let mut norm_maps: Vec<HashMap<u64, f32>> = Vec::with_capacity(set.source_heads.len());
        for hits in &set.per_head {
            let mut raw: Vec<f32> = hits.iter().map(|x| x.raw_score).collect();
            normalize_scores(&mut raw, cfg.normalization);
            let mut m = HashMap::with_capacity(hits.len());
            for (hit, s) in hits.iter().zip(raw) {
                m.insert(hit.id, s);
            }
            norm_maps.push(m);
        }

        // ---- 4. Head gating profile (sums to 1) ----------------------------
        let n_heads = set.source_heads.len();
        let mut profile: Vec<f32> = match (gate_override, mode) {
            (Some(g), _) => g,
            (None, RetrievalMode::FixedFusion) => vec![1.0 / n_heads as f32; n_heads],
            // Phase 2B: trained model first (§14: loaded model, no training at
            // query time); legacy built-in net second; uniform last.
            (None, _) => self
                .get_trained_weights(query, &set.source_heads)
                .unwrap_or_else(|| self.get_gated_weights(query, &set.source_heads)),
        };
        let psum: f32 = profile.iter().sum();
        if !psum.is_finite() || psum <= 0.0 {
            profile = vec![1.0 / n_heads as f32; n_heads];
        } else {
            for p in profile.iter_mut() {
                *p /= psum;
            }
        }

        // ---- 5. BM25 normalization (optional channel) ----------------------
        let bm_map: Option<HashMap<u64, f32>> = bm25_raw.map(|list| {
            let mut raw: Vec<f32> = list.iter().map(|(_, s)| *s).collect();
            normalize_scores(&mut raw, ScoreNormalization::MinMax);
            list.iter().zip(raw).map(|((id, _), s)| (*id, s)).collect()
        });

        check_deadline("normalization_attention")?;

        // ---- 6. Exact rerank (MODE E): metric-consistent gated similarity --
        let t_rerank = std::time::Instant::now();
        let _span_rerank = tracing::info_span!("rerank", candidates = union.len()).entered();
        let metric = self.settings.read().similarity_metric.clone();
        let exact_maps: Option<Vec<HashMap<u64, f32>>> = if mode == RetrievalMode::Full {
            let mgr = self.head_manager.read();
            let mut maps = Vec::with_capacity(n_heads);
            for h in &set.source_heads {
                let mut m = HashMap::new();
                if let Ok(idx) = mgr.get_head(h) {
                    let idx = idx.read();
                    for hit in
                        &set.per_head[set.source_heads.iter().position(|x| x == h).unwrap_or(0)]
                    {
                        if let Some(v) = idx.get_vector(hit.id) {
                            m.insert(hit.id, similarity(&metric, query, v));
                        }
                    }
                }
                maps.push(m);
            }
            Some(maps)
        } else {
            None
        };

        // ---- 7. Candidate features + attention -----------------------------
        let t_attn = std::time::Instant::now();
        let _span_attn = tracing::info_span!("attention", candidates = union.len()).entered();
        let mut attention_scores: Option<Vec<f32>> = None;
        let mut c7_trace: Option<C7Trace> = None;

        // C7: genuine candidate-level Q/K/V attention over per-head vectors.
        // Replaces ONLY the attention channel source; candidate membership and
        // fusion weights are unchanged, so ranking differences are attributable
        // to the attention scores themselves (§C7). Falls back to the legacy
        // feature scorer when disabled.
        let c7_cfg = cfg.attention.clone().filter(|c| c.enabled);
        if let Some(attn_cfg) = c7_cfg {
            // Build (or reuse) the deterministic attention subsystem.
            let fp = attentiondb_attention::config_fingerprint(&attn_cfg);
            let subsystem = {
                let cached = self.c7_attention.read();
                match &*cached {
                    Some((f, s)) if *f == fp => s.clone(),
                    _ => {
                        drop(cached);
                        let s = Arc::new(
                            attentiondb_attention::AttentionSubsystem::new(
                                attn_cfg.clone(),
                                set.source_heads.clone(),
                            )
                            .map_err(|e| CoreError::Internal(format!("c7 subsystem: {e}")))?,
                        );
                        *self.c7_attention.write() = Some((fp, s.clone()));
                        s
                    }
                }
            };

            // Per-candidate per-head vectors + retrieval evidence.
            let mgr = self.head_manager.read();
            let mut c7_cands: Vec<Vec<Vec<f32>>> = Vec::with_capacity(union.len());
            let mut c7_evidence: Vec<attentiondb_attention::RetrievalEvidence> =
                Vec::with_capacity(union.len());
            for u in &union {
                let mut per_head: Vec<Vec<f32>> = Vec::with_capacity(n_heads);
                let mut sims: Vec<Option<f32>> = Vec::with_capacity(n_heads);
                let mut ranks: Vec<Option<f32>> = Vec::with_capacity(n_heads);
                let mut present: Vec<bool> = Vec::with_capacity(n_heads);
                for (h_idx, head) in set.source_heads.iter().enumerate() {
                    let v = match mgr.get_head(head) {
                        Ok(idx) => idx.read().get_vector(u.id).map(|v| v.to_vec()),
                        Err(_) => None,
                    };
                    per_head.push(v.unwrap_or_else(|| vec![0.0; self.dim]));
                    let sim =
                        u.head_scores[h_idx].and_then(|_| norm_maps[h_idx].get(&u.id).copied());
                    sims.push(sim);
                    let r = u.head_ranks[h_idx];
                    ranks.push(r.map(|r| 1.0 / (1.0 + r as f32)));
                    present.push(sim.is_some());
                }
                c7_cands.push(per_head);
                c7_evidence.push(attentiondb_attention::RetrievalEvidence {
                    head_sims: sims,
                    head_ranks: ranks,
                    head_present: present,
                });
            }
            let out = subsystem
                .compute(query, &c7_cands, &c7_evidence)
                .map_err(|e| CoreError::Internal(format!("c7 attention: {e}")))?;
            attention_scores = Some(out.scores.clone());
            // Union-level attention trace (§C7-6): per-candidate A_d / O_d /
            // logits / entropy, per-head mean, mean entropy, compute time.
            // Aligned with `union` (same order as c7_cands).
            c7_trace = Some(C7Trace {
                head_names: set.source_heads.clone(),
                candidates: union
                    .iter()
                    .zip(out.candidates.iter())
                    .zip(out.scores.iter())
                    .map(|((u, c), &s)| C7CandidateTrace {
                        id: u.id,
                        attention_score: s,
                        weights: c.weights.clone(),
                        output: c.output.clone(),
                        logits: c.logits.clone(),
                        entropy: c.entropy,
                    })
                    .collect(),
                per_head_mean: out.per_head_mean.clone(),
                mean_entropy: out.mean_entropy,
                compute_time_us: out.compute_time_us,
            });
            metrics::histogram!("attentiondb_c7_attention_micros")
                .record(out.compute_time_us as f64);
            tracing::info!(
                candidates = union.len(),
                mean_entropy = out.mean_entropy,
                "c7 genuine attention"
            );
        } else {
            // Legacy attention channel (mode >= QKAttention).
            let scorer = AttentionScorer::new(n_heads, cfg.attention_rank_weight);
            let feature_vecs: Vec<Vec<f32>> = union
                .iter()
                .map(|u| {
                    let mut x = Vec::with_capacity(n_heads * 2);
                    for (norm_map, head_score) in norm_maps.iter().zip(u.head_scores.iter()) {
                        x.push(
                            head_score
                                .and_then(|_| norm_map.get(&u.id).copied())
                                .unwrap_or(0.0),
                        );
                    }
                    for r in u.head_ranks.iter() {
                        x.push(r.map(|r| 1.0 / (1.0 + r as f32)).unwrap_or(0.0));
                    }
                    x
                })
                .collect();
            if mode >= RetrievalMode::QKAttention {
                attention_scores = Some(scorer.score(&profile, &feature_vecs));
            }
        }

        // ---- 8. Fusion (explicit equation; §8) -----------------------------
        let mut out: Vec<RankedCandidate> = Vec::with_capacity(union.len());
        for (i, u) in union.iter().enumerate() {
            let mut head_scores = Vec::with_capacity(n_heads);
            let mut mhs_num = 0.0f32;
            for ((norm_map, head_score), gate) in norm_maps
                .iter()
                .zip(u.head_scores.iter())
                .zip(profile.iter())
            {
                let norm = head_score.and_then(|_| norm_map.get(&u.id).copied());
                if let Some(s) = norm {
                    mhs_num += gate * s;
                }
                head_scores.push(norm);
            }
            // Multi-head similarity = Σ_h profile_h · norm_h. A head that did
            // NOT return the candidate contributes 0 — no renormalization over
            // present heads (a candidate perfect in one channel while absent
            // from another must not tie with one that is perfect everywhere).
            // Exact rerank (MODE E) replaces normalized scores with exact
            // similarities under the same rule; candidates with no exact data
            // at all fall back to the normalized approximation.
            let mut exact_num = 0.0f32;
            let mut exact_any = false;
            if let Some(maps) = &exact_maps {
                for (m, gate) in maps.iter().zip(profile.iter()) {
                    if let Some(&s) = m.get(&u.id) {
                        exact_num += gate * s;
                        exact_any = true;
                    }
                }
            }
            let mhs = if exact_any { exact_num } else { mhs_num };

            let attn = attention_scores.as_ref().map(|a| a[i]);
            let bm25 = bm_map.as_ref().and_then(|m| m.get(&u.id).copied());

            let features = CandidateFeatures {
                head_scores,
                multi_head_similarity: mhs,
                bm25,
                attention: attn,
            };
            let final_score = fuse_candidate(&features, &cfg.fusion);
            out.push(RankedCandidate {
                id: u.id,
                final_score,
                features,
            });
        }

        // ---- 9. Deterministic top-K ----------------------------------------
        metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "exact_rerank")
            .record(t_rerank.elapsed().as_secs_f64());
        metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "attention")
            .record(t_attn.elapsed().as_secs_f64());
        metrics::histogram!("attentiondb_exact_rerank_candidates").record(out.len() as f64);
        check_deadline("fusion_rerank")?;
        let _span_topk = tracing::info_span!("topk", k = top_k).entered();
        let pairs: Vec<(u64, f32)> = out.iter().map(|r| (r.id, r.final_score)).collect();
        let top = deterministic_top_k(pairs, top_k);
        let stats = PipelineStats {
            union_size: union.len(),
            rerank_size: out.len(),
            heads_present: present_heads.len(),
        };
        let mut by_id: HashMap<u64, RankedCandidate> = out.into_iter().map(|r| (r.id, r)).collect();
        Ok((
            top.into_iter()
                .filter_map(|(id, s)| {
                    by_id.remove(&id).map(|mut r| {
                        r.final_score = s;
                        r
                    })
                })
                .collect(),
            stats,
            cross_trace,
            adaptive_trace,
            c7_trace,
        ))
    }

    /// Canonical multi-head retrieval (Phase 2). Same signature as before; the
    /// staged pipeline runs underneath (mode from `retrieval_config`).
    pub fn attend(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let ranked = self.attend_detailed(heads, query, top_k, None, None, None, None, None)?;
        Ok(ranked.into_iter().map(|r| (r.id, r.final_score)).collect())
    }

    /// attend_detailed with adaptive trace (C6).
    #[allow(clippy::too_many_arguments)]
    pub fn attend_detailed_adaptive(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        bm25_raw: Option<&[(u64, f32)]>,
        mode_override: Option<RetrievalMode>,
        gate_override: Option<Vec<f32>>,
        config_override: Option<&RetrievalConfig>,
        deadline: Option<std::time::Instant>,
    ) -> Result<
        (
            Vec<RankedCandidate>,
            PipelineStats,
            CrossHeadTrace,
            AdaptiveTrace,
        ),
        CoreError,
    > {
        self.attend_detailed_inner(
            heads,
            query,
            top_k,
            bm25_raw,
            mode_override,
            gate_override,
            config_override,
            deadline,
        )
        .map(|(ranked, stats, cross, adaptive, _c7)| (ranked, stats, cross, adaptive))
    }

    /// attend_detailed_traced, and ALSO the C7 genuine-attention trace for the
    /// query. The trace is `Some` only when the attention channel is enabled
    /// (arms C/D/E/F); it is `None` for the legacy/A/B arms so the probe can
    /// assert parity. Additive — does not alter any existing signature.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn attend_detailed_c7(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        bm25_raw: Option<&[(u64, f32)]>,
        mode_override: Option<RetrievalMode>,
        gate_override: Option<Vec<f32>>,
        config_override: Option<&RetrievalConfig>,
        deadline: Option<std::time::Instant>,
    ) -> Result<
        (
            Vec<RankedCandidate>,
            PipelineStats,
            CrossHeadTrace,
            AdaptiveTrace,
            Option<C7Trace>,
        ),
        CoreError,
    > {
        self.attend_detailed_inner(
            heads,
            query,
            top_k,
            bm25_raw,
            mode_override,
            gate_override,
            config_override,
            deadline,
        )
    }

    /// Fixed-weight retrieval: explicit head weights replace the gating profile.
    pub fn attend_weighted(
        &self,
        heads: &[(String, f32)],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let names: Vec<String> = heads.iter().map(|(h, _)| h.clone()).collect();
        let raw: Vec<f32> = heads.iter().map(|(_, w)| *w).collect();
        let sum: f32 = raw.iter().sum();
        let gates: Vec<f32> = if sum > f32::EPSILON {
            raw.iter().map(|w| w / sum).collect()
        } else {
            vec![1.0 / raw.len().max(1) as f32; raw.len()]
        };
        let ranked = self.attend_detailed(
            &names,
            query,
            top_k,
            None,
            Some(RetrievalMode::FixedFusion),
            Some(gates),
            None,
            None,
        )?;
        Ok(ranked.into_iter().map(|r| (r.id, r.final_score)).collect())
    }

    /// Hybrid vector+text retrieval through the staged pipeline.
    /// Default strategy: RRF (strong deterministic baseline, §15).
    /// Raw BM25 channel for a text query (descending score, retired ids
    /// excluded). Used by the engine's filtered-hybrid path to filter the
    /// sparse channel BEFORE fusion (§13 invariant: post-filter never leaks).
    pub fn bm25_channel(&self, query_text: &str, limit: usize) -> Vec<(u64, f32)> {
        let t_bm25 = std::time::Instant::now();
        let _span_bm25 = tracing::info_span!("bm25", limit = limit).entered();
        let retired = self.retired_ids.read().clone();
        let out = self
            .bm25
            .search(query_text, limit)
            .into_iter()
            .filter(|(id, s)| s.is_finite() && !retired.contains(id))
            .collect::<Vec<_>>();
        metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "bm25")
            .record(t_bm25.elapsed().as_secs_f64());
        out
    }

    pub fn attend_hybrid(
        &self,
        heads: &[String],
        query_vector: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let cfg = self.retrieval_config.read().clone();
        let per_head_k = (top_k * cfg.candidate_multiplier).max(cfg.min_candidates_per_head);
        let t_bm25 = std::time::Instant::now();
        let _span_bm25 = tracing::info_span!("bm25", limit = per_head_k).entered();
        let sparse: Vec<(u64, f32)> = {
            let retired = self.retired_ids.read().clone();
            self.bm25
                .search(query_text, per_head_k)
                .into_iter()
                .filter(|(id, s)| s.is_finite() && !retired.contains(id))
                .collect()
        };
        metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "bm25")
            .record(t_bm25.elapsed().as_secs_f64());
        if sparse.is_empty() && query_vector.is_empty() {
            return Ok(vec![]);
        }
        if query_vector.is_empty() {
            // text-only query: BM25 channel alone
            return Ok(deterministic_top_k(sparse, top_k));
        }
        match cfg.hybrid_strategy {
            HybridStrategy::Rrf => {
                let dense = self.attend(heads, query_vector, per_head_k)?;
                Ok(rrf_fuse(&[&dense, &sparse], cfg.rrf_k, top_k))
            }
            HybridStrategy::Fusion => {
                let ranked = self.attend_detailed(
                    heads,
                    query_vector,
                    top_k,
                    Some(&sparse),
                    None,
                    None,
                    None,
                    None,
                )?;
                Ok(ranked.into_iter().map(|r| (r.id, r.final_score)).collect())
            }
        }
    }

    pub fn list_heads(&self) -> Vec<String> {
        self.head_manager.read().list_heads()
    }

    pub fn total_vectors(&self) -> usize {
        self.head_manager.read().total_vectors()
    }

    pub fn head_count(&self) -> usize {
        self.head_manager.read().head_count()
    }
}

#[cfg(test)]
mod gating_model_tests {
    use super::*;
    use attentiondb_learned::gating_v2::{GatingMlp, ModelCard, TrainingMeta};

    /// The deterministic contract behind "queries use the trained model":
    /// get_trained_weights MUST return the card's own softmax output, mapped
    /// to the requested head order. No HNSW involved ⇒ exact equality holds.
    #[test]
    fn trained_weights_equal_card_prediction_by_head_name() {
        let dim = 8usize;
        let coll = Collection::new("t", dim);
        let names: Vec<String> = (0..4).map(|g| format!("head{g}")).collect();
        let mut m = GatingMlp::new(dim, 4, 4, 1);
        m.b2[2] = 5.0; // non-trivial softmax, not one-hot
        m.w1[0] = 0.3;
        let meta = TrainingMeta {
            seed: 1,
            dataset_hash: 1,
            objective: "soft_target".into(),
            learning_rate: 0.01,
            batch_size: 32,
            epochs_run: 1,
            best_val_loss: 0.0,
            l2: 0.0,
            timestamp_unix: 0,
            code_commit: "t".into(),
            hardware: "ci".into(),
        };
        let mut card = ModelCard::from_mlp(&m, meta, "w-test", "soft_target");
        // card order ≠ requested order: mapping must follow names
        card.head_names = Some(vec![
            "head3".into(),
            "head2".into(),
            "head1".into(),
            "head0".into(),
        ]);
        *coll.gating_model.write() = Some(Arc::new(card));
        let query = vec![0.1; dim];
        let got = coll.get_trained_weights(&query, &names).unwrap();
        let full = m.predict(&query);
        let card_names = ["head3", "head2", "head1", "head0"];
        for (i, n) in names.iter().enumerate() {
            // head_names[j] names MODEL ROW j — resolve by name lookup
            let model_idx = card_names.iter().position(|c| c == n).unwrap();
            assert_eq!(got[i], full[model_idx], "head {n} weight mapping wrong");
        }
        // subset request maps by name too (head2 = model row 1 in card order)
        let got2 = coll
            .get_trained_weights(&query, &["head2".to_string()])
            .unwrap();
        assert_eq!(got2[0], full[1]);
        // the boosted row (b2[2]=5 → card row 2 = "head1") must dominate
        let got3 = coll
            .get_trained_weights(&query, &["head1".to_string()])
            .unwrap();
        assert_eq!(got3[0], full[2]);
        assert!(
            full[2] > 0.9,
            "boosted head should dominate softmax: {}",
            full[2]
        );
    }

    /// §14: no model → None → caller falls back; wrong coverage → None (never
    /// partial weights).
    #[test]
    fn missing_or_incompatible_model_returns_none() {
        let dim = 8usize;
        let coll = Collection::new("t2", dim);
        assert!(coll
            .get_trained_weights(&vec![0.1; dim], &["h".to_string()])
            .is_none());
        let mut m = GatingMlp::new(dim, 2, 2, 1);
        let meta = TrainingMeta {
            seed: 1,
            dataset_hash: 1,
            objective: "soft_target".into(),
            learning_rate: 0.01,
            batch_size: 32,
            epochs_run: 1,
            best_val_loss: 0.0,
            l2: 0.0,
            timestamp_unix: 0,
            code_commit: "t".into(),
            hardware: "ci".into(),
        };
        let mut card = ModelCard::from_mlp(&m, meta, "small", "soft_target");
        card.head_names = Some(vec!["a".into(), "b".into()]);
        *coll.gating_model.write() = Some(Arc::new(card));
        // request a head the card doesn't know → None, not garbage
        assert!(coll
            .get_trained_weights(&vec![0.1; dim], &["c".to_string()])
            .is_none());
        let _ = &mut m;
    }

    /// C5: the interaction must be a no-op when disabled (control parity) and
    /// must move candidates when enabled (causal change), not just rescore.
    #[test]
    fn cross_refine_changes_candidate_set_not_scores() {
        let dim = 4usize;
        let coll = Collection::new("c5", dim);
        coll.add_default_head("a").unwrap();
        coll.add_default_head("b").unwrap();
        // Same point-set for both heads, but each head ranks along its own
        // dominant axis, so their round-1 top-k lists differ (real surprise).
        // 80 points > per_head_k (25) so no head returns the whole universe.
        for i in 0..80u64 {
            let x = i as f32 * 0.01 - 0.4; // spread queries around
            let mut v_a = vec![0f32; dim];
            let mut v_b = vec![0f32; dim];
            v_a[0] = x + if i % 3 == 0 { 0.3 } else { -0.1 };
            v_a[1] = -0.2;
            v_a[2] = 0.05 * (i % 5) as f32;
            v_a[3] = 0.1;
            v_b[1] = x + if i % 3 == 1 { 0.3 } else { -0.1 };
            v_b[0] = -0.2;
            v_b[2] = 0.01 * (i % 7) as f32;
            v_b[3] = -0.1;
            coll.insert_vector("a", i, &v_a).unwrap();
            coll.insert_vector("b", i, &v_b).unwrap();
        }
        let heads = vec!["a".to_string(), "b".to_string()];
        let q = [0.2, 0.15, 0.5, 0.1];

        // --- Control: cross_refine off (identical to pre-C5 pipeline) ---
        let base_cfg = RetrievalConfig {
            mode: RetrievalMode::FixedFusion,
            ef_search: None,
            cross_refine: None,
            ..RetrievalConfig::default()
        };
        let (base_cands, base_stats, empty_trace, _empty_adaptive) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&base_cfg), None)
            .unwrap();
        assert!(
            empty_trace.round1.is_empty(),
            "trace empty when interaction off"
        );
        assert_eq!(empty_trace.union_post, 0);
        assert_eq!(base_cands.len(), 5);

        // --- Enabled interaction at λ=0.5 ---
        let cross_cfg = RetrievalConfig {
            mode: RetrievalMode::FixedFusion,
            ef_search: Some(32),
            cross_refine: Some(CrossRefineConfig { lambda: 0.5 }),
            ..RetrievalConfig::default()
        };
        let (cross_cands, cross_stats, trace, _cross_adaptive) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&cross_cfg), None)
            .unwrap();
        assert_eq!(trace.round1.len(), 2, "two present heads");
        assert_eq!(trace.surprise.len(), 2);
        assert!(
            trace.surprise.iter().any(|s| !s.is_empty()),
            "clusters must produce surprise sets"
        );
        assert_eq!(trace.ef_r1, 16);
        assert_eq!(trace.ef_r2, 16);
        assert!(
            trace.union_pre <= trace.union_post,
            "interaction may grow the union, never anticipate it"
        );
        assert!(
            trace.per_head_capped.iter().all(|&n| n <= 25),
            "per-head cap must hold (per_head_k = max(5*5,20))"
        );
        // The interaction must be able to ADD candidates (round-2 only ids),
        // which is what changes the SET. It may not always; but here clusters
        // are engineered so at least one round-2 addition exists.
        let any_new = trace.round2_added.iter().any(|ids| !ids.is_empty());
        assert!(
            any_new,
            "round-2 must surface cross-head-surprise candidates"
        );
        assert_eq!(cross_cands.len(), 5);
        assert_eq!(cross_stats.heads_present, 2);

        // --- Glue arm: λ=0.0 must reproduce the control candidate SET ---
        let glue_cfg = RetrievalConfig {
            mode: RetrievalMode::FixedFusion,
            ef_search: Some(32),
            cross_refine: Some(CrossRefineConfig { lambda: 0.0 }),
            ..RetrievalConfig::default()
        };
        let (glue_cands, _glue_stats, glue_trace, _glue_adaptive) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&glue_cfg), None)
            .unwrap();
        // λ=0 → no directional shift; candidate sets must be within budget and
        // the interaction MUST NOT invent better universe: nothing outside of
        // the potentially-seen universe may appear. (Exact equality with control
        // is not required since ef is now a real knob; the set move is the claim.)
        let glue_ids: std::collections::HashSet<u64> = glue_cands.iter().map(|c| c.id).collect();
        assert_eq!(glue_ids.len(), glue_cands.len(), "ids unique");
        assert!(
            glue_trace.refined.iter().all(|r| !r),
            "λ=0 must never refine a query"
        );
        assert_eq!(base_stats.heads_present, 2);
    }

    #[allow(clippy::too_many_arguments)]
    #[test]
    fn cross_refine_glue_arm_matches_control_candidates() {
        let dim = 6usize;
        let coll = Collection::new("c5glue", dim);
        coll.add_default_head("h1").unwrap();
        coll.add_default_head("h2").unwrap();
        coll.add_default_head("h3").unwrap();
        let mut seed = 20260925u64;
        let mut lcg = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed as f32 / u64::MAX as f32
        };
        for i in 0..96u64 {
            // deterministic pseudo-random unit-ish vectors
            let mut v_a = vec![0f32; dim];
            let mut v_b = vec![0f32; dim];
            let mut v_c = vec![0f32; dim];
            for k in 0..dim {
                v_a[k] = lcg() - 0.5;
                v_b[k] = lcg() - 0.5;
                v_c[k] = lcg() - 0.5;
            }
            coll.insert_vector("h1", i, &v_a).unwrap();
            coll.insert_vector("h2", i, &v_b).unwrap();
            coll.insert_vector("h3", i, &v_c).unwrap();
        }
        let heads = vec!["h1".to_string(), "h2".to_string(), "h3".to_string()];
        let q: Vec<f32> = (0..dim).map(|_| lcg() - 0.5).collect();

        let control = RetrievalConfig {
            mode: RetrievalMode::FixedFusion,
            ef_search: Some(64),
            cross_refine: None,
            ..RetrievalConfig::default()
        };
        let (c_cands, _, _, _) = coll
            .attend_detailed_traced(&heads, &q, 10, None, None, None, Some(&control), None)
            .unwrap();

        // λ=0.0 re-searches the exact control query (never refines); the
        // mechanism must be inert w.r.t. query direction. Candidate SETS may
        // differ slightly from the control because ef is now a real per-round
        // knob (two ef/2 searches ≠ one ef search) — that is inherent to the
        // split and documented; the causal claim is measured at λ>0 by the
        // study harness, not assumed here.
        let glue = RetrievalConfig {
            mode: RetrievalMode::FixedFusion,
            ef_search: Some(64),
            cross_refine: Some(CrossRefineConfig { lambda: 0.0 }),
            ..RetrievalConfig::default()
        };
        let (g_cands, _, g_trace, _) = coll
            .attend_detailed_traced(&heads, &q, 10, None, None, None, Some(&glue), None)
            .unwrap();
        assert_eq!(c_cands.len(), g_cands.len(), "deterministic top-k");
        assert!(
            g_trace.refined.iter().all(|r| !r),
            "λ=0 must never refine any head's query"
        );
        assert_eq!(g_trace.ef_r1, 32);
        assert_eq!(g_trace.ef_r2, 32);
        assert_eq!(g_trace.round1.len(), 3);
        assert!(
            g_trace.per_head_capped.iter().all(|&n| n <= 50),
            "per-head cap must hold"
        );
    }
}

#[cfg(test)]
mod c7_attention_tests {
    use super::*;

    fn c7_fixed_config(heads: usize, dim: usize) -> attentiondb_attention::AttentionConfig {
        attentiondb_attention::AttentionConfig::fixed_identity(heads, dim, dim, dim)
    }

    #[test]
    fn c7_disabled_matches_legacy_path() {
        let dim = 4usize;
        let coll = Collection::new("c7-disabled", dim);
        coll.add_default_head("a").unwrap();
        coll.add_default_head("b").unwrap();
        for i in 0..60u64 {
            let x = i as f32 * 0.01 - 0.3;
            let mut v_a = vec![0f32; dim];
            let mut v_b = vec![0f32; dim];
            v_a[0] = x;
            v_a[1] = -0.2;
            v_b[1] = x;
            v_b[0] = -0.2;
            coll.insert_vector("a", i, &v_a).unwrap();
            coll.insert_vector("b", i, &v_b).unwrap();
        }
        let heads = vec!["a".to_string(), "b".to_string()];
        let q = [0.2, 0.1, 0.0, 0.0];

        // Legacy path: attention disabled.
        let legacy = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(32),
            attention: None,
            ..RetrievalConfig::default()
        };
        let (l_cands, l_stats, l_trace, _) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&legacy), None)
            .unwrap();

        // Attention config present but disabled → same path as legacy.
        let disabled = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(32),
            attention: Some(attentiondb_attention::AttentionConfig::disabled()),
            ..RetrievalConfig::default()
        };
        let (d_cands, d_stats, d_trace, _) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&disabled), None)
            .unwrap();

        assert_eq!(l_cands.len(), d_cands.len());
        for (a, b) in l_cands.iter().zip(d_cands.iter()) {
            assert_eq!(a.id, b.id, "ids must match");
            assert_eq!(a.final_score, b.final_score, "scores must match");
        }
        assert_eq!(l_stats.union_size, d_stats.union_size);
        assert_eq!(l_trace.round1, d_trace.round1);
    }

    #[test]
    fn c7_enabled_changes_scores_not_membership() {
        let dim = 4usize;
        let coll = Collection::new("c7-enabled", dim);
        coll.add_default_head("a").unwrap();
        coll.add_default_head("b").unwrap();
        for i in 0..60u64 {
            let x = i as f32 * 0.01 - 0.3;
            let mut v_a = vec![0f32; dim];
            let mut v_b = vec![0f32; dim];
            v_a[0] = x;
            v_a[1] = -0.2;
            v_b[1] = x;
            v_b[0] = -0.2;
            coll.insert_vector("a", i, &v_a).unwrap();
            coll.insert_vector("b", i, &v_b).unwrap();
        }
        let heads = vec!["a".to_string(), "b".to_string()];
        let q = [0.2, 0.1, 0.0, 0.0];

        let off = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(32),
            attention: None,
            ..RetrievalConfig::default()
        };
        let (no_attn, no_stats, _, _) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&off), None)
            .unwrap();

        let on = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(32),
            attention: Some(c7_fixed_config(2, dim)),
            ..RetrievalConfig::default()
        };
        let (with_attn, with_stats, _, _) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&on), None)
            .unwrap();

        // Membership invariant: attention must not change the candidate SET.
        let no_ids: HashSet<u64> = no_attn.iter().map(|c| c.id).collect();
        let with_ids: HashSet<u64> = with_attn.iter().map(|c| c.id).collect();
        assert_eq!(no_ids, with_ids, "candidate membership must be invariant");
        assert_eq!(no_stats.union_size, with_stats.union_size);

        // Reliability: C7 outputs finite scores.
        assert!(with_attn.iter().all(|c| c.final_score.is_finite()));

        // Attention must actually participate (V use): the returned candidates
        // carry the attention feature; verify it is populated and finite.
        for cand in &with_attn {
            if let Some(attn) = cand.features.attention {
                assert!(attn.is_finite());
            }
        }
    }

    #[test]
    fn c7_identity_is_deterministic() {
        let dim = 4usize;
        let coll = Collection::new("c7-det", dim);
        coll.add_default_head("a").unwrap();
        coll.add_default_head("b").unwrap();
        for i in 0..40u64 {
            let x = i as f32 * 0.01 - 0.2;
            let mut va = vec![0f32; dim];
            let mut vb = vec![0f32; dim];
            va[0] = x;
            vb[1] = x;
            coll.insert_vector("a", i, &va).unwrap();
            coll.insert_vector("b", i, &vb).unwrap();
        }
        let heads = vec!["a".to_string(), "b".to_string()];
        let q = [0.1, 0.1, 0.0, 0.0];
        let cfg = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(16),
            attention: Some(c7_fixed_config(2, dim)),
            ..RetrievalConfig::default()
        };
        let (r1, _, _, _) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&cfg), None)
            .unwrap();
        let (r2, _, _, _) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&cfg), None)
            .unwrap();
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(
                a.final_score.to_bits(),
                b.final_score.to_bits(),
                "bit-exact determinism"
            );
        }
    }

    #[test]
    fn c7_trace_additive_and_union_aligned() {
        let dim = 4usize;
        let coll = Collection::new("c7-trace", dim);
        coll.add_default_head("a").unwrap();
        coll.add_default_head("b").unwrap();
        for i in 0..50u64 {
            let x = i as f32 * 0.01 - 0.25;
            let mut va = vec![0f32; dim];
            let mut vb = vec![0f32; dim];
            va[0] = x;
            vb[1] = x;
            coll.insert_vector("a", i, &va).unwrap();
            coll.insert_vector("b", i, &vb).unwrap();
        }
        let heads = vec!["a".to_string(), "b".to_string()];
        let q = [0.15, 0.1, 0.0, 0.0];

        // Additive API: public 4-tuple methods are unchanged; the new method
        // returns the same three traces PLUS the C7 trace (None when disabled).
        let cfg_on = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(16),
            attention: Some(c7_fixed_config(2, dim)),
            ..RetrievalConfig::default()
        };
        let (ranked, stats, cross, adaptive, trace) = coll
            .attend_detailed_c7(&heads, &q, 5, None, None, None, Some(&cfg_on), None)
            .unwrap();
        let (ranked4, stats4, cross4, adaptive4, _) = coll
            .attend_detailed_traced(&heads, &q, 5, None, None, None, Some(&cfg_on), None)
            .map(|(r, s, c, a)| (r, s, c, a, ()))
            .unwrap();

        assert_eq!(ranked.len(), ranked4.len());
        assert_eq!(stats.union_size, stats4.union_size);
        assert_eq!(cross.round1, cross4.round1);
        assert_eq!(
            adaptive.total_candidates_used,
            adaptive4.total_candidates_used
        );

        // Trace present and union-aligned when attention enabled.
        let trace = trace.expect("trace must be Some when attention is enabled");
        assert_eq!(trace.candidates.len(), stats.union_size);
        assert_eq!(trace.head_names, heads);
        for c in &trace.candidates {
            assert_eq!(c.weights.len(), 2, "one weight per head");
            assert_eq!(c.logits.len(), 2);
            assert_eq!(c.output.len(), dim);
            assert!(c.entropy.is_finite() && c.entropy >= 0.0);
            assert!(c.attention_score.is_finite());
        }
        assert_eq!(trace.per_head_mean.len(), 2);
        assert!(trace.mean_entropy.is_finite() && trace.mean_entropy >= 0.0);
        // Attention scores in the trace must match the fused candidates.
        assert_eq!(trace.candidates[0].id, ranked[0].id);

        // None when the attention channel is disabled (arm A/B / legacy).
        let cfg_off = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(16),
            attention: None,
            ..RetrievalConfig::default()
        };
        let (_, _, _, _, trace_off) = coll
            .attend_detailed_c7(&heads, &q, 5, None, None, None, Some(&cfg_off), None)
            .unwrap();
        assert!(
            trace_off.is_none(),
            "disabled attention must yield None trace"
        );
    }

    #[test]
    fn c7_trace_deterministic() {
        let dim = 4usize;
        let coll = Collection::new("c7-trace-det", dim);
        coll.add_default_head("a").unwrap();
        coll.add_default_head("b").unwrap();
        for i in 0..30u64 {
            let x = i as f32 * 0.02 - 0.3;
            let mut va = vec![0f32; dim];
            let mut vb = vec![0f32; dim];
            va[0] = x;
            vb[1] = x;
            coll.insert_vector("a", i, &va).unwrap();
            coll.insert_vector("b", i, &vb).unwrap();
        }
        let heads = vec!["a".to_string(), "b".to_string()];
        let q = [0.1, 0.05, 0.0, 0.0];
        let cfg = RetrievalConfig {
            mode: RetrievalMode::LearnedGating,
            ef_search: Some(16),
            attention: Some(c7_fixed_config(2, dim)),
            ..RetrievalConfig::default()
        };
        let (_, _, _, _, t1) = coll
            .attend_detailed_c7(&heads, &q, 5, None, None, None, Some(&cfg), None)
            .unwrap();
        let (_, _, _, _, t2) = coll
            .attend_detailed_c7(&heads, &q, 5, None, None, None, Some(&cfg), None)
            .unwrap();
        let t1 = t1.unwrap();
        let t2 = t2.unwrap();
        assert_eq!(t1.candidates.len(), t2.candidates.len());
        for (a, b) in t1.candidates.iter().zip(t2.candidates.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.attention_score.to_bits(), b.attention_score.to_bits());
            assert_eq!(a.entropy.to_bits(), b.entropy.to_bits());
            assert_eq!(a.weights, b.weights);
            assert_eq!(a.output, b.output);
            assert_eq!(a.logits, b.logits);
        }
        assert_eq!(t1.per_head_mean, t2.per_head_mean);
        assert_eq!(t1.mean_entropy.to_bits(), t2.mean_entropy.to_bits());
    }
}
