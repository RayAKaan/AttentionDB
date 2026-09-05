use crate::bm25::Bm25Index;
use crate::error::CoreError;
use crate::retrieval::{
    candidate_union, deterministic_top_k, fuse_candidate, normalize_scores, rrf_fuse,
    AttentionScorer, CandidateFeatures, CandidateSet, FusionWeights, HeadHit, RankedCandidate,
    ScoreNormalization,
};
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
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(Vec<RankedCandidate>, PipelineStats), CoreError> {
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
            return Ok((vec![], PipelineStats::default()));
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
                .search(query, top_k, None)
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
            ));
        }

        // ---- 1. Candidate generation (parallel, bounded) -------------------
        let per_head_k = (top_k * cfg.candidate_multiplier).max(cfg.min_candidates_per_head);
        let retired = self.retired_ids.read().clone();
        let head_indexes: Vec<Option<Arc<RwLock<attentiondb_hnsw::HNSWIndex>>>> = {
            let mgr = self.head_manager.read();
            heads.iter().map(|h| mgr.get_head(h).ok()).collect()
        };

        let search_one = |idx: &Arc<RwLock<attentiondb_hnsw::HNSWIndex>>| {
            // Clamp k to the head's element count: hnsw_rs misbehaves when
            // k exceeds the number of indexed elements.
            let head_len = idx.read().len().max(1);
            let k = per_head_k.min(head_len);
            let res = idx.read().search(query, k, None).ok()?;
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

        // Stage instrumentation (§56/§57): per-stage latency histograms with
        // STATIC labels only (stage name) — never doc ids or query text.
        let t_hnsw = std::time::Instant::now();
        let _span_head_search = tracing::info_span!("head_search", heads = heads.len()).entered();
        let mut present_heads: Vec<String> = Vec::new();
        let mut present_hits: Vec<Vec<HeadHit>> = Vec::new();
        if cfg.parallel && heads.len() > 1 {
            // Bounded concurrency: at most min(heads, max_search_threads) scoped
            // threads; results assembled in head order (deterministic).
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
            for (h, r) in heads.iter().zip(results) {
                if let Some(hits) = r {
                    if !hits.is_empty() {
                        present_heads.push(h.clone());
                        present_hits.push(hits);
                    }
                }
            }
        } else {
            for (h, o) in heads.iter().zip(head_indexes.iter()) {
                if let Some(idx) = o {
                    if let Some(hits) = search_one(idx) {
                        if !hits.is_empty() {
                            present_heads.push(h.clone());
                            present_hits.push(hits);
                        }
                    }
                }
            }
        }
        check_deadline("candidate_generation")?;
        metrics::histogram!("attentiondb_stage_latency_seconds", "stage" => "hnsw")
            .record(t_hnsw.elapsed().as_secs_f64());
        if present_heads.is_empty() {
            return Ok((vec![], PipelineStats::default()));
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
        let attention_scores = if mode >= RetrievalMode::QKAttention {
            Some(scorer.score(&profile, &feature_vecs))
        } else {
            None
        };

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
}
