//! C8 in-memory document-side K/V cache.
//!
//! C7 measured candidate attention at roughly two orders of magnitude more
//! latency than the baseline fusion, and the dominant cost is projecting *every
//! candidate* through `W_K` and `W_V` on *every* query. Those projections depend
//! only on the document and the model, never on the query — so they are
//! computed once and reused.
//!
//! Scope (C8 decision Q3): research/runtime cache only. No durable SSTable
//! format, no versioning, no on-disk persistence — that belongs to a later
//! phase. What C8 needs to answer is narrowly *"does caching the document-side
//! K/V computation reduce query-time attention cost, without changing a single
//! ranking decision?"*, and that question is fully answered by an in-memory
//! cache whose validity is provable from the fingerprints below.
//!
//! Correctness contract: a cached entry is used only when every field of
//! [`CacheFingerprint`] matches. `model_fingerprint` covers the config and the
//! effective QKV weights, `projection_fingerprint` covers `W_K`/`W_V` alone, and
//! the remaining fields cover the geometry (attention/key/value dimensions and
//! head order). A model retrain, a dimension change, or a reordering of the
//! retrieval heads therefore invalidates the cache instead of silently serving
//! stale vectors.
//!
//! Determinism: `HashMap` iteration is never used to produce scores — lookups
//! are by explicit key and the stats counters are plain integers.

use crate::projection::QkvProjection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Everything that must match for cached K/V vectors to be valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheFingerprint {
    /// Fingerprint of the whole model (config + effective QKV weights).
    pub model_fingerprint: u64,
    /// Fingerprint of `W_K` and `W_V` (the cached matrices) alone.
    pub projection_fingerprint: u64,
    /// Attention dimension `d_a`.
    pub attention_dim: usize,
    /// Key dimension `d_k`.
    pub key_dim: usize,
    /// Value dimension `d_v`.
    pub value_dim: usize,
    /// Number of retrieval heads covered by each entry.
    pub head_count: usize,
}

impl CacheFingerprint {
    /// Build a fingerprint for a projection. `head_names` is not hashed: only
    /// its length matters, because the head *order* is baked into the entry
    /// layout and any reorder must change `model_fingerprint` (which the caller
    /// derives from the ordered head list).
    pub fn from_projection(model_fingerprint: u64, qkv: &QkvProjection, head_count: usize) -> Self {
        Self {
            model_fingerprint,
            projection_fingerprint: projection_fingerprint(&qkv.w_k, &qkv.w_v),
            attention_dim: qkv.attention_dim,
            key_dim: qkv.key_dim,
            value_dim: qkv.value_dim,
            head_count,
        }
    }

    /// Can entries fingerprinted with `other` be served for this model?
    pub fn accepts(&self, other: &CacheFingerprint) -> bool {
        self == other
    }
}

/// FNV-1a over the raw bits of two matrices.
pub fn projection_fingerprint(w_k: &[f32], w_v: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for w in w_k.iter().chain(w_v.iter()) {
        for b in w.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

/// Cached keys and values for one document: one (K, V) pair per retrieval head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedCandidateKV {
    /// Per-head key vectors, `head_count x d_k`.
    pub keys: Vec<Vec<f32>>,
    /// Per-head value vectors, `head_count x d_v`.
    pub values: Vec<Vec<f32>>,
}

impl CachedCandidateKV {
    /// Project an aligned candidate `Z_d ∈ R^{H x d_a}` into per-head K/V.
    pub fn project(z_d: &[Vec<f32>], qkv: &QkvProjection) -> Option<Self> {
        if z_d.is_empty() {
            return None;
        }
        let keys = qkv.project_k(z_d).ok()?;
        let values = qkv.project_v(z_d).ok()?;
        Some(Self { keys, values })
    }
}

/// Cache hit/miss accounting for one query or one whole experiment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of document lookups served from the cache.
    pub hits: u64,
    /// Number of document lookups computed from scratch.
    pub misses: u64,
    /// Entries currently resident.
    pub entries: usize,
}

impl CacheStats {
    /// Total lookups.
    pub fn lookups(&self) -> u64 {
        self.hits + self.misses
    }

    /// Hit rate in `[0, 1]`; `0.0` when nothing was looked up.
    pub fn hit_rate(&self) -> f32 {
        let total = self.lookups();
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// Merge another stats block into this one (accumulates counters).
    pub fn merge(&mut self, other: &CacheStats) {
        self.hits += other.hits;
        self.misses += other.misses;
        self.entries = other.entries;
    }
}

/// In-memory document-side K/V cache, keyed by document id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionKVCache {
    fingerprint: CacheFingerprint,
    entries: HashMap<u64, CachedCandidateKV>,
}

impl AttentionKVCache {
    /// Create an empty cache bound to `fingerprint`.
    pub fn new(fingerprint: CacheFingerprint) -> Self {
        Self {
            fingerprint,
            entries: HashMap::new(),
        }
    }

    /// The fingerprint this cache is valid for.
    pub fn fingerprint(&self) -> CacheFingerprint {
        self.fingerprint
    }

    /// Is this cache valid for `fingerprint`? Callers must consult this before
    /// serving a hit; [`Self::lookup`] assumes it already did.
    pub fn is_valid_for(&self, fingerprint: &CacheFingerprint) -> bool {
        self.fingerprint.accepts(fingerprint)
    }

    /// Number of resident entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every entry (keeps the fingerprint).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Drop entries for documents `keep` rejects, returning how many were removed.
    ///
    /// The vector store this cache fronts (`hnsw_rs`) cannot physically remove
    /// graph nodes, so deletes are logical: a retired document stops appearing
    /// in search results but its K/V would otherwise stay in this cache forever
    /// and be served to any future lookup of that id. Callers must purge
    /// retired ids through here on the same path that retires them.
    pub fn retain(&mut self, mut keep: impl FnMut(u64) -> bool) -> usize {
        let before = self.entries.len();
        self.entries.retain(|&id, _| keep(id));
        before - self.entries.len()
    }

    /// Document ids currently held, in unspecified order.
    pub fn document_ids(&self) -> Vec<u64> {
        self.entries.keys().copied().collect()
    }

    /// Fetch cached K/V for a document, counting the lookup in `stats`.
    pub fn lookup(&self, doc_id: u64, stats: &mut CacheStats) -> Option<&CachedCandidateKV> {
        match self.entries.get(&doc_id) {
            Some(e) => {
                stats.hits += 1;
                Some(e)
            }
            None => {
                stats.misses += 1;
                None
            }
        }
    }

    /// Insert (or overwrite) the K/V for a document.
    pub fn insert(&mut self, doc_id: u64, kv: CachedCandidateKV) {
        self.entries.insert(doc_id, kv);
    }

    /// Refresh `stats.entries` after mutations.
    pub fn seal_stats(&self, stats: &mut CacheStats) {
        stats.entries = self.entries.len();
    }

    /// Verify every resident entry matches the declared geometry. Used by the
    /// verifier to prove a cache was not filled by a differently shaped model.
    pub fn validate_geometry(&self) -> Result<(), String> {
        let fp = &self.fingerprint;
        for (doc_id, e) in &self.entries {
            if e.keys.len() != fp.head_count {
                return Err(format!(
                    "doc {doc_id}: {} keys, expected {} heads",
                    e.keys.len(),
                    fp.head_count
                ));
            }
            if e.values.len() != fp.head_count {
                return Err(format!(
                    "doc {doc_id}: {} values, expected {} heads",
                    e.values.len(),
                    fp.head_count
                ));
            }
            for (h, k) in e.keys.iter().enumerate() {
                if k.len() != fp.key_dim {
                    return Err(format!(
                        "doc {doc_id} head {h}: key dim {} != {}",
                        k.len(),
                        fp.key_dim
                    ));
                }
            }
            for (h, v) in e.values.iter().enumerate() {
                if v.len() != fp.value_dim {
                    return Err(format!(
                        "doc {doc_id} head {h}: value dim {} != {}",
                        v.len(),
                        fp.value_dim
                    ));
                }
            }
        }
        Ok(())
    }

    /// Build a fully populated cache for a document set.
    ///
    /// `aligned` maps document id to its aligned `Z_d ∈ R^{H x d_a}`. Documents
    /// missing from the map are simply not cached (a later query pays a miss).
    pub fn build(
        fingerprint: CacheFingerprint,
        qkv: &QkvProjection,
        aligned: &HashMap<u64, Vec<Vec<f32>>>,
    ) -> Self {
        let mut cache = Self::new(fingerprint);
        for (doc_id, z_d) in aligned {
            if let Some(kv) = CachedCandidateKV::project(z_d, qkv) {
                cache.insert(*doc_id, kv);
            }
        }
        cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj(d: usize) -> QkvProjection {
        QkvProjection::identity(d)
    }

    fn aligned(seed: f32, heads: usize, d: usize) -> Vec<Vec<f32>> {
        (0..heads).map(|h| vec![seed + h as f32; d]).collect()
    }

    fn fp_for(qkv: &QkvProjection, heads: usize) -> CacheFingerprint {
        CacheFingerprint::from_projection(0xABCD, qkv, heads)
    }

    #[test]
    fn hit_then_miss_accounting() {
        let qkv = proj(4);
        let mut cache = AttentionKVCache::new(fp_for(&qkv, 2));
        let mut stats = CacheStats::default();
        assert!(cache.lookup(1, &mut stats).is_none());
        cache.insert(
            1,
            CachedCandidateKV::project(&aligned(1.0, 2, 4), &qkv).unwrap(),
        );
        assert!(cache.lookup(1, &mut stats).is_some());
        assert!(cache.lookup(2, &mut stats).is_none());
        cache.seal_stats(&mut stats);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.lookups(), 3);
        assert_eq!(stats.entries, 1);
        assert!((stats.hit_rate() - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn empty_stats_hit_rate_is_zero_not_nan() {
        let s = CacheStats::default();
        assert_eq!(s.hit_rate(), 0.0);
        assert_eq!(s.lookups(), 0);
    }

    #[test]
    fn retain_purges_retired_documents() {
        // The vector store cannot remove graph nodes, so a deleted document's
        // K/V must be purged explicitly or it keeps being served.
        let qkv = proj(4);
        let mut cache = AttentionKVCache::new(fp_for(&qkv, 2));
        for id in 1..=5u64 {
            cache.insert(
                id,
                CachedCandidateKV::project(&aligned(id as f32, 2, 4), &qkv).unwrap(),
            );
        }
        assert_eq!(cache.len(), 5);
        let removed = cache.retain(|id| id % 2 == 0);
        assert_eq!(removed, 3, "1, 3, 5 are retired");
        assert_eq!(cache.len(), 2);
        let mut ids = cache.document_ids();
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 4]);
        let mut stats = CacheStats::default();
        assert!(cache.lookup(3, &mut stats).is_none());
        assert!(cache.lookup(2, &mut stats).is_some());
    }

    #[test]
    fn retain_keeps_the_fingerprint() {
        let qkv = proj(4);
        let fp = fp_for(&qkv, 2);
        let mut cache = AttentionKVCache::new(fp.clone());
        cache.insert(
            7,
            CachedCandidateKV::project(&aligned(1.0, 2, 4), &qkv).unwrap(),
        );
        cache.retain(|_| false);
        assert!(cache.is_empty());
        assert!(
            cache.is_valid_for(&fp),
            "purge must not invalidate the cache"
        );
    }

    #[test]
    fn cached_vectors_equal_uncached_projection() {
        let qkv = proj(4);
        let z = aligned(2.0, 3, 4);
        let direct_k = qkv.project_k(&z).unwrap();
        let direct_v = qkv.project_v(&z).unwrap();
        let cached = CachedCandidateKV::project(&z, &qkv).unwrap();
        assert_eq!(cached.keys, direct_k);
        assert_eq!(cached.values, direct_v);
    }

    #[test]
    fn weight_change_invalidates() {
        let qkv = proj(4);
        let fp = fp_for(&qkv, 2);
        let mut retrained = qkv.clone();
        retrained.w_k[0] = 0.5;
        assert!(!fp.accepts(&fp_for(&retrained, 2)));
    }

    #[test]
    fn dimension_change_invalidates() {
        let fp = fp_for(&proj(4), 2);
        assert!(!fp.accepts(&fp_for(&proj(4), 3))); // head count
        let mut narrow = proj(4);
        narrow.key_dim = 2;
        narrow.w_k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!fp.accepts(&fp_for(&narrow, 2)));
    }

    #[test]
    fn model_fingerprint_change_invalidates() {
        let qkv = proj(4);
        let a = CacheFingerprint::from_projection(1, &qkv, 2);
        let b = CacheFingerprint::from_projection(2, &qkv, 2);
        assert!(!a.accepts(&b));
    }

    #[test]
    fn geometry_validation_catches_foreign_entries() {
        let qkv = proj(4);
        let mut cache = AttentionKVCache::new(fp_for(&qkv, 2));
        cache.insert(
            1,
            CachedCandidateKV::project(&aligned(1.0, 3, 4), &qkv).unwrap(),
        );
        assert!(cache.validate_geometry().is_err());
        let mut good = AttentionKVCache::new(fp_for(&qkv, 2));
        good.insert(
            1,
            CachedCandidateKV::project(&aligned(1.0, 2, 4), &qkv).unwrap(),
        );
        assert!(good.validate_geometry().is_ok());
    }

    #[test]
    fn build_is_deterministic_and_populated() {
        let qkv = proj(4);
        let mut docs = HashMap::new();
        docs.insert(7u64, aligned(1.0, 2, 4));
        docs.insert(8u64, aligned(2.0, 2, 4));
        let a = AttentionKVCache::build(fp_for(&qkv, 2), &qkv, &docs);
        let b = AttentionKVCache::build(fp_for(&qkv, 2), &qkv, &docs);
        assert_eq!(a.len(), 2);
        assert!(a.validate_geometry().is_ok());

        let mut stats = CacheStats::default();
        for id in [7u64, 8] {
            let ka = a.lookup(id, &mut stats).unwrap().clone();
            let kb = b.lookup(id, &mut stats).unwrap();
            assert_eq!(&ka, kb);
        }
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate(), 1.0);
    }

    #[test]
    fn fingerprint_changes_with_matrix_bits() {
        let a = [1.0f32, 2.0];
        let mut b = a;
        assert_eq!(
            projection_fingerprint(&a, &[]),
            projection_fingerprint(&b, &[])
        );
        b[1] = 2.5;
        assert_ne!(
            projection_fingerprint(&a, &[]),
            projection_fingerprint(&b, &[])
        );
    }
}
