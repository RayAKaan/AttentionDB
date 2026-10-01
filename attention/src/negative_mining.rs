//! C8 deterministic hard-negative mining.
//!
//! C7 trained against a single fixed negative pattern, which gave the QKV
//! projection almost no information about *which* candidates the baseline
//! actually finds confusing. C8 mines a real, reproducible negative set per
//! query from the retrieval union, stratified into four hardness sources:
//!
//! ```text
//! HighBaseline      50%  the candidates the baseline already scores highest
//! HeadDisagreement  20%  heads disagree most about the candidate
//! NearPositive      20%  baseline scores closest to the weakest positive
//! UniformUnion      10%  a deterministic stride sample of the whole union
//! ```
//!
//! Cross-head disagreement is the **variance of the per-head normalized
//! similarities** (C8 decision Q4):
//!
//! ```text
//! D(d) = Var(r_{d,1}, ..., r_{d,H})
//! ```
//!
//! computed only over heads that are both flagged present *and* carry a finite
//! score — a missing head is never imputed as zero. With fewer than
//! `min_heads_for_disagreement` usable heads the candidate has no defined
//! disagreement and cannot enter that stratum; it is still mined, but ranked
//! by its baseline score instead.
//!
//! Determinism: no RNG anywhere. Ties break on candidate id, the uniform
//! stratum is a fixed stride over the id-sorted pool, and quota allocation uses
//! the largest-remainder method. The same pool always yields byte-identical
//! output.

use crate::errors::Result;
use crate::scorer::RetrievalEvidence;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Default minimum number of usable heads before cross-head variance is defined.
pub const MIN_HEADS_FOR_DISAGREEMENT: usize = 2;

/// Which hardness source a mined negative came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NegSource {
    /// Highest baseline scores in the union.
    HighBaseline,
    /// Highest cross-head disagreement.
    HeadDisagreement,
    /// Baseline scores closest to the weakest positive.
    NearPositive,
    /// Deterministic uniform stride sample of the union.
    UniformUnion,
}

impl NegSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            NegSource::HighBaseline => "high_baseline",
            NegSource::HeadDisagreement => "head_disagreement",
            NegSource::NearPositive => "near_positive",
            NegSource::UniformUnion => "uniform_union",
        }
    }

    /// Stratum index: 0 is the most important source, and it is also the
    /// tie-break order when a candidate qualifies for several strata.
    pub fn stratum(&self) -> u8 {
        match self {
            NegSource::HighBaseline => 0,
            NegSource::HeadDisagreement => 1,
            NegSource::NearPositive => 2,
            NegSource::UniformUnion => 3,
        }
    }

    /// Fixed stratum order for quota allocation.
    pub fn ordered() -> [NegSource; 4] {
        [
            NegSource::HighBaseline,
            NegSource::HeadDisagreement,
            NegSource::NearPositive,
            NegSource::UniformUnion,
        ]
    }
}

/// One mined negative with full provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardNegative {
    /// Candidate document id.
    pub candidate_id: u64,
    /// Baseline score `S_base` of the candidate (teacher signal).
    pub baseline_score: f32,
    /// Hardness source that selected it.
    pub source: NegSource,
    /// Within-stratum hardness, always oriented so that larger = harder.
    pub hardness: f32,
    /// `Var(r_{d,1..H})` over usable heads; `None` when undefined.
    pub disagreement: Option<f32>,
    /// Number of heads that contributed a usable similarity.
    pub usable_heads: usize,
}

/// A candidate offered to the miner.
#[derive(Debug, Clone)]
pub struct PoolEntry {
    pub id: u64,
    /// Baseline score `S_base`.
    pub baseline_score: f32,
    /// Per-head retrieval evidence (full vectors are preserved, not a scalar).
    pub evidence: RetrievalEvidence,
    /// Whether this candidate is a known relevant document for the query.
    pub is_positive: bool,
}

/// Mining configuration (stratum fractions must sum to 1 within tolerance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeConfig {
    /// Maximum negatives to return.
    pub max_negatives: usize,
    /// Stratum fractions, in [`NegSource::ordered`] order.
    pub high_baseline_fraction: f32,
    pub head_disagreement_fraction: f32,
    pub near_positive_fraction: f32,
    pub uniform_union_fraction: f32,
    /// Minimum usable heads before variance is defined.
    pub min_heads_for_disagreement: usize,
}

impl Default for HardNegativeConfig {
    fn default() -> Self {
        Self {
            max_negatives: 8,
            high_baseline_fraction: 0.50,
            head_disagreement_fraction: 0.20,
            near_positive_fraction: 0.20,
            uniform_union_fraction: 0.10,
            min_heads_for_disagreement: MIN_HEADS_FOR_DISAGREEMENT,
        }
    }
}

impl HardNegativeConfig {
    fn fractions(&self) -> [f32; 4] {
        [
            self.high_baseline_fraction,
            self.head_disagreement_fraction,
            self.near_positive_fraction,
            self.uniform_union_fraction,
        ]
    }

    fn validate(&self) -> Result<()> {
        let f = self.fractions();
        if f.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(crate::errors::AttentionError::Config(
                "hard-negative fractions must be finite and >= 0".into(),
            ));
        }
        let total: f32 = f.iter().sum();
        if (total - 1.0).abs() > 1e-6 {
            return Err(crate::errors::AttentionError::Config(format!(
                "hard-negative fractions must sum to 1.0 (got {total})"
            )));
        }
        if self.min_heads_for_disagreement < 2 {
            return Err(crate::errors::AttentionError::Config(
                "min_heads_for_disagreement must be >= 2".into(),
            ));
        }
        Ok(())
    }
}

/// Deterministic hard-negative miner.
#[derive(Debug, Clone)]
pub struct HardNegativeMiner {
    cfg: HardNegativeConfig,
}

impl HardNegativeMiner {
    pub fn new(cfg: HardNegativeConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &HardNegativeConfig {
        &self.cfg
    }

    /// Mine negatives from `pool`.
    ///
    /// `positives` is the authoritative set of relevant ids: a candidate is
    /// skipped if it appears there *or* if `pool` flags it relevant, so a
    /// leaked or mis-flagged relevance label can never become a negative.
    /// `exclude_ids` removes ids that must not be trained on for other reasons
    /// (e.g. previously selected positives of another split).
    ///
    /// Ordering of the returned vector: by stratum, then hardness descending,
    /// then candidate id ascending.
    pub fn mine(
        &self,
        pool: &[PoolEntry],
        positives: &BTreeSet<u64>,
        exclude_ids: &BTreeSet<u64>,
    ) -> Result<Vec<HardNegative>> {
        self.cfg.validate()?;
        if self.cfg.max_negatives == 0 {
            return Ok(Vec::new());
        }

        // --- Candidate set: drop positives, excluded ids, and non-finite scores.
        let mut candidates: Vec<&PoolEntry> = Vec::with_capacity(pool.len());
        for e in pool {
            if e.is_positive || positives.contains(&e.id) || exclude_ids.contains(&e.id) {
                continue;
            }
            if !e.baseline_score.is_finite() {
                continue;
            }
            candidates.push(e);
        }
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Baseline threshold for the near-positive stratum: the weakest known
        // positive. Falls back to the strongest pool score when no positive
        // baseline is available, which keeps the stratum well defined.
        let weak_positive = pool
            .iter()
            .filter(|e| e.is_positive || positives.contains(&e.id))
            .filter(|e| e.baseline_score.is_finite())
            .map(|e| e.baseline_score)
            .fold(f32::INFINITY, f32::min);
        let threshold = if weak_positive.is_finite() {
            weak_positive
        } else {
            candidates
                .iter()
                .map(|e| e.baseline_score)
                .fold(f32::NEG_INFINITY, f32::max)
        };

        // --- Per-candidate derived statistics (computed once, deterministically).
        let stats: Vec<EntryStat> = candidates
            .iter()
            .map(|e| EntryStat {
                id: e.id,
                baseline: e.baseline_score,
                disagreement: e
                    .evidence
                    .head_similarity_variance(self.cfg.min_heads_for_disagreement),
                usable_heads: e.evidence.usable_head_count(),
                near_positive_hardness: 1.0 / (1.0 + (e.baseline_score - threshold).abs()),
            })
            .collect();

        // --- Rank each stratum (higher hardness first, id ascending on ties).
        let by_baseline = ranked(&stats, |s| s.baseline);
        let by_disagreement = ranked(&stats, |s| s.disagreement.unwrap_or(f32::NEG_INFINITY));
        let by_near_positive = ranked(&stats, |s| s.near_positive_hardness);

        // --- Quotas by largest remainder so the total is exact.
        let quotas = allocate_counts(self.cfg.max_negatives, self.cfg.fractions());

        let mut selected: Vec<HardNegative> = Vec::with_capacity(self.cfg.max_negatives);
        let mut taken: BTreeSet<u64> = BTreeSet::new();

        let strata: [(&[usize], NegSource); 4] = [
            (&by_baseline, NegSource::HighBaseline),
            (&by_disagreement, NegSource::HeadDisagreement),
            (&by_near_positive, NegSource::NearPositive),
            (
                &uniform_stride(&stats, self.cfg.max_negatives),
                NegSource::UniformUnion,
            ),
        ];

        for (order, source) in strata {
            let mut taken_in_stratum = 0usize;
            for &idx in order {
                if taken_in_stratum >= quotas[source.stratum() as usize] {
                    break;
                }
                let s = &stats[idx];
                if !taken.insert(s.id) {
                    continue;
                }
                selected.push(HardNegative {
                    candidate_id: s.id,
                    baseline_score: s.baseline,
                    source,
                    hardness: match source {
                        NegSource::HighBaseline | NegSource::UniformUnion => s.baseline,
                        NegSource::HeadDisagreement => s.disagreement.ok_or_else(|| {
                            crate::errors::AttentionError::NonFinite(
                                "disagreement stratum requires a defined variance".into(),
                            )
                        })?,
                        NegSource::NearPositive => s.near_positive_hardness,
                    },
                    disagreement: s.disagreement,
                    usable_heads: s.usable_heads,
                });
                taken_in_stratum += 1;
            }
        }

        // --- Backfill so `max_negatives` is always met when the pool allows,
        // in hardest-baseline order (keeps stratum identity honest: backfilled
        // entries are labelled HighBaseline because that is how they ranked).
        if selected.len() < self.cfg.max_negatives {
            for &idx in &by_baseline {
                if selected.len() >= self.cfg.max_negatives {
                    break;
                }
                let s = &stats[idx];
                if !taken.insert(s.id) {
                    continue;
                }
                selected.push(HardNegative {
                    candidate_id: s.id,
                    baseline_score: s.baseline,
                    source: NegSource::HighBaseline,
                    hardness: s.baseline,
                    disagreement: s.disagreement,
                    usable_heads: s.usable_heads,
                });
            }
        }

        selected.sort_by(|a, b| {
            a.source
                .stratum()
                .cmp(&b.source.stratum())
                .then_with(|| {
                    b.hardness
                        .partial_cmp(&a.hardness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.candidate_id.cmp(&b.candidate_id))
        });
        Ok(selected)
    }

    /// Stable FNV-1a fingerprint of a mined negative set (dataset provenance).
    pub fn fingerprint(negatives: &[HardNegative]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        let mut push = |bytes: &[u8]| {
            for b in bytes {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        for n in negatives {
            push(&n.candidate_id.to_le_bytes());
            push(&n.baseline_score.to_le_bytes());
            push(&[n.source.stratum()]);
        }
        hash
    }
}

/// Per-candidate derived statistics used by every stratum.
struct EntryStat {
    id: u64,
    baseline: f32,
    disagreement: Option<f32>,
    usable_heads: usize,
    near_positive_hardness: f32,
}

/// Indices into the candidate list sorted by `key` descending, ties by id
/// ascending. Indices with a non-finite key are dropped (never sorted as 0).
fn ranked(stats: &[EntryStat], key: impl Fn(&EntryStat) -> f32) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..stats.len()).collect();
    idx.retain(|&i| key(&stats[i]).is_finite());
    idx.sort_by(|&a, &b| {
        key(&stats[b])
            .partial_cmp(&key(&stats[a]))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| stats[a].id.cmp(&stats[b].id))
    });
    idx
}

/// Deterministic uniform stride sample of the id-sorted candidate list.
///
/// A stride rather than an RNG draw: the stratum must cover the whole union and
/// must be reproducible byte-for-byte across processes and platforms.
fn uniform_stride(stats: &[EntryStat], want: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..stats.len()).collect();
    ids.sort_by_key(|&i| stats[i].id);
    let n = ids.len();
    let take = want.min(n);
    if take == 0 {
        return Vec::new();
    }
    if take == n {
        return ids;
    }
    (0..take)
        .map(|i| {
            // Centered stride keeps the sample spread over the id space.
            let pos = (2 * i + 1) * n / (2 * take);
            ids[pos.min(n - 1)]
        })
        .collect()
}

/// Largest-remainder allocation of `total` items across `fractions`.
/// Deterministic; ties in the remainder go to the lower index.
fn allocate_counts(total: usize, fractions: [f32; 4]) -> [usize; 4] {
    let mut counts = [0usize; 4];
    let t = total as f32;
    let mut remainders: Vec<(usize, f32)> = Vec::with_capacity(4);
    let mut assigned = 0usize;
    for (i, &f) in fractions.iter().enumerate() {
        let exact = t * f;
        let floor = exact.floor();
        let c = floor as usize;
        counts[i] = c;
        assigned += c;
        remainders.push((i, exact - floor));
    }
    remainders.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut leftover = total.saturating_sub(assigned);
    for (i, _) in remainders {
        if leftover == 0 {
            break;
        }
        counts[i] += 1;
        leftover -= 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(sims: &[Option<f32>]) -> RetrievalEvidence {
        RetrievalEvidence {
            head_sims: sims.to_vec(),
            head_ranks: vec![Some(0.5); sims.len()],
            head_present: sims.iter().map(|s| s.is_some()).collect(),
        }
    }

    fn pool(n: u64, positives: &[u64]) -> Vec<PoolEntry> {
        (0..n)
            .map(|i| PoolEntry {
                id: i,
                // Descending baseline so ids and baseline scores agree in order.
                baseline_score: 1.0 - (i as f32 / n as f32),
                evidence: evidence(&[
                    Some(1.0 - (i as f32 / n as f32)),
                    Some(0.1 * (i as f32 / n as f32)),
                ]),
                is_positive: positives.contains(&i),
            })
            .collect()
    }

    #[test]
    fn excludes_positives_always() {
        let miner = HardNegativeMiner::new(HardNegativeConfig {
            max_negatives: 8,
            ..Default::default()
        });
        let positives: BTreeSet<u64> = [0u64, 1, 2].into_iter().collect();
        let negatives = miner
            .mine(&pool(40, &[0, 1, 2]), &positives, &BTreeSet::new())
            .unwrap();
        assert!(!negatives.is_empty());
        for n in &negatives {
            assert!(
                !positives.contains(&n.candidate_id),
                "leaked positive {}",
                n.candidate_id
            );
        }
    }

    #[test]
    fn exclude_ids_also_filtered() {
        let miner = HardNegativeMiner::new(HardNegativeConfig {
            max_negatives: 6,
            ..Default::default()
        });
        let excluded: BTreeSet<u64> = [3u64, 4, 5].into_iter().collect();
        let negatives = miner
            .mine(&pool(40, &[]), &BTreeSet::new(), &excluded)
            .unwrap();
        for n in &negatives {
            assert!(!excluded.contains(&n.candidate_id));
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let cfg = HardNegativeConfig {
            max_negatives: 8,
            ..Default::default()
        };
        let positives: BTreeSet<u64> = [0u64].into_iter().collect();
        let a = HardNegativeMiner::new(cfg.clone())
            .mine(&pool(50, &[0]), &positives, &BTreeSet::new())
            .unwrap();
        let b = HardNegativeMiner::new(cfg)
            .mine(&pool(50, &[0]), &positives, &BTreeSet::new())
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(
            HardNegativeMiner::fingerprint(&a),
            HardNegativeMiner::fingerprint(&b)
        );
    }

    #[test]
    fn honors_max_negatives_and_no_duplicates() {
        let miner = HardNegativeMiner::new(HardNegativeConfig {
            max_negatives: 7,
            ..Default::default()
        });
        let negatives = miner
            .mine(&pool(30, &[0]), &BTreeSet::new(), &BTreeSet::new())
            .unwrap();
        assert!(negatives.len() <= 7);
        let ids: BTreeSet<u64> = negatives.iter().map(|n| n.candidate_id).collect();
        assert_eq!(ids.len(), negatives.len(), "duplicate negative ids");
    }

    #[test]
    fn disagreement_stratum_is_used_when_heads_agree_variably() {
        let miner = HardNegativeMiner::new(HardNegativeConfig {
            max_negatives: 4,
            high_baseline_fraction: 0.0,
            head_disagreement_fraction: 1.0,
            near_positive_fraction: 0.0,
            uniform_union_fraction: 0.0,
            min_heads_for_disagreement: 2,
        });
        let entries = vec![
            PoolEntry {
                id: 10,
                baseline_score: 0.9,
                evidence: evidence(&[Some(1.0), Some(1.0)]),
                is_positive: false,
            },
            PoolEntry {
                id: 11,
                baseline_score: 0.1,
                evidence: evidence(&[Some(1.0), Some(0.0)]),
                is_positive: false,
            },
        ];
        let negatives = miner
            .mine(&entries, &BTreeSet::new(), &BTreeSet::new())
            .unwrap();
        // The disagreement stratum is quota 4 but the pool only holds two
        // candidates, so backfill returns both; the ranking is what matters.
        assert_eq!(negatives.len(), 2);
        assert_eq!(
            negatives[0].candidate_id, 11,
            "high variance must rank first"
        );
        assert_eq!(negatives[0].source, NegSource::HeadDisagreement);
        assert!(negatives[0].disagreement.unwrap() > 0.2);
        // The agreeing candidate is still mined, but not via the stratum.
        assert_eq!(negatives[1].candidate_id, 10);
        assert!(negatives[1].disagreement.unwrap().abs() < 1e-9);
    }

    #[test]
    fn missing_heads_are_not_imputed_as_zero() {
        // Two candidates, identical mean over present heads; the one with a
        // missing head must NOT be credited with zero-similarity variance.
        let dense = PoolEntry {
            id: 1,
            baseline_score: 0.5,
            evidence: evidence(&[Some(0.5), Some(0.5)]),
            is_positive: false,
        };
        let sparse = PoolEntry {
            id: 2,
            baseline_score: 0.5,
            evidence: evidence(&[Some(0.5), None]),
            is_positive: false,
        };
        assert!(dense.evidence.head_similarity_variance(2).unwrap().abs() < 1e-9);
        assert!(sparse.evidence.head_similarity_variance(2).is_none());
        assert_eq!(dense.evidence.usable_head_count(), 2);
        assert_eq!(sparse.evidence.usable_head_count(), 1);
    }

    #[test]
    fn strata_appear_in_priority_order() {
        let negatives = HardNegativeMiner::new(HardNegativeConfig {
            max_negatives: 10,
            ..Default::default()
        })
        .mine(&pool(20, &[0]), &BTreeSet::new(), &BTreeSet::new())
        .unwrap();
        let mut last = 0u8;
        for n in &negatives {
            let s = n.source.stratum();
            assert!(s >= last, "strata must be emitted in priority order");
            last = s;
        }
        // Default fractions (50/20/20/10) over 10 slots.
        let counts: Vec<usize> = NegSource::ordered()
            .iter()
            .map(|src| negatives.iter().filter(|n| n.source == *src).count())
            .collect();
        assert_eq!(counts[0], 5, "high-baseline quota");
        assert_eq!(counts[1], 2, "disagreement quota");
        assert_eq!(counts[2], 2, "near-positive quota");
        assert_eq!(counts[3], 1, "uniform quota");
    }

    #[test]
    fn rejects_fractions_that_do_not_sum_to_one() {
        let miner = HardNegativeMiner::new(HardNegativeConfig {
            max_negatives: 4,
            high_baseline_fraction: 0.5,
            head_disagreement_fraction: 0.2,
            near_positive_fraction: 0.2,
            uniform_union_fraction: 0.2,
            ..Default::default()
        });
        assert!(miner
            .mine(&pool(10, &[]), &BTreeSet::new(), &BTreeSet::new())
            .is_err());
    }

    #[test]
    fn empty_pool_yields_nothing() {
        let miner = HardNegativeMiner::new(HardNegativeConfig::default());
        assert!(miner
            .mine(&[], &BTreeSet::new(), &BTreeSet::new())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn all_positive_pool_yields_nothing() {
        let positives: BTreeSet<u64> = (0..10).collect();
        let negatives = HardNegativeMiner::new(HardNegativeConfig::default())
            .mine(&pool(10, &[]), &positives, &BTreeSet::new())
            .unwrap();
        assert!(negatives.is_empty());
    }
}
