//! C2 regression pin for the BM25 tie-order defect (TEST-C2-008 leg B5).
//!
//! Pre-fix: `search`/`search_phrase` ranked strictly tied BM25 scores with a
//! stable sort over a per-call HashMap, so the top-k doc-id ORDER (and RRF's
//! fused top-k membership / max-score id) varied per call. The contract is
//! "score desc, id asc". This file pins: repeated calls over tied docs must
//! return exactly one ordering, and `reciprocal_rank_fusion` must be
//! deterministic on identical inputs.

use attentiondb_core::bm25::{reciprocal_rank_fusion, Bm25Index};

const N_DOCS: u64 = 12;
const TOP_K: usize = 4;
const SAMPLES: usize = 100;

fn tied_docs() -> Bm25Index {
    let ix = Bm25Index::default();
    let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima";
    for d in 0..N_DOCS {
        ix.insert(d, text);
    }
    ix
}

#[test]
fn search_tie_order_is_deterministic_score_desc_id_asc() {
    let ix = tied_docs();
    let mut orderings: Vec<Vec<u64>> = Vec::new();
    let mut all_scores_tied = true;
    for _ in 0..SAMPLES {
        let res = ix.search("alpha", TOP_K);
        assert_eq!(res.len(), TOP_K);
        if res.windows(2).any(|w| w[0].1 != w[1].1) {
            all_scores_tied = false;
        }
        orderings.push(res.iter().map(|x| x.0).collect());
    }
    assert!(all_scores_tied, "scores must be strictly tied for this stimulus");
    let first = orderings[0].clone();
    for o in orderings.iter().skip(1) {
        assert_eq!(&first, o, "tied-score doc order must be stable across calls");
    }
    let mut expected: Vec<u64> = (0..N_DOCS).collect();
    expected.truncate(TOP_K);
    assert_eq!(first, expected, "tie rule must be score desc, id asc");
}

#[test]
fn search_phrase_tie_order_is_deterministic() {
    let ix = tied_docs();
    let mut orderings: Vec<Vec<u64>> = Vec::new();
    for _ in 0..SAMPLES {
        let res = ix.search_phrase("alpha bravo", TOP_K, 5);
        orderings.push(res.iter().map(|x| x.0).collect());
    }
    let first = orderings[0].clone();
    for o in orderings.iter().skip(1) {
        assert_eq!(&first, o, "phrase tie-order must be stable across calls");
    }
    let mut expected: Vec<u64> = (0..N_DOCS).collect();
    expected.truncate(TOP_K);
    assert_eq!(first, expected, "phrase tie rule must be score desc, id asc");
}

#[test]
fn reciprocal_rank_fusion_is_deterministic_and_id_asc_on_ties() {
    let ix = tied_docs();
    let dense: Vec<(u64, f32)> = (0..N_DOCS).rev().map(|d| (d, 100.0 - d as f32)).collect();
    let sparse = ix.search("alpha", N_DOCS as usize);
    let first = reciprocal_rank_fusion(&dense, &sparse, TOP_K);
    for _ in 0..SAMPLES {
        let sp = ix.search("alpha", N_DOCS as usize);
        let fused = reciprocal_rank_fusion(&dense, &sp, TOP_K);
        assert_eq!(first, fused, "RRF training order instabilities must not resurface");
    }
    let ids: Vec<u64> = first.iter().map(|x| x.0).collect();
    // fused(d) = 1/(12-d) + 1/(d+1) pairs {0,11}, {1,10}, {2,9}, {3,8} on ties:
    // top-4 by fused score desc with id-asc tie rule = [0,11,1,10].
    assert_eq!(ids, vec![0, 11, 1, 10], "RRF tie rule must be fused score desc, id asc");
}