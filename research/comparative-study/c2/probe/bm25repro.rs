// BM25 tie-order nondeterminism reproduction (TEST-C2-008 leg).
// Builds a Bm25Index over docs engineered so a query yields strictly tied
// scores, then observes the returned top-k doc-id order across repeated
// calls. Prints observations; RC 0 always. Nondeterminism here is the
// engine behavior under audit, not a harness failure.
use attentiondb_core::bm25::{reciprocal_rank_fusion, Bm25Index};

const N_DOCS: u64 = 12;
const TOP_K: usize = 4;
const SAMPLES: usize = 200;

fn tied_docs() -> Bm25Index {
    let ix = Bm25Index::default();
    let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima";
    for d in 0..N_DOCS {
        ix.insert(d, text);
    }
    ix
}

fn main() {
    let json_path = std::env::args().nth(1);
    let ix = tied_docs();

    let q1 = "alpha";
    let mut first = None;
    let mut orderings: std::collections::BTreeSet<Vec<u64>> = std::collections::BTreeSet::new();
    let mut score_ties_observed = true;
    for _i in 0..SAMPLES {
        let res = ix.search(q1, TOP_K);
        if res.len() != TOP_K {
            score_ties_observed = false;
        }
        if res.windows(2).any(|w| w[0].1 != w[1].1) {
            score_ties_observed = false;
        }
        let ids: Vec<u64> = res.iter().map(|x| x.0).collect();
        if first.is_none() {
            first = Some(ids.clone());
        }
        orderings.insert(ids);
    }
    println!("search('{}', top_k={}) over {} tied docs, {} samples", q1, TOP_K, N_DOCS, SAMPLES);
    println!("strict_score_ties_observed: {}", score_ties_observed);
    println!("first_call_doc_order: {:?}", first.unwrap());
    println!("distinct_doc_orderings_seen: {}", orderings.len());
    for (oi, o) in orderings.iter().enumerate() {
        println!("  ordering[{}]: {:?}", oi, o);
    }

    let q2 = "alpha bravo";
    let mut sets: std::collections::BTreeSet<Vec<u64>> = std::collections::BTreeSet::new();
    for _ in 0..SAMPLES {
        let res = ix.search(q2, TOP_K);
        sets.insert(res.iter().map(|x| x.0).collect());
    }
    println!();
    println!("search('{}', top_k={}): distinct_doc_orderings_seen: {}", q2, TOP_K, sets.len());
    for (oi, o) in sets.iter().enumerate() {
        println!("  ordering[{}]: {:?}", oi, o);
    }

    // RRF propagation: sparse ranks come from a tied engine search, so the
    // fused outcome depends on the nondeterministic sparse order.
    let dense: Vec<(u64, f32)> = (0..N_DOCS).rev().map(|d| (d, 100.0 - d as f32)).collect();
    let mut fused_membership: std::collections::BTreeSet<Vec<u64>> = std::collections::BTreeSet::new();
    let mut fused_top1: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for _ in 0..SAMPLES {
        let sparse = ix.search(q1, N_DOCS as usize);
        let fused = reciprocal_rank_fusion(&dense, &sparse, TOP_K);
        fused_membership.insert(fused.iter().map(|x| x.0).collect());
        if let Some((id, _)) = fused.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) {
            fused_top1.insert(*id);
        }
    }
    println!();
    println!("reciprocal_rank_fusion(dense, tied_search) fused outputs:");
    println!(" distinct_fused_topk_memberships: {}", fused_membership.len());
    println!(" distinct_fused_max_score_ids: {:?}", fused_top1.iter().collect::<Vec<_>>());
    for (oi, o) in fused_membership.iter().enumerate() {
        println!("  fused[{}]: {:?}", oi, o);
    }

    if let Some(p) = json_path {
        let summary = serde_json::json!({
            "search_distinct_orderings_seen": orderings.len(),
            "phrase_search_distinct_orderings_seen": sets.len(),
            "strict_score_ties_observed": score_ties_observed,
            "rrf_distinct_fused_topk_memberships": fused_membership.len(),
            "rrf_distinct_fused_max_score_ids": fused_top1.iter().copied().collect::<Vec<_>>(),
            "n_samples": SAMPLES,
            "n_docs": N_DOCS,
            "top_k": TOP_K,
        });
        std::fs::write(p, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    }
}