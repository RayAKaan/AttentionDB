//! Phase 2B corpora (§8–9, §34–37). All generators are deterministic for a
//! given seed. Each corpus defines: collection dim, head names, docs with
//! per-head vectors, queries with per-view query representations, and ground
//! truth from data-generating vectors (exact cosine, same method as Phase 2).

use attentiondb_core::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn f32_01(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn gauss(&mut self) -> f32 {
        // sum of 4 uniforms, centered: σ≈0.577 — scaled by callers
        (self.f32_01() + self.f32_01() + self.f32_01() + self.f32_01() - 2.0) * 0.866
    }
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_u64() % (i + 1) as u64) as usize;
            xs.swap(i, j);
        }
    }
}

pub fn l2normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 1e-20 || nb <= 1e-20 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// A query in a corpus: `vectors` holds one representation per head (§34:
/// genuinely multi-view corpora give DIFFERENT representations per view; §36:
/// the gating input is the concatenation of these — never the type label).
pub struct BenchQuery {
    pub id: u64,
    pub group: u32,
    /// representation per head (same order as head_names)
    pub vectors: Vec<Vec<f32>>,
    /// gating input (for single-view corpora: the query itself)
    pub gating_input: Vec<f32>,
    /// ground truth: doc ids best-first
    pub ground_truth: Vec<u64>,
}

pub struct BenchDoc {
    pub numeric_hint: u64,
    pub fields: HashMap<String, serde_json::Value>,
    /// vector per head (same order as head_names)
    pub head_vecs: Vec<Vec<f32>>,
    /// true data-generating vector (ground-truth scoring)
    pub true_vec: Vec<f32>,
}

pub struct Corpus {
    pub name: &'static str,
    pub dim: usize,
    pub head_names: Vec<String>,
    pub docs: Vec<BenchDoc>,
    pub queries: Vec<BenchQuery>,
}

fn open_engine() -> (tempdir::TempDirGuard, AttentionEngine) {
    let dir = std::env::temp_dir().join(format!("phase2b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    (tempdir::TempDirGuard(dir), e)
}

mod tempdir {
    pub struct TempDirGuard(pub std::path::PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// Insert a corpus into a fresh engine; returns numeric ids in doc order.
pub fn insert_corpus(corpus: &Corpus) -> (tempdir::TempDirGuard, AttentionEngine, Vec<u64>) {
    let (guard, e) = open_engine();
    let heads: Vec<&str> = corpus.head_names.iter().map(|s| s.as_str()).collect();
    e.create_collection("bench", corpus.dim, &heads).unwrap();
    let mut ids = Vec::with_capacity(corpus.docs.len());
    for doc in &corpus.docs {
        let mut r = Record::new(doc.fields.clone());
        for (h, v) in doc.head_vecs.iter().enumerate() {
            r.k_vecs.insert(corpus.head_names[h].clone(), v.clone());
        }
        e.insert_document("bench", r).unwrap();
        ids.push(doc.numeric_hint);
    }
    (guard, e, ids)
}

// ---------------------------------------------------------------------------
// Corpus 1 — CONTROLLED (§8/§9): best head = f(query group)
// ---------------------------------------------------------------------------

/// `groups` query groups; group g's docs are clean ONLY in head g (others
/// noisy). Gating input = the query vector itself (single view). Known
/// optimal head per query by construction.
pub fn controlled(
    n_groups: usize,
    clusters_per_group: usize,
    n_queries: usize,
    seed: u64,
) -> Corpus {
    let dim = 32usize;
    let heads: Vec<String> = (0..n_groups).map(|g| format!("head{g}")).collect();
    let mut rng = Rng::new(seed);
    let mut docs = Vec::new();
    let mut queries = Vec::new();
    let mut id = 0u64;

    // centroids per group: block-g signal
    let block = dim / n_groups.max(1);
    let centroids: Vec<Vec<Vec<f32>>> = (0..n_groups)
        .map(|g| {
            (0..clusters_per_group)
                .map(|_| {
                    let mut v: Vec<f32> = (0..dim)
                        .map(|d| {
                            if d / block == g {
                                rng.gauss() * 0.5 + 0.6
                            } else {
                                rng.gauss() * 0.1
                            }
                        })
                        .collect();
                    l2normalize(&mut v);
                    v
                })
                .collect()
        })
        .collect();

    for (g, group_centroids) in centroids.iter().enumerate() {
        for (c, centroid) in group_centroids.iter().enumerate() {
            for _doc in 0..8 {
                // true vector = centroid + distinct direction (well-defined GT)
                let mut u: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
                l2normalize(&mut u);
                let mut tv: Vec<f32> = centroid
                    .iter()
                    .zip(u.iter())
                    .map(|(a, b)| a + 0.35 * b)
                    .collect();
                l2normalize(&mut tv);
                let mut head_vecs = Vec::with_capacity(n_groups);
                for h in 0..n_groups {
                    let sigma = if h == g { 0.02f32 } else { 0.30 };
                    let mut v: Vec<f32> = tv.iter().map(|x| x + rng.gauss() * sigma).collect();
                    l2normalize(&mut v);
                    head_vecs.push(v);
                }
                let mut fields = HashMap::new();
                fields.insert("idx".to_string(), serde_json::json!(id));
                fields.insert("group".to_string(), serde_json::json!(g as i64));
                fields.insert("cluster".to_string(), serde_json::json!(c as i64));
                docs.push(BenchDoc {
                    numeric_hint: id,
                    fields,
                    head_vecs,
                    true_vec: tv,
                });
                id += 1;
            }
        }
    }

    // queries: noised centroids; GT = exact cosine over true vectors
    for qi in 0..n_queries {
        let g = qi % n_groups;
        let c = (qi / n_groups) % clusters_per_group;
        let mut q: Vec<f32> = centroids[g][c]
            .iter()
            .map(|x| x + rng.gauss() * 0.05)
            .collect();
        l2normalize(&mut q);
        let mut scored: Vec<(u64, f32)> = docs
            .iter()
            .map(|d| (d.numeric_hint, cosine(&d.true_vec, &q)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let gt: Vec<u64> = scored.iter().take(10).map(|x| x.0).collect();
        queries.push(BenchQuery {
            id: qi as u64,
            group: g as u32,
            vectors: vec![q.clone(); n_groups],
            gating_input: q,
            ground_truth: gt,
        });
    }

    Corpus {
        name: "controlled",
        dim,
        head_names: heads,
        docs,
        queries,
    }
}

// ---------------------------------------------------------------------------
// Corpus 2 — PHASE 2 NOISE (§7): one globally dominant head
// ---------------------------------------------------------------------------

/// Same shape as the accepted Phase 2 ablation corpus (σ ladder ⇒ head 0
/// globally best). Purpose: measure what gating learns when there is NO
/// query-conditional signal (collapse investigation).
pub fn noise_ladder(n_queries: usize, seed: u64) -> Corpus {
    let dim = 32usize;
    let n_heads = 8usize;
    let sigmas = [0.02f32, 0.04, 0.07, 0.12, 0.05, 0.08, 0.11, 0.14];
    let heads: Vec<String> = (0..n_heads).map(|h| format!("head{h}")).collect();
    let mut rng = Rng::new(seed);
    let n_clusters = 100usize;
    let n_docs = 10_000usize;

    let mut centroids = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let mut c: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
        l2normalize(&mut c);
        centroids.push(c);
    }
    let mut docs = Vec::with_capacity(n_docs);
    let mut doc_cluster = Vec::with_capacity(n_docs);
    let mut queries = Vec::new();
    for i in 0..n_docs {
        let c = (rng.next_u64() as usize) % n_clusters;
        doc_cluster.push(c);
        let mut u: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
        l2normalize(&mut u);
        let mut tv: Vec<f32> = centroids[c]
            .iter()
            .zip(u.iter())
            .map(|(a, b)| a + 0.35 * b)
            .collect();
        l2normalize(&mut tv);
        let mut head_vecs = Vec::with_capacity(n_heads);
        for sigma in sigmas.iter() {
            let mut v: Vec<f32> = tv.iter().map(|x| x + rng.gauss() * sigma).collect();
            l2normalize(&mut v);
            head_vecs.push(v);
        }
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i as u64));
        fields.insert("cluster".to_string(), serde_json::json!(c as i64));
        docs.push(BenchDoc {
            numeric_hint: i as u64,
            fields,
            head_vecs,
            true_vec: tv,
        });
        if i < n_queries {
            let q: Vec<f32> = centroids[c].iter().map(|x| x + rng.gauss() * 0.1).collect();
            // GT is filled after the doc loop (needs the full corpus)
            queries.push(BenchQuery {
                id: i as u64,
                group: 0,
                vectors: vec![q.clone(); n_heads],
                gating_input: q.clone(),
                ground_truth: Vec::new(),
            });
            queries.last_mut().unwrap().vectors = vec![q; n_heads];
        }
    }
    // fill GT now that all docs exist
    for q in queries.iter_mut() {
        let qv = q.vectors[0].clone();
        let mut scored: Vec<(u64, f32)> = docs
            .iter()
            .map(|d| (d.numeric_hint, cosine(&d.true_vec, &qv)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        q.ground_truth = scored.iter().take(10).map(|x| x.0).collect();
    }

    Corpus {
        name: "noise_ladder",
        dim,
        head_names: heads,
        docs,
        queries,
    }
}

// ---------------------------------------------------------------------------
// Corpus 3 — MULTI-VIEW (§34–37): genuinely different views
// ---------------------------------------------------------------------------

/// Three views over the same docs (§34–37), fine-grained:
/// every view vector = coarse topic part + doc-UNIQUE random part, so only
/// the matching view can identify the exact target doc. A type-V query is
/// built from doc d's view-V representation (+noise); ground truth = exact
/// cosine over that view's doc vectors — the same GT rule as the other
/// corpora, applied per view. The best head for a type-V query is view V by
/// CONSTRUCTION; the other heads see only the coarse topic. Gating input =
/// concatenation of all three query representations (§36: no label input).
pub fn multiview(n_queries: usize, seed: u64) -> Corpus {
    const VIEW_DIM: usize = 64;
    const N_TOPICS: usize = 40;
    const VOCAB: usize = 400;
    const N_CATS: usize = 8;

    let dim = VIEW_DIM;
    let head_names = vec![
        "semantic".to_string(),
        "lexical".to_string(),
        "metadata".to_string(),
    ];
    let mut rng = Rng::new(seed);
    let n_docs = 3_000usize;

    let mut topics = Vec::with_capacity(N_TOPICS);
    for _ in 0..N_TOPICS {
        let mut v: Vec<f32> = (0..VIEW_DIM).map(|_| rng.gauss()).collect();
        l2normalize(&mut v);
        topics.push(v);
    }
    let topic_words: Vec<Vec<usize>> = (0..N_TOPICS)
        .map(|t| {
            let start = (t * 9) % (VOCAB - 10);
            (start..start + 10).collect()
        })
        .collect();
    let topic_cat: Vec<usize> = (0..N_TOPICS).map(|t| t % N_CATS).collect();
    let hash = |word: usize, salt: u64| -> usize {
        let mut x = (word as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51AFD7ED558CCD);
        (x >> 32) as usize % VIEW_DIM
    };

    let mut docs = Vec::with_capacity(n_docs);
    for i in 0..n_docs {
        let t = i % N_TOPICS;
        // doc-unique directions (the fine-grained identity per view)
        let mut u_sem: Vec<f32> = (0..VIEW_DIM).map(|_| rng.gauss()).collect();
        l2normalize(&mut u_sem);
        let mut u_lex: Vec<f32> = (0..VIEW_DIM).map(|_| rng.gauss()).collect();
        l2normalize(&mut u_lex);
        let mut u_meta: Vec<f32> = (0..VIEW_DIM).map(|_| rng.gauss()).collect();
        l2normalize(&mut u_meta);
        // semantic: topic + unique
        let mut sem: Vec<f32> = topics[t]
            .iter()
            .zip(u_sem.iter())
            .map(|(a, b)| a + 0.5 * b)
            .collect();
        l2normalize(&mut sem);
        // lexical: hashed topic words + unique lexical identity
        let mut lex = vec![0.0f32; VIEW_DIM];
        for &w in &topic_words[t] {
            lex[hash(w, 1)] += 1.0;
        }
        for d in 0..VIEW_DIM {
            lex[d] += 0.6 * u_lex[d];
        }
        l2normalize(&mut lex);
        // metadata: category one-hot + unique structured part
        let mut meta = vec![0.0f32; VIEW_DIM];
        meta[topic_cat[t]] = 1.0;
        for d in 0..VIEW_DIM {
            meta[d] += 0.6 * u_meta[d];
        }
        l2normalize(&mut meta);

        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i as u64));
        fields.insert("topic".to_string(), serde_json::json!(t as i64));
        fields.insert(
            "category".to_string(),
            serde_json::json!(topic_cat[t] as i64),
        );
        docs.push(BenchDoc {
            numeric_hint: i as u64,
            fields,
            head_vecs: vec![sem, lex, meta],
            true_vec: Vec::new(), // unused: GT is view-defined
        });
    }

    let mut queries = Vec::with_capacity(n_queries);
    for qi in 0..n_queries {
        let qtype = qi % 3;
        // target a SPECIFIC doc; the query arrives IN ONE MODALITY (§34):
        // view qtype gets the doc-level (fine) representation; the OTHER views
        // only get the coarse topic-level projection — what a cross-modal
        // extractor could infer. Only the matching head can rank the target's
        // fine-grained neighborhood; that asymmetry IS the learning signal.
        let target = (qi * 7919) % n_docs;
        let t = target % N_TOPICS;
        let mut q_views: Vec<Vec<f32>> = Vec::with_capacity(3);
        for v in 0..3 {
            let mut q = if v == qtype {
                docs[target].head_vecs[v]
                    .iter()
                    .map(|x| x + rng.gauss() * 0.03)
                    .collect::<Vec<f32>>()
            } else {
                match v {
                    0 => topics[t]
                        .iter()
                        .map(|x| x + rng.gauss() * 0.05)
                        .collect::<Vec<f32>>(),
                    1 => {
                        let mut lx = vec![0.0f32; VIEW_DIM];
                        for &w in &topic_words[t] {
                            lx[hash(w, 1)] += 1.0;
                        }
                        for v in lx.iter_mut() {
                            *v += rng.gauss() * 0.05;
                        }
                        lx
                    }
                    _ => {
                        let mut mx = vec![0.0f32; VIEW_DIM];
                        mx[topic_cat[t]] = 1.0;
                        for v in mx.iter_mut() {
                            *v += rng.gauss() * 0.05;
                        }
                        mx
                    }
                }
            };
            l2normalize(&mut q);
            q_views.push(q);
        }
        // GT in the QUERY'S view
        let mut scored: Vec<(u64, f32)> = docs
            .iter()
            .map(|d| (d.numeric_hint, cosine(&d.head_vecs[qtype], &q_views[qtype])))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let gt: Vec<u64> = scored.iter().take(10).map(|x| x.0).collect();

        queries.push(BenchQuery {
            id: qi as u64,
            group: qtype as u32,
            vectors: q_views.clone(),
            gating_input: q_views.concat(),
            ground_truth: gt,
        });
    }

    Corpus {
        name: "multiview",
        dim,
        head_names,
        docs,
        queries,
    }
}
