//! Sanity probe: is low measured recall a retrieval failure or a GT/decode
//! artifact? Prints the cluster distribution of returned docs vs the query's
//! cluster, plus raw scores, for a handful of queries.

use attentiondb_core::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

const DIM: usize = 32;
const N_DOCS: usize = 10_000;
const N_CLUSTERS: usize = 100;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn gauss(&mut self) -> f32 {
        (self.next_f32() + self.next_f32() + self.next_f32() + self.next_f32()) * 0.866
    }
}

fn main() {
    // regenerate the exact same corpus RNG sequence as main.rs
    let mut rng = Rng::new(0xC0FFEE);
    let mut centroids = Vec::new();
    for _ in 0..N_CLUSTERS {
        let mut c: Vec<f32> = (0..DIM).map(|_| rng.gauss()).collect();
        let n: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in c.iter_mut() {
            *x /= n;
        }
        centroids.push(c);
    }
    let head_sigmas = vec![0.15f32, 0.35, 0.6, 0.9];
    let mut doc_cluster = Vec::with_capacity(N_DOCS);
    let mut queries = Vec::new();
    for i in 0..N_DOCS {
        let c = (rng.next_u64() as usize) % N_CLUSTERS;
        doc_cluster.push(c);
        if i < 100 {
            let mut q = centroids[c].clone();
            for x in q.iter_mut() {
                *x += rng.gauss() * 0.1;
            }
            let n: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in q.iter_mut() {
                *x /= n;
            }
            queries.push(q);
        }
    }

    // build ONLY head-0 docs for a small corpus and query it — with cluster labels
    let dir = std::env::temp_dir().join(format!("probe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    e.create_collection("bench", DIM, &["default"]).unwrap();
    let mut rng2 = Rng::new(0xC0FFEE ^ 0xB10B);
    // consume RNG identically to main.rs (4 heads × docs): regenerate all
    for i in 0..N_DOCS {
        let mut head_vecs: Vec<Vec<f32>> = Vec::new();
        for sigma in &head_sigmas {
            let mut v: Vec<f32> = centroids[doc_cluster[i]]
                .iter()
                .map(|x| x + rng2.gauss() * sigma)
                .collect();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= n;
            }
            head_vecs.push(v);
        }
        if i >= 2000 {
            continue; // small corpus for the probe; vectors discarded
        }
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        fields.insert(
            "cluster".to_string(),
            serde_json::json!(doc_cluster[i] as i64),
        );
        let mut r = Record::new(fields);
        r.k_vecs.insert("default".to_string(), head_vecs[0].clone());
        e.insert_document("bench", r).unwrap();
    }
    println!("probe corpus: 2000 docs, head0 sigma=0.15");
    let mut in_cluster = 0usize;
    for qi in 0..10 {
        let q = &queries[qi];
        let res = e
            .get_collection("bench")
            .unwrap()
            .attend(&["default".to_string()], q, 10)
            .unwrap();
        let mut desc = String::new();
        let mut hits = 0;
        for (id, s) in &res {
            let f = e.get_document_fields(*id);
            let cl: i64 = f.get("cluster").and_then(|v| v.parse().ok()).unwrap_or(-1);
            if cl == doc_cluster[qi] as i64 {
                hits += 1;
            }
            desc.push_str(&format!(" id={id} cl={cl} s={s:.3}"));
        }
        println!(
            "q{qi} (cluster {}) hits_in_cluster={hits}/10:{desc}",
            doc_cluster[qi]
        );
        in_cluster += hits;
    }
    println!("TOTAL in-cluster hits: {in_cluster}/100");
}
