use attentiondb_hnsw::HNSWConfig;
use attentiondb_multihead::{HeadConfig, HeadType, MultiHeadManager};
use attentiondb_query::parse_aql;
use std::collections::HashMap;
use std::time::Instant;

fn generate_distinct(seed: u64, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|x| {
            let f = (seed as f32) * 0.1 + (x as f32) * 0.3;
            f.sin() * (seed % 7 + 1) as f32 * 0.15
        })
        .collect();
    let cluster = (seed % 10) as usize;
    for i in (cluster..dim).step_by(10) {
        v[i] += 0.5;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           AttentionDB Comprehensive Demo (Per-Head Settings)               ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    // Step 1: Parse per-head settings from AQL
    println!("▶ Step 1: Parsing per-head settings from AQL...\n");

    let create_stmt = r#"
        CREATE COLLECTION papers (
            title TEXT,
            abstract TEXT,
            year INT
        ) WITH (
            ef_search = 128,
            semantic.ef_search = 256,
            semantic.max_connections = 32,
            temporal.ef_search = 64,
            temporal.max_connections = 16
        )
    "#;

    let (collection_name, head_settings) = match parse_aql(create_stmt) {
        Ok(attentiondb_query::AQLStatement::CreateCollection(c)) => {
            println!("   Collection '{}' parsed successfully", c.collection);
            println!("   Global settings: ef_search={}", c.settings.ef_search);
            println!("\n   Per-head settings:");
            for (head, s) in &c.head_settings {
                println!(
                    "     - {} → ef_search={}, max_connections={}",
                    head, s.ef_search, s.max_nb_connection
                );
            }
            (c.collection.clone(), c.head_settings.clone())
        }
        _ => {
            println!("   Failed to parse CREATE COLLECTION statement");
            return;
        }
    };

    // Step 2: Create heads with custom settings
    println!("\n▶ Step 2: Creating heads with per-head configurations...\n");

    let base_config = HNSWConfig {
        max_nb_connection: 16,
        ef_construction: 400,
        ef_search: 128,
        store_vectors: true,
        max_elements: 10_000,
    };

    let mut manager = MultiHeadManager::new(256, 2);

    let sem_config = head_settings.get("semantic").cloned().unwrap_or_default();
    let mut sem_head = HeadConfig::new("semantic", HeadType::Semantic, 256);
    sem_head.settings = Some(sem_config);
    manager.add_head(sem_head);

    let tmp_config = head_settings.get("temporal").cloned().unwrap_or_default();
    let mut tmp_head = HeadConfig::new("temporal", HeadType::Temporal, 256);
    tmp_head.settings = Some(tmp_config);
    manager.add_head(tmp_head);

    let mut sem_idx = manager
        .create_hnsw_index_for_head("semantic", 256, base_config.clone())
        .unwrap();
    let mut tmp_idx = manager
        .create_hnsw_index_for_head("temporal", 256, base_config.clone())
        .unwrap();

    println!("   semantic: ef_search=256, max_connections=32 (high recall)");
    println!("   temporal: ef_search=64, max_connections=16 (faster queries)");

    // Step 3: Index documents
    println!("\n▶ Step 3: Indexing 2,000 documents...\n");

    let start = Instant::now();
    for i in 0..2000 {
        let v = generate_distinct(i, 256);
        let _ = sem_idx.insert(i, &v);
        let mut tv = v.clone();
        if i % 4 == 0 {
            tv[0] = (tv[0] + 0.15).min(1.0);
        }
        let _ = tmp_idx.insert(i, &tv);
    }
    println!("   Indexed 2,000 documents in {:.2?}\n", start.elapsed());

    // Step 4: Run queries per-head
    println!("▶ Step 4: Running per-head queries...\n");

    let query_vec = generate_distinct(350, 256);

    let sem_start = Instant::now();
    let sem_results = sem_idx.search(&query_vec, 5, Some(256)).unwrap();
    let sem_latency = sem_start.elapsed();

    let tmp_start = Instant::now();
    let tmp_results = tmp_idx.search(&query_vec, 5, Some(64)).unwrap();
    let tmp_latency = tmp_start.elapsed();

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║                        SEMANTIC HEAD (ef=256, {:.2?})                      ║",
        sem_latency
    );
    println!("╠════════════════════════════════════════════════════════════════════════════╣");
    for (rank, (id, score)) in sem_results.iter().enumerate() {
        println!(
            "║   {:<3}   ID: {:>6}   Similarity: {:.4}                                ║",
            rank + 1,
            id,
            score
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║                        TEMPORAL HEAD (ef=64, {:.2?})                        ║",
        tmp_latency
    );
    println!("╠════════════════════════════════════════════════════════════════════════════╣");
    for (rank, (id, score)) in tmp_results.iter().enumerate() {
        println!(
            "║   {:<3}   ID: {:>6}   Similarity: {:.4}                                ║",
            rank + 1,
            id,
            score
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    // Step 5: Fuse results
    println!("▶ Step 5: Fusing results (semantic weight=1.0, temporal weight=0.7)...\n");

    let mut fused: HashMap<u64, f32> = HashMap::new();
    for (id, s) in &sem_results {
        *fused.entry(*id).or_insert(0.0) += s;
    }
    for (id, s) in &tmp_results {
        *fused.entry(*id).or_insert(0.0) += s * 0.7;
    }

    let max_val = fused.values().cloned().fold(0.0f32, f32::max);
    if max_val > 0.0 {
        for v in fused.values_mut() {
            *v /= max_val;
        }
    }

    let mut final_results: Vec<_> = fused.into_iter().collect();
    final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    final_results.truncate(5);

    let sem_map: HashMap<u64, f32> = sem_results.iter().map(|(id, s)| (*id, *s)).collect();
    let tmp_map: HashMap<u64, f32> = tmp_results.iter().map(|(id, s)| (*id, *s)).collect();

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           FUSED RESULTS (Normalized)                       ║");
    println!("╠════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║   {:<5} {:>8} {:>12} {:>12} {:>12} ║",
        "Rank", "ID", "Fused", "Semantic", "Temporal"
    );
    println!(
        "║   {}  {}  {}  {}  {} ║",
        "-----", "--------", "------------", "------------", "------------"
    );
    for (rank, (id, score)) in final_results.iter().enumerate() {
        let ss = sem_map.get(id).copied().unwrap_or(0.0);
        let ts = tmp_map.get(id).copied().unwrap_or(0.0);
        println!(
            "║   {:<5} {:>8} {:>12.4} {:>12.4} {:>12.4} ║",
            rank + 1,
            id,
            score,
            ss,
            ts
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    println!("✅ Comprehensive demo completed successfully.");
    println!(
        "   Collection '{}' with 2 per-head settings, 2000 documents indexed.",
        collection_name
    );
    println!("   Per-head settings demonstrate tunable recall/performance tradeoffs.\n");
}
