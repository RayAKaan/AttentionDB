use attentiondb_hnsw::{HeadIndexManager, HNSWConfig};
use std::time::Instant;
use std::path::Path;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB Phase 2 — HNSW Index Layer (Amazing)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut manager = HeadIndexManager::new(256);

    let config = HNSWConfig::new()
        .with_ef_search(64)
        .with_vector_storage(true);

    manager.add_head_with_config("semantic", config.clone());
    manager.add_head_with_config("temporal", config.clone());
    manager.add_head_with_config("structural", config);

    println!("→ Inserting 10,000 vectors across 3 heads...");
    let start = Instant::now();
    for i in 0..10_000 {
        let vec: Vec<f32> = (0..256).map(|x| ((i + x) as f32).sin() * 0.1).collect();
        manager.insert("semantic", i, &vec).unwrap();
        if i % 2 == 0 {
            manager.insert("temporal", i + 100_000, &vec).unwrap();
        }
        if i % 5 == 0 {
            manager.insert("structural", i + 200_000, &vec).unwrap();
        }
    }
    println!("   Inserted in {:.2?}", start.elapsed());
    println!("   Total vectors: {}", manager.total_vectors());
    println!("   Heads: {:?}\n", manager.list_heads());

    println!("→ Per-head stats:");
    for head in manager.list_heads() {
        if let Ok(index) = manager.get_head(&head) {
            println!("   {:<12}  {:>6} vectors",
                     head, index.len());
        }
    }

    let query: Vec<f32> = (0..256).map(|x| (x as f32).cos() * 0.1).collect();

    println!("\n→ Raw HNSW Search (semantic):");
    if let Ok(index) = manager.get_head("semantic") {
        let results = index.search(&query, 5, None).unwrap();
        for (id, dist) in results {
            println!("   ID: {:>6}  Distance: {:.4}", id, dist);
        }
    }

    println!("\n→ Search + Exact Rerank (semantic):");
    if let Ok(index) = manager.get_head("semantic") {
        let results = index.search_with_rerank(&query, 5, None).unwrap();
        for (id, score) in results {
            println!("   ID: {:>6}  Score: {:.4}", id, score);
        }
    }

    println!("\n→ Weighted Multi-Head Search:");
    let weighted = manager.search_multi_weighted(
        &[("semantic", 1.0), ("temporal", 0.7), ("structural", 0.4)],
        &query, 6, None,
    ).unwrap();
    for (id, score) in weighted {
        println!("   ID: {:>6}  Weighted Score: {:.4}", id, score);
    }

    println!("\n→ Persistence Demo:");
    std::fs::create_dir_all("hnsw_graphs").ok();
    manager.save_all(Path::new("hnsw_graphs")).unwrap();
    println!("   Saved all heads to hnsw_graphs/");

    let path = Path::new("hnsw_graphs/semantic.hnsw");
    let loaded = attentiondb_hnsw::HNSWIndex::load(path, "semantic", 256).unwrap();
    println!("   Loaded semantic head: {} vectors\n", loaded.len());

    println!("✅ Phase 2 demo completed successfully!");
}
