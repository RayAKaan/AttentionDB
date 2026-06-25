//! End-to-End Demo: AttentionDB HNSW + Multi-Head + Query Flow
//!
//! Run with: cargo run --example end_to_end_demo -p attentiondb-hnsw --release

use attentiondb_hnsw::{HNSWConfig, HNSWIndex};
use std::collections::HashMap;
use std::time::Instant;

fn sim(distance: f32) -> f32 {
    1.0 - distance * 0.5
}

fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║           AttentionDB End-to-End Demo (HNSW)               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let dim = 256;
    let n_docs = 2000;
    let config = HNSWConfig {
        max_nb_connection: 16,
        ef_construction: 400,
        ef_search: 128,
        store_vectors: true,
        max_elements: 10_000,
    };

    // Create attention heads with different perspectives on the same data
    let mut semantic = HNSWIndex::new("semantic", dim, config.clone());
    let mut temporal = HNSWIndex::new("temporal", dim, config.clone());
    let mut structural = HNSWIndex::new("structural", dim, config.clone());

    println!("▶ Building 3 attention heads over {n_docs} documents...\n");

    let topics = [
        "attention mechanisms",
        "convolutional networks",
        "recurrent architectures",
        "optimization methods",
        "representation learning",
        "graph neural networks",
        "reinforcement learning",
        "NLP",
        "computer vision",
        "generative models",
    ];

    let start = Instant::now();
    let mut rng = 1u64;

    for i in 0..n_docs {
        let topic_idx = i % topics.len();
        let phase = topic_idx as f32 * 0.2;

        // Base embedding: topic-grounded vector
        let mut base = vec![0.0f32; dim];
        for x in 0..dim {
            base[x] = ((phase + x as f32 * 0.05) * (topic_idx + 1) as f32).sin()
                + ((phase + x as f32 * 0.03) * (topic_idx + 1) as f32 * 0.5).cos() * 0.5;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
        let base = normalize(&base);

        // All three heads index the SAME document with DIFFERENT encodings
        let _ = semantic.insert(i as u64, &base);

        // Temporal head: boosts recent signals (dim 0)
        let mut tv = base.clone();
        tv[0] += 0.25;
        let tv = normalize(&tv);
        let _ = temporal.insert(i as u64, &tv);

        // Structural head: boosts schema patterns (dim 10)
        let mut sv = base.clone();
        sv[10] += 0.20;
        let sv = normalize(&sv);
        let _ = structural.insert(i as u64, &sv);
    }

    println!(
        "   Indexed {n_docs} documents across all heads in {:.2?}\n",
        start.elapsed()
    );

    // Query for topic 0: "attention mechanisms"
    println!("▶ Multi-head search query: \"attention mechanisms\"\n");

    let q_phase = 0.0f32;
    let mut q = vec![0.0f32; dim];
    for x in 0..dim {
        q[x] = ((q_phase + x as f32 * 0.05) * 1.0).sin()
            + ((q_phase + x as f32 * 0.03) * 1.0 * 0.5).cos() * 0.5;
    }
    let q = normalize(&q);

    // Each head returns results with its own perspective
    let sem_results = semantic.search(&q, 10, Some(128)).unwrap();
    let temp_results = temporal.search(&q, 10, Some(128)).unwrap();
    let struct_results = structural.search(&q, 10, Some(128)).unwrap();

    // Build per-head similarity maps
    let sem_sim: HashMap<u64, f32> = sem_results.iter().map(|(id, d)| (*id, sim(*d))).collect();
    let temp_sim: HashMap<u64, f32> = temp_results.iter().map(|(id, d)| (*id, sim(*d))).collect();
    let struct_sim: HashMap<u64, f32> = struct_results
        .iter()
        .map(|(id, d)| (*id, sim(*d)))
        .collect();

    // Fuse: weighted sum of similarity scores
    println!("▶ Fusing scores (semantic×1.0 + temporal×0.7 + structural×0.5)...\n");

    let mut fused: HashMap<u64, f32> = HashMap::new();
    for (&id, &s) in &sem_sim {
        *fused.entry(id).or_insert(0.0) += s * 1.0;
    }
    for (&id, &s) in &temp_sim {
        *fused.entry(id).or_insert(0.0) += s * 0.7;
    }
    for (&id, &s) in &struct_sim {
        *fused.entry(id).or_insert(0.0) += s * 0.5;
    }

    let mut ranked: Vec<(u64, f32)> = fused.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(8);

    // Display
    println!(
        "┌──────┬────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────┐"
    );
    println!(
        "│ Rank │    ID  │    Fused │ Semantic │ Temporal │ Struct.  │ Topic                │"
    );
    println!(
        "├──────┼────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────┤"
    );

    for (rank, (id, score)) in ranked.iter().enumerate() {
        let sem = sem_sim.get(id).copied().unwrap_or(0.0);
        let tmp = temp_sim.get(id).copied().unwrap_or(0.0);
        let stc = struct_sim.get(id).copied().unwrap_or(0.0);
        let topic = topics[*id as usize % topics.len()];

        println!(
            "│ {:<4} │ {:>6} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:<20} │",
            rank + 1,
            id,
            score,
            sem,
            tmp,
            stc,
            topic
        );
    }
    println!(
        "└──────┴────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────┘\n"
    );

    println!("▶ Per-head statistics:");
    println!(
        "   Semantic:   {} total, returned {} results",
        semantic.len(),
        sem_results.len()
    );
    println!(
        "   Temporal:   {} total, returned {} results",
        temporal.len(),
        temp_results.len()
    );
    println!(
        "   Structural: {} total, returned {} results",
        structural.len(),
        struct_results.len()
    );

    println!("\n▶ Score fusion captures signals across all three heads.");
    println!("   A document ranking high in multiple heads rises to the top.\n");

    println!("✅ End-to-end demo completed successfully.\n");
}
