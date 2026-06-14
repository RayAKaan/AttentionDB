use attentiondb_multihead::{MultiHeadManager, HeadConfig, HeadType};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB Phase 4 — Multi-Head + Score Fusion       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut manager = MultiHeadManager::new(256, 3);

    manager.add_head(HeadConfig::new("semantic", HeadType::Semantic, 256).with_weight(1.0));
    manager.add_head(HeadConfig::new("temporal", HeadType::Temporal, 256).with_weight(0.8));
    manager.add_head(HeadConfig::new("structural", HeadType::Structural, 256).with_weight(0.6));

    println!("→ Registered {} heads:", manager.head_count());
    for (name, config) in &manager.heads {
        println!("   {:12}  type: {:?}  weight: {:.1}", name, config.head_type, config.weight);
    }

    let query_emb: Vec<f32> = (0..256).map(|i| (i as f32).sin() * 0.1).collect();

    let head_results = vec![
        ("semantic".to_string(),   vec![(1, 0.92), (2, 0.85), (3, 0.71), (4, 0.65)]),
        ("temporal".to_string(),   vec![(2, 0.88), (4, 0.79), (1, 0.65)]),
        ("structural".to_string(), vec![(1, 0.81), (3, 0.77), (5, 0.60)]),
    ];

    println!("\n→ Raw results per head:");
    for (head, results) in &head_results {
        let ids: Vec<String> = results.iter().map(|(id, s)| format!("{}:{:.2}", id, s)).collect();
        println!("   {:12}  [{}]", head, ids.join(", "));
    }

    println!("\n→ Computing gating weights for query...");
    let gate_weights = manager.gating.forward(&query_emb);
    println!("   Gate weights:");
    for (head, weight) in manager.list_heads().iter().zip(gate_weights.iter()) {
        println!("     {:12}  {:.4}", head, weight);
    }

    println!("\n→ Fusing scores...");
    let fused = manager.fuse(&query_emb, &head_results).unwrap();

    println!("\n   Final fused results:");
    for (i, (id, score)) in fused.iter().enumerate() {
        println!("   {:>3}.  ID: {:>4}   Fused Score: {:.4}", i + 1, id, score);
    }

    // Weighted fusion bypassing gate
    println!("\n→ Weighted fusion (bypassing gate):");
    let explicit_weights = vec![
        ("semantic".to_string(), 1.0),
        ("temporal".to_string(), 0.7),
        ("structural".to_string(), 0.5),
    ];
    let weighted = manager.fuse_weighted(&head_results, &explicit_weights);
    for (i, (id, score)) in weighted.iter().enumerate().take(3) {
        println!("   {:>3}.  ID: {:>4}   Score: {:.4}", i + 1, id, score);
    }

    println!("\n✅ Phase 4 demo completed successfully.");
}
