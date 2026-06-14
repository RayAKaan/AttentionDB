use attentiondb_learned::ProjectionTrainer;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB Phase 6 — Learned Projections             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut trainer = ProjectionTrainer::new(256, 0.001);

    let dim = 256;
    let positive_pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..100)
        .map(|i| {
            let base: Vec<f32> = (0..dim).map(|x| ((i + x) as f32).sin() * 0.1).collect();
            let mut pos = base.clone();
            pos[0] += 0.01;
            (base, pos)
        })
        .collect();

    let negatives: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..dim).map(|x| ((i * 7 + x) as f32).cos() * 0.1).collect())
        .collect();

    println!("→ Training contrastive projection (100 steps)...");
    println!("   Positive pairs: {}", positive_pairs.len());
    println!("   Negatives per step: {}", negatives.len());
    println!("   Learning rate: {}\n", trainer.learning_rate);

    for epoch in 0..100 {
        let loss = trainer.train_step(&positive_pairs, &negatives).unwrap();
        if epoch % 20 == 0 || epoch == 99 {
            println!("   Epoch {:>3}: Loss = {:.6}", epoch, loss);
        }
    }

    println!("\n→ Projecting sample vector with learned w_k...");
    let sample: Vec<f32> = (0..dim).map(|x| (x as f32).cos() * 0.1).collect();
    let projected = trainer.get_projection().project_key(&sample);
    println!("   Input dim:  {}", sample.len());
    println!("   Output dim: {}", projected.len());
    println!("   First 5 values: {:?}", &projected[..5.min(projected.len())]);

    println!("\n   w_k shape: {:?}", trainer.get_projection().w_k.shape());
    println!("   w_v shape: {:?}", trainer.get_projection().w_v.shape());

    println!("\n✅ Phase 6 demo completed successfully.");
}
