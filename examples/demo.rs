use attentiondb_hnsw::HNSWConfig;
use attentiondb_multihead::{HeadConfig, HeadType, MultiHeadManager};
use attentiondb_query::{parse_aql, AQLStatement};
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
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║           AttentionDB End-to-End Demo                      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Step 1: CREATE COLLECTION
    println!(">> Step 1: Creating collection with settings...\n");

    let create_stmt = r#"
        CREATE COLLECTION papers (
            title TEXT,
            abstract TEXT,
            year INT
        ) WITH (
            ef_search = 128,
            ef_construction = 400,
            max_connections = 16,
            similarity = "cosine",
            exact_rerank = true
        )
    "#;

    match parse_aql(create_stmt) {
        Ok(AQLStatement::CreateCollection(c)) => {
            println!("   Collection '{}' created successfully", c.collection);
            println!(
                "   Settings -> ef_search={}, ef_construction={}, max_connections={}",
                c.settings.ef_search, c.settings.ef_construction, c.settings.max_nb_connection
            );
        }
        _ => {
            println!("   Failed to parse CREATE COLLECTION");
            return;
        }
    }

    // Step 2: Create heads
    println!("\n>> Step 2: Creating attention heads...\n");

    let base_config = HNSWConfig {
        max_nb_connection: 16,
        ef_construction: 400,
        ef_search: 128,
        store_vectors: true,
        max_elements: 10_000,
    };

    let mut manager = MultiHeadManager::new(256, 2);
    manager.add_head(HeadConfig::new("semantic", HeadType::Semantic, 256));
    manager.add_head(HeadConfig::new("temporal", HeadType::Temporal, 256));

    let mut sem_idx = manager
        .create_hnsw_index_for_head("semantic", 256, base_config.clone())
        .unwrap();
    let mut tmp_idx = manager
        .create_hnsw_index_for_head("temporal", 256, base_config.clone())
        .unwrap();

    println!("   Created heads: semantic, temporal");

    // Step 3: Insert documents
    println!("\n>> Step 3: Inserting 1,000 documents...\n");

    let start = Instant::now();
    for i in 0..1000 {
        let v = generate_distinct(i, 256);
        let _ = sem_idx.insert(i, &v);
        let mut tv = v.clone();
        if i % 4 == 0 {
            tv[0] = (tv[0] + 0.15).min(1.0);
        }
        let _ = tmp_idx.insert(i, &tv);
    }
    println!("   Indexed 1,000 documents in {:.2?}\n", start.elapsed());

    // Step 4: Parse and run query
    println!(">> Step 4: Running AQL query...\n");

    let query_aql = r#"ATTEND TO papers WHERE QUERY "attention mechanisms" TOP_K 5"#;
    match parse_aql(query_aql) {
        Ok(AQLStatement::Query(q)) => {
            println!("   Collection: {}", q.collection);
            println!("   Query text: {}\n", q.query_text);

            let qv = generate_distinct(350, 256);

            let sem = sem_idx.search(&qv, 10, Some(128)).unwrap();
            let tmp = tmp_idx.search(&qv, 8, Some(128)).unwrap();

            // Fusion with normalization
            let mut fused: HashMap<u64, f32> = HashMap::new();
            for (id, s) in &sem {
                *fused.entry(*id).or_insert(0.0) += s;
            }
            for (id, s) in &tmp {
                *fused.entry(*id).or_insert(0.0) += s * 0.7;
            }

            let max_val = fused.values().cloned().fold(0.0f32, f32::max);
            if max_val > 0.0 {
                for v in fused.values_mut() {
                    *v /= max_val;
                }
            }

            let mut results: Vec<_> = fused.into_iter().collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results.truncate(5);

            // Build per-head score maps for the results table
            let sem_map: HashMap<u64, f32> = sem.iter().map(|(id, s)| (*id, *s)).collect();
            let tmp_map: HashMap<u64, f32> = tmp.iter().map(|(id, s)| (*id, *s)).collect();

            println!("   TOP 5 RESULTS (normalized fusion):\n");
            println!(
                "   {:>4}  {:>6}  {:>8}  {:>8}  {:>8}",
                "Rank", "ID", "Fused", "Sem", "Tmp"
            );
            println!(
                "   {}  {}  {}  {}  {}",
                "----", "------", "--------", "--------", "--------"
            );
            for (rank, (id, score)) in results.iter().enumerate() {
                let ss = sem_map.get(id).copied().unwrap_or(0.0);
                let ts = tmp_map.get(id).copied().unwrap_or(0.0);
                println!(
                    "   {:>4}  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}",
                    rank + 1,
                    id,
                    score,
                    ss,
                    ts
                );
            }
            println!();
        }
        _ => {
            println!("   Failed to parse query");
        }
    }

    println!("   Demo completed successfully.");
}
