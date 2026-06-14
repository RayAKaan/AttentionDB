use attentiondb_query::{parse_aql, plan_query, QueryExecutor};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB Phase 3 — Query Engine + AQL Parser       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let aql = r#"
        ATTEND TO papers
        WHERE QUERY "attention mechanisms in transformers"
        HEADS [semantic, temporal, structural]
        TOP_K 10
        MIN_WEIGHT 0.05
        TEMPORAL_DECAY 0.3
    "#;

    println!("→ Input AQL:");
    for line in aql.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            println!("   {}", trimmed);
        }
    }

    println!("\n→ Parsing...");
    let parsed = parse_aql(aql).unwrap();
    println!("   Collection: {}", parsed.collection);
    println!("   Query:      \"{}\"", parsed.query_text);
    println!("   Heads:      {:?}", parsed.heads);
    println!("   Top-K:      {}", parsed.top_k);
    println!("   MinWeight:  {:.3}", parsed.min_weight);
    println!("   Temporal:   {:?}", parsed.temporal_decay);

    println!("\n→ Planning...");
    let plan = plan_query(parsed).unwrap();
    println!("   HNSW heads:     {:?}", plan.hnsw_search.heads);
    println!("   ef:             {}", plan.hnsw_search.ef);
    println!("   Overfetch:      {}", plan.hnsw_search.k);
    println!("   Exact rerank:   {}", if plan.exact_rerank.is_some() { "yes" } else { "no" });

    println!("\n→ Executing...");
    let query_vector = vec![0.1; 256];
    let (result, status) = QueryExecutor::execute_with_status(&plan, &query_vector).unwrap();

    println!("   Status: {}", status);
    println!("\n   Results:");
    for (i, (id, score)) in result.ids.iter().zip(result.scores.iter()).enumerate() {
        println!("   {:>3}.  ID: {:>6}  Score: {:.4}", i + 1, id, score);
    }

    println!("\n✅ Phase 3 demo completed successfully.");
}
