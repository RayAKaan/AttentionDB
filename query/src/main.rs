use attentiondb_query::{parse_aql, plan_query, AQLStatement, QueryExecutor};
use std::collections::HashMap;

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

    println!("→ Input AQL (ATTEND query):");
    for line in aql.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            println!("   {}", trimmed);
        }
    }

    println!("\n→ Parsing...");
    let parsed = parse_aql(aql).unwrap();

    match &parsed {
        AQLStatement::Query(query) => {
            println!("   Collection: {}", query.collection);
            println!("   Query:      \"{}\"", query.query_text);
            println!("   Heads:      {:?}", query.heads);
            println!("   Top-K:      {}", query.top_k);
            println!("   MinWeight:  {:.3}", query.min_weight);
            println!("   Temporal:   {:?}", query.temporal_decay);

            println!("\n→ Planning...");
            let plan = plan_query(query.clone()).unwrap();
            println!("   HNSW heads:     {:?}", plan.hnsw_search.heads);
            println!("   ef:             {}", plan.hnsw_search.ef);
            println!("   Overfetch:      {}", plan.hnsw_search.k);
            println!(
                "   Exact rerank:   {}",
                if plan.exact_rerank.is_some() {
                    "yes"
                } else {
                    "no"
                }
            );

            println!("\n→ Executing...");
            let index = attentiondb_hnsw::HNSWIndex::new(
                "papers",
                256,
                attentiondb_hnsw::HNSWConfig::default(),
            );
            let query_vector = vec![0.1; 256];
            let result = QueryExecutor::execute(&plan, &index, &query_vector).unwrap();
            println!(
                "   Status: Heads: {} | Top-K: {} | Results: {} | Latency: {:.3}ms",
                plan.hnsw_search.heads.len(),
                plan.top_k,
                result.ids.len(),
                result.latency_ms
            );
            println!("\n   Results:");
            for (i, (id, score)) in result.ids.iter().zip(result.scores.iter()).enumerate() {
                println!("   {:>3}.  ID: {:>6}  Score: {:.4}", i + 1, id, score);
            }
        }
        AQLStatement::CreateCollection(coll) => {
            println!("   Collection: {}", coll.collection);
            println!("   Fields:     {:?}", coll.fields);
            println!("   Settings:   {:?}", coll.settings);
        }
        AQLStatement::AlterCollection(alter) => {
            println!("   Collection: {}", alter.collection);
            println!("   Settings:   {:?}", alter.settings);
        }
    }

    let ddl = r#"CREATE COLLECTION papers (title TEXT, body TEXT, year INT) WITH (ef_search = 256, similarity = "cosine")"#;

    println!("\n→ Input AQL (CREATE COLLECTION DDL):");
    println!("   {}", ddl);

    let ddl_parsed = parse_aql(ddl).unwrap();
    if let AQLStatement::CreateCollection(coll) = &ddl_parsed {
        println!("   Collection: {}", coll.collection);
        println!("   Fields:     {:?}", coll.fields);
        println!("   Settings:   {:?}", coll.settings);
    }

    let alter = r#"ALTER COLLECTION papers SET (ef_search = 512, max_connections = 64, exact_rerank = false)"#;

    println!("\n→ Input AQL (ALTER COLLECTION DDL):");
    println!("   {}", alter);

    let alter_parsed = parse_aql(alter).unwrap();
    if let AQLStatement::AlterCollection(a) = &alter_parsed {
        println!("   Collection: {}", a.collection);
        println!(
            "   New Settings: ef_search={}, max_connections={}, exact_rerank={}",
            a.settings.ef_search, a.settings.max_nb_connection, a.settings.enable_exact_reranking
        );
    }

    let mut empty_indexes = HashMap::new();
    let mut empty_managers = HashMap::new();
    let alter_result = QueryExecutor::execute_statement(
        &alter_parsed,
        &mut empty_indexes,
        &mut empty_managers,
        None,
    )
    .unwrap();
    println!("\n→ Executor result: {}", alter_result.message);

    println!("\n✅ Phase 3 demo completed successfully.");
}
