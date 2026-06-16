use attentiondb_query::{parse_aql, plan_query, QueryExecutor, AQLStatement, ExecuteResult, execute_statement};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB Phase 3 — Query Engine + AQL Parser       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // --- ATTEND query demo ---
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
            println!("   Exact rerank:   {}", if plan.exact_rerank.is_some() { "yes" } else { "no" });

            println!("\n→ Executing...");
            let query_vector = vec![0.1; 256];
            let (result, status) = QueryExecutor::execute_with_status(&plan, &query_vector).unwrap();

            println!("   Status: {}", status);
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

    // --- CREATE COLLECTION DDL demo ---
    let ddl = r#"CREATE COLLECTION papers (title TEXT, body TEXT, year INT) WITH (ef_search = 256, similarity = "cosine")"#;

    println!("\n→ Input AQL (CREATE COLLECTION DDL):");
    println!("   {}", ddl);

    let ddl_parsed = parse_aql(ddl).unwrap();
    match &ddl_parsed {
        AQLStatement::CreateCollection(coll) => {
            println!("   Collection: {}", coll.collection);
            println!("   Fields:     {:?}", coll.fields);
            println!("   Settings:   {:?}", coll.settings);
        }
        _ => {}
    }

    // --- ALTER COLLECTION DDL demo ---
    let alter = r#"ALTER COLLECTION papers SET (ef_search = 512, max_connections = 64, exact_rerank = false)"#;

    println!("\n→ Input AQL (ALTER COLLECTION DDL):");
    println!("   {}", alter);

    let alter_parsed = parse_aql(alter).unwrap();
    match &alter_parsed {
        AQLStatement::AlterCollection(a) => {
            println!("   Collection: {}", a.collection);
            println!("   New Settings: ef_search={}, max_connections={}, exact_rerank={}",
                     a.settings.ef_search, a.settings.max_nb_connection, a.settings.enable_exact_reranking);
        }
        _ => {}
    }

    // Execute ALTER via the executor
    let alter_result = execute_statement(&alter_parsed, None, None).unwrap();
    println!("\n→ Executor result: {}", match &alter_result {
        ExecuteResult::DdlResult { message, .. } => message,
        _ => "unknown",
    });

    println!("\n✅ Phase 3 demo completed successfully.");
}
