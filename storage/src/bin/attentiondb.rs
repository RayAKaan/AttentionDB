//! Simple CLI for AttentionDB Phase 1

use attentiondb::{DocumentStore, Record};
use clap::{Parser, Subcommand};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "attentiondb")]
#[command(about = "AttentionDB Phase 1 Storage Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Insert a record
    Insert { name: String },
    /// Show stats
    Stats,
}

fn main() {
    let cli = Cli::parse();
    let mut store = DocumentStore::new();

    match &cli.command {
        Commands::Insert { name } => {
            let mut fields = HashMap::new();
            fields.insert("name".to_string(), serde_json::json!(name));

            let record = Record::new(fields);
            let id = store.insert(record).unwrap();
            println!("Inserted record: {}", id);
        }
        Commands::Stats => {
            println!("Records in store: {}", store.len());
        }
    }
}
