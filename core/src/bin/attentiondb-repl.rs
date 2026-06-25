use attentiondb_core::AttentionEngine;
use std::io::{self, Write};

const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";

fn print_banner() {
    println!(
        "{}{}╔════════════════════════════════════════════════════════════╗{}",
        BOLD, CYAN, RESET
    );
    println!(
        "{}{}║                    AttentionDB REPL                        ║{}",
        BOLD, CYAN, RESET
    );
    println!(
        "{}{}╚════════════════════════════════════════════════════════════╝{}",
        BOLD, CYAN, RESET
    );
    println!();
    println!("Type 'help' for available commands, 'quit' to exit.");
    println!();
}

fn print_help() {
    println!("{}Available commands:{}", BOLD, RESET);
    println!();
    println!(
        "  {}create <name>{}              Create a new collection",
        CYAN, RESET
    );
    println!(
        "  {}list / ls{}                  List all collections",
        CYAN, RESET
    );
    println!(
        "  {}insert <collection>{}        Insert a vector interactively",
        CYAN, RESET
    );
    println!(
        "  {}query <collection> <text>{}  Run a simple query",
        CYAN, RESET
    );
    println!(
        "  {}aql <statement>{}            Execute a raw AQL statement",
        CYAN, RESET
    );
    println!(
        "  {}status{}                     Show current engine state",
        CYAN, RESET
    );
    println!(
        "  {}clear{}                      Clear the screen",
        CYAN, RESET
    );
    println!(
        "  {}help / ?{}                   Show this help message",
        CYAN, RESET
    );
    println!(
        "  {}quit / exit{}                Exit the REPL",
        CYAN, RESET
    );
    println!();
    println!("{}Examples:{}", BOLD, RESET);
    println!("  create papers");
    println!("  insert papers");
    println!("  query papers attention mechanisms");
    println!("  aql ATTEND TO papers WHERE QUERY 'transformers' TOP_K 5");
    println!();
}

fn print_success(msg: &str) {
    println!("{}✓ {}{}", GREEN, msg, RESET);
}

fn print_error(msg: &str) {
    println!("{}✗ {}{}", RED, msg, RESET);
}

fn main() {
    print_banner();

    let engine = AttentionEngine::new();
    let mut collections: Vec<String> = Vec::new();
    let mut next_id: u64 = 1;

    loop {
        print!("{}>{} ", CYAN, RESET);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            print_error("Error reading input. Exiting.");
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let command = parts.first().copied().unwrap_or("");

        match command {
            "quit" | "exit" => {
                println!("Goodbye!");
                break;
            }

            "help" | "?" => {
                print_help();
            }

            "clear" => {
                print!("\x1b[2J\x1b[H");
                io::stdout().flush().unwrap();
            }

            "status" => {
                println!("{}Engine Status:{}", BOLD, RESET);
                println!("  Collections: {}", collections.len());
                println!(
                    "  Persistence: {}",
                    if engine.is_persistent() {
                        "Enabled"
                    } else {
                        "In-memory only"
                    }
                );
            }

            "create" => {
                if parts.len() < 2 {
                    print_error("Usage: create <collection_name>");
                    continue;
                }
                let name = parts[1];
                if collections.contains(&name.to_string()) {
                    print_error(&format!("Collection '{}' already exists.", name));
                    continue;
                }
                let stmt = format!("CREATE COLLECTION {} ()", name);
                match engine.execute_aql(&stmt) {
                    Ok(msg) => {
                        print_success(&msg);
                        collections.push(name.to_string());
                    }
                    Err(e) => print_error(&e.to_string()),
                }
            }

            "list" | "ls" => {
                if collections.is_empty() {
                    println!("No collections created yet. Use 'create <name>' to create one.");
                } else {
                    println!("{}Collections ({}):{}", BOLD, collections.len(), RESET);
                    for (i, name) in collections.iter().enumerate() {
                        println!("  {:>2}. {}", i + 1, name);
                    }
                }
            }

            "insert" => {
                if parts.len() < 2 {
                    print_error("Usage: insert <collection_name>");
                    continue;
                }
                let collection = parts[1];

                if !collections.contains(&collection.to_string()) {
                    print_error(&format!(
                        "Collection '{}' does not exist. Create it first with 'create {}'.",
                        collection, collection
                    ));
                    continue;
                }

                println!(
                    "Enter vector data in the format {}head_name=value1,value2,...{}.",
                    CYAN, RESET
                );
                println!("Example: {}semantic=0.1,0.2,0.3,0.4{}", CYAN, RESET);
                println!("Type 'done' when finished, or 'cancel' to abort.");

                loop {
                    print!("  {}data>{} ", CYAN, RESET);
                    io::stdout().flush().unwrap();

                    let mut field_input = String::new();
                    io::stdin().read_line(&mut field_input).unwrap();
                    let field_input = field_input.trim();

                    if field_input == "done" {
                        break;
                    }
                    if field_input == "cancel" {
                        println!("Insertion cancelled.");
                        break;
                    }

                    if let Some((key, values_str)) = field_input.split_once('=') {
                        let head_name = key.trim();
                        let vec: Vec<f32> = values_str
                            .split(',')
                            .filter_map(|s| s.trim().parse::<f32>().ok())
                            .collect();

                        if vec.is_empty() {
                            print_error(
                                "No valid numbers found. Use format: head_name=0.1,0.2,0.3",
                            );
                            continue;
                        }

                        match engine.insert_vector(collection, head_name, next_id, &vec) {
                            Ok(_) => {
                                print_success(&format!(
                                    "Inserted id={} into head '{}' (dim={})",
                                    next_id,
                                    head_name,
                                    vec.len()
                                ));
                                next_id += 1;
                            }
                            Err(e) => print_error(&e.to_string()),
                        }
                    } else {
                        print_error("Invalid format. Use head_name=value1,value2,...");
                    }
                }
            }

            "query" => {
                if parts.len() < 3 {
                    print_error("Usage: query <collection> <query_text>");
                    continue;
                }
                let collection_name = parts[1];
                let query_text = parts[2..].join(" ");

                let stmt = format!(
                    "ATTEND TO {} WHERE QUERY \"{}\" TOP_K 5",
                    collection_name, query_text
                );

                match engine.execute_aql(&stmt) {
                    Ok(msg) => print_success(&msg),
                    Err(e) => print_error(&e.to_string()),
                }
            }

            "aql" => {
                if parts.len() < 2 {
                    print_error("Usage: aql <statement>");
                    continue;
                }
                let aql = parts[1..].join(" ");
                match engine.execute_aql(&aql) {
                    Ok(msg) => print_success(&msg),
                    Err(e) => print_error(&e.to_string()),
                }
            }

            _ => {
                print_error(&format!("Unknown command: '{}'", command));
                println!("  Type 'help' for available commands.");
            }
        }
    }
}
