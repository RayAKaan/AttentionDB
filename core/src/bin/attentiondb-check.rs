//! `attentiondb-check` — offline consistency checker (§23).
//!
//! Opens a database directory read-only-equivalently (recovery runs in this
//! process, nothing is mutated beyond standard recovery) and reports coded,
//! actionable issues. Exit codes: 0 = clean (warnings allowed), 1 = errors
//! found, 2 = could not run the check.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("ATTENTIONDB_DATA_DIR")
                .ok()
                .map(std::path::PathBuf::from)
        })
        .unwrap_or_else(|| std::path::PathBuf::from("/data"));

    println!("attentiondb check — {}", dir.display());

    match attentiondb_core::checker::check_db_dir(&dir) {
        Ok(issues) => {
            let errors = issues.iter().filter(|i| i.severity.is_error()).count();
            for i in &issues {
                match i.severity {
                    attentiondb_core::Severity::Error => {
                        println!(
                            "ERROR   {} collection={} {}",
                            i.code, i.collection, i.detail
                        );
                    }
                    _ => {
                        println!(
                            "WARNING {} collection={} {}",
                            i.code, i.collection, i.detail
                        );
                    }
                }
            }
            if errors == 0 {
                println!(
                    "OK — no consistency errors ({} warning(s))",
                    issues.len() - errors
                );
                ExitCode::SUCCESS
            } else {
                println!("FAILED — {errors} consistency error(s)");
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("check could not run: {e}");
            ExitCode::from(2)
        }
    }
}
