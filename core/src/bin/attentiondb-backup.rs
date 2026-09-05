//! `attentiondb-backup` — create a consistent full backup of a database directory.
//!
//! Opens the database (runs recovery), checkpoints, then copies the
//! authoritative state set into the destination. Internally consistent by
//! construction. Exit 0 on success.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: attentiondb-backup <db_dir> <dest_dir>");
        std::process::exit(2);
    }
    let db_dir = std::path::PathBuf::from(&args[1]);
    let dest = std::path::PathBuf::from(&args[2]);

    let durability = attentiondb_storage::Durability::Sync;
    let engine = match attentiondb_core::AttentionEngine::open_dir(&db_dir, durability) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("backup failed: cannot open database: {e}");
            std::process::exit(2);
        }
    };
    match engine.backup_to(&dest) {
        Ok(info) => {
            println!(
                "backup complete: dest={} checkpoint_seq={} collections={} documents={}",
                dest.display(),
                info.checkpoint_seq,
                info.collections,
                info.documents
            );
        }
        Err(e) => {
            eprintln!("backup failed: {e}");
            std::process::exit(1);
        }
    }
}
