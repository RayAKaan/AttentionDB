//! `attentiondb-restore` — restore a backup into a (new) directory and validate
//! it by opening it. Exit 0 on success.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: attentiondb-restore <backup_dir> <dest_db_dir>");
        std::process::exit(2);
    }
    let src = std::path::PathBuf::from(&args[1]);
    let dest = std::path::PathBuf::from(&args[2]);

    match attentiondb_core::backup::restore_backup(&src, &dest) {
        Ok(meta) => {
            println!(
                "restore complete: dest={} checkpoint_seq={} collections={}",
                dest.display(),
                meta.checkpoint_seq,
                meta.collections.len()
            );
        }
        Err(e) => {
            eprintln!("restore failed: {e}");
            std::process::exit(1);
        }
    }
}
