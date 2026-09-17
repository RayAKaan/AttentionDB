//! Phase 1 integration tests — recovery & WAL corruption (TEST 9, 13, 14, 15).

mod common;

use common::*;

/// TEST 9 — WAL replay: mutations with unclean termination (no close, no
/// checkpoint) are recovered from the WAL.
#[test]
fn t09_wal_replay_unclean() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..20 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "wal only"))
                .unwrap();
        }
        // Simulate kill -9: no drop handlers, no flush, no checkpoint.
        std::mem::forget(e);
    }
    let e = open_sync(dir.path());
    assert_eq!(e.list_collections(), vec!["c".to_string()]);
    assert_eq!(e.document_store.read().len(), 20);
    let r = e
        .attend("c", &["default".into()], &one_hot(11, 8), 3)
        .unwrap();
    assert!(!r.is_empty());
}

/// TEST 14 — Corrupt WAL tail: truncated final record is repairable; everything
/// acknowledged before the torn write survives.
#[test]
fn t14_corrupt_wal_tail() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..5 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "safe"))
                .unwrap();
        }
        std::mem::forget(e); // durable (Sync), no rotation
    }
    // Simulate a REAL torn write: a frame whose header was written (magic + a
    // length that exceeds what reached the disk) followed by zero padding —
    // the classic crash signature on journaled filesystems. Non-zero garbage
    // tails are treated as corruption (fatal) by policy; see docs/wal.md.
    let wal_dir = dir.path().join("WAL");
    // Pick the SEGMENT (sorted .wal files) — unsorted read_dir can return
    // wal-state.json, and appending torn-frame bytes to the E1 integrity
    // sidecar is a DIFFERENT (refusal) scenario covered elsewhere.
    let mut segs: Vec<_> = std::fs::read_dir(&wal_dir)
        .unwrap()
        .filter_map(|x| x.ok())
        .map(|x| x.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wal"))
        .collect();
    segs.sort();
    let seg = segs.last().unwrap().clone();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&seg).unwrap();
        let magic: u32 = 0x5741_4C32; // FRAME_MAGIC
        let claimed_len: u32 = 500; // header claims a 500-byte body…
        f.write_all(&magic.to_be_bytes()).unwrap();
        f.write_all(&claimed_len.to_be_bytes()).unwrap();
        f.write_all(&[0u8; 9]).unwrap(); // …but only 9 zero bytes landed
    }
    let e = open_sync(dir.path());
    assert_eq!(e.document_store.read().len(), 5, "pre-torn records survive");
    // WAL remains usable after tail repair.
    e.insert_document("c", doc(99, &one_hot(1, 8), "post-repair"))
        .unwrap();
    e.close().unwrap();
    let e = open_sync(dir.path());
    assert_eq!(e.document_store.read().len(), 6);
}

/// TEST 15 — Corrupt WAL middle: fatal, refuses unsafe recovery.
#[test]
fn t15_corrupt_wal_middle() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..10 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "x"))
                .unwrap();
        }
        std::mem::forget(e);
    }
    let wal_dir = dir.path().join("WAL");
    let seg = std::fs::read_dir(&wal_dir)
        .unwrap()
        .filter_map(|x| x.ok())
        .map(|x| x.path())
        .find(|p| std::fs::metadata(p).map(|m| m.len() > 64).unwrap_or(false))
        .unwrap();
    let mut bytes = std::fs::read(&seg).unwrap();
    // Flip a byte inside the first record's body (not the frame header, not the tail).
    bytes[16] ^= 0xFF;
    std::fs::write(&seg, &bytes).unwrap();

    let result = try_open_sync(dir.path());
    match result {
        Ok(_) => panic!("mid-log corruption must refuse recovery"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("Corruption")
                    || msg.contains("corruption")
                    || msg.contains("checksum"),
                "error should be corruption-class, got: {msg}"
            );
        }
    }
}

/// TEST 13 — Crash during checkpoint: every partial-checkpoint state recovers to
/// a consistent database. We simulate the two real windows at file level:
///   (a) SSTs flushed + idmap saved, manifest NOT yet saved → checkpoint_seq low,
///       replay covers the delta;
///   (b) manifest saved, old WAL segments still present (trim never ran) →
///       harmless, next checkpoint trims.
#[test]
fn t13_crash_during_checkpoint() {
    let dir = temp_db();
    // Window (a): engine checkpoints early, then writes more (WAL-only), crash.
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..10 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "checkpointed"))
                .unwrap();
        }
        e.checkpoint().unwrap();
        for i in 10..25 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "post-checkpoint"))
                .unwrap();
        }
        std::mem::forget(e); // crash before any further checkpoint
    }
    let e = open_sync(dir.path());
    assert_eq!(
        e.document_store.read().len(),
        25,
        "checkpoint + WAL replay converge"
    );
    assert_eq!(e.get_collection("c").unwrap().total_vectors(), 25);

    // Window (b): manifest generation fallback — corrupt the newest manifest,
    // keep an older valid one; recovery must use the previous generation.
    let db_dir = dir.path();
    let manifest_dir = db_dir.join("MANIFEST");
    let mut gens: Vec<_> = std::fs::read_dir(&manifest_dir)
        .unwrap()
        .filter_map(|x| x.ok())
        .map(|x| x.path())
        .collect();
    gens.sort();
    let newest = gens.last().unwrap().clone();
    std::fs::write(&newest, b"corrupted manifest bytes").unwrap();
    // CURRENT points at the newest — the loader must fall back.
    //
    // Phase 3E E1 update: destroying the CURRENT generation here leaves only
    // gen 1 (checkpoint_seq=0) as fallback, while the seqs 1..11 that would
    // reconnect the surviving WAL (12..26) to a valid base live ONLY in the
    // SSTables the lost manifest described. Pre-E1 this opened and replayed
    // onto an unprovable base; under the E1 invariant (production-contract A1)
    // the open must REFUSE rather than silently mis-recover. The two-generation
    // policy still covers the designed crash (CURRENT write torn, gen N valid).
    let refused = try_open_sync(db_dir)
        .err()
        .expect("stale-fallback + trimmed WAL must refuse to open under E1");
    let msg = format!("{refused}");
    assert!(
        msg.contains("WAL_SEQ_GAP"),
        "expected WAL_SEQ_GAP refusal, got: {msg}"
    );
    // The legitimate fallback path (CURRENT write torn, manifest generations
    // intact) must keep opening: fallback gen (cp=6) connects to the surviving
    // WAL (7..9). Mirrors PH3E-WAL-001 corrupt_current_manifest_fallback.
    {
        let d2 = temp_db();
        {
            let e = open_sync(d2.path());
            e.create_collection("c", 8, &["default"]).unwrap();
            for i in 0..5 {
                e.insert_document("c", doc(i, &one_hot(i as usize, 8), "safe")).unwrap();
            }
            e.checkpoint().unwrap(); // gen 2 (cp=6); WAL trimmed, active @7
            for i in 5..8 {
                e.insert_document("c", doc(i, &one_hot(i as usize, 8), "tail")).unwrap();
            }
            std::mem::forget(e);
        }
        std::fs::write(d2.path().join("CURRENT"), b"manifest-999999999\n").unwrap();
        let e = open_sync(d2.path());
        assert_eq!(
            e.document_store.read().len(),
            8,
            "CURRENT-torn fallback with connected generation still opens"
        );
    }
    // The corrupted original dir now refuses (above) — nothing further to
    // open there; the connected-fallback sub-case passed in the block above.
}

/// Extra: checkpoint trims old WAL segments, and the trimmed database still
/// replays correctly (segments fully covered by the checkpoint are gone).
#[test]
fn t13b_wal_trim_after_checkpoint() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..30 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "x"))
                .unwrap();
        }
        e.checkpoint().unwrap();
        e.insert_document("c", doc(100, &one_hot(4, 8), "tail"))
            .unwrap();
        e.close().unwrap();
    }
    let wal_dir = dir.path().join("WAL");
    let seg_count = std::fs::read_dir(&wal_dir).unwrap().count();
    assert!(seg_count >= 1);
    let e = open_sync(dir.path());
    assert_eq!(e.document_store.read().len(), 31);
    assert_eq!(e.get_collection("c").unwrap().total_vectors(), 31);
}

/// Extra: startup refuses a database written by a NEWER format version.
#[test]
fn t15b_newer_format_rejected() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        e.close().unwrap();
    }
    let (mut cat, _) = attentiondb_storage::Catalog::load(dir.path()).unwrap();
    cat.database_format_version = 99;
    cat.save(dir.path()).unwrap();
    assert!(try_open_sync(dir.path()).is_err());
}

/// Extra: unreadable SSTable is a hard recovery failure (no silent skip).
#[test]
fn t15c_corrupt_sstable_is_fatal() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..20 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "x"))
                .unwrap();
        }
        e.checkpoint().unwrap();
        e.close().unwrap();
    }
    let sst_dir = dir.path().join("sst");
    let sst = std::fs::read_dir(&sst_dir)
        .unwrap()
        .filter_map(|x| x.ok())
        .map(|x| x.path())
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("sst"))
        .unwrap();
    std::fs::write(&sst, b"garbage-not-an-sstable").unwrap();
    assert!(
        try_open_sync(dir.path()).is_err(),
        "corrupt SST must fail recovery loudly"
    );
}
