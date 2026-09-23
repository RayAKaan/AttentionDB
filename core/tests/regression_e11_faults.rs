//! E11 regression pins (final phase, fault-injection closure).
//!
//! 1. The E11 crash gates (backup_mid_copy / backup_after_copy /
//!    restore_mid_copy / rebuild_mid) must be INERT in ordinary runs — no env,
//!    no abort, no behavior change (spec §7 requirement).
//! 2. The measured F01 tolerances are pinned: a regressed sidecar watermark
//!    opens with full recovery (records authoritative — PH3E-FAULT-038), and a
//!    torn WAL tail recovers the intact prefix (G4).

use attentiondb_core::engine::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

const DIM: usize = 32;
const HEAD: &str = "h";

fn rec(idx: u32) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(idx));
    fields.insert("cat".to_string(), serde_json::json!("a"));
    fields.insert("num".to_string(), serde_json::json!(idx as i64));
    let mut r = Record::new(fields);
    let mut v = vec![0.0f32; DIM];
    v[idx as usize % DIM] = 1.0;
    r.k_vecs.insert(HEAD.to_string(), v);
    r
}

fn build(dir: &std::path::Path, n: u32) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = AttentionEngine::open_dir(dir, Durability::Sync).unwrap();
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for i in 0..n {
        e.insert_document("bench", rec(i)).unwrap();
    }
}

fn count_live(e: &AttentionEngine) -> usize {
    e.scan_filtered("bench", None, 100_000).unwrap().len()
}

#[test]
fn e11_fault_gates_inert_without_env() {
    // PH3E_CRASH_AT deliberately unset for the whole test process: every new
    // E11 gate hit() must be a no-op and the workload must behave normally.
    std::env::remove_var("PH3E_CRASH_AT");
    attentiondb_storage::crashgate::GATE_BACKUP_MID_COPY.hit();
    attentiondb_storage::crashgate::GATE_BACKUP_AFTER_COPY.hit();
    attentiondb_storage::crashgate::GATE_RESTORE_MID_COPY.hit();
    attentiondb_storage::crashgate::GATE_REBUILD_MID.hit();
    let dir = std::env::temp_dir().join("e11reg-gates-inert");
    build(&dir, 50);
    let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    assert_eq!(count_live(&e), 50);
    e.checkpoint().unwrap();
    let bdir = std::env::temp_dir().join("e11reg-gates-inert-bak");
    let _ = std::fs::remove_dir_all(&bdir);
    e.backup_to(&bdir).unwrap(); // backup gate would abort here if armed
    drop(e);
    let rdir = std::env::temp_dir().join("e11reg-gates-inert-restore");
    let _ = std::fs::remove_dir_all(&rdir);
    attentiondb_core::backup::restore_backup(&bdir, &rdir).unwrap(); // restore gate too
    let e2 = AttentionEngine::open_dir(&rdir, Durability::Sync).unwrap();
    assert_eq!(count_live(&e2), 50); // rebuild gate too (recovery rebuild ran)
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&bdir);
    let _ = std::fs::remove_dir_all(&rdir);
}

#[test]
fn e11_regressed_sidecar_opens_with_full_recovery() {
    // Measured tolerance (PH3E-FAULT-007/038): the sidecar is derived
    // bookkeeping; a REGRESSED high_watermark must not lose data on open.
    let dir = std::env::temp_dir().join("e11reg-sidecar-regress");
    build(&dir, 20);
    {
        let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
        e.checkpoint().unwrap(); // sidecar exists after checkpoint
        for i in 20..30u32 {
            e.insert_document("bench", rec(i)).unwrap();
        }
    }
    let state = dir.join("WAL").join("wal-state.json");
    assert!(state.exists(), "sidecar must exist after checkpoint");
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&state).unwrap()).unwrap();
    let hw = v["high_watermark"].as_u64().unwrap();
    v["high_watermark"] = serde_json::json!(hw.saturating_sub(10));
    std::fs::write(&state, serde_json::to_vec_pretty(&v).unwrap()).unwrap();

    let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    assert_eq!(count_live(&e), 30, "regressed sidecar must not lose records");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn e11_torn_tail_recovers_intact_prefix() {
    // G4: a torn WAL tail is truncated and the intact prefix recovers; the
    // loss is bounded by the destroyed bytes (F01, PH3E-FAULT-002).
    let dir = std::env::temp_dir().join("e11reg-torn-tail");
    build(&dir, 20);
    let wal = dir.join("WAL");
    let mut segs: Vec<_> = std::fs::read_dir(&wal)
        .unwrap()
        .flatten()
        .map(|p| p.path())
        .filter(|p| p.extension().map(|x| x == "wal").unwrap_or(false))
        .collect();
    segs.sort();
    let last = segs.last().unwrap().clone();
    let mut b = std::fs::read(&last).unwrap();
    let cut = b.len() * 3 / 5;
    b.truncate(cut);
    std::fs::write(&last, b).unwrap();

    let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    let live = count_live(&e);
    assert!(live > 0 && live <= 20, "intact prefix must recover, got {live}");
    assert_ne!(live, 20, "test must actually destroy records to be meaningful");
    // repeated restart deterministic
    drop(e);
    let e2 = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    assert_eq!(count_live(&e2), live, "recovery must be deterministic");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn e11_upsert_dual_live_uuid_is_documented_semantics() {
    // Pinned by the E10 D41 probe and re-confirmed in E11 D56: insert_new on
    // an already-live logical idx mints a SECOND live uuid; the engine stores
    // both faithfully and filtered scans return both. Callers own upsert
    // semantics. This test freezes that contract so it cannot silently change.
    let dir = std::env::temp_dir().join("e11reg-dual-live");
    build(&dir, 5);
    let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    e.insert_document("bench", rec(0)).unwrap(); // same logical idx, new uuid? No:
    // Record::new generates a fresh uuid; rec(0) keeps fields idx=0.
    assert_eq!(count_live(&e), 6, "second live record for idx 0 must be stored");
    let got = e.scan_filtered("bench", None, 100).unwrap();
    let idx0 = got
        .iter()
        .filter(|(uid, _)| {
            let store = e.document_store.read();
            store
                .list_all_records()
                .into_iter()
                .any(|r| &r.id.to_string() == uid
                    && r.fields.get("idx").and_then(|v| v.as_u64()) == Some(0))
        })
        .count();
    assert_eq!(idx0, 2, "both live versions of idx 0 must be scannable");
    e.checkpoint().unwrap();
    drop(e);
    let e2 = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    assert_eq!(count_live(&e2), 6, "both live versions must survive restart");
    let _ = std::fs::remove_dir_all(&dir);
}
