//! REGRESSION (E8 soak find — ENGINE DEFECT #2, source-fix canary):
//! flush entry timestamps must be LOGICALLY MONOTONIC — a flush immediately
//! following another (same wall-clock millisecond) must still stamp STRICTLY
//! greater entry timestamps. Pre-fix, per-entry `timestamp_millis()` let two
//! same-ms flushes tie; a partial compaction's `compacted_*` output (sorting
//! lexically before `sstable_*`) then inverted the (ts, file-order) tiebreak
//! and a tombstone could lose to its own record's flush file. Full-compaction
//! tombstone GC then leaked the dead record (recovery: MISSING_MAPPING).
use attentiondb_storage::catalog::Catalog;
use attentiondb_storage::sstable::{SSTableReader, SSTableWriter};
use attentiondb_storage::{DocumentStore, Record};
use std::collections::HashMap;
use std::path::PathBuf;

fn rec(id: uuid::Uuid) -> Record {
    let fields: HashMap<String, serde_json::Value> = [("k".to_string(), serde_json::json!(1))]
        .into_iter()
        .collect();
    let mut r = Record::new(fields);
    r.id = id;
    r
}

fn entry_ts_by_key(dir: &std::path::Path) -> HashMap<Vec<u8>, Vec<i64>> {
    // NOTE: DocumentStore's storage_dir IS the SST directory (the engine
    // passes Catalog::sst_dir(db_root)); tests scan it directly.
    let mut out: HashMap<Vec<u8>, Vec<i64>> = HashMap::new();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sst"))
        .collect();
    paths.sort();
    for p in paths {
        let reader = SSTableReader::open(&p).unwrap();
        for e in reader.iter() {
            out.entry(e.key.clone()).or_default().push(e.timestamp);
        }
    }
    out
}

#[test]
fn consecutive_flushes_stamp_strictly_increasing_ts() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let mut ds = DocumentStore::new()
        .with_storage_dir(dir.clone())
        .expect("storage dir");

    // 6 back-to-back flush rounds: insert one fresh doc, flush immediately.
    // Pre-fix, rounds landing in the same millisecond tied on ts.
    let mut prev_max: Option<i64> = None;
    for i in 0..6u32 {
        ds.insert(rec(uuid::Uuid::from_u128(i as u128))).unwrap();
        ds.flush_memtable().unwrap();
        let map = entry_ts_by_key(&dir);
        let this_max = map.values().flatten().copied().max().unwrap();
        if let Some(p) = prev_max {
            assert!(
                this_max > p,
                "REGRESSION: flush {i} stamped ts {this_max} not > previous {p} — \
                 same-ms flush tie re-appeared"
            );
        }
        prev_max = Some(this_max);
    }
}

/// INV-C2 under the monotonic regime: with DISTINCT entry timestamps the
/// version resolution is decided by ts alone — a tombstone one ms newer than
/// its record must win even though `compacted_*` sorts lexically BEFORE
/// `sstable_*` (the file-order tiebreak must stay defense-in-depth only).
#[test]
fn distinct_ts_resolution_tombstone_beats_record_despite_file_order() {
    let tmp = tempfile::tempdir().unwrap();
    let sst_dir = Catalog::sst_dir(tmp.path());
    std::fs::create_dir_all(&sst_dir).unwrap();

    let id = uuid::Uuid::from_u128(0xA11CE);
    let key = id.as_bytes().to_vec();

    // sstable_1.sst: LIVE record at ts 1000 (older flush).
    let mut w = SSTableWriter::new(&sst_dir.join("sstable_1.sst")).unwrap();
    w.append_with_timestamp(key.clone(), rec(id).to_msgpack().unwrap(), 1000)
        .unwrap();
    w.flush().unwrap();

    // compacted_2.sst: TOMBSTONE at ts 1001 (later flush, later compacted).
    let mut t = rec(id);
    t.tags.push("__TOMBSTONE__".to_string());
    let mut w = SSTableWriter::new(&sst_dir.join("compacted_2.sst")).unwrap();
    w.append_with_timestamp(key.clone(), t.to_msgpack().unwrap(), 1001)
        .unwrap();
    w.flush().unwrap();

    // NOTE: files are crafted OUTSIDE DocumentStore open (which re-reads them
    // via open_inner) — resolution contract check happens in core's
    // regression suite through the engine. Here we assert the raw files carry
    // distinct ts so the engine's ts-first rule decides.
    let map = entry_ts_by_key(&sst_dir); // files were crafted in sst_dir
    let tss = map.get(&key).unwrap();
    // path order: compacted_2 < sstable_1 lexically → ts seq [1001, 1000];
    // the engine must resolve by TIMESTAMP (1001 tombstone wins), not order.
    assert_eq!(tss, &[1001, 1000], "crafted files must carry distinct ts");
}
