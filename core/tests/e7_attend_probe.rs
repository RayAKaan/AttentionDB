//! E7 investigation #2: under concurrent readers (>=8 threads on 2 vCPUs),
//! attend() occasionally returns doc id 120 for a 120-doc collection
//! (ids must be 0..=119). Read-only workload: no writers at all. Harness
//! evidence: PH3E-CONC-001 E7a rows (kinds breakdown: attend_bad_id only).
use attentiondb_core::engine::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;
use std::sync::Arc;

fn rec(idx: u32) -> Record {
    let mut f = HashMap::new();
    f.insert("idx".to_string(), serde_json::json!(idx));
    f.insert("num".to_string(), serde_json::json!(idx));
    let mut r = Record::new(f);
    r.id = uuid::Uuid::from_u128(((7000u64 + idx as u64) as u128) << 64 | 1);
    let mut v = vec![0.0f32; 32];
    v[(idx as usize) % 32] = 1.0;
    r.k_vecs.insert("h".to_string(), v);
    r
}

#[test]
fn attend_concurrent_readers_never_return_invalid_ids() {
    let dir = std::env::temp_dir().join("e7-attend-probe");
    let _ = std::fs::remove_dir_all(&dir);
    let e = Arc::new(AttentionEngine::open_dir(&dir, Durability::Sync).unwrap());
    e.create_collection("bench", 32, &["h"]).unwrap();
    for i in 0..120u32 {
        e.insert_document("bench", rec(i)).unwrap();
    }
    // single-threaded exhaustive probe BEFORE checkpoint: does id 120 exist?
    {
        let mut seen_pre = std::collections::BTreeSet::new();
        for r in 0..200u32 {
            let mut v = vec![0.0f32; 32];
            v[(r as usize) % 32] = 1.0;
            if let Ok(res) = e.attend("bench", &["h".to_string()], &v, 10) {
                for (id, _) in res {
                    seen_pre.insert(id);
                }
            }
        }
        println!(
            "PRE-checkpoint exhaustive: has-120={} distinct={}",
            seen_pre.contains(&120),
            seen_pre.len()
        );
        // offender forensics (pre-checkpoint, single-threaded)
        for r in 0..200u32 {
            let mut v = vec![0.0f32; 32];
            v[(r as usize) % 32] = 1.0;
            let det = e
                .get_collection("bench")
                .unwrap()
                .attend_detailed(&["h".to_string()], &v, 10, None, None, None, None, None)
                .unwrap();
            if det.iter().any(|c| c.id == 0 || c.id > 120) {
                let bad: Vec<_> = det
                    .iter()
                    .filter(|c| c.id == 0 || c.id > 120)
                    .map(|c| (c.id, c.final_score))
                    .collect();
                println!("offending query r={r}: results={}", det.len());
                println!("  bad={bad:?}");
                break;
            }
        }
    }
    e.checkpoint().unwrap();
    // single-threaded exhaustive probe: does id 120 exist at all?
    {
        let mut seen = std::collections::BTreeSet::new();
        for r in 0..200u32 {
            let mut v = vec![0.0f32; 32];
            v[(r as usize) % 32] = 1.0;
            if let Ok(res) = e.attend("bench", &["h".to_string()], &v, 10) {
                for (id, _) in res {
                    seen.insert(id);
                }
            }
        }
        let out_of_range: Vec<u64> = seen
            .iter()
            .filter(|&&id| id == 0 || id > 120)
            .copied()
            .collect();
        println!(
            "single-thread exhaustive: distinct={} out-of-range={:?}",
            seen.len(),
            out_of_range
        );
        // find the first offending query + its scores via attend_detailed stats
        for r in 0..200u32 {
            let mut v = vec![0.0f32; 32];
            v[(r as usize) % 32] = 1.0;
            let det = e
                .get_collection("bench")
                .unwrap()
                .attend_detailed(&["h".to_string()], &v, 10, None, None, None, None, None)
                .unwrap();
            if det.iter().any(|c| c.id >= 120) {
                let bad: Vec<_> = det
                    .iter()
                    .filter(|c| c.id >= 120)
                    .map(|c| (c.id, c.final_score))
                    .collect();
                println!("offending query r={r}: results={}", det.len());
                println!("  bad={bad:?}");
                println!(
                    "  all ids={:?}",
                    det.iter().map(|c| c.id).collect::<Vec<_>>()
                );
                break;
            }
        }
        println!(
            "POST-checkpoint exhaustive: has-120={}",
            seen.contains(&120)
        );
    }
    let mut qv = vec![0.0f32; 32];
    qv[3] = 1.0;
    let q: Arc<Vec<f32>> = Arc::new(qv);
    let bad = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let bad_ids = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let mut handles = Vec::new();
    for _t in 0..16usize {
        let e2 = e.clone();
        let bad2 = bad.clone();
        let ids2 = bad_ids.clone();
        let q2 = q.clone();
        handles.push(std::thread::spawn(move || {
            let mut local = 0usize;
            for _ in 0..20_000usize {
                if let Ok(res) = e2.attend("bench", &["h".to_string()], &q2, 10) {
                    // valid numeric ids: 1..=120 (mapper mints from 1; the
                    // 120-doc collection owns exactly this range — verified
                    // against id_mapper in e7_tiny_probe.rs)
                    for (id, _) in res {
                        if id == 0 || id > 120 {
                            local += 1;
                            let mut g = ids2.lock().unwrap();
                            if g.len() < 8 && !g.contains(&id) {
                                g.push(id);
                            }
                        }
                    }
                }
                std::thread::yield_now();
            }
            bad2.fetch_add(local, std::sync::atomic::Ordering::Relaxed);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let total = bad.load(std::sync::atomic::Ordering::Relaxed);
    let ids = bad_ids.lock().unwrap().clone();
    e.close().unwrap();
    assert_eq!(
        total, 0,
        "attend returned invalid ids {total} times under 16-thread read load; ids={ids:?}"
    );
}
