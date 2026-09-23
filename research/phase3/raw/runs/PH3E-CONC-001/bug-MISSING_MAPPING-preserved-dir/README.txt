PRESERVED FAILING STATE — E7 engine defect #1 (found & fixed during PH3E-CONC-001)

Defect: committed transaction [Delete U; Insert same U] applied the reinserted
document under numeric id 0 with NO id mapping (the txn's own Delete arm had
already retired the mapping; commit-apply used uuid_to_id().unwrap_or(0)).
Live commits returned Ok; a later checkpoint persisted the orphan; fresh
open_dir REFUSED with "Recovery failed: 1 consistency error(s)" and checker
code MISSING_MAPPING (orphan record; uuid 00000000-0000-17d4-0000-000000000000).

This directory is the EXACT on-disk state produced by the pre-fix engine
(core/tests/e7_replay_probe.rs::replay_same_uuid_delete_reinsert_two_txns,
Durability::Sync): plain insert (uuid = 6100<<64), checkpoint, two committed
same-uuid [Delete,Insert] txns (num 100, 200), clean close (checkpoint at
close). Fresh open on this state fails; live total_vectors was 3 pre-close.

Fix (core/src/engine.rs, commit-apply TxnOp::Insert): re-register a fresh
numeric id when the mapping is absent. Regression: e7_replay_probe (2 tests,
incl. WAL no-close variant). Discovered 2026-09-17 during E7g/E7w harness
same-uuid rewrite; minimal repro + diagnosis in phase3e-e7-deviations.md.
