# Phase 3E — E8 Specification: Soak / Long-Running Reliability (FROZEN before measurement)

Date frozen: 2026-09-17. Commit under test: `3e65d50f8fd1a81cda339d4608260d6be98ba708`
(tree dirty with sealed E7 artifacts, tree sha16 `7ccdc6e15450bb23`).
E7 verdict (D, single-op linearizability subset) is the entering contract; E8 tests
whether it SURVIVES sustained operation. E8 measures and documents; it does not
optimize (memory issues belong to E9; scale expansion to E10; fault injection to E11).

## Part I — Reliability properties S1–S18 (what is measured, per property)

- S1 state correctness — engine post-recovery consistency checker clean at every
  verification point (in-process AND fresh-process).
- S2 reference-model equality — exact equality of the harness's independent model
  (per-collection: idx → (uuid version, num, live)) against engine export state,
  at every verification point. The model is mutated in lock-step with issued ops,
  NEVER read back from the engine (anti-circular, §8).
- S3 WAL integrity — fresh open replays without error; sequence continuity across
  rotations; no duplicate/missing committed records after restart.
- S4 SST integrity — checker SST validation clean; compaction output loads.
- S5 ID-map integrity — every live uuid maps to a numeric id; retired uuids never
  remap (spot-verified against the model's full per-idx version history, 50 sampled
  retired versions per verification).
- S6 tombstone correctness — deleted uuids remain absent forever (model: deleted
  (idx,version) never reappears; verified via export + read probes).
- S7 transaction atomicity — after ANY restart (graceful or groupkill), every
  modeled committed transaction is fully present, every rolled-back/staged one
  fully absent. Never partial.
- S8 recovery stability — repeated open/recovery produces identical state
  (restart equality of state hashes) and never degrades.
- S9 checkpoint stability — repeated checkpoint cycles keep watermark/trim
  behavior stable; WAL does not grow unboundedly across checkpoints.
- S10 compaction stability — repeated compactions preserve logical state; no
  resurrection; SST count bounded by policy.
- S11 backup stability — repeated backup→restore→checker→model-snapshot-equality
  cycles succeed; generations never cross-contaminate.
- S12 concurrency progress — reader/writer workers complete their op budgets.
- S13 absence of deadlock — watchdog-distinguished: progress (ops advancing)
  vs. stuck (no heartbeats for the stall interval); latency spikes alone are
  never classified as deadlock (E7 p99 spikes were scheduler noise).
- S14 resource stability — FD count, thread count, file counts tracked; stepwise
  FD/thread growth or unexplained file accumulation flagged (§31/§32 classes).
- S15 bounded file growth — WAL/SST/db-dir bytes tracked vs ops; classified
  against the documented lifecycle (WAL trim at checkpoint; SST compaction policy).
- S16 bounded memory behavior — RSS/VmSize sampled on a fixed interval and
  classified (§31): bounded plateau / linear / stepwise / unexplained. E8 does
  not fix memory findings; they are documented for E9.
- S17 deterministic restart behavior — same logical prefix + same restart point
  ⇒ same recovered state (kill-point matrix R1–R7, §59).
- S18 long-run retrieval correctness — deleted documents never returned by
  attend/scan; live docs remain addressable; filters sound (spot checks at
  verification points; no new retrieval benchmark).

## Part II — Families and operation budgets (fixed BEFORE running; wall-clock is
secondary to the deterministic op budget; §44 adaptation documented in deviations)

| family | run ID | budget | workload | cadence (deterministic) |
|---|---|---|---|---|
| E8a smoke | PH3E-SOAK-001 | 30 000 ops | mixed 70/15/15 on 500 keys, 2 readers | ckpt 5k, compact 10k, backup 15k, restart 10k, verify 3k |
| E8b mutation | PH3E-SOAK-002 | 200 000 ops | 2 000 keys, hot set 40 (2 %), 70 % update/upsert (30 % of updates on hot set), 15 % insert, 15 % delete | ckpt 25k, compact 50k, restart 100k, backup 150k, verify 10k |
| E8c mixed rw | PH3E-SOAK-003 | 2×60 000 writer ops + 3 hot readers, 120 s cap | disjoint writer ranges, readers validate shape per E7 contract | verify 20 s; exact census at 2 quiesced barriers |
| E8d txn | PH3E-SOAK-004 | ladder sizes 1/2/5/10/25/50/100 × 40 commits each + 15 000 mixed txns (25 % rollback) + 1 000 single ops | E6 TxnOp model; 2 KiB WAL segments in phase 3 | verify per phase; restart between phases |
| E8e maintenance | PH3E-SOAK-005 | 12 cycles × 1 000 ops | cadence §21: 100→ckpt, 200→rotation, 500→compact, 750→backup, 1000→restart | 2 KiB segments; census + verify per cycle |
| E8f restart/recovery | PH3E-SOAK-006 | 60 graceful cycles × 2 000 ops + kill matrix R1–R7 × 2 reps + async Case A/B | child process + groupkill at deterministic points; model checkpoint JSON = expected state | verify after EVERY restart |
| E8g lifecycle | PH3E-SOAK-007 | 25 cycles × (200 ops + INSERT→UPDATE→DELETE→REINSERT→TXN→CKPT→ROTATE→COMPACT→BACKUP→RESTART→VERIFY) | 3 collections (300 keys each); per-collection independent comparison | full §39 pipeline per cycle |
| E8h resource | PH3E-SOAK-008 | 240 s continuous steady-state | FIXED 800-key space, live count pinned ≈600 by balanced churn; ckpt 10k, compact 30k | telemetry 2 s |
| E8i extended | PH3E-SOAK-009 | 600 s continuous mixed integrated | multi-collection + txns + maintenance cadence + readers + telemetry | verify 60 s |

## Part III — Harness rules

- Operation log (`oplog.csv`): seq, ts_us, thread, op, coll, idx, ver, num,
  txn, argsha, result — append-only, buffered, flushed every 2 000 ops. The
  workload is fully determined by (seed, budget, cadence); the log records
  every realized operation with its result (§9).
- Model checkpoints (`model_ckpt_NNNN.json`) every 10 000 ops and at every
  maintenance/restart boundary: {seq, state_hash, live_count, per_coll counts,
  live_set_hash} (§10). The oplog + seed regenerate everything else.
- Resource telemetry (`resource.csv`) sampled every 2 s by a dedicated thread:
  elapsed_s, op_seq, vm_rss_kb, vm_size_kb, threads, fds, wal_bytes, sst_bytes,
  db_bytes, wd_checks (§30). WD watchdog thread: samples op progress every 5 s;
  no progress for 120 s ⇒ dump diagnostics to `stall.txt`, abort (§16/§17/§52).
- State hash: FNV-1a-64 over canonical sorted `coll|idx|ver|num` lines (§34) —
  independent of map/SST/thread order.
- Verification point = model equality (all collections) + engine checker +
  mapper S5 checks + S6 spot probes + optional retrieval spot check (S18).
  Results appended to `verif.csv`.
- Failure protocol: first failure ⇒ preserve everything (§36), stop, classify.
  INVALIDATED runs stay in raw (§49). Engine bug ⇒ §50 protocol (new IDs for
  reruns; E1–E7 regression coverage). Memory findings ⇒ measure/classify only
  (§51).
- Registry: PH3E-SOAK-001..009. Status vocabulary §48. Final report structure
  §62 with capability matrix §63 and numerical summary §64.
