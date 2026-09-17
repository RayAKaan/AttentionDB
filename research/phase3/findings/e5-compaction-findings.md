# E5 Findings — Compaction (PH3E-COMPACT-004, 2026-09-17)

Generated from results/e5-compaction.csv + results/e5-crash-windows.csv.

1. **Coordinated compaction (C1) is real and verified**: `Engine::compact_storage`
   holds the mutation gate across boundary->publish; all 25 cells MATCH with the
   independent sidecar/value model, not just the checker.
2. **Writers pause, bounded and measured** (max op 1.8-2.3 ms incl. gate wait at
   tested sizes); **readers block, never error** (p50 53 us, p99 116 us, n=604,
   0 errors during a real merge).
3. **Tombstone GC is provably safe only under full merge**; partial merge retains
   tombstones (verified at the storage API: merge 4 of 5 files -> tombstones
   retained in output).
4. **Equal-millisecond same-key versions**: latent bug fixed (compaction now
   resolves ties exactly like open/recovery: later file wins). Regression test
   added; fails under the old rule.
5. **Fresh-process crash recovery at 4 instrumented windows** (incl. the two
   dangerous ones: output-present/inputs-present, inputs-removed/readers-old)
   always restarts to the same valid logical state; no tmp artifacts; second
   compaction accepted.
6. **Publication order output -> unlink -> reader swap** is justified by
   scan-based recovery + in-memory reader materialization and is crash-verified.
7. C2 (fully online, zero writer pause) remains **UNSUPPORTED** by the gate
   architecture; nothing in E5 changed that, and no claim is made.
