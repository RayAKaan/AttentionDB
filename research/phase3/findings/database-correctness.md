# PH3D Finding — Database Correctness

Status: **SUPPORTED** (within the tested envelope) · Run: PH3D-STATE-001..004, PH3D-FILTER-001 · Hardware/config: sandbox Linux x86_64, DIM=32, 1 head `h`, Durability=Async for model runs, PH3D harness build of HEAD.

## Finding
The engine maintains a reference-model-equivalent logical state under a mixed deterministic
operation stream. Across 3 × 1000-op and 1 × 100-op runs (seeds 42/7/1), with weighted
insert/update/upsert/delete/query/filter/checkpoint/compact/restart operations:

- observed state after restart, after every offline compaction, and at the end exactly equals
  the reference model (`expected-state.json == observed-state.json`, 926/936/925 logged ops);
- no dead (deleted) document was ever returned (`inv_live_only`, `final_dead_not_retrievable`);
- repeated queries return identical results (`query_determinism`);
- a second collection's documents never leak into the first
  (`collection_isolation`; the +1-doc export artifact that motivated collection-membership
  filtering in the harness was a harness defect, not an engine defect);
- filters never returned a non-matching document in ~380 filtered probes
  (`inv_filter_leak` = 0), and zero-match filters always returned ∅.

## Evidence
`results/state-machine.csv` (0 failures / 24 gated checks), `results/mutations.csv`
(state EQUAL on all three 1000-op runs), `results/filtering.csv`.
Audit closure extends evidence: PH3D-INTEGRATION-001 (multi-collection isolation with
namespaced ids across live/restart/compaction/restore, 16/16; documents the global-uuid
membership semantics) and PH3D-CONC-003 (merged concurrent op logs replay to the EXACT
observed state, 450 docs; same-key contention 0 torn records).

## Limitations
Single-node, single collection pair, 200-doc working set, DIM=32, one head. Retrieval
*quality* is out of scope here (Phase 3B/3C cover it). No isolation/linearizability claim —
see the concurrency finding. Soundness of filters is asserted for the probe shapes exercised;
completeness is candidate-bound by design (0.967–1.0 measured recall).
