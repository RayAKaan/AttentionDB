# Phase 3E / E5 — Deviations, Boundary Notes & Post-run Corrections

Experiment: **PH3E-COMPACT-004** (supersedes PH3E-COMPACT-003) — online compaction,
tombstone safety & concurrent storage lifecycle. Date: 2026-09-17. All deviations are
documented addenda; no raw data was altered.

## D1. PH3E-COMPACT-003 superseded by PH3E-COMPACT-004 (harness completeness gap)

The first recorded run (PH3E-COMPACT-003, 24/24 cells MATCH) was recorded before two
harness-completeness items were recognized as required by the E5 specification: the
**collection-isolation-under-compaction family** and **reader-latency percentiles**
(p50/p99 + blocked-vs-errored semantics). Engine code was unchanged between the two
runs; the harness gained one cell and latency capture. Per the raw-immutability rule,
PH3E-COMPACT-003 is preserved (registry: SUPERSEDED) and PH3E-COMPACT-004
(25/25 MATCH) is the sole basis for E5 claims.

## D2. Equal-timestamp bug — where the proof lives

The fixed INV-C2 tie-break bug is proven at the **storage API level**
(`test_compaction_timestamp_tie_breaks_to_later_file`): timestamps are injectable
there, so the equal-millisecond condition is deterministic. At the engine level the
same-millisecond collision is timing-dependent and cannot be forced deterministically;
the engine-level multigen cell covers the same-uuid cross-generation update chain
model-checked against the sidecar. This split (API-level deterministic proof +
engine-level model check) is the honest boundary of the evidence.

## D3. Publication order differs from the prompt's illustrative sketch

The prompt sketched "install/reload reader state → retire obsolete SSTs". The
implementation publishes **output → unlink inputs → reader-list swap**, because
recovery SCANS the sst/ directory (no manifest names SSTs) and readers materialize
entries in memory at open: any order is recovery-safe, and cleanup-then-reload keeps
the published reader list exactly equal to the on-disk set (a reload-first list would
transiently name files that cleanup then removes — harmless but inelegant).
The choice is justified by crash tests at all four instrumented windows, including
the two dangerous ones (output+inputs present; inputs removed + old reader list).

## D4. Engine-level equal-ms collision frequency not measured

No claim is made about how often equal-millisecond ties occur in production; the
claim is that IF they occur, compaction and open resolve them identically (proven).

## D5. Compaction failure injection = crash gates only

There are no injectable write-failure hooks in the compaction path. Per the spec, the
crash-gate groupkill mechanism is the strongest safe failure injection used; a general
fault-injection matrix is E11 scope and was NOT started. NOT_INSTRUMENTED boundaries:
none — all four planned gates exist and fired (marker-verified).

## D6. Generator defects fixed post-E4 (results-identical)

While wiring E5 into the results pipeline, two generator defects were found and
fixed: (1) `_gen_e4(mismatches)` received the mismatch counter BY VALUE, so E4
mismatches could never propagate to the generator's exit code (E4 results were all
MATCH, so no recorded outcome changes — but the FAIL-on-mismatch guarantee was
weaker than documented); (2) `gen_e5` initially returned the row COUNT instead of the
mismatch count. Both fixed; the generator now fails on any E4/E5 mismatch. No raw
data affected; results regenerated and identical except the corrected exit
semantics.

## D7. Mode assignment

Concurrency cells (readers/writer/multi-writer/rotation/collections) run under group
mode; deterministic cells under sync. The mode dimension of compaction itself is
covered by design: compaction is mode-independent (it touches SSTs, not the commit
path); the WAL interaction cell forces rotations during compaction under group mode.

## D8. In-process vs fresh-process crash evidence

Every crash-window verification opens the crashed directory in a FRESH process
(the controller) after the child process group was SIGKILLed at the gate; the
in-process assertions inside the crashed child are never used as evidence.
