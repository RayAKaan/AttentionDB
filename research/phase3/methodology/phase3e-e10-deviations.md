# Phase 3E — E10 Deviations & Adaptations

Author: Rayyan Kaan (RayAKaan). All deviations documented at or before
corrective measurement; raw runs immutable; failures preserved.

## D38 — Ladder adaptation to the resource class
The suggested ladder (…300k) was adapted to the 1.9 GiB host: tiers above
80k (primary) are M3 budget-BLOCKED from the measured 17.2 KB/doc marginal,
and the slim configuration hit a kernel OOM at ~100k. The executed ladder is
40k/60k/80k + budget-blocked 100k/150k/200k rows (§8's adaptation clause).

## D39 — SCALE-006 pre-flight used an unmeasured marginal
The slim-120k pre-flight passed on a 6.0 KB/doc GUESS; the run was OOM-killed
at ~100k docs (kernel evidence preserved). Corrective action: slim 60k rung
measured the true marginal (16.5 KB — payload-independent), the harness
telemetry was made crash-durable (incremental flush), and no higher tier was
attempted without a measured marginal. Classification OBSERVED LIMIT.

## D40 — SCALE-DEFECT #1: head-count-blind hygiene trigger (engine policy bug, fixed)
E9's INV-E9-HYGIENE deadness ratio (`vstore×2 > mapped×3`) ignored that every
head stores every live doc — any multi-head collection with zero dead entries
fired a full deterministic rebuild at EVERY checkpoint (measured 77,809 ms at
2 heads / 157,232 ms at 4 heads / 40k docs, PH3E-SCALE-008/009). Fix:
deadness = vstore − heads×mapped, degraded when > INDEX_REBUILD_MIN_VSTORE
and > 1.5× mapped. Post-fix: ckpt 24-32 ms, peaks −33-37 % (SCALE-012/013).
Regression: `e9_hygiene_multi_head_fresh_checkpoint_must_not_rebuild` (suite
330/0). Semantics preserved: rebuilds remain the sealed recovery-path
operation; single-head E9 behavior unchanged (existing regression green).

## D41 — SCALE-014 INVALIDATED (harness upsert defect; engine exonerated)
The churn harness's "reinsert" branch minted a second live uuid for an
already-live idx; the engine stored both versions faithfully while the model
tracked one (counts 81,419 vs 48,148). A minimal probe proved scan_filtered
excludes deleted documents pre/post checkpoint and all model-tracked docs
were exact. Fixed to strict upsert semantics with asserted deletes; rerun as
SCALE-018 (counts exact 48,148/48,148).

## D42 — SCALE-015 configuration error (OBSERVED LIMIT)
The first concurrency run used a 100k base (130k docs projected ≈2.2 GB),
violating the frozen M3 discipline, and its writer loop lacked the M4 guard;
the wall-clock terminated it. Preserved (partial telemetry + compressed db
inventory); redesigned (40k+20k, guards, paced reader) as SCALE-019. Two
operational lessons recorded: long runs must be bounded by an inner timeout,
and the reader must pace itself (a spin-loop reader starved the writer).

## D43 — Async-WAL-tail loss observed at SCALE-019 (documented boundary, not a bug)
The first 019 attempt reopened WITHOUT a final checkpoint in async durability
and lost 7 acked writer docs — the E8f documented buffered-WAL-tail boundary,
reproduced at scale on a graceful reopen. The sealed E8 pattern (final
checkpoint before reopen) was applied; the rerun lost nothing (60,000/60,000).
Sync-mode guarantees are untouched (E9 regression soaks were sync and
count-identical). This CONFIRMS the E8f boundary at scale; it does not weaken
E2's contract vocabulary.

## D44 — SCALE-017 classification gap (harness); rerun as SCALE-021
The retrieval run's classifier initially applied only the M4 guard, not the
M6(e) recall gate; 017 measured self-hit 93/100 (<95%) but stamped VERIFIED.
The original artifacts are preserved untouched with NOTE-CLASSIFICATION.md;
the harness now honors the gate; SCALE-021 (99/100, VERIFIED) plus 017
establish that grown-graph recall at 80k is NONDETERMINISTIC at the gate
(93-99 across runs; recovery/hygiene rebuilds restore 99-100).

## D45 — Compact timing semantics
SCALE-020's explicit compact_storage measured 0 ms: the preceding checkpoint
already flushed AND internally compacted, so there was nothing to reclaim.
A dedicated maintenance run (SCALE-016) with 6k deletes measured real
reclamation: 426 ms, SST 3→1, 37.1→35.0 MB at 60k docs.

## D46 — Artifact policy under the persistence cap
Green-run db payloads pruned (regenerable, spec M15); failure-run db states
compressed then inventoried (ls + sha256) before payload removal — the OOM
db content is deterministically regenerable from the recorded seed/kill
point. Failure notes (OBSERVED.md / INVALIDATED.md / NOTE-CLASSIFICATION.md)
are the primary classification records. TRIAGE updated (entries 10-11).

## D47 — No threshold tuning
The E9 hygiene thresholds were NOT tuned for performance: the multi-head fix
is a correctness fix for a misfiring predicate (D40), motivated by the defect,
not by benchmark results; its effect on legitimate rebuilds is unchanged.
No other engine constant was touched.

## D48 — Post-run lint/hygiene repairs (no behavior change)
After the last E10 run closed, three compile-hygiene repairs landed before the
final validation: an unnecessary same-type cast and a `%5==0` →
`is_multiple_of(5)` (both clippy lints in the D40 fix line and the telemetry
flush), and restoration of the `regression_e9_index_hygiene.rs` test
attributes that the E10 insertion had scrambled (a duplicated `#[test]` on the
new regression had exactly masked the E10-insertion's removal of the
original E9 test's `#[test]`, so the suite count stayed 330 while the original
skipped). Both hygiene tests now execute; suite re-verified 330/0, clippy 0,
both checkers PASS at the final tree sha16 35ea3b7e5c7c0427. No engine or
harness behavior changed; all prior run classifications stand.
