# Phase 3E — E10 Final Report
## Scale-Envelope Expansion & Capacity Characterization

Author: Rayyan Kaan (RayAKaan). Baseline: commit `3e65d50f8fd1a81cda339d4608260d6be98ba708`,
E9 tree sha16 `a4ce3224bc9a3b3d`; E10 closes at tree sha16 `35ea3b7e5c7c0427`
(one engine-policy fix + harness work, all regression-sealed). All scale claims
are scoped to the tested 2-vCPU / 1.9 GiB RAM resource class.

## 1. Executive Summary

E10 answered the charter question — what can AttentionDB sustain while
preserving its verified guarantees — with 21 registered runs across 14
families on a frozen methodology (spec M1–M15). The verified envelope on this
host: **40,000 documents (primary config, full gate battery) and 60,000
documents (slim config and the integrated 82,000-op lifecycle with 2,000/2,000
transactions), all counts exact, checkers clean, retrieval self-hit 97–100/100**.
80,000 documents reproduced with every correctness gate green but crossed the
85%-memory guard during measurement (peak 1.32–1.38 GB; per-doc marginal
16.5–17.2 KB/doc, payload-independent); grown-graph retrieval recall at 80k is
nondeterministic at the 95% gate (93–99/100 across runs) and is restored to
99–100 by any recovery/hygiene rebuild. 100k+ is budget-blocked for the
primary config; the slim config hit a preserved kernel OOM at ~100k. E10 found
and fixed one genuine engine-policy defect (SCALE-DEFECT #1: the E9 hygiene
trigger was head-count-blind, forcing 78–157 s rebuilds at every multi-head
checkpoint; post-fix 24–32 ms, peaks −33–37%), exonerated the engine on two
harness defects, and measured a real concurrency bottleneck (writer ~24 ops/s
under a scan-heavy reader at 60k, vs ~700 solo). Verdict: **envelope
characterized with high confidence; B** (bounded by RAM, honest boundaries
everywhere, one defect found+fixed+sealed). E10 COMPLETE; E11 NOT STARTED.

## 2. Baseline and Repository State

HEAD `3e65d50f…` (main), E9 working tree intact, registry at E9 close = 143
entries, suite 329/0, clippy clean, both phase checkers PASS at entry. The
sandbox reset between E9 and E10 wiped toolchain + target; restored (rustc
1.98.1 — identical version), rebuilt, and all gates re-verified BEFORE any
scale experiment (§4 of the charter). One gate item surfaced: the emptied
PH3E-SOAK-010 directory shell had been dropped by the platform cap;
recreated with PLATFORM-LOSS-NOTE.md (its record lives in registry + report);
checker PASS after.

## 3. Environment

2 vCPU (kernel 6.1.158+), 1.9 GiB RAM (1.4 GiB available at capture), 20 GiB
disk, ext2/ext3, rustc/cargo 1.98.1, python 3.13.14. Durability Sync for all
scale runs unless noted. Every run records config.json + telemetry.csv
(2 s cadence, crash-durable) + summary.json + the read-only component census.

## 4. Frozen Methodology

`phase3e-e10-spec.md` frozen before any official result: M1 baseline
reproduction; M2 tier order; M3 pre-flight budget gate (launch forbidden if
projected peak > 80% of MemAvailable, using the MEASURED per-doc marginal of
the same configuration); M4 live guard (halt at 85%); M5 OOM discipline;
M6 correctness gates (checker, exact counts, ≥5,000-sample model equality,
reopen equality, ≥95% self-hit at k=10); M7 primary configuration (shipped
HNSW defaults: M=16, efC=400, efS=64, max_elements=100k, store_vectors=true);
M8 clustered+near-duplicate dataset with mixed-length payloads and dataset
hashing; M9 frozen 100-self/100-noise query sets; M10 family→ID map;
M11 telemetry reuse (E9 instruments); M12 anti-circular harness-side model;
M13 statistics; M14 stop conditions; M15 artifact policy. No threshold was
changed after seeing results (the one trigger change is D40's correctness
fix, with regression + new-ID reruns, not tuning).

## 5. Dataset Generation

Deterministic synthetic corpus per M8: 50 hash-derived cluster centroids,
member vectors = normalize(0.8·centroid + 0.2·noise), every 16th document a
near-duplicate of its predecessor, five payload fields with mixed title
lengths (1–7 words), a hot subset (idx%1000<50) for churn families, uuid =
version-stamped FNV scheme. dataset_hash recorded per run (e.g. 80k primary =
`70d13928e663cf55`). The generator is pure hash math — independent of any
retrieval result (anti-circular).

## 6. Resource Budget

M3/M4 as frozen. Measured per-doc marginal: 17.2 KB (primary), 16.5 KB (slim)
— payload size is NOT the driver; the cost lives in record plumbing, index
structures, and the read-path caches. Projections from these measurements
blocked 100k/150k/200k launches before any memory pressure could occur, and
the 120k slim attempt that launched on a guess became the preserved OOM
boundary (D39). Live headroom at the largest completed rungs: 399–868 MB
MemAvailable remaining.

## 7. Baseline Reproduction

PH3E-SCALE-001 re-ran the E9-sealed 80k scale with the new harness: 80,000
docs in 104.4 s (~767 docs/s), checkpoint 46 ms, reopen 83.7 s, peak
1,381,616 KB, all correctness gates green (counts exact, 5,000/5,000 samples,
checker clean, self-hit 97/100 post-build and 99/100 post-reopen), hygiene
census exactly at live (vstore 80,000). M1 SATISFIED; the guard fired during
the measurement phase (scan materialization + caches crossed 85% of
available), honestly classified OBSERVED_LIMIT-with-green-gates.

## 8. Document-Count Scaling

Ladder: 40k VERIFIED (peak ~700 MB class), 80k reproduced-with-boundary
(§7), 100k/150k/200k M3-blocked with explicit projections
(1.75/2.61/3.47 GB vs 80% ≈ 1.16 GB). The count ceiling on this host is
~95–100k docs (kernel OOM at ~100k on the slim run, PH3E-SCALE-006).
Throughput: ~767–1,100 docs/s insert at 32-dim; dim256 builds ~40% slower.

## 9. Memory Scaling

Peak RSS: 40k ≈ 0.70 GB, 60k ≈ 0.99 GB, 80k ≈ 1.32–1.38 GB — linear in
documents at 16.5–17.2 KB/doc across payload variants (R² ≈ 1 on the three
rungs). Composition (census): HNSW store+graph per head, DocumentStore
read-through record cache (unbounded by design, == live docs), bounded block
cache (50k), BM25 postings (~5/doc), INV-6 retired sets (churn-proportional).
Post-restart memory is repopulated from disk (read caches) — restart does NOT
reset the doc-proportional floor. The E9 hygiene keeps DEAD retention at ~0
at every checkpoint boundary even after 200k mutations (SCALE-018:
vstore 48,148 == live 48,148).

## 10. Storage Scaling

db dir at 80k: 56.0 MB (SST-dominant after checkpoint trim); WAL trims at
checkpoints (0 bytes steady-state); compaction reclamation measured at 60k
with 6k deletes: SST 3→1 files, 37.1→35.0 MB, 426 ms (SCALE-016).
Bytes/live-doc on disk ≈ 0.7 KB — storage is NOT the binding resource
(memory is, ~24× more per doc).

## 11. Index-Construction Scaling

Fresh build: 40k in ~50 s, 60k in 60–74 s, 80k in 94–104 s (~767–1,100
docs/s) — near-linear in the measured range with the deterministic
insert-order HNSW. Update-heavy reconstruction is the REOPEN path:
reopen = 44.6 s (60k) / 59.9 s (60k slim) / 83.7 s (80k) — dominated by
`rebuild_all_indexes` (the recovery contract); superlinearity was not
observed within the measured range (0.55–1.05 ms/doc steady).

## 12. Retrieval Scaling

Frozen query sets. Latency is FLAT with scale on the primary config:
p50 428–655 µs from 20k to 80k docs (single head), p99 0.8–1.4 ms, QPS
~1,900–2,460 single-client. Grown-graph self-hit recall at 80k:
93/100 (017) vs 99/100 (021) — NONDETERMINISTIC at the 95% gate; post-reopen
(rebuilt graph) self-hit is 99–100/100 everywhere (SCALE-001/016/020). This
is the hnsw_rs grown-graph shape effect near its 100k cap — a measured
reliability boundary, mitigated by the existing E9 hygiene rebuilds, NOT
tuned (ef_search frozen).

## 13. Head-Count Scaling

At 40k docs: heads 1/2/4 → build 50/85–89/166–172 s, vstore 40k/80k/160k,
retrieval p50 ~490/634/1,215 µs (per-head scan cost adds linearly).
Pre-fix checkpoints were catastrophic (77.8 s / 157.2 s — SCALE-DEFECT #1);
post-fix 32 ms / 24 ms with peaks −33–37% (833→561 MB, 1,371→862 MB).
Per-head cost is linear and predictable; correctness gates green at every
head count.

## 14. Vector-Dimension Scaling

At 20k docs: dim 32/128/256 → build 26.4/–/40.4 s equivalents (dim32 rung
from the ladder family), peak 0.28/0.35 GB class, p50 475/683 µs, ckpt
24–50 ms, self-hit 98–100/100. Memory and build scale ~linearly with vector
bytes (128→256 adds ~26% peak at the same doc count). Head-count and
dimension effects were measured in separate configurations (no conflated
runs).

## 15. Mutation-Churn Scaling

SCALE-018 (post-fix): 60k base + 200k mixed mutation ops (updates with
numeric remap, asserted deletes, reinserts; hot-key weighting): 260k ops
total, peak 914,932 KB (bounded — churn does NOT push RSS past the
doc-proportional envelope), hygiene held vstore == live (48,148) at every
checkpoint, retired ids 169,460 (INV-6), counts exact 48,148/48,148,
checker clean, self-hit 97/100. The first attempt (SCALE-014) is preserved
INVALIDATED — a harness upsert defect (duplicate live uuids), engine
exonerated by a minimal insert/delete/scan probe (D41).

## 16. E9 Hygiene at Scale

Hygiene behavior at 60–80k: purge+rebuild triggers exactly under the fixed
predicate; dead retention ≈ 0 at checkpoint boundaries after 200k mutations;
recall POST-hygiene 97–100/100 at all scales. The one defect was the
multi-head trigger (D40) — found BY scale testing, fixed, regression-sealed
(`e9_hygiene_multi_head_fresh_checkpoint_must_not_rebuild`), reruns under new
IDs. Rebuild cost when legitimately triggered remains significant (the 78–157 s
pre-fix measurements at 40k×2/4 heads bound a full multi-head rebuild;
single-head 60k rebuild ≈ 45–60 s inside reopen) — recorded as the documented
availability tradeoff of correctness-preserving hygiene.

## 17. Checkpoint Scaling

Checkpoint durations: 24–50 ms at 40k (single/multi-head, post-fix),
46 ms at 80k fresh, 436 ms at 60k after 20k churn ops (purge of ~20k dead
entries + flush), 50 ms at 60k with 6k tombstones. Checkpoint time is
dominated by (a) flush+fsync of the memtable and (b) purge/rebuild when
deadness triggers — both bounded and predictable under the fixed policy.

## 18. Compaction Scaling

Explicit compaction at 60k with real garbage: 426 ms, SST 3→1, −2.1 MB
(SCALE-016). Post-checkpoint compaction is a no-op (0 ms — flush already
compacted internally; SCALE-020), which is the documented internal-compact
behavior, not a missing phase. Checker + model equality verified after every
compaction; restart after compaction verified in 016/018/020.

## 19. Recovery Scaling

Reopen (WAL/SST load + id-map + deterministic index rebuild): 44.6 s (60k),
59.9 s (60k slim), 83.7 s (80k) — ~0.7–1.05 ms/doc, correctness-verified
after every reopen (exact counts + samples + checker). Recovery peak RSS is
the doc-proportional floor plus rebuild transients. Async durability: the
buffered-WAL-tail boundary was REPRODUCED at scale (7 acked docs lost across
a reopen without final checkpoint, SCALE-019 pre-fix attempt) and sealed by
the E8 final-checkpoint pattern — consistent with the E2/E8f vocabulary;
sync guarantees untouched (D43).

## 20. Bounded Concurrency at Scale

SCALE-019 (60k total: 40k base + 20k writer, 1 scan-heavy reader, periodic
checkpoints + one compaction): reader 4,581 batch-checks, latency p50
2.9 ms / p99 5.7 ms under concurrent writes; zero reader errors; stability
spots green throughout; final state exact (60,000/60,000) with self-hit
100/100. Measured degradation: writer throughput fell to ~24 ops/s under the
reader's continuous scan load (vs ~700–1,100 solo) — RwLock write-lock
contention on the shared store is the identified bottleneck (characterized,
NOT optimized — no E10 optimization campaign). The first attempt (015) is a
preserved OBSERVED_LIMIT (configuration error per D42).

## 21. Integrated Scale Validation

SCALE-020 at 60k: build 59.7 s → 20k churn ops → 2,000/2,000 committed
2-op transactions → checkpoint 436 ms (hygiene) → compaction (no-op) →
backup 28 ms → restart → verification: counts exact 60,197/60,197, checker
clean, self-hit 99/100, peak 1.08 GB. The full lifecycle preserves every
guarantee at the largest fully-verified integrated scale.

## 22. Correctness Results

Every VERIFIED run: consistency checker clean post-build AND post-reopen;
exact live counts; 5,000-sample (2,000 at small tiers) model equality with
0 bad; reopen equality; retrieval gate ≥95%. Every OBSERVED_LIMIT run has
green correctness where the gate battery ran (001, 017's caveat documented).
One engine-policy defect (D40) found, fixed, regression-sealed (suite
330/0). Two harness defects (D41, D42) exonerated the engine with probes and
were fixed with new-ID reruns. The async WAL-tail boundary (D43) is a
documented durability edge, not a correctness violation of any sealed
contract.

## 23. Resource-Boundary Findings

1. RAM is the binding resource: ~16.5–17.2 KB/doc marginal, ceiling ~95–100k
   docs on 1.9 GiB (kernel OOM evidence preserved).
2. Grown-graph recall at 80k is nondeterministic at the 95% gate (93–99),
   restored to 99–100 by rebuild — the reliability boundary of the default
   HNSW configuration near max_elements.
3. Read-lock contention: scan-heavy readers throttle writers ~30× at 60k.
4. Reopen cost (~45–84 s at 60–80k) is the recovery contract's deterministic
   rebuild — an availability consideration at scale.
5. Compaction/backup/checkpoint costs are milliseconds-to-sub-second at these
   tiers — not bottlenecks.

## 24. Bugs Found

- **SCALE-DEFECT #1 (engine policy, FIXED):** head-count-blind E9 hygiene
  trigger forced full rebuilds at every multi-head checkpoint (D40).
  Preserve→minimize→regression→fix→new-ID reruns→E1–E9 regression coverage —
  complete; A9's guarantee statement unchanged (the predicate implements its
  intent correctly now).
- Harness defects (engine exonerated, both fixed with new-ID reruns):
  SCALE-014 upsert semantics (D41); SCALE-015 sizing/guard design (D42);
  SCALE-017 classification gap (D44).
- Boundary confirmations (not bugs): async WAL-tail loss at graceful
  reopen-without-checkpoint (D43); grown-graph recall variance near the
  hnsw_rs cap (§12).

## 25. Invalidated Runs

PH3E-SCALE-014 (harness upsert defect; preserved with INVALIDATED.md; engine
exonerated by probe; rerun as 018). No other INVALIDATED. OBSERVED_LIMIT
runs preserved: 001 (guard-at-measurement), 002/003/004 (M3-blocked),
006 (kernel OOM), 015 (config error), 017 (recall gate, classification note).

## 26. E1–E9 Regression Results

Full workspace suite **330 passed / 0 failed** at E10 close (adds the
multi-head hygiene regression; all sealed E1–E9 chains green). Clippy
`-D warnings` clean. E9's four count-identical soak families remain sealed;
the engine's only E10 change is the D40 predicate fix, covered by the new
regression plus the existing E9 hygiene tests (2/2) and the 012/013 new-ID
reruns. Phase-2 and phase-3 consistency checkers PASS at close (gate 27
added for E10 evidence).

## 27. Production-Contract Amendment

**A10 — Scale Envelope** added (evidence-backed, scope-limited to the tested
resource class): verified tiers, measured costs, bottlenecks, and explicit
non-claims (no extrapolation beyond measured rungs; no production-readiness;
no orders-of-magnitude-larger document claims).

## 28. Supported Scale Envelope

On 2 vCPU / 1.9 GiB RAM, default config, Sync durability:
- **VERIFIED operating envelope: 60,000 documents** (integrated lifecycle
  82,000 ops incl. 2,000 txns, exact counts, checker clean, recall 99/100);
  40,000 documents on the primary config with the full gate battery; 60,000
  slim-config build+verify. Operation counts: 260,000-op churn run and
  82,000-op integrated run VERIFIED; 150,000-op baseline reproduced.
- **80,000 documents: reproduced with all correctness gates green; resource
  guard and recall-variance boundaries observed** — usable, with the two
  documented caveats.
- **>100k documents: OBSERVED boundary** (M3-blocked primary; kernel OOM at
  ~100k slim). Dimensions verified to 256; heads verified to 4; concurrency
  verified at 60k total with a measured writer-degradation caveat.

## 29. Unsupported Claims

Explicitly NOT claimed: any behavior beyond 80k documents / 260k operations /
dim 256 / 4 heads on this resource class; production readiness from scale
evidence; linear scaling beyond the measured rungs; massively-larger document
capacity; write scalability under concurrent readers (measured degraded);
recovery latency bounds beyond the tested tiers; any distributed/replicated/
sharded behavior; power-loss semantics beyond the E2-tested model.

## 30. Limitations

Single-host, single-writer concurrency shape (one reader tested); HNSW
max_elements=100k is a shipped default that the primary config approaches at
80k docs (multi-version churn reaches it sooner — hygiene rebuilds reset the
count); reopen cost grows with doc count (deterministic rebuild contract);
the harness model lives in-process (harness RSS shares the measured budget);
2-vCPU timing variance (single-run observations labeled; the 40k pair and
80k pair provide reproduced comparisons).

## 31. Reproducibility

`dbtest e10run --exp NAME --out DIR [--docs N]` with frozen seeds; dataset
hashes recorded per run; telemetry/census/summary per run;
`generate_results_ph3e.py` regenerates results/tables/figures;
`verify_consistency.py` gate 27 FAILS on registry/raw/report drift, on any
scale claim without a matching run, and on the E10 defect's regression
missing. Failure runs preserve notes + inventories + kernel evidence.

## 32. Final E10 Verdict

**Outcome: envelope characterized; verdict B.** The verified envelope
(60k integrated / 40k primary-full-gates / 80k reproduced-with-boundaries,
260k-op churn, dim 256, 4 heads, bounded concurrency, integrated lifecycle
exact) is established with reproducible evidence and explicit boundaries:
RAM ceiling ~95–100k docs (OOM evidence), grown-graph recall variance at 80k,
a measured read-contention bottleneck, and reopen costs documented. One
engine-policy defect was found by scale testing and is fixed and
regression-sealed; two harness defects were exonerated and corrected; all
E1–E9 guarantees re-verified (330/0, clippy clean, checkers PASS). The stop
condition checklist is complete. **E10 COMPLETE; E11 NOT STARTED.**
