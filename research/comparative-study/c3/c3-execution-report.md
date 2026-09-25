# C3 — Controlled Benchmark Execution Report (Pilot Run)

Study `comparative-study-001` · C3 pilot stage · protocol v1.0.0
Registered under `raw/RUN-INDEX.yaml` (append-only).

## 1. Scope

Bounded preregistered slice of the eligibility matrix: 7 pilot runs executed on
the Windows host (host-as-execution-envelope per user decision). Every run has a
NEW C3-namespaced ID, its own `environment.yaml` (host snapshot + input hashes +
guardrail policy), immutable `artifacts/` (raw f32 inputs, per-query rows, exact
oracle), and an append-only registry entry.

| Run ID | Cell | Mode | Dataset | Subset | Status |
|---|---|---|---|---|---|
| C3-W01-NFC-B0-001 | W01-NFCORPUS-B0 | B0 (exact oracle) | DS-NFCORPUS | qrels-test-323 SAMPLED-subset (100) | PASS |
| C3-W01-NFC-B1-001 | W01-NFCORPUS-B1 | B1 (SingleHead/CANONICAL) | DS-NFCORPUS | qrels-test-323 SAMPLED-subset (100) | PASS |
| C3-W02-NFC-B2-001 | W02-NFCORPUS-B2 | B2 (FixedFusion, H=3) | DS-NFCORPUS | qrels-test-323 SAMPLED-subset (100) | PASS |
| C3-W07-NFC-B7-001 | W07-NFCORPUS-B7 | B7 (Full default) | DS-NFCORPUS | qrels-test-323 SAMPLED-subset (100) | PASS |
| C3-W03-SCI-B3-001 | W03-SCIFACT-B3 | B3 (LearnedGating) | DS-SCIFACT | qrels-test-300 SAMPLED-subset (100) | PASS |
| C3-W08-SYN-B1-001 | W08-SYNTH | B1 | DS-SYNTH-PH2B (DIAGNOSTIC) | generator-query-set (200) | PASS |
| C3-W10-SYN-DUP-B0-001 | W10-SYNTH-DUP | B0 | DS-SYNTH-PH2B-dup (DIAGNOSTIC) | generator-query-set (200) | PASS |

## 2. Execution machinery (new, additive)

- Driver: `c2/probe/c2pilot.rs` → `target/release/c2pilot.exe` (additive binary in
  the audited probe crate; `main.rs` and all audited code untouched). Drives the
  REAL AttentionDB engine over raw LE f32 external embeddings with C1 protocol
  (seeded order, warmup discard, per-query latency, execution of modes
  B0/B1/B2/B3/B7).
- Orchestrator: `c3/harness/c3_pilot_run.py` (real-data cells) +
  `c3/harness/c3_synth_pilot.py` (synthetic diagnostic cells). Windows-safe
  500 ms process-tree RSS sampler (child process working set via
  `GetProcessMemoryInfo`), qrels mapping from BEIR `.tsv`, seeded subsample
  drawing, 5 fresh-process reps, aggregation, registration.
- Exact oracle: brute-force cosine top-k over every query's CANONICAL view
  (score desc, id ASC — engine tie order), emitted as `exact_top10` for every
  run query. B0 cell reports this as its own outputs and establishes the
  quality ceiling and the subsample list used by all paired cells.

## 3. Results (subsamples: 100 queries per real-data cell)

Latency: mean-of-rep-p50 across the 5 fresh-process reps. Recall@10 vs qrels
(mean-of-rep-means) and vs the in-process exact oracle.

| Cell | Recall@10 vs qrels | Recall@10 vs exact | p50 latency (µs) | p95 latency (µs) | peak RSS (MB) |
|---|---|---|---|---|---|
| NFC B0 (oracle ceiling) | 0.142 | — | (oracle, untimed) | — | ~2 |
| NFC B1 (SingleHead) | 0.134 | 0.958 | 882 | 1269 | 109 |
| NFC B2 (FixedFusion) | 0.168 | 0.680 | 1738 | 2790 | 194 |
| NFC B7 (Full) | 0.155 | 0.620 | 2177 | 3597 | 194 |
| SCI B3 (LearnedGating) | 0.813 | 0.685 | 1910 | 2758 | 304 |

Synthetic diagnostics (n=200, generator GT verified in full):

| Cell | Recall@10 vs GT | Recall@10 vs exact | GT-oracle checks |
|---|---|---|---|
| SYN B1 (conflicting signals) | 0.394 | 0.988 | 200/200 |
| SYN B0 dup (dedupe semantics) | 0.175 | — | 200/200 |

Notes:
- B2 (FixedFusion) outperforms the single CANONICAL-view ceiling (0.168 > 0.142):
  multi-view union recovers title/body/cite signals the canonical projection
  cannot — the expected value-add of the multi-head design.
- B7 (production default Full) lands between B1 and B2 on NFCorpus qrels
  (0.155) at the highest latency — consistent with "unfavorable evidence
  evaluated as-is" (W-07).
- B3 transfers the NFCorpus-derived LODO gating card to SciFact-headline with
  high qrels recall (0.813) — a positive pilot signal for the learned-gating
  contribution (RQ3), pending confirmation-scale runs.

## 4. Fairness & protocol compliance

- Paired by query: identical seeded subsample lists across all four NFCorpus
  cells (verified byte-identical in `config.subsample`); same seeded in-process
  order across reps; warmup 20 queries executed but excluded from latency.
  Identical subsets across modes enable the paired per-query bootstrap planned
  in C1.
- Fresh-process reps: 5 per real-data cell; query order re-seeded per process
  (same seed → same order), so reps measure process-level variance rather than
  order interaction.
- Determinism: query order and exact oracle identical across reps within a cell.
  Top-10 identity across reps is ~0.92 Jaccard on B3 (tail-rank flap from the
  approximate index) with per-rep recall means tight (0.800–0.825 range 0.025);
  this variance is captured by the rep protocol. No silent-wrong-result
  indicator: generator-GT oracle soundness checks passed 200/200 in both
  synthetic cells.
- Guardrails: child RSS never approached the 85% of preflight MemAvailable
  threshold (peaks 2–304 MB vs ~13.5 GB budget); no aborts during measurement.
  The DS-SYNTH / B0 / B3-4-head aborted attempts noted in §6 are pre-measurement
  harness corrections, preserved with evidence.

## 5. Environment (host-as-execution-envelope; disclosed)

Windows 11 Pro build 26200, AMD Ryzen 3 3100 4C/8T, 15.9 GiB RAM; `c2pilot`
built from working-tree source (r1.96.0) and hashed into each run's
`environment.yaml`; SciFact inputs = `C3-EMBED-SCIFACT-001` (on-own-hashes;
cross-env float drift of ~2–3e-7 vs C2 manifests documented in the C3 preflight).
NFCorpus inputs = `C2-EMBED-NFCORPUS-003` re-verified on this host by
`C3DATA-NFC-VERIFY-001` (PASS). B3 modelcard = `b3-lodo-nf-v2-s20260925-h32-lr0.01`
(input_dim 384, 3 heads, C2-validated; applied to SciFact-headline test queries
never touched during its training).

## 6. Corrections made during execution (all preserved, all pre-measurement)

1. Guardrail sampler initially measured host-global MEMORYLOAD (86% on this
   desktop regardless of the run) instead of the child process-tree RSS per
   `c1/environment-guardrails.md`. Fixed to child RSS vs 85% of preflight
   MemAvailable. Evidence: `C3-W01-NFC-B0-001_ABORT-ATTEMPT-{0,1,2}`.
2. B3 cell initially created the engine collection with 4 heads; the engine
   refused (§15: card trained for 3 heads). Corrected to a 3-head collection.
   Evidence: `C3-W03-SCI-B3-001_ABORT-ATTEMPT-0`.
3. B2/B7 cells initially created a 4-head collection (CANONICAL included);
   retrieval attended only the 3 HEAD views so results were unaffected, but run
   config did not match the C1 mode registry (H=3). Re-run under the same
   planned IDs with a corrected 3-head config. Evidence:
   `C3-W02-NFC-B2-001_CONFIG-CORRECTION-0`, `C3-W07-NFC-B7-001_CONFIG-CORRECTION-0`.
4. RUN-INDEX therefore contains FAILED/FAILED-HARNESS entries for
   C3-W01-NFC-B0-001 / C3-W03-SCI-B3-001 and duplicate PASS entries for B2/B7;
   those refer to the pre-measurement attempts above, not to evidence loss.
   Full reconciliation in `run-reconciliation.csv`.

## 7. Budget

180-minute pilot budget; execution used under an hour of wall time across all
cells. Disk footprint per run held to ~run-scoped artifacts only.

---
## Change history (documentation-only; no raw measurements modified)

- 2026-09-25 (C4 preflight): corrected §4 guardrail peak figure from "2–284 MB"
  to "2–304 MB" to match the authoritative verified value (C3-W03-SCI-B3-001
  peak_child_rss_bytes_by_rep max = 303.8–304.2 MiB). Documentation only; no
  raw artifact, hash, run registry, or metrics.json value was altered.