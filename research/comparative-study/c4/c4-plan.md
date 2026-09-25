# C4 — Confirmed-Scale Benchmark Execution (plan)

Study `comparative-study-001`, protocol v1.0.0. Branch
`comparative-study/c4-confirmed-benchmark` cut from the C3 closure commit
`5195133ca5fab7a660d403b009b565408d6a09b3` (tree sha16 `93c1ccdef076aeca`).
This document is the C4 operating plan + gate contract; the
`c4-environment.md` and `c4-dataset-validation.md` record what was actually
measured at start.

## 1. Objectives

1. Re-verify the C3 closure is the correct C4 base (commit, branch, tree,
   correction) and re-validate datasets/embeddings on this host — done:
   `c4-environment.md` (§1 handoff), `c4-dataset-validation.md`
   (`C4DATA-VERIFY-001` PASS 18/18).
2. Advance the C3 pilot to **confirmed scale** per the C1 protocol: full test
   query sets (SciFact 300, NFCorpus 323), 5 fresh-process latency reps,
   warmup 20, seeded paired query order; quality matching on VALIDATION only.
3. Execute the preregistered Track A confirmed cells (W-01/W-02/W-03/W-04/
   W-07 on both primary datasets: B0, B1, B2, B3, B4, B7) with
   BUDGET-CAND/EF/MEM/TIME matching sweeps as required by
   `recall-and-budget-matching.md`.
4. Run the preregistered statistical analysis (8 primary paired contrasts,
   paired bootstrap 95% CI, Wilcoxon, Holm correction, effect sizes) purely
   from immutable raw per-query rows; no winner labels, no composite scores.
5. Produce the C4 reporting deliverables with a gate audit and a final
   COMPLETE vs BLOCKED/PARTIAL decision; do NOT claim conclusions beyond the
   evidence.

## 2. Source-of-truth anchors (immutable)

- C1 content commit `963170b` + registration stamp `7788067`; C1 files are
  authoritative (protocol, mode registry, workload registry, statistical
  plan, recall/budget matching, guardrails). C4 populates matrices from these
  files, never from memory.
- C2 closure commit `7cbd16d` (BM25 tie-order fix) + C2 final-audit docs;
  C3 closure commit `5195133` (base for this branch).
- Mode definitions: `c1/mode-registry.yaml` (B0 exact oracle; B1 single-head;
  B2 fixed-fusion multi-head; B3 learned-gating; B4 documented no-op; B7
  production default Full). Equivalences stated: B2≡B3 without an activated
  trained card; B4 identity-init QK attention previously measured a no-op vs
  B3 — neither pair is presented as independent.
- Raw run evidence is append-only under `../raw/<run_id>/` + `RUN-INDEX.yaml`;
  run dirs are immutable, never overwritten; any rerun = new ID; failed
  attempts are preserved with status (FAILED / FAILED-HARNESS / ABORTED /
  INVALID-STARTUP / BLOCKED-* / SUPERSEDED).

## 3. Execution host & envelope (see `c4-environment.md`)

This Windows 11 workstation (8 logical CPUs / ~15.9 GiB RAM (MemAvailable
3,004,496 KB @ preflight) / ~455 GiB free; rustc 1.96.0, Python 3.14.4).
The preregistered constrained envelope (Linux 2 vCPU / 1.9 GiB / 20 GB) is
NOT the C4 host; C4 runs are NEW run IDs with their own `environment.yaml`;
deviation disclosed, not silenced. Guardrails per C1 (85% of preflight
MemAvailable, disk floor, sampler abort, one external server at a time,
timeouts, PH3E envs unset). Budget axes absolute and unchanged.

## 4. Eligibility (C4-G3..G5) — summary of the CSVs

- `c4-system-eligibility.csv`: SYS-ADB, SYS-B0, SYS-B3 ELIGIBLE. Qdrant /
  Weaviate / Milvus-lite / ES CONDITIONAL (must be re-smoked on THIS host with
  sampler before use, else BLOCKED at C4). pgvector BLOCKED (C4) pending
  Windows install + credential re-authorization (PGVECTOR-003). Pinecone /
  Mongo BLOCKED-AUTH. B6 adapter BLOCKED (requires-implementation).
- `c4-dataset-eligibility.csv`: DS-SCIFACT ELIGIBLE-ON-OWN-HASHES (decision A,
  `c4-dataset-validation.md` §3); DS-NFCORPUS ELIGIBLE; GloVe/ESCI/COCO
  CONDITIONAL (outside C4 primary scope); SYNTH DIAGNOSTIC-ONLY.
- `c4-workload-eligibility.csv`: W-01, W-02, W-03, W-04, W-07 ELIGIBLE on both
  primary datasets; W-19 ADB-leg ELIGIBLE; W-05 SciFact-only optional;
  external legs CONDITIONAL/BLOCKED per system CSV.

## 5. Phases C4.0–C4.7

| Phase | Work | Entry evidence |
|---|---|---|
| C4.0 | C3 closure verification + branch creation + doc correction | all PASS (this doc, §1) |
| C4.1 | Environment + dataset validation + eligibility freeze | `c4-environment.md`, `c4-dataset-validation.md`, CSVs, `c4-resource-budget.yaml`, `C4DATA-VERIFY-001` PASS |
| C4.2 | Validation tuning on validation queries only (ef_search × candidate grids per `recall-and-budget-matching.md`; tolerance ±0.01 Recall@10; fastest p50 recorded; TARGET-UNREACHABLE never interpolated) | frozen `c4-run-plan.csv`; tuning runs registered with new IDs |
| C4.3 | Parameter/budget matching (BUDGET-CAND/EF/MEM/TIME single-axis sweeps on validation; same-RSS-abort discipline; OBSERVED-LIMIT recorded, never skipped silently) | matched config table, recorded verbatim |
| C4.4 | Confirmed test execution on full frozen test sets (5 fresh-process reps, warmup 20, seeded paired order); every run registered pre-execution with NEW C4 IDs; hashes + status per run | PASS cells in RUN-INDEX |
| C4.5 | Statistical analysis from raw per-query rows (paired bootstrap 10k seeded, Wilcoxon signed-rank, Holm within family; BH q=0.10 exploratory; Cohen's dz; power reported for non-significant cells; never mean-only) | `c4-statistical-analysis.md` |
| C4.6 | Reconciliation (run-vs-plan, failed attempts preserved, cross-rep determinism, exact-top10 cross checks) + report + gate audit | `c4-execution-report.md`, `run-reconciliation.csv`, `c4-gate-audit.md` |
| C4.7 | Closure decision COMPLETE vs BLOCKED/PARTIAL + commit C4 docs | `c4-closure-decision.md` |

## 6. Run schema & registration (C4-G10)

Every C4 run: unique `C4-…` ID (never reuse C3 IDs), `environment.yaml`,
`config.yaml`, `manifest.yaml`, `status.txt`, `metrics.json`, per-query rows,
exact-oracle reference, hashes; registered in append-only `RUN-INDEX.yaml`
before execution. QoS/failure classification per C4 §18 (PASS / FAILED /
FAILED-HARNESS / ABORTED / INVALID-STARTUP / BLOCKED-AUTH / BLOCKED-RESOURCE /
SUPERSEDED). Per-query schema (§16) records query_id, system, mode, dataset,
split, rep, latency_us, candidate_count, returned_ids, exact_top10,
relevant_ids, recall10_exact, recall10_qrels, configuration_id (+ head-level
candidate counts and head/fusion/rerank latencies for multi-head modes).

## 7. C4-G1..G20 acceptance contract

- C4-G1 C3 closure provenance (commit/branch/tree verified) — DONE.
- C4-G2 host/toolchain capture — DONE (`c4-environment.md`).
- C4-G3 system eligibility with evidence — DONE (CSV; externals CONDITIONAL).
- C4-G4 dataset/embedding hashes — DONE (`C4DATA-VERIFY-001` PASS 18/18).
- C4-G5 exact oracle + metric validated — DONE via C2 oracle battery +
  C3 0/100 exact-top10 checks; reconfirmed on C4 rows in C4.4.
- C4-G6 Track A/B separation — plan keeps A (matched) and B (practical)
  reporting separate; never merged into one ranking.
- C4-G7 confirmed workloads from C1 registry (W-01..W-07 + W-19 ADB leg).
- C4-G8 parameter matching/fairness: validation-only tuning, ±0.01 Recall@10.
- C4-G9 resource guardrails enforced per-run (85% MemAvailable abort, disk
  floor, timeouts, PH3E unset).
- C4-G10 pre-execution registration of all C4 run IDs.
- C4-G11 failed/aborted runs preserved, never rewritten; new-ID reruns.
- C4-G12 valid artifacts + hashes for every PASS cell.
- C4-G13 statistical pipeline on known inputs (reproducible derivation).
- C4-G14 reporting verifier rejects incomplete evidence (extended
  `verify_c3.py`-style; no silent COMPLETE).
- C4-G15 C0/Phase3E untouched; B3 leak guards rechecked; test never tuned.
- C4-G16 auditable go/no-go from pilot evidence.
- C4-G17 confirmed-scale query sets frozen (300/323) before measurement.
- C4-G18 per-query schema + multi-head cost accounting present.
- C4-G19 equal-budget / equal-quality matching tables honest (never fake match).
- C4-G20 closure decision COMPLETE/BLOCKED/PARTIAL with evidence.

## 8. Reporting discipline

`c2/harness/build_reports.py` is draft-only and MUST NOT be the authority.
C4 extends `verify_c3.py` into `verify_c4.py` (rejects nonterminal statuses,
unknown/duplicate IDs, missing metrics; never marks COMPLETE without artifact
hashes + per-query evidence). Headline numbers recomputed independently in the
C4.G20 closure audit.

## 9. Stop conditions (any → halt, preserve evidence, report)

- C3/C1 anchor mismatch; dataset hash mismatch after re-procurement; exact-
  oracle discrepancy; unpreregistered workload/config; unavailable guardrails;
  missing pre-execution registration; harness silently changes config/metric;
  raw evidence overwritten; C0/Phase3E modified; guardrail abort; 3
  consecutive harness crashes on same config; silent-wrong-result indicator
  (soundness failure) → freeze that comparison pending defect review (§15:
  preserve → diagnose → ledger → new-ID rerun).
- A blocked scope isolates only that scope; execution continues where the
  protocol permits.
- C4 HARD STOP: the C4 phase never auto-promotes to a larger/commercial run;
  any scope beyond the frozen `c4-run-plan.csv` requires a user amendment.

## 10. C4 hard rules (never violated)

1. No test-set tuning or analysis-before-freeze; test sets locked (§4 of
   dataset validation).
2. No winner labels / rankings / composite scores; separate Track A/B tables.
3. No post-hoc run exclusion; failures preserved and classified; reruns new-ID.
4. No byte-identity claims for SciFact vs C2 (own-hashes + disclosure).
5. External systems only after this-host smoke + sampler; otherwise BLOCKED.
6. No reduced-heap Elasticsearch retry (preregistration forbids).
7. No paid/cloud services without explicit user authorization.

## 11. Deliverables

`c4-plan.md` (this), `c4-dataset-validation.md`, `c4-environment.md`,
`c4-resource-budget.yaml`, `c4-system-eligibility.csv`,
`c4-dataset-eligibility.csv`, `c4-workload-eligibility.csv`,
`c4-run-plan.csv` (frozen), `c4-execution-report.md`,
`c4-statistical-analysis.md`, `c4-results.md`, `c4-gate-audit.md`,
`run-reconciliation.csv`, `c4-closure-decision.md`, `harness/`, `raw/`.