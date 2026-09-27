# C3 — Controlled Benchmark Execution & Pilot Validation (plan)

Study `comparative-study-001`, protocol v1.0.0 (C1 content commit `963170b`,
registration stamp `7788067`). This document is the C3 operating plan; the
`c3-preflight-report.md` records what was actually measured at start.

## 1. Objectives

1. Verify the C2 handoff is provably intact (branch, commit, tree, phase3
   diff, verifier rerun) — done; recorded in `c3-preflight-report.md`.
2. Convert C2 readiness into an executable C3 run pipeline on THIS host, with
   an honest preflight record (system/dataset/workload eligibility, resource
   budget) so that every decision point is auditable before any pilot run.
3. Regenerate + sha256-fix any materialized dataset/embedding artifact that is
   not byte-present on this host, against the C2 manifest hashes (C3-G6).
4. Validate the exact-oracle + metric pipeline on the C3 execution path
   (C3-G4) — the C2 oracle battery and engine-agreement are the seed; a
   bounded revalidation runs fresh on this host.
5. Register and execute a **bounded preregistered pilot** on eligible cells
   only, then verify fairness (Recall@10 ±0.01 quality matching on validation
   queries only; CAND/EF/MEM/TIME budget-matched axes), resource guardrails,
   artifact hashes, and produce `c3-gate-audit.md` + `c3-closure-decision.md`
   (go/no-go for full C4 execution).
6. Do NOT begin C4.

## 2. Source-of-truth anchors (immutable)

- C1 content commit `963170b` is the preregistration text (protocol,
  registries, guardrails). C3 populates matrices FROM these files, never from
  memory.
- C2 registration stamp `7788067`; C2 closure (fix) commit `7cbd16d` on
  `comparative-study/c2-validation`. C2 final-audit docs are the binding
  disposition for BM25 tie ordering, harness tolerances, and PGVECTOR-003.
- Raw run evidence is append-only under `../raw/<run_id>/` + `RUN-INDEX.yaml`;
  C3 run dirs are immutable, never overwritten; reruns take new IDs.

## 3. Execution host & envelope policy (decided at preflight, see report)

- Execution host = this Windows 11 workstation (8 logical CPUs / ~16 GiB RAM /
  ~455 GB free; rustc/cargo 1.96.0, Python 3.14.4).
- The preregistered constrained envelope is the Linux sandbox (2 vCPU / 1.9 GiB
  / 20 GB) captured in every C2 `environment.yaml`. Per C1 fairness + C2
  environment-report standing note, C3 runs execute on the actual C3 host as
  NEW run IDs with their own `environment.yaml`; no cross-host conflation; the
  deviation is disclosed, not silenced.
- Resource envelopes are enforced per C1 guardrails (85 % MemAvailable at
  preflight, disk floor, sampler abort), with the preregistered budget axes
  (BUDGET-CAND/EF/MEM/TIME) kept as the absolute matching caps. No new
  "larger memory budget" is invented; a system that cannot fit its own
  envelope + the matching caps is recorded BLOCKED or OBSERVED-LIMIT.

## 4. Eligibility gates (C3 preflight matrices)

Three CSVs (system / dataset / workload) carry one row per candidate with
status ELIGIBLE / CONDITIONAL / BLOCKED / EXCLUDED, plus evidence reference.

- System eligibility keys off C2 smokes + mode-battery + adapters readiness.
- Dataset eligibility keys off what is byte-present under `../raw/datasets/`
  on this host vs the C2 manifest hashes; absent files are CONDITIONAL until
  re-fetched + sha256-verified (never BLOCKED on provenance alone).
- Workload eligibility = combination of its datasets and modes, per W-01..W-19
  registry cells; a workload cell is eligible only if every dataset (for the
  mode's needed views) and the mode are eligible.

## 5. C3-G1..G16 acceptance contract (summary; full text in C3 prompt)

- G1 handoff provenance; G2 host/toolchain capture; G3 system eligibility
  with evidence; G4 dataset/embedding manifests + hashes; G5 exact oracle +
  metric validation on footprint; G6 Track A/B separation; G7 preregistered
  pilot workloads (from C1 registry); G8 parameter matching/fairness
  (validation-only tuning, ±0.01 Recall@10); G9 resource guardrails;
  G10 pre-execution registration; G11 failed/aborted runs preserved;
  G12 valid artifacts + hashes; G13 statistical pipeline on known inputs;
  G14 reporting verifier rejects incomplete evidence; G15 C0/Phase3E
  untouched; G16 auditable go/no-go.

## 6. Reporting discipline (differs from C2)

- `c2/harness/build_reports.py` is draft-only and MUST NOT be used as the
  authority (stale IDs, hardcoded Linux paths, silent-COMPLETE risk). C3
  extends the deterministic verifier ideas from `verify_c2.py` into a
  `verify_c3.py` that: rejects nonterminal statuses, unknown/duplicate run
  IDs, missing metrics, and never marks COMPLETE without artifact hashes +
  metric evidence.

## 7. Pilot scope (registered in `c3-run-plan.csv`)

The first C3 pilot is a bounded, preregistered slice of eligible cells —
exact-oracle/latency-floor + mode validation cells plus a matched-quality
sweep — sized to complete on this host within the resource budget. Its full
parameter grid is frozen before any run executes; runs are registered
pre-execution with new IDs.

## 8. Stop conditions (any → halt, preserve evidence, report)

- C2 closure commit cannot be verified (already ruled out — PASS).
- Unexplained working-tree changes; dataset hash mismatch after
  re-procurement; exact-oracle discrepancy; unpreregistered workload/config;
  unavailable guardrails; missing pre-execution registration; harness silently
  changes config/metric; raw evidence overwritten; C0/Phase3E modified.
- A blocked scope isolates that scope only; execution continues where the
  protocol permits.

## 9. Deliverables

`c3-plan.md`, `c3-preflight-report.md`, `c3-system-eligibility.csv`,
`c3-dataset-eligibility.csv`, `c3-workload-eligibility.csv`,
`c3-resource-budget.yaml`, `c3-run-plan.csv`, `c3-execution-report.md`,
`c3-gate-audit.md`, `c3-closure-decision.md`, `run-reconciliation.csv`,
`artifacts/`, `raw/`.