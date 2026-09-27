# C1 — Environment & Resource Guardrails (2 vCPU / ~1.9 GiB RAM / ~20 GB disk, no Docker/systemd)

Study `comparative-study-001`, protocol v1.0.0.

## Preflight (every run)

1. Snapshot: MemAvailable, disk free on the run volume, load average →
   recorded into the run's environment.json.
2. Dataset/workspace budget check: required bytes × 1.3 must fit disk; a
   projected peak-RSS estimate (dataset + index overhead recorded per system
   from C2 smokes) must fit ≤85% of MemAvailable or the run is
   PRE-DECLARED-BLOCKED (not attempted, not retried silently at a smaller
   size without a documented amendment).
3. Phase3E isolation check: assert `PH3E_CRASH_AT`/`PH3E_CRASH_HIT`/
   `PH3E_CRASH_MODEL`/`PH3E_CRASH_MARKER` are UNSET in the run env (hard
   abort otherwise). The comparative study never activates crashgates.
4. Process budget: at most ONE external server + harness + sampler at a
   time; server startup health-checked before measurement.

## Live thresholds

- RAM: sampler (500 ms, process tree) aborts the run when total RSS ≥85% of
  MemAvailable at preflight → the run is classified OBSERVED-LIMIT (resource
  boundary), logs + last-good stage preserved. AttentionDB runs reuse the
  same discipline the E10 harness pioneered (pre-flight projection +
  in-run guard) but under the NEW registry and run IDs.
- Disk: abort at <500 MiB free; cleanup rules: only artifacts inside the
  run's own directory are ever deleted, and only after inventory+checksum
  (retention policy mirrors the charter's immutability rules).
- Timeouts: per-stage wall caps recorded in the run config (build ≤ 60 min,
  query phase ≤ 30 min, single query hard cap per BUDGET-TIME). A timeout
  kill uses process-group SIGKILL to the CHILD ONLY (never self/parent;
  lesson inherited from Phase 3E ops deviations).
- OOM detection: child killed by SIGKILL without our abort + kernel log line
  if accessible → classified OOM-LIMIT (resource boundary, NOT a retrieval-
  quality failure — charter §12).

## Safe progression & abort criteria

- Dataset-size ladder: 10k → 25k → 40k → 60k → 80k, advancing ONLY if the
  previous rung completed with ≥20% headroom under the RAM threshold.
  External systems size independently by their own smokes.
- Abort criteria (any → stop, preserve, classify): OOM-LIMIT; disk <500 MiB;
  3 consecutive harness crashes on the same config; external server
  unhealthy after 2 restart attempts; any silent-wrong-result indicator
  (soundness check failure) → that comparison is frozen pending defect
  review (§15 defect protocol: preserve → diagnose → ledger → new-ID rerun
  if fixed).
- Evidence after abort: environment snapshot, partial metrics with stage
  markers, server logs, sampler trace, checksums — registered with the
  classification.

## Host etiquette

- No stress tests, no large ingestion, no dataset downloads in C1 (this
  document is the plan; C2 runs the smokes).
- One measurement process at a time; no concurrent campaigns (2 vCPU makes
  concurrency itself a confounder).
