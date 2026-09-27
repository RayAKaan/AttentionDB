# Elasticsearch — Readiness Report

Study `comparative-study-001` · §17 (host-bounded feasibility).

## Status

**ABORTED / OBSERVED-LIMIT — terminal for this host.** C1's CONDITIONAL-LOW
classification unchanged. NOT feasible at ~1.9 GiB RAM.

## Evidence — C2-SMOKE-ES-003 (terminal, ABORTED)

- version `8.15.2`, default JVM heap (preregistered expectation: guardrail abort)
- the server ran **under** `guarded_popen` (500 ms tree-RSS sampler); at ~10 s
  into JVM startup the tree reached **1,149,344 KB ≥ 1,089,169 KB budget**
  (85% of MemAvailable); sampler SIGKILL'd the child only (row 20); 20 sampler
  rows + ES logs preserved as run artifacts
- second independent host constraint recorded: `vm.max_map_count=65530 < 262144`
  (production-mode ES would hit this too)
- policy per prereg: **no reduced-heap retry**; the -002 metrics note announcing
  one is **RETRACTED** (see smoke-registry.yaml)
- superseded ledger: -001 FAILED (OSError 28: 993 MB tmpfs cannot fit 1.5 GiB
  extraction), -002 ABORTED (server outside sampler; chown defect;
  evidence hollow → superseded, label retracted)

## Verdict

Result is the result: documented, preserved, terminal. Any future ES work
requires a larger-envelope host and must re-state guardrails; no attempt was
made to hide or retry beyond prereg.