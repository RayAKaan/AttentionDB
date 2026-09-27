# Report-Builder Audit — `c2/harness/build_reports.py`

Scope: the C2 harness report generator (draft). Reviewed at closure (2026-09-24)
together with the deterministic verifier `verify_c2.py`.

## Findings
1. **NEVER executed during C2.** No output file produced; no run references it.
   C2 reports were authored manually and cross-checked with the reconciliation
   verifier.
2. **Hardcoded Linux paths** (`/home/user/...`, `/proc`, unix-only commands):
   not runnable on the Windows audit host without edits.
3. **Stale run IDs / hardcoded target lists** — would emit a report that
   contradicts the RUN-INDEX ledger if run against the current 67-run registry
   (e.g., no awareness of `C2-MODES-TEST-002`, `C2-BM25-REPRO-00{1,2}`).
4. **`run_status()` accepts `NOT-RUN` and `CONDITIONAL`** as reportable
   outcomes ⇒ a run with no terminal status could be silently rendered as
   "COMPLETE". This is the flagged **silent-COMPLETE risk**.
5. **No metrics/artifact cross-check** — it reports status strings without
   verifying artifact hashes, manifest presence, or index consistency.

## Disposition
- **Not part of C2 evidence.** The authoritative reporter is the deterministic
  reconciliation verifier (`verify_c2.py` → `final-audit/run-reconciliation.csv`),
  which hash-checks artifacts, validates manifests, and reconciles RUN-INDEX;
  at closure it reports **0 inconsistencies**.
- Recommended (out of C2 scope, pre-C3 housekeeping): delete `build_reports.py`
  or rewrite it to consume `run-reconciliation.csv` and to treat missing
  terminal statuses as errors, not as COMPLETE.

## Verifier status (authoritative)
- 67 run dirs; RUN-INDEX authoritative-status counts: PASS 24, FAILED 14,
  INVALID-STARTUP 17, SUPERSEDED 4, FAILED-HARNESS-* 7, ABORTED 1.
- Inconsistencies: **0**. Dataset files: **14/14 MATCH**.
- New-run artifacts verified: C2-BM25-REPRO-001 (2/2), C2-BM25-REPRO-002
  (2/2), C2-MODES-TEST-002 (9/9), all sha256 match.
- 12 pre-existing dirs carry undeclared files on disk (LEAK exports, split
  indices, DS manifests, conditional-gates.yaml, b3-validate.json); all
  previously documented as accepted extras, none affecting verdicts.