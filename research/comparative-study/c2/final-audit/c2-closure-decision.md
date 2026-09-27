# C2 Closure Decision

Date: 2026-09-24 · Study: comparative-study-001 (protocol v1.0.0) ·
Branch: `comparative-study/c2-validation` (fix `7cbd16d`)

## Decision
**STATUS: COMPLETE** — with recorded findings, a verified engine-core fix, and
preserved FAILED/observation runs per discipline.

## Basis
1. **Finding BM25-TIE-ORDER-001** — confirmed nondeterministic tied-order
   top-k in BM25 search/phrase/RRF; fixed at `7cbd16d`; regression-pinned
   (`core/tests/regression_bm25_tie_order.rs` 3/3 PASS); reproduced FAILED
   (`C2-BM25-REPRO-001`) → PASS (`C2-BM25-REPRO-002`) on identical stimuli.
2. **Modes battery re-verified post-fix** — `C2-MODES-TEST-002` PASS across
   8/8 runs with corrected harness contracts; TEST-C2-008 now asserts a
   meaningful token-doc-set property; TEST-C2-003/-009 assert determinism and
   exact-equality when candidate pools are exhaustive, and record coverage
   otherwise.
3. **New recorded observation (not a defect introduced by the fix)**
   **HNSW-RECALL-OBS-001**: `hnsw_rs` search at k == element count is not
   exhaustive (0..7 docs/head unreachable, process-variable). Evidenced by
   `C2-MODES-TEST-002/artifacts/hrecall-recall-observations.txt`. Mode A
   records approximate agreement by design (`exact_set_agreement_rate`).
4. **Gates G1–G15 re-verified PASS/CONFIRMED** (`final-audit/c2-gate-audit.md`).
   Ledger: 67 run dirs, 0 inconsistencies, 14/14 dataset hashes MATCH,
   artifact hashes verified.
5. **PGVECTOR-003 classified**: environment `su - postgres` credential issue
   (install/cluster OK; SQL blocked); not a product defect.
6. **Immutability honored**: all run dirs append-only; superseded attempts
   preserved; new runs created only via the migration script; staged repro
   artifacts relocated under `c2/_staging/` (removed from `raw/`).

## Open/unresolved carries (none blocking)
- `build_reports.py` is draft-only and must not be used as evidence
  (`final-audit/report-builder-audit.md`); recommended removal/replacement
  pre-C3.
- GloVe HDF5 / SciFact npy binaries remain recorded-hash-consistent only (not
  byte-verifiable on this host); recorded in report §E.

## Pre-C3 handoff requirements confirmed
- Main benchmark **not begun** (G13).
- Phase3E/C0 tree **unmodified** (G14).
- Fix branch `comparative-study/c2-validation` contains exactly the BM25 fix,
  its regression pin, and the C2 harness files; C3 work should branch from it.

— verifier: `verify_c2.py` (authoritative) · closure docs in `c2/final-audit/`.