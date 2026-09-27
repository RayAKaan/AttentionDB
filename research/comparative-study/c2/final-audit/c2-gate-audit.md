# C2 Gate Audit — G1–G15 (re-verified at closure, 2026-09-24)

Re-verification performed after the BM25 tie-order finding and fix
(`7cbd16d`). Every gate below references a registered/recorded artifact; a
gate whose supporting run did not exist is marked NOT-VERIFIED (none are).

Current authoritative ledger: 67 run dirs, **0 inconsistencies** between
RUN-INDEX and manifests, all artifact hashes match (`verify_c2.py`).

| # | Gate | Re-verified result | Support |
|---|---|---|---|
| G1 | environment captured | **PASS** | `environment-report.md`, `toolchain-report.md`, per-run `environment.yaml` |
| G2 | harness foundation | **PASS** | `c2/harness` + `c2probe` (release build exit 0); `verify_c2.py` deterministic verifier |
| G3 | exact oracle validated | **PASS** | `c2/oracle/validation-report.md` — 7/7 battery + engine agreement + NN-GT set-equality 1.0 |
| G4 | AttentionDB modes executed + findings preserved | **PASS (updated)** | `C2-MODES-TEST-001` FAILED retained (008 engine defect; 003/009 tolerance obs); **`C2-MODES-TEST-002` PASS (8/8)** post-fix; `C2-BM25-REPRO-001` FAILED (defect capture), `C2-BM25-REPRO-002` PASS; report §G amended — see `bm25-finding-and-disposition.md` |
| G5 | dataset hashes/manifests validated | **PASS** | per-run sha256 manifests; TES-C2-013 integrity re-hash PASS; re-checked at closure: dataset files 14/14 MATCH |
| G6 | primary datasets materialized | **PASS** | SciFact + NFCorpus (+ glove splits materialized for C2-B3); qrels-granularity discrepancy documented |
| G7 | ≥1 external baseline smoke completed | **PASS** | qdrant (PASS 001/006), pgvector (`C2-SMOKE-PGVECTOR-005` PASS), milvus-lite-002 PASS, weaviate-004 PASS; ES ABORTED-recorded |
| G8 | every declared external baseline has a real status | **PASS** | all declared baselines terminal (PASS/FAILED/ABORTED/BLOCKED-AUTH/EXCLUDED); pgvector-003 additionally diagnosed this closure (below) |
| G9 | B3 trained or honestly blocked | **PASS** | trained on preregistered source only (TRAIN-011, best_val_loss 1.0985422); activation validated |
| G10 | conditional dataset gates explicit outcomes | **PASS** | SciDocs/FiQA recorded in `c2-final-audit` reconciliation and report §K–M; COCO/ESCI liabilities unchanged |
| G11 | resource guardrails verified | **PASS** | samplers active; aborts preserved (ES row 20); guardrails recorded in run manifests |
| G12 | raw runs immutable | **PASS** | no-overwrite enforced; superseded attempts preserved with reasons (REPRO/MODES dirs created only via migration script) |
| G13 | no main benchmark sweep | **CONFIRMED** | zero recall/latency/quality/budget-matched runs; C2 is preparatory only |
| G14 | no C0/Phase3E mutation | **CONFIRMED** | `git diff fe4f92b..HEAD -- research/phase3` = 0; C0 docs unmodified by this closure |
| G15 | C2 artifacts internally consistent | **PASS (updated)** | `C2-INTEGRITY-001` PASS; smoke registry + RUN-INDEX reconcile; **closure re-check: 67 dirs, 0 inconsistencies, 14/14 dataset MATCH, new-run artifact hashes 2/2/2/9 verified** |

## PGVECTOR-003 verification (previously flagged)
`C2-SMOKE-PGVECTOR-003` FAILED: metrics.json shows install succeeded
(`install_rc` 0, `postgresql-17` + `postgresql-17-pgvector`), cluster start 0,
but **every SQL step blocked by `Password: su: Authentication failure`**
(`sql_errors`, `ann_query_rc` 1). Verdict: **environment credential issue**
(passwordless `su postgres` unavailable), not a pgvector product defect.

## Gate conclusion
All 15 gates pass/confirm with recorded support; the two gates affected by the
finding (G4, G15) are re-verified PASS on the post-fix evidence. No gate is
left NOT-VERIFIED.