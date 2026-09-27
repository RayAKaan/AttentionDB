# C2 Validation Report — study comparative-study-001 (protocol v1.0.0)

Generated 2026-09-24 from run artifacts. All statuses below are terminal
statuses read from immutable run dirs under `../raw/` and the RUN-INDEX;
nothing was overwritten. This report supersedes the draft outline in
`harness/build_reports.py`, which never executed (Linux-path template) and
referenced run IDs that differ from the registered evidence — where they
differed, this report uses the ACTUAL registered runs.

## A. Repository state

- Branch `comparative-study/c0-audit`; registration/working commit
  `7788067c1437ae8c75ced4dc1ffc2f49b2c22a88` matches every run's manifest.
- C0 documents + Phase 3E untouched (`git diff fe4f92b HEAD -- research/phase3`
  = 0 lines at C1 build time; C2 added new files only under
  `research/comparative-study/`).
- Working tree: `research/comparative-study/c2/` and `research/comparative-study/raw/`
  are the only untracked additions.

## B. Environment state

Two execution environments, each recorded per-run in `environment.yaml`
(see `environment-report.md`):
- Sandbox-originated runs: Linux 6.1.158+, Intel Xeon 2.60 GHz, 2 vCPU,
  1,932,608 KB RAM total, ~14,094 MB disk free, Python 3.13.14.
- Fresh re-runs (2026-09-24, this host): Windows 11 Pro (build 26200), AMD
  Ryzen 3 3100 4C/8T, 16,700,964 KB RAM, NTFS, ~471 GB free, Python 3.14.4.
- Guardrails active in every run: 85%-MemAvailable abort, 500 MB disk floor,
  500 ms process-tree sampler, `phase3e_crash_env_asserted_unset: true`.

## C. Toolchain

Pinned in `toolchain-report.md`: rustc/cargo 1.96.0 (stable via
rust-toolchain.toml), Python 3.13.14 (sandbox runs) / 3.14.4 (re-runs),
sentence-transformers 6.1.0, torch 2.14.0+cpu, transformers 5.17.0, model
all-MiniLM-L6-v2 revision `1110a243…`. Protoc at
`C:\Users\Rayyan Khan\.cargo\protoc\bin\protoc.exe` for the probe rebuild.

## D. Dataset manifests

- **DS-SCIFACT** (C2-DATA-SCIFACT-007, **PASS**): 5,183 docs; 300 test-qrel
  queries; qrels 919 train / 339 test, positives-only score=1. C1-vs-data
  discrepancy recorded (C1 said "graded 0/1/2"; shipped files are binary) —
  vocabulary vs shipped levels; acceptance rule unaffected.
- **DS-NFCORPUS** (C2-DATA-NFCORPUS-003, **PASS**): 3,633 docs; 323 test-qrel
  queries; qrels 110,575 / 11,385 / 12,334 (levels 1–2, positives-only).
- Full counts, qrels provenance, hashes: `dataset-manifests/README.md` +
  per-run metrics.
- **ANN track**: DS-ANN-GLOVE-25 (C2-DATA-GLOVE25-003, **PASS**) and
  DS-ANN-GLOVE-50 (C2-DATA-GLOVE50-003, **PASS**); official ann-benchmarks
  HDF5, file sha256 recorded; **full exact NN-GT recompute over all 10,000
  queries × top-100: set mismatches 0, match_rate 1.0** (2 / 1 exact-boundary
  tie swaps recorded, not hidden). Labeled NN-GROUND-TRUTH / EFFICIENCY
  (never semantic).

## E. Dataset integrity / embedding determinism

- Re-hash at materialization, cardinality checks, qrels→doc closure
  (0 dangling ids both primaries), schema checks.
- Embeddings (C2-EMBED-SCIFACT-003 / C2-EMBED-NFCORPUS-003, both **PASS**):
  dim 384, float32, finite/no-nan/inf on every view; per-view export sha256
  recorded; `determinism_full_view_reencode_byte_equal: true`;
  batch-1-vs-batch-32 padding note documented.
- **TEST-C2-013 integrity re-hash: C2-INTEGRITY-001 — PASS** (manifest
  re-hash across all C2 run dirs; no missing/mismatched artifact; first
  execution of this run_id, ran on the current host).

## F. Oracle correctness

- C2-ORACLE-TESTS-002 **PASS** (7/7 battery: hand-check, independent
  numpy-vs-pure-python cross-check, determinism, id-ascending ties, top-k
  boundaries, empty input, monotonicity).
- Engine cross-check C2-ORACLE-AGREE-001 **PASS**: over 20 queries, engine
  mode A scores consistent with stored vectors (`true`), set equality 1.0,
  order equality 1.0 (mode A approximate by design; score consistency is the
  correctness assertion).
- Detail: `oracle/validation-report.md`.

## G. AttentionDB adapter correctness (C2-MODES-TEST-001, FAILED 5/8)

| test | what it verifies | status |
|---|---|---|
| TEST-C2-002 | single-head ANN (B1): soundness, score consistency, determinism | **PASS** |
| TEST-C2-003 | multi-head equal fusion matches pipeline reference | **FAILED** (strict-1e-5 only; IDs agree 100%) |
| TEST-C2-004 | head membership semantics: unknown heads skipped, empty→empty | **PASS** |
| TEST-C2-005 | candidate-budget: invalid refused, min budget sound | **PASS** |
| TEST-C2-006 | deterministic tie ordering (score desc, id asc) | **PASS** |
| TEST-C2-008 | hybrid fusion correctness (B5) | **FAILED** (BM25 tie-order nondeterminism) |
| TEST-C2-009 | Full-mode metric-consistent exact rerank (B7) | **FAILED** (strict-1e-5 only; IDs agree 100%) |
| TEST-C2-B4-NOOP | B4 identity-QK: deterministic untrained scorer; C-vs-D overlap 1.0 | **PASS** |

B3 engine activation + rejection (TEST-C2-010/010a/010b/011a/011b/011c) ran in
C2-B3-VALID-004 (**PASS**, 6/6). TEST-C2-001 (oracle hand-check + independent
cross-check + engine agreement) = section F. B4 evidence: measured-contribution
claim recorded; byte identity NOT asserted because fusion renormalizes present
channels. Findings (recorded, not hidden):
1. **TEST-C2-008 engine findinng**: BM25 tie-ordering nondeterministic per call
   (`core/src/bm25.rs` sort with `unwrap_or(Equal)` and no id tiebreaker over a
   per-call HashMap) — dense paths deterministic (score desc, id asc verified).
2. **TEST-C2-003/009 harness-tolerance observations**: mismatches are ~1e-3
   score deltas (e.g. 0.799783 vs 0.800715) with **100% top-k ID agreement**;
   strictly a 1e-5 score-tolerance mismatch, not an engine result defect.

## H. External baseline smoke results

| system | run | status |
|---|---|---|
| Qdrant 1.12.4 | C2-SMOKE-QDRANT-006 | **PASS** (exact self-top1; **named-vector collection validated** — B6 config feasible) |
| pgvector 0.8.0 | C2-SMOKE-PGVECTOR-005 | **PASS** (distro-repo; HNSW + IVFFlat; exact `<=>` self-top1) |
| Elasticsearch 8.15.2 | C2-SMOKE-ES-003 | **ABORTED / OBSERVED-LIMIT** (default-JVM tree RSS 1.10 GiB ≥ 85% budget 1.04 GiB; terminal for host, not retried per prereg) |
| Milvus-Lite 3.2.1 | C2-SMOKE-MILVUSLITE-002 | **PASS** (`pymilvus[milvus-lite]`; create/insert/search OK) |
| Weaviate 1.39.6 | C2-SMOKE-WEAVIATE-004 | **PASS** (ready @5.5 s; nearVector 200; peak RSS 107 MiB) |
| Pinecone | — | BLOCKED-AUTH (unchanged, C1 BLK-1) |
| MongoDB Atlas | — | BLOCKED-AUTH (unchanged, C1 BLK-1) |
| Milvus standalone | — | EXCLUDED (C1) |

Per-run evidence incl. logs + sampler CSVs in `raw/`. No failed smoke was
converted to READY. B6 note: Qdrant named-vector/prefetch mapping assessed in
`adapters/qdrant.md` against the C1 B6 definition (feasibility only, never an
AttentionDB adapter).

## I. B3 training results

| run | arm | status |
|---|---|---|
| C2-B3-LEAK-002 | INV-L1..L5 leakage negative tests | **PASS** |
| C2-B3-DATA-LODO-002 | LODO NFCorpus-derived gating dataset (exact targets) | **PASS** |
| C2-B3-TRAIN-009 | b3-synth-concat-v2 (grid + primary seed) | **PASS** |
| C2-B3-TRAIN-010 | b3-synth-view0-v2 (grid + primary seed) | **PASS** |
| C2-B3-TRAIN-011 | b3-lodo-nf-v2 (grid + primary seed; winner h32/lr0.005) | **PASS** |
| C2-B3-VALID-004 | ModelCard engine activation (TEST-C2-010..011) | **PASS** |

Selection on the trainer's validation split only; ModelCards carry TrainingMeta
(`-v2` set, head_names present); bitwise reproducibility recorded
(`reproducible_bitwise: true`). Superseded/INVALID-STARTUP attempts preserved
(TRAIN-001/002/004/005, VALID-002/003 — see `b3/training-validation.md`).
Disclosure: NFCorpus TEST queries used as LODO TRAINING inputs (C1 LODO rule);
SciFact untouched.

## J. Leakage validation

C2-B3-LEAK-002 **PASS**. INV-L1 refuses headline/test-dataset paths; INV-L2
config hash frozen in every training run (8283c127…); INV-L3 targets computed
only from exported training datasets; INV-L4 `ModelCard::validate()` rejects
cards without TrainingMeta (tampered format test PASS); INV-L5 bitwise
reproducibility per run. Headline test sets never passed to training.

## K–M. Conditional dataset gates (SciDocs/FiQA)

Terminal run C2-GATE-SCIDOCS-FIQA-003 **PASS** (earlier -001/-002 attempts:
FAILED → INVALID-STARTUP/FAILED-HARNESS-DEFECT, superseded, preserved).
Findings: SciDocs corpus carries only title/text + metadata — the claimed
facet structure is NOT present as distinct fields; C1's conditional status
stands (no invented citation-head semantics). FiQA: title+text present —
Track-B single-vector role validated. COCO and ESCI gates were NOT registered
as runs this session (BLK-2/BLK-4 host-bound, unchanged liabilities from C1;
see preregistration).

## N. Resource failures

All sampler aborts preserved (resource.csv + logs). Elasticsearch default-JVM
behavior is the recorded OBSERVED-LIMIT result (C2-SMOKE-ES-003) — not relaxed
to make it pass (#16). Milvus/pgvector/Weaviate/Qdrant all below envelope.

## O. Blockers status

- BLK-1 (Pinecone/Atlas): BLOCKED-AUTH unchanged — needs user authorization;
  nothing requested/transmitted/spent.
- BLK-3 resolved per-smoke: pgvector = PASS (distro repo), Milvus-Lite = PASS.
- BLK-2 (COCO CLIP-on-CPU) and BLK-4 (ESCI RAM): not executed this session;
  host-bound liabilities recorded (unchanged from C1).

## P. Accepted/rejected conditional systems

No status upgraded without its preregistered gate; CONDITIONAL statuses carry
the exact unresolved gate (see `adapters/README.md` + `smoke/results.md`).
BLOCKED readers: BLOCKED-AUTH / EXCLUDED are unchanged.

## Q. C2 acceptance gates (§34 G1–G15)

| # | Gate | Result |
|---|---|---|
| G1 | environment captured | **PASS** — environment-report.md + per-run environment.yaml (both hosts) |
| G2 | harness foundation | **PASS** — registry, sampler, manifest writer, oracle, drivers (c2/harness + c2probe, built release exit 0) |
| G3 | exact oracle validated | **PASS** — 7/7 battery + engine agreement + NN-GT set-equality 1.0 |
| G4 | AttentionDB modes executed + findings preserved | **PASS** — C2-MODES-TEST-001 registered; 5/8 PASS, findings recorded (008 engine defect; 003/009 tolerance observations) |
| G5 | dataset hashes/manifests validated | **PASS** — per-run sha256 manifests + TES-C2-013 integrity re-hash PASS |
| G6 | primary datasets materialized | **PASS** — SciFact + NFCorpus (qrels-granularity discrepancy documented) |
| G7 | ≥1 external baseline smoke completed | **PASS** — qdrant/pgvector/milvus-lite/weaviate PASS; ES ABORTED-recorded |
| G8 | every declared external baseline has a real status | **PASS** — all 8 declared have terminal status (PASS/ABORTED/BLOCKED-AUTH/EXCLUDED) |
| G9 | B3 trained or honestly blocked | **PASS** — trained on preregistered source only; activation validated |
| G10 | conditional dataset gates explicit outcomes | **PASS** — SciDocs/FiQA; COCO/ESCI liabilities unchanged, recorded |
| G11 | resource guardrails verified | **PASS** — samplers active; aborts preserved (ES row 20) |
| G12 | raw runs immutable | **PASS** — no-overwrite enforced; superseded attempts preserved with reasons |
| G13 | no main benchmark sweep | **CONFIRMED** — zero sweeps/ranks/quality runs |
| G14 | no C0/Phase3E mutation | **CONFIRMED** — diff evidence in A |
| G15 | C2 artifacts internally consistent | **PASS** — C2-INTEGRITY-001 PASS; smoke registry + RUN-INDEX reconcile |

**C2 STATUS: COMPLETE** (all gates G1–G15 pass or confirmed; failed runs are
preserved terminal results with recorded causes, per discipline).

## R. Main benchmark NOT begun

No recall/latency sweeps, no quality/budget-matched runs, no Track A/B
campaigns, no concurrency tests, no statistical comparisons. C2 is
preparatory; smokes are correctness/feasibility only.

## S. Phase3E / C0 untouched

`git diff fe4f92b HEAD -- research/phase3` = 0 lines; C0 audit docs and C1
artifacts unmodified (C2 added new files only). C2 never set `PH3E_CRASH_*`.
## T. Closure addendum (2026-09-24) — BM25 tie-order finding fixed & re-verified

Follow-up to §G/§Q (which predate the fix and remain preserved as recorded
history). The C2 modes battery surfaced an engine-core defect in the BM25
tie path (B5); it has been fixed, pinned, and re-verified.

- **Finding BM25-TIE-ORDER-001**: `core/src/bm25.rs` sorted tied BM25 scores
  without a doc-id tiebreaker over a per-call `HashMap`, making top-k doc-id
  order nondeterministic under strict ties and destabilizing fused top-k.
- **Fix `7cbd16d`** (branch `comparative-study/c2-validation`): append
  id-ascending tiebreaker (score desc, id asc) at all three sort sites
  (search, search_phrase, reciprocal_rank_fusion). Regression pin
  `core/tests/regression_bm25_tie_order.rs` **3/3 PASS**.
- **Evidence**: `C2-BM25-REPRO-001` FAILED (198/197 distinct orderings, 153
  distinct RRF memberships) vs `C2-BM25-REPRO-002` PASS (1/1/1). **`C2-MODES-
  TEST-002` PASS across 8/8 runs** with corrected harness contracts
  (TEST-C2-008 v2; TEST-C2-003/-009 determinism + conditional exact-equality).
- **Recorded observation HNSW-RECALL-OBS-001** (pre-existing, not caused by
  the fix): hnsw_rs search at k == element count is not exhaustive
  (0..7 docs/head unreachable, process-variable); this is why 003/009 assert
  exact equality only when per-head pools cover all docs, and it motivates the
  recorded-coverage (`exact_set_agreement_rate`) treatment of mode A.
- **Closure additions**: `c2/final-audit/` → `bm25-finding-and-disposition.md`,
  `c2-gate-audit.md` (G1–G15 re-verified; G4/G15 updated), `report-builder-audit.md`
  (build_reports.py draft-only; deterministic verifier authoritative),
  `c2-closure-decision.md`. Ledger re-check: 67 dirs, 0 inconsistencies,
  14/14 dataset hashes MATCH. PGVECTOR-003 verified as environment `su`
  credential issue (not a product defect).

**C2 STATUS: COMPLETE** (all gates G1–G15 pass/confirm post-fix; failed runs
preserved as terminal evidence; main benchmark not begun).
