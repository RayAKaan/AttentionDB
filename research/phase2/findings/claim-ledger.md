# Claim Ledger (§21, §22)

Every major claim maps to experiment IDs in `raw/experiment-index.json`.
Statuses: SUPPORTED / NOT SUPPORTED / OPEN QUESTION. Statistical confidence
is reported as mean ± std over training seeds where a multi-seed run exists;
all other entries are single-run point estimates (stated).

## SUPPORTED

**C1.** "Query-dependent learned gating improves retrieval over fixed
multi-head fusion on the evaluated controlled and multiview corpora."
→ PH2B-GATING-004 (0.9533 vs 0.6244, +33pp R@10), PH2B-MULTIVIEW-005
(0.5322 vs 0.2128, +32pp), PH2B-MULTISEED-002 (0.4365 ± 0.0361 vs 0.2128).
Limitation: synthetic corpora, small query counts, single split per corpus.

**C2.** "Gating can recover oracle head-selection behavior on the
controlled corpus." → PH2B-GATING-004: gating R@10/NDCG@10 identical to
oracle (0.9533/0.9700). PH2B-MULTISEED-001: 0.9556 ± 0.0000 across seeds.
Limitation: 4 heads, strong block-structured signal.

**C3.** "RRF loses to trained gating on the evaluated corpora." → F7
evidence chain; margins +6.9pp to +28.4pp R@10. Limitation: RRF k fixed at
60 (the standard constant [CITE: RRF]); no k sweep.

**C4.** "Trained gating adds negligible inference cost." → PH2B-LATENCY-001:
0.87 µs/query, 1188 params, 20 288 bytes serialized, CPU-only. Limitation:
micro-benchmark on one machine class; end-to-end adds one matmul per query.

**C5.** "On a globally-dominated corpus, trained gating reduces to selecting
the dominant head." → PH2B-NOISE-003 (avg weight 0.894, selection 100%,
quality ≈ global best). Limitation: one corpus family.

**C6.** "Parallel head execution preserves semantics and speeds up
multi-head search." → Phase 2 `heads-scaling.csv` (identical recall serial
vs parallel; 1.39×/1.72× p50 at 4/8 heads) + unit tests (parallel==serial).
Limitation: shared VM, cpu-count dependent.

**C7.** "Ground-truth construction and id-mapping bugs were found, fixed,
and re-run." → PH2B-GATING-001/002 (invalidated), harness-corrections.md,
PH2B-GATING-003+. Limitation: none (documented).

## NOT SUPPORTED

**N1.** "Candidate-level QK attention improves retrieval." → No trained-QK
experiment exists (PH2C-QK-* pending). The only QK data point is the
UNTRAINED identity scorer (Phase 2 mode D = mode C), which supports only:
"identity-init QK adds nothing" (F8). Claim status until trained results
exist: NOT SUPPORTED (insufficient evidence).

**N2.** "More heads improve retrieval." → Phase 2 ablation: 8-head fixed
fusion (0.672) LOSES to the best single head (0.763) on its own corpus.
Multi-head helps only as robustness-to-wrong-head-choice, not accuracy.

**N3.** "Exact reranking as implemented in the Phase 2 pipeline improves
retrieval." → Phase 2 mode E 0.558 < mode D 0.672 (regression). The offline
study attributes it to equal-head weighting of exact scores
(PH2C-RERANK-*), but the pipeline itself has not been re-weighted and
re-evaluated — the pipeline-level fix remains unvalidated.

## OPEN QUESTIONS

**Q1.** Does learned gating generalize to real-world multi-view retrieval
workloads (real embeddings, real lexical features, real structured fields)?

**Q2.** Where exactly between 420 and 840 training queries does the
multiview transition occur, and is it a true threshold or a smooth curve?

**Q3.** Can a rerank mixture with learned/oracle head weights fix mode E
inside the live pipeline (the offline evidence says the scores are fine;
the pipeline change is untested)?

**Q4.** Does trained QK attention add value beyond learned gating (§11–13)?

**Q5.** Does temperature calibration transfer (one T per corpus vs per
query-band), and is validation-fitted T stable across datasets?

**Q6.** How do these results scale beyond 10k docs / 300–1200 queries?

---

## Dated addenda (append-only; never rewrite entries above)

**Addendum to N1 (2026-09-04, PH2C-QK-001).** A trained-QK data point now
exists — on the SYNTHETIC sanity dataset only: linear QK reaches test
R@1 = 1.0000 where every gating variant is ≤ chance (0.0027 / 0.1000),
confirming the machinery can learn candidate-level interaction that the
gating class provably cannot express [PH2C-QK-001]. N1 itself is UNCHANGED
for real corpora: no trained-QK result on controlled/noise/multiview
exists yet, so "candidate-level QK improves retrieval (on the real
corpora)" remains NOT SUPPORTED until PH2C-QK-002+ measures it (Rule
Zero: outcome open).

**Addendum 2 (2026-09-05, PH2C-QK-002-MULTIVIEW).** N1 RESOLVED for the
frozen-pool protocol: trained candidate-level QK LOSES to trained gating on
multiview (0.1113 vs 0.4983 R@10, seeds 42/7/1; candidate recall 0.9975 —
ranking failure, not generation failure). "Candidate-level QK attention
improves retrieval" = NOT SUPPORTED (multiview; controlled/noise
untestable on frozen caches per HC-6). §29 STOP condition engaged: no
deeper/cross-attention architectures. Gating retained as the architecture;
QK optional (no measured benefit anywhere).

**Addendum 3 (2026-09-17, PH3D).** Database validation (Phase 3D): new claims, each
run-backed — correctness of mixed-workload state vs a reference model (PH3D-STATE-001..004,
0/24 gated checks failed); filter SOUNDNESS guaranteed / completeness candidate-bound,
recall 0.967–1.0 (PH3D-FILTER-001); acked-write durability per selected mode with
GroupCommit/Sync ALL_ACKED at 7 crash points and Async proper-prefix + explicit
committed-txn-loss semantics (PH3D-CRASH-001..003, 21/21 contract-consistent, zero
resurrection); WAL corruption detect-and-refuse (torn tail warned+truncated; corrupt
frame/gapped segment refuses open; pre-checkpoint segment deletion undetectable — OPEN);
transactions all-or-nothing under crash and injected failure, NO update op, NO isolation
claim (PH3D-TX-001); concurrency stable with zero errors, p99 tail growth documented, NO
linearizability claim (PH3D-CONC-001/002); backup/restore exact on the quiescent path,
online backup UNSUPPORTED/documented (PH3D-BACKUP-001). Two product fixes with regression
tests: `compact_all` now resolves `db_dir/sst` (was an unreachable silent no-op) and
`check_db_dir` WAL-gap invariant replaces a post-checkpoint false positive. No ACID/
linearizability/production-ready claims beyond what these runs justify; production
readiness recorded only as a 25-capability matrix (results/production-readiness.csv),
never a single score. Memory optimization NOT attempted (PH3D-MEM-OPT-001 reserved).

**Addendum 4 (2026-09-17, PH3D audit closure).** Re-audit of Phase 3D against the full
spec: registered the spec-conforming IDs PH3D-MUTATION-001, PH3D-RECOVERY-001,
PH3D-COMPACTION-001, PH3D-CONCURRENCY-001 (child runs mapping to the delivered families;
parent-map.json in each run dir) and ran three NEW experiments — PH3D-INTEGRATION-001
(multi-collection isolation across restart/compaction/backup-restore 16/16; graceful
close→reopen durability for all four mutation kinds; filter × multi-head soundness;
discovered + documented: engine uuid identity is GLOBAL, collections are membership tags,
same-uuid insert re-members the document — multi-collection callers must namespace ids),
PH3D-CONC-003 (deterministic concurrent mutation logs; merged-log replay == observed state
exactly, 450 docs; same-key contention 100 keys × 150 rounds, 0 torn records — visibility
model documented, no linearizability claim), PH3D-BACKUP-002 (per-file size+sha256 backup
inventory; independent integrity: source==backup and backup==restored, 0 mismatches).
Crash-point mapping to the spec's 10 points, txn-failure injection granularity, §21
UNSUPPORTED update-combos, §34 fuzz scope, and §41–42 numbering recorded in
methodology/ph3d-spec-deviations.md. Deterministic fuzz regression tests added for the WAL
parser (random bytes + truncated valid WALs: never panic, strict prefix or error) and
filter validate/eval. All prior PH3D claims unchanged; tables renumbered to spec order
(1–8 + extras 9–11); figures renamed figure-ph3d-1..3 (PH3C 1–9 untouched).

## Addendum 5 (2026-09-17) — Phase 3E E1: WAL-integrity refusal invariant

**Change:** durable rotation-time WAL high-water sidecar (`WAL/wal-state.json`) +
`open_dir` refusal of databases missing required WAL history (`WAL_LOST_SEGMENT`,
`WAL_SEQ_GAP`, `WAL_STATE_CORRUPT`). Closes the PH3D-WALCORRUPT-001 `delete_segment`
OPEN hole (previously: deleted pre-checkpoint segment → apparently-valid empty DB).

**Claim now justified:** "a single-node AttentionDB database whose required pre-checkpoint
WAL history has been deleted or gapped REFUSES to open, and reports the loss — it never
opens as an apparently-valid partial database." Evidence: PH3E-WAL-001 (11 cases, 6
refusals, 5 legitimate opens, 0 expectation mismatches) + 9 engine unit tests.

**Boundary (documented, not hidden):** watermark anchors at segment rotation, not per
append — loss of the post-last-rotation ACTIVE segment remains undetectable while no
watermark covers it (bounded by the segment-size threshold). Legacy sidecar-less DBs open
with pre-E1 semantics. The catalog gains NO field (bincode v1 positional) — sidecar only.

**Not claimed:** durability level changes (E2 pending); crash-machine coverage (E3
pending); no ACID/linearizability/exactly-once vocabulary introduced by E1.
