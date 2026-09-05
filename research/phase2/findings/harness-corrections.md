# Harness Corrections (§5)

Full record of benchmark/harness defects. Rule: a correction names the
invalidated runs and their replacements; invalidated raw numbers stay in
the registry (status INVALIDATED), never deleted.

## HC-1: Ground truth in hint-ids instead of engine numeric ids

- **Original behavior**: `generate_dataset` wrote ground truth as corpus
  numeric_hint ids (0-based) while candidates carry engine numeric ids
  (IdMapper starts at 1, INV-6).
- **Why wrong**: every recall/NDCG/MRR compared shifted id sets.
- **Detection**: noise corpus returned R@10 = 0.000 for ALL approaches
  INCLUDING the oracle (PH2B-NOISE-001). An oracle scoring zero is a
  harness signature, not a model result. On controlled, the shift partially
  masked itself (same-block docs are id-adjacent) — measured recalls were
  plausible-but-wrong (PH2B-GATING-001/002), which is why the all-zeros
  run was the real tell.
- **Correction**: map hint → engine id via `document_store` +
  `id_mapper.uuid_to_id` at dataset generation; recompute stored per-head
  quality metrics against mapped ids.
- **Invalidated results**: PH2B-GATING-001, PH2B-GATING-002,
  PH2B-NOISE-001 (status INVALIDATED in the registry; metrics preserved
  verbatim).
- **Valid?** As evidence about retrieval: no. As evidence about the
  harness: yes (documented above).
- **Replacement runs**: PH2B-GATING-003 → PH2B-GATING-004 (current),
  PH2B-NOISE-002 → PH2B-NOISE-003 (current).

## HC-2: Multiview GT tie lottery

- **Original behavior**: doc "identity" per view was its topic centroid
  (75 docs/topic); view-level cosine tied exactly within a topic.
- **Why wrong**: top-10 among 75 exact ties = insertion-order lottery;
  oracle itself scored 0.26 (PH2B-MULTIVIEW-001).
- **Detection**: oracle far below 1.0 on a corpus where the matching view
  should be near-perfect.
- **Correction**: doc-unique fine structure per view (0.6·u_v term);
  GT = exact cosine in the query's view.
- **Invalidated**: PH2B-MULTIVIEW-001. **Replacement**:
  PH2B-MULTIVIEW-002+ (which exposed HC-3).

## HC-3: Multiview queries carried full-fidelity information in every view

- **Original behavior**: a type-V query was built from the target doc's
  view-V vector, and NON-target views got the same full-fidelity vector.
- **Why wrong**: every view then identified the target doc equally well;
  "which view is useful" became statistically undetectable from the query.
  The gate correctly refused to learn: corr = −0.34, weights ≈ uniform
  (PH2B-MULTIVIEW-002). Real cross-modal queries arrive in ONE modality.
- **Detection**: near-uniform predicted weights + negative weight–quality
  correlation while the oracle scored 0.99 — an information-theoretic
  contradiction, i.e. the input lacked the signal.
- **Correction**: non-target views receive only the coarse topic-level
  projection (+noise), simulating a cross-modal extractor.
- **Invalidated**: PH2B-MULTIVIEW-002 (design), superseded by
  PH2B-MULTIVIEW-003/004/005.

## HC-4: Phase 2 ground truth (pre-Phase-2B) — same-centroid tie lottery

- **Original behavior**: all docs in a cluster shared the centroid vector.
- **Detection**: `benchmarks/phase2/src/bin/probe.rs` — engine retrieved
  99/100 same-cluster docs while scored recall was ≈ 0.10 (chance-level).
- **Why accepted results were NOT invalidated**: the Phase 2 ablation
  compared modes against the SAME (degenerate) GT; the ranking ORDER of
  modes was confirmed stable across three runs, and the maintainer
  accepted the benchmark with the known limitation recorded in
  `benchmarks/phase2/ablation.md`. Absolute recall values from that
  harness are not comparable to Phase 2B absolute values — cross-phase
  numeric comparison is forbidden.
- **Correction**: per-doc true vectors (adopted by all Phase 2B corpora).

## HC-5: `rank_metrics` treats slice order as the ranking (Phase 2C)

- **Original behavior**: `learned::eval::rank_metrics` computes ranks from
  the ORDER of its input slice — it does not sort by score. Callers that
  pass an unsorted (pool-order, engine-order) candidate/score zip get
  metrics of the arbitrary input order, silently.
- **Detection**: PH2C-QK-001 — `qk_untrained` measured R@1 = 1.0000 in the
  first run, impossible for a random bilinear scorer on a 10-way chance
  task. An independent Python replication of the generator RNG (exact
  xorshift/Box–Muller sequence) measured chance-level R@1 (0.08–0.13) for
  the same random weights, isolating the defect to the evaluator.
- **Why the defect matters**: any unsorted caller silently reports metrics
  of pool order, not score order. In PH2C-QK-001 this affected (a) the QK
  evaluation path, (b) the generation-time per-head quality metrics (which
  feed gating training targets), (c) the oracle construction.
- **Correction**: every caller sorts by score (desc, id-asc tie-break)
  before calling `rank_metrics`; the anti-cosine invariant test
  (`sanity_head_quality_targets_reflect_true_head_ranking`) now pins the
  generation path, and the learnability test pins the QK eval path.
- **Invalidated**: no recorded numbers — the defect was caught and fixed
  before PH2C-QK-001 entered the registry (first recording used the fixed
  evaluator; reproduction verified byte-identical after a subsequent
  clippy-only change).

## HC-6: controlled/noise corpus builders are not content-reproducible (PH2C-QK-002)

- **Original assumption**: re-running `corpora::controlled`/`noise_ladder`
  with the recorded seeds rebuilds the same doc content (only HNSW graph
  construction was known OS-seeded — see dataset-as-unit doctrine).
- **Detection**: PH2C-QK-002 §4 gate (`qk-cache`). GT/doc-id mapping and
  doc-id picks DO reproduce (gt_verified 300/300), but doc/query VECTOR
  content does not: recomputed exact cosines differ from cached exact
  scores (e.g. 0.9900 vs 0.9468). `corpora::multiview` reproduces exactly.
- **Why it matters**: attaching rebuilt content to frozen pools would have
  scored QK against vectors unrelated to the pools — silently invalid.
- **Correction**: content attached only where the exact-score linkage check
  passes (multiview). controlled/noise QK arms recorded NOT_EXECUTABLE;
  the cached dataset.json remains the sole canonical record for those
  corpora. No numbers were invalidated (the check ran BEFORE training).
