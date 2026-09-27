# C1 — Dataset Decisions and Rationale

Companion to `dataset-matrix.csv`. No dataset was downloaded during C1 except
where a tiny metadata verification was needed (none was: all size/license
facts below come from the cited official pages, verified 2026-09-25, and each
carries a verify-at-download flag for the C2 manifest step).

## The multi-head validity problem (charter §6)

C0 established that AttentionDB heads are independent indexes over
CALLER-SUPPLIED per-head vectors. A dataset therefore supports meaningful
multi-head retrieval only if its head views are either (a) NATIVE distinct
representations (different modalities/spaces) or (b) DERIVED from native
field structure with an externally-defined, version-pinned model — never
constructed post hoc to flatter multi-head fusion. Both accepted primaries
use (b) with a pinned model; DS-COCO-CAP-5K provides the (a) case, gated on
a C2 feasibility smoke.

## Accepted datasets

### DS-SCIFACT — ACCEPT-PRIMARY
- Source: BEIR (https://huggingface.co/datasets/BeIR/scifact; original
  SciFact: https://openreview.net/forum?id=wF5DuRiKfMq). 5,183 corpus docs,
  300 test queries, graded qrels (0/1/2). License: BEIR HF cards state
  cc-by-sa-4.0; the underlying SciFact dataset is CC BY-NC 4.0 — both
  recorded; verify-at-download flag set (research use).
- Head construction (preregistered): HEAD-TITLE = embed(title),
  HEAD-BODY = embed(abstract/context), HEAD-CITE = embed(structured
  key-sentences) [optional third head, fixed before any run]; canonical
  single-vector view = embed(title + " " + text) — the view every external
  single-vector system receives. Model: all-MiniLM-L6-v2, version + file
  hashes in the C2 dataset manifest. Cosine metric.
- Ground truth, two kinds kept separate: (1) NN-GT from B0 exact search on
  each view (candidate-quality analysis); (2) qrels-based relevance metrics
  (nDCG@10 etc. — the PRIMARY quality metric because graded human labels
  exist). NN-GT is never presented as relevance.
- Feasibility: fully within host envelope; 300 queries is a fixed sample —
  power analysis in statistical-plan.md (paired design, detectable effect
  computed before running).

### DS-NFCORPUS — ACCEPT-SECONDARY
Same mechanics (3,633 docs / 323 queries / graded 0–2; title/body views).
Second domain for the leave-one-dataset-out transfer test of B3 gating.

### DS-ANN-GLOVE-25 / DS-ANN-GLOVE-50 — ACCEPT (efficiency/scale track only)
- Source: ann-benchmarks official pre-split HDF5 (train/test + exact top-100
  NN ground truth), https://github.com/erikbern/ann-benchmarks; mirror
  huggingface.co/datasets/hhy3/ann-datasets (sizes 121/235 MB confirmed on
  the mirror page).
- Purpose: RQ4/RQ8 recall–latency curves and the dataset-size ladder with
  exact NN-GT at host-feasible scale. Announced in every table as
  NN-GROUND-TRUTH (no semantic claim). 1M-vector SIFT rejected on RAM;
  smaller rungs (Fashion-MNIST 60k, NYTimes 290k) conditional pending C2
  RAM smokes.

## Conditional datasets

### DS-ESCI-EN-SUB — CONDITIONAL (Track B centerpiece if admitted)
- Source: Amazon "Shopping Queries Data Set"
  (https://github.com/amazon-science/esci-data; license pointer
  https://github.com/amazon-research/esci-code — reported CC BY-4.0, verify
  at repo). Native multi-field products (title/description/bullet_point/
  brand) with manual 4-grade E/S/C/I judgments (~20 judgments/query).
- Required preregistered subsampling protocol (before any download in C2):
  keep ONLY queries whose FULL judgment closure (all judged products) falls
  inside the sampled catalog; sample catalog from the official TRAIN split
  queries; official test split queries untouched for headline reporting;
  corpus capped at what the host RAM smoke allows (target 50–100k products).
- Risk recorded: subset-of-catalog retrieval is easier than full-catalog
  (judged products overrepresented) — disclose in every table.

### DS-COCO-CAP-5K — CONDITIONAL (the NATIVE two-head case)
- Source: COCO (https://cocodataset.org), CC BY 4.0; val2017 5k-image subset.
- Heads: CLIP ViT-B/32 image embedding vs CLIP text embedding (genuinely
  different input spaces; MIT-licensed model, pinned). Queries: held-out
  captions; ground truth = target image (exact match), the standard
  caption→image task. Near-duplicate images exist in COCO — documented
  caveat; strict GT = own image.
- GATE: CLIP-on-CPU encode time for 5k images (estimated 25–60 min on 2
  vCPU) must pass a C2 smoke; otherwise the native-head requirement stands
  unmet on public data and that fact is REPORTED (acceptance-rule fallback
  described below).

### DS-SCIDOCS / DS-FIQA-2018 — CONDITIONAL (limited roles)
SciDocs: citation-graph facets need a facet-validity check before multi-head
use. FiQA: no field structure — Track B single-vector practical workload
only.

## Rejected

- DS-MSMARCO-FULL: 8.8M passages exceeds host RAM by an order of magnitude;
  no official small variant with qrels closure. (A future, properly resourced
  run may add it; recorded as deferred-by-environment, not skipped.)
- DS-FLICKR8K: download gated behind a UIUC research agreement — cannot be
  reproduced in this sandbox; COCO (CC BY 4.0) covers the two-modality
  design.

## Diagnostic only

- DS-SYNTH-PH2B: the existing in-repo deterministic generators (new seeds,
  new registry under comparative-study/raw/) are used for HARNESS VALIDATION
  (C2 correctness checks, ground-truth cross-verification) and as B3
  TRAINING corpora. They are never counted as real-world evidence; every
  table sourced from them is labeled DIAGNOSTIC/SYNTHETIC.

## Acceptance-rule check (charter §6)

Primary requirement: ≥1 dataset with defensible relevance labels AND a clear
reproducible per-head method. DS-SCIFACT + DS-NFCORPUS satisfy this: graded
human qrels (official), field-split head views from a pinned public model,
full reproducibility from cited sources. **The rule is MET without the COCO
conditional**; COCO would strengthen the native-head case but its gate
failing would not block the study. No blocker to report.
