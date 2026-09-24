# C1 — Benchmark Protocol (Comparative Benchmarking Study)

Study ID: `comparative-study-001` · Protocol version: 1.0.0 · Status: FROZEN at
the C1 commit. Post-freeze changes require a versioned amendment in
`preregistration.yaml` (§ protocol_change_policy) and, where outcomes were
already examined, reruns under new run IDs.

Entry commit (C0): `fe4f92b2187bf631125e975a5db8a202b5316d00` · C0 tree sha16
`d12fa093bc294977` · C0 commit `79b2837` · This protocol lives on branch
`comparative-study/c0-audit` (branch carries C0+C1; a dedicated C2+ branch may
be cut from the C1 commit).

## 1. What is being compared (Question A)

Two pre-registered, never-merged comparisons:

- **TRK-A (Matched retrieval-mode comparison):** retrieval/fusion behavior
  under controlled conditions — identical records, identical queries,
  identical exported embeddings, identical requested top-k, identical
  distance semantics (cosine where the system supports it; documented
  substitution otherwise). Systems/modes receive the SAME dataset files from
  a hash-manifested export. Focus: AttentionDB mode ladder (B1–B5, B7) vs
  the exact oracle (B0) and vs equivalent-mode configurations of feasible
  external systems.
- **TRK-B (Practical configuration comparison):** documented, realistic
  configurations of each system (vendor-recommended indexing + native
  features: filtering, hybrid). No forced architectural equivalence. All
  configuration and tuning decisions recorded. Setup/ingest/index/storage
  costs in scope.

Conclusions from the tracks are reported separately and never combined into
one ranking (charter §4, Rule 12).

## 2. Systems and modes evaluated (Question B)

Modes: MODE-B0..MODE-B7 per `mode-registry.yaml` (definitions grounded in the
C0 audit; equivalences stated plainly — e.g., B2 ≡ B3 without an activated
trained ModelCard; B4's attention is identity-init and previously measured a
no-op vs B3). Baselines/systems: BASE-ADB, BASE-QDRANT, BASE-PGVECTOR,
BASE-ES, BASE-MILVUSLITE, BASE-WEAVIATE, BASE-PINECONE (blocked:
credentials/budget), BASE-MONGO (blocked: Atlas-only for vector search),
plus BASE-ORACLE (harness-side exact kNN, engine-independent).
Feasibility status per `baseline-feasibility.csv`. No system is
benchmark-ready by declaration — each requires a C2 validation smoke.

## 3. Datasets (Questions C, D)

Registry + decisions in `dataset-matrix.csv` / `dataset-decisions.md`.

- **Primary (accepted):** `DS-SCIFACT` (BEIR; 5,183 docs, 300 queries, graded
  qrels 0–2) and `DS-NFCORPUS` (BEIR; 3,633 docs, 323 queries, graded qrels).
  Per-head representations are DERIVED from native document fields (title /
  body-text / [metadata]) via a version-pinned sentence-transformer
  (default: `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, 384-d);
  the canonical single-vector view (title+text concatenation, same model)
  is the like-for-like view for all systems. Sources cited in
  dataset-decisions.md; licenses recorded with verify-at-download flags.
- **Conditional:** `DS-SCIDOCS` (faceted citations), `DS-ESCI-EN-SUB`
  (multi-field product catalog, 4-grade E/S/C/I labels, subsampling protocol
  required), `DS-COCO-CAP-5K` (NATURAL two-modality heads — CLIP image vs
  text embeddings; CC BY 4.0; gated on a C2 CLIP-on-CPU feasibility smoke).
- **Efficiency/scale track only (ANN ground truth, NOT semantic):**
  `DS-ANN-GLOVE-25`, `DS-ANN-GLOVE-50` (ann-benchmarks HDF5, pre-split,
  top-100 exact ground truth); `DS-ANN-FMNIST`, `DS-ANN-NYTIMES` conditional
  for the size ladder.
- **Rejected:** `DS-MSMARCO-FULL` (host RAM); Flickr8K (gated research
  agreement — cannot download in-sandbox).
- **Diagnostic only:** `DS-SYNTH-PH2B` (existing deterministic generators,
  NEW seeds and registry) — never presented as real-world evidence.

Splits (Question D): BEIR sets have no official train split → the leakage
protocol (`training-and-leakage.md`) uses (i) held-out training corpora
(synthetic/auxiliary, train-split-only supervision for B3), (ii)
leave-one-dataset-out transfer across accepted BEIR micro-sets, and (iii)
strictly untouched test queries. No model sees test queries, test labels, or
test-corpus statistics.

## 4. Resource and index cost accounting (Question E)

`fairness-and-resource-accounting.md`: two reporting views (retrieval-only;
end-to-end), with explicit per-record vector counts, per-head dimension,
total indexed dimensions, index build time/size, storage amplification, peak
and steady RSS, and (when applicable) representation-generation cost.
Multi-head configurations are NEVER compared against single-vector
configurations without the ×heads disclosure line in the same table cell.

## 5. Quality-vs-latency comparison (Question F)

`recall-and-budget-matching.md`: (a) quality-matched comparison — tune
candidate-generation parameters ON VALIDATION QUERIES ONLY to hit a
pre-declared Recall@10 target (±0.01), then compare frozen-config test
latency; (b) budget-matched comparison — fixed candidate/ef/memory budgets
from a preregistered grid. If a system cannot reach the target recall, the
full quality–latency curve is reported and the mismatch declared — a false
match is never forced.

## 6. Environment feasibility (Question G)

C0 host: 2 vCPU, ~1.9 GiB RAM, ~20 GB disk, no Docker/systemd. Feasibility
statuses and smoke-test requirements are in `baseline-feasibility.csv`;
guardrails (preflight, thresholds, abort evidence) in
`environment-guardrails.md`. Managed systems requiring credentials/payment
(Pinecone, MongoDB Atlas) are BLOCKED pending explicit user authorization;
no data leaves the sandbox.

## 7. Evidence required before a result is accepted (Question H)

A benchmark result is admissible iff: (1) it has a unique immutable run
directory under `research/comparative-study/raw/runs/` with the full run
schema (§12 of charter; validator-enforced); (2) its dataset hash matches
the dataset manifest; (3) its configuration is registered
(mode-registry/baseline-feasibility or an amendment); (4) the metric
implementation is the registry one (recomputed from raw per-query rows);
(5) classification (VALID/INVALIDATED/FAILED/BLOCKED/…) is present with
reason; (6) the claim tracing to it passes the chain
CLAIM→FINDING→ANALYSIS→TABLE→RAW→CONFIG→DATASET-HASH→COMMIT. Anything else
is reported as a limitation, not evidence.

## 8. What C1 does NOT authorize

No performance claims; no readiness claims; no result values anywhere in
these documents; C2 implementation has not begun. C1 is complete as a
preregistration only.
