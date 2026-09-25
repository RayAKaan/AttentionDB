# B3 — Learned Gating: Training, Provenance, Leakage Controls (C2)

## What B3 is (binding, from C1)

B3 = AttentionDB mode C (LearnedGating) with a REAL trained and activated
`ModelCard`. Uniform weights are NEVER called learned (charter §7). The MLP
input at inference is the vector handed to `attend()` (engine
`collection.rs::get_trained_weights`); for Track A semantic datasets that is
the canonical pinned-MiniLM query embedding (384-d) and the heads are the
preregistered document-side views (HEAD-TITLE / HEAD-BODY / HEAD-CITE).

## Training sources (preregistered, C1 training-and-leakage.md)

1. `DS-SYNTH-PH2B` multiview corpora — the in-repo deterministic generator
   (`phase2b_bench::corpora::multiview`), 3,000 docs × 3 heads (semantic /
   lexical / metadata, dim 64), NEW seed 20260925, exported by `c2probe
   corpus`. Query split: phase2b scheme (seeded 70/15/15, seed 20260925).
   Two input modes: `concat` (3×64, phase2b offline scheme — pipeline
   validation) and `view0` (64-d, engine-compatible — activation card).
2. LODO transfer arm `b3-lodo-nfcorpus`: NFCorpus-derived 384-d dataset —
   features = canonical NFCorpus test-query embeddings; targets = B0 exact
   per-head recall@10 on the preregistered head views vs NFCorpus qrels.
   **Disclosed per C1**: NFCorpus test queries serve as TRAINING inputs; the
   held-out headline dataset for this arm is SciFact, which is never read by
   this pipeline.

## Targets

Per-query per-head B0 exact top-k (score DESC, id ASC) → recall@10 /
nDCG@10 / MRR per head vs ground truth (generator GT for synthetic; qrels
positives for NFCorpus). Objective `soft_target` (cross-entropy against
softmax(quality/τ)), trainer = in-repo `learned::gating_v2::train_gating`
(Adam, batch 32, patience 15, early stopping on VALIDATION loss only).
Grid + seeds: see `training-config.yaml` (hashed into every run).

## ModelCards

`modelcards/` — engine `ModelCard` JSON (format v1) with TrainingMeta
(seed, FNV-1a dataset hash, objective, lr, batch, epochs_run, best_val_loss,
l2, timestamp, code commit, hardware). Selection rule (validation-only) is
recorded in each `C2-B3-TRAIN-*` run. The activation card
(`b3-synth-view0`) is validated against the real engine in `C2-B3-VALID-001`
(install → inspect → attend == attend_weighted(card softmax) → head-name
mapping under permutation → deactivate → uniform fallback).

## Leakage rules (INV-L1..L5) — all negatively tested in C2-B3-LEAK-001

- INV-L1: `c2probe train` accepts only `synth-multiview*`/`lodo-train*`
  corpus_desc prefixes; a headline-test-marked dataset and a
  `test-exports/` path are REFUSED (exit 2). Structural: the trainer reads
  only Train/Val rows.
- INV-L2: `training-config.yaml` hash frozen into every run config.
- INV-L3: targets computed only from exported training datasets (provenance
  recorded per dataset file).
- INV-L4: engine `ModelCard::validate()` refuses invalid provenance
  (TEST-C2-011c).
- INV-L5: bitwise reproducibility retrain recorded per training invocation.

## Honest-status rule

If any link fails under host resources, B3 is reported BLOCKED-TRAINING with
the exact reason. B2 is never relabeled as B3.
