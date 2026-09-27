# C1 — Training & Leakage Controls (B3 and any learned mode)

Study `comparative-study-001`, protocol v1.0.0. Binding for C2+; violations
invalidate the run (preregistered invalidation rule INV-L1..L5 below).

## The problem

The accepted primary test sets (SciFact 300 / NFCorpus 323 queries) have NO
official train split, and the test sets are fixed public benchmarks. Training
B3's gating on them is impossible without leakage. This protocol defines how
B3 is trained honestly — or marked blocked.

## Split policy (preregistered)

1. **Training corpora (supervision source):** DS-SYNTH-PH2B multiview
   corpora (deterministic in-repo generators) with explicit generator-level
   train/val/test splits, PLUS leave-one-dataset-out (LODO) transfer:
   train on {NFCorpus-derived synthetic analog + synthetic multiview},
   validate on the generator validation split, test ONLY on the untouched
   official test set (SciFact or NFCorpus, whichever is held out).
   - LODO detail: for a SciFact-headline experiment, gating supervision may
     use NFCorpus-derived data ONLY (its test queries included as TRAINING
     inputs is permitted and disclosed — the held-out dataset remains
     SciFact); for an NFCorpus-headline experiment, vice versa. Headline
     datasets are never each other's contamination because supervision never
     touches the headline dataset's queries, corpus, or labels.
2. **Supervision targets:** per-query per-head targets derived from ORACLE
   head rankings on training corpora (B0 exact per-view recall as the
   target signal), computed ONLY on training data. Target-construction code
   is part of the harness and is deterministic (seeded).
3. **No test usage:** the headline test set (queries, corpus, qrels) is not
   read by any training, target-construction, calibration, or
   checkpoint-selection process. Enforced procedurally: the training
   pipeline never receives the test export path.

## Fitting/selection isolation

- Normalization statistics: per-head normalizers are engine-internal
  (min-max over result sets — no dataset-level fitting exists in the engine;
  verified in C0 stage 3). Any harness-side statistic (e.g., temperature
  scaling) is fit on training data only.
- Hyperparameters (MLP hidden size, lr, epochs, patience): selected on the
  VALIDATION split of the training corpora only; ONE default grid,
  preregistered in the C2 run configs, no test-informed changes.
- Checkpoint selection: best validation loss (early stopping patience per
  learned/ GatingTrainer defaults); the chosen checkpoint + seed + training
  metadata are written into the ModelCard (TrainingMeta) — provenance field
  is part of the artifact.
- Seeds: training seed fixed at C2 run config (primary 20260925, secondary
  seeds only for the multiseed variance run — 3 seeds minimum, reported as
  mean ± std).
- Training compute/duration: recorded per run (wall time, process CPU time);
  the 2-vCPU host makes training minutes-scale — measured, never assumed.

## Duplicate / near-duplicate handling

- BEIR corpora: near-duplicate handling is the dataset's own (documented as a
  caveat, e.g., COCO near-duplicate images); no additional dedup is applied
  to official sets (preserves comparability with published numbers).
- Synthetic training corpora: generator-enforced distinct records (existing
  behavior); verified in C2 harness validation.
- ESCI (if admitted): subset protocol deduplicates product_ids; queries are
  the official ones (no synthetic queries).

## Preregistered invalidation rules

- INV-L1: any artifact showing the headline test set path accessed by the
  training pipeline → all B3 runs INVALIDATED.
- INV-L2: hyperparameters or checkpoints changed after any test metric was
  computed → affected runs INVALIDATED (rerun under new IDs with the frozen
  config).
- INV-L3: supervision targets computed on anything but training corpora →
  INVALIDATED.
- INV-L4: ModelCard missing TrainingMeta provenance (seed/data hash/
  hyperparams) → run INVALIDATED (unverifiable provenance).
- INV-L5: an "activated ModelCard" that cannot be reproduced from the
  recorded config → the B3 arm is reclassified BLOCKED and any results
  relying on it are marked unsupported.

## Honest-status rule

If the training pipeline cannot be built and verified credibly under host
resources in C2, B3 is reported **BLOCKED-TRAINING** with the exact reason.
Under NO circumstance is uniform-weight B2 relabeled or reported as learned
gating (charter §7; C0 finding 3). B2-vs-B3 contrast is then reported as
not-established with the blocker documented.
