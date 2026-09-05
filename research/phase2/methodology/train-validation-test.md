# Train / Validation / Test Protocol (§3, §26)

- Split: seeded Fisher–Yates (seed 42 ^ 0x5EED) over query indices, 70/15/15.
  Recorded per run in `run_info.txt` and the registry.
- **Training** sees only Train queries (loss + epoch order shuffling).
- **Validation** drives: (a) early stopping, (b) objective/hyperparameter
  selection across the grid, (c) temperature calibration.
- **Test** is evaluated exactly once per run after all selection is frozen.
  No test-derived decision exists in the pipeline; the code has no code path
  that reads Test before evaluation.

Observed overfitting chain (`results/*-overfit-chain.csv`): train ≥ val ≥
test in the expected ordering on every corpus; no case of train-up /
test-flat was observed at the selected configurations.

Multi-seed (§12): the split is held fixed (seed 42) and training seeds
{42, 7, 1} vary model init + batch order. Rationale: seed robustness of the
LEARNER is the question; re-splitting would confound learner variance with
query-sample variance.

Sample efficiency (§13): corpus query count is varied (150/300/600/1200),
which changes all three splits together — different test sets per point.
Point-to-point comparisons are therefore directional (learning vs
not-learning), not fine-grained; this is stated on the table.
