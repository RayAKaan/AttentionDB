# B3 Training & ModelCard Validation Report (C2)

Study `comparative-study-001`, protocol v1.0.0 (commit `7788067`).
B3 verifies that the in-repo gating trainer runs on preregistered sources with
leakage guards, that ModelCards carry valid TrainingMeta, and that the engine
can activate/use them. **No benchmark results are produced.** Config was
preregistered before any training run (C1 §fitting-isolation).

## Preregistered grid & arms (b3/training-config.yaml, sha256 8283c127…)

- trainer: `learned::gating_v2::train_gating` (in-repo, reused unmodified)
- objective soft_target vs softmax(quality/tau); quality_target recall_at_k
  (B0 exact per-view recall); tau 1.0; batch 32; max_epochs 200; patience 15;
  min_delta 1e-4; l2 1e-4
- grid: hidden [16,32] × lr [0.005,0.01,0.02] (ONE grid, selection on the
  trainer's val split only)
- seeds: primary 20260925; variance [1,2] on the selected config only
- arms: b3-synth-concat, b3-synth-view0, b3-lodo-nf (NFCorpus LODO, disclosed)
- leakage guards INV-L1..L5 (see config)

## Run ledger (terminal statuses)

| Run | Status | Meaning |
|---|---|---|
| C2-B3-LEAK-001 | FAILED | first leakage-guard execution (superseded by -002) |
| C2-B3-LEAK-002 | PASS | INV-L1..L5 negative tests PASS |
| C2-B3-DATA-LODO-002 | PASS | LODO gating dataset materialized (exact targets) |
| C2-B3-TRAIN-001/-002/-004/-005 | INVALID-STARTUP | SIGPIPE, harness KeyError, id collisions — preserved, superseded |
| C2-B3-TRAIN-006/-007/-008 | SUPERSEDED | first-batch arms (superseded by -v2 reruns for card schema) |
| C2-B3-TRAIN-009 | PASS | arm b3-synth-concat-v2 (grid + primary seed) |
| C2-B3-TRAIN-010 | PASS | arm b3-synth-view0-v2 (grid + primary seed) |
| C2-B3-TRAIN-011 | PASS | arm b3-lodo-nf-v2 (grid + primary seed) |
| C2-B3-VALID-002/-003 | FAILED | exact-f32 equality too strict vs re-normalized attend_weighted; switched to ids + 1e-5 (superseded by -004) |
| C2-B3-VALID-004 | PASS | ModelCard activation validation 6/6 PASS (TEST-C2-010/010a/010b/011a/011b/011c) |

## Training evidence (TRAIN-011, arm b3-lodo-nf-v2)

- grid winner: hidden 32, lr 0.005, best_val_loss 1.098542 (min across grid;
  ties → lower lr then smaller hidden rule)
- seeds_trained [20260925, 1, 2]; wall_s 14.7
- each card meta: `reproducible_bitwise: true`, `stopped_early: true`
  (best_epoch 0), dataset_hash_fnv1a64 pinned, `rows.test_unused_by_trainer: 0`,
  curves recorded
- modelcards written under `c2/b3/modelcards/` (`-v2` set carries head_names +
  TrainingMeta; s1/s2 variance cards exist for lodo-nf-v2)

## ModelCard engine activation (VALID-004 = TEST-C2-010..011)

| Test | Name | Status |
|---|---|---|
| TEST-C2-010 | ModelCard activate + inspect | PASS |
| TEST-C2-010a | mode C results == attend_weighted(card softmax); weights match engine | PASS |
| TEST-C2-010b | deactivate → documented uniform fallback | PASS |
| TEST-C2-011a | num_heads mismatch refused | PASS |
| TEST-C2-011b | head-name coverage refusal | PASS |
| TEST-C2-011c | tampered format rejected by validate() | PASS |

## Verdict

B3 verified on preregistered sources only; leakage guards PASS; activations
installed, weights checked against the engine, refusals enforced, fallback
uniform. Train/val/test isolation and honest-status rules hold.