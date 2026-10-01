# C8 Dataset Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`

## 1. Datasets (frozen splits; identical to C7)

| Dataset | Role | qids | Notes |
|---|---|---|---|
| SciFact | primary | 300 TEST | 3 heads, 384-d canonical |
| NFCorpus | primary | 323 TEST | 3 heads, 384-d canonical |

TRAIN/VALIDATION splits are disjoint from TEST (leakage gate #5 enforced in
`c8_verify.py`: TUNE row sets ∩ TEST row sets = ∅).

## 2. Training-example hashes (TUNE, VALIDATION)

Arm hashing is content-based over the mined training examples, so arms that share a
training-example design share a hash, and evidence/disagreement arms differ:

| Dataset | Arm(s) | dataset_hash |
|---|---|---|
| SciFact | D, E, H | `8141147154856472173` |
| SciFact | F, G | `14864224517566290257` |
| NFCorpus | D, E, H | `17744417282313574642` |
| NFCorpus | F, G | `3686909789714422379` |

D/E/H share a hash (same example construction; D learns the unrestricted projection,
E/H the residual, H adds a KL-distillation loss), while F/G add retrieval-evidence
features, changing the example content and thus the hash. This is the expected
sensitivity and confirms the evidence arms are not silently identical to E.

## 3. Example counts

| Dataset | examples | epochs × batch | optimizer steps |
|---|---|---|---|
| SciFact | 192 | 5 × 8 | 120 |
| NFCorpus | 303 | 5 × 8 | 190 |

Negative mining produces this many `(query, candidate)` contrastive examples from the
VALIDATION split's candidate traces; the exact counts replay deterministically from
the frozen seed (report 9).
