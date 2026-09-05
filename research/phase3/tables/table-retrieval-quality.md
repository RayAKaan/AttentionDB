# Table: retrieval quality — PH3-DS-FM-S (Fashion-MNIST 10K, TEST, agg seeds 42/7/1)

Caption draft: *Frozen architecture vs baselines on real-image multi-view retrieval; exact reference = ground-truth definition (sanity anchor). Trained gating collapses to the dominant full-image view and matches single-head ANN; uniform fusion degrades.* [PH3-QUAL-FM-S]

| System | R@1 | R@5 | R@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Single-vector ANN (full view) | 0.1000 | 0.5000 | 0.9860 | 0.9911 | 1.0000 |
| Uniform multi-head fusion | 0.0927 | 0.3927 | 0.6293 | 0.6991 | 0.9622 |
| Trained gating (frozen arch.) | 0.1000 | 0.4998 | 0.9853 | 0.9906 | 1.0000 |
| Exact/brute-force reference | 0.1000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 |

Candidate recall (5-head pool union): 0.9992 — ranking, not generation, is the residual gap to 1.0.
