# Table: complementary multi-view retrieval — PH3B-COMP-001 (AG News 10K, TEST n=61)

Caption draft: *Query-dependent gating recovers 44% of the oracle-head gap on real multi-field text where no view dominates; uniform fusion does not beat the best single view; exact reference anchors the harness at 1.0.* [PH3B-COMP-001]

| System | R@1 | R@5 | R@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Global-best single view (val-selected) | 0.0557 | 0.2344 | 0.4033 | 0.4397 | 0.6080 |
| Uniform multi-view | 0.0721 | 0.2525 | 0.3885 | 0.4569 | 0.8025 |
| **Trained gating (frozen arch.)** | 0.0781 | 0.3525 | 0.6448 | 0.6752 | 0.8189 |
| Oracle head (empirical, ceiling) | 0.1000 | 0.5000 | 0.9672 | 0.9790 | 1.0000 |
| Exact reference (anchor) | 0.1000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 |

Gating seed stability: 0.6448 ± 0.0025 (seeds 42/7/1). Gap recovered = (gating−uniform)/(oracle−uniform) = 0.443.
