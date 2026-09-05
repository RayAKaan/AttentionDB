# Table: sample efficiency (multiview)

Caption draft: *Test R@10 as training-set size grows. Each point regenerates the corpus (HNSW candidate pools are OS-seeded), so points are directional, not paired. The transition lies between 420 and 840 training queries.* [PH2B-SAMPLE-001]


| train | val | test | uniform R@10 | gating R@10 | gating NDCG@10 | oracle R@10 |
|---|---|---|---|---|---|---|
| 105 | 22 | 23 | 0.2391 | 0.2565 | 0.3134 | 0.9957 |
| 210 | 45 | 45 | 0.1978 | 0.2422 | 0.2993 | 0.9933 |
| 420 | 90 | 90 | 0.1944 | 0.2533 | 0.3002 | 0.9944 |
| 840 | 180 | 180 | 0.2106 | 0.4928 | 0.5223 | 0.9861 |
