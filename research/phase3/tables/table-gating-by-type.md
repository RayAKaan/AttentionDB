# Table: retrieval by query type — PH3B-COMP-001

Caption draft: *Per-query-type breakdown. The defining head is near-perfect by construction (oracle ≈ 0.97); gating must infer the query type from per-head query vectors. Gating beats both static baselines on every type but leaves headroom on title queries.* [PH3B-COMP-001]

| Arm | Query type | n | R@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| global_best_single | title | 22 | 0.0364 | 0.0481 | 0.1369 |
| global_best_single | body | 18 | 0.9667 | 0.9786 | 1.0000 |
| global_best_single | mixed | 21 | 0.3048 | 0.3881 | 0.7656 |
| uniform_multihead | title | 22 | 0.2864 | 0.3088 | 0.6013 |
| uniform_multihead | body | 18 | 0.4000 | 0.5011 | 0.9028 |
| uniform_multihead | mixed | 21 | 0.4857 | 0.5741 | 0.9274 |
| trained_gating (s42) | title | 22 | 0.5409 | 0.5494 | 0.5921 |
| trained_gating (s42) | body | 18 | 0.7056 | 0.7350 | 0.9167 |
| trained_gating (s42) | mixed | 21 | 0.6952 | 0.7331 | 0.9076 |
| trained_gating | title | 22 | 0.6000 | 0.6101 | 0.6580 |
| trained_gating | body | 18 | 0.6889 | 0.7288 | 0.9389 |
| trained_gating | mixed | 21 | 0.6540 | 0.6976 | 0.8844 |
| bm25 | title | 22 | 0.0955 | 0.1251 | 0.2870 |
| bm25 | body | 18 | 0.1944 | 0.2575 | 0.5805 |
| bm25 | mixed | 21 | 0.2524 | 0.3002 | 0.5186 |
| hybrid_rrf_k60 | title | 22 | 0.1136 | 0.1515 | 0.3823 |
| hybrid_rrf_k60 | body | 18 | 0.3389 | 0.3854 | 0.6380 |
| hybrid_rrf_k60 | mixed | 21 | 0.5000 | 0.5539 | 0.8210 |
| hybrid_engine | title | 22 | 0.1091 | 0.1464 | 0.3633 |
| hybrid_engine | body | 18 | 0.3333 | 0.3907 | 0.6713 |
| hybrid_engine | mixed | 21 | 0.5286 | 0.5882 | 0.8508 |
| oracle_head_empirical | title | 22 | 0.9727 | 0.9825 | 1.0000 |
| oracle_head_empirical | body | 18 | 0.9667 | 0.9786 | 1.0000 |
| oracle_head_empirical | mixed | 21 | 0.9619 | 0.9755 | 1.0000 |
| oracle_head_group | title | 22 | 0.9727 | 0.9825 | 1.0000 |
| oracle_head_group | body | 18 | 0.9667 | 0.9786 | 1.0000 |
| oracle_head_group | mixed | 21 | 0.9619 | 0.9755 | 1.0000 |
| exact_reference | title | 22 | 1.0000 | 1.0000 | 1.0000 |
| exact_reference | body | 18 | 1.0000 | 1.0000 | 1.0000 |
| exact_reference | mixed | 21 | 1.0000 | 1.0000 | 1.0000 |
