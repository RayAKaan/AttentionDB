# Table: final comparison (§29 central table)

Caption draft: *Retrieval quality of fixed multi-head fusion, learned query-dependent gating, RRF, and oracle head selection across the controlled, noise, and multiview corpora. Values are measured on held-out test queries (single-run protocol; multiview multi-seed mean 0.4365 ± 0.0361 — see table-multiseed).* [PH2B-GATING-004, PH2B-NOISE-003, PH2B-MULTIVIEW-005]


| Method | Controlled R@10 | Noise R@10 | Multiview R@10 | Controlled NDCG@10 | Noise NDCG@10 | Multiview NDCG@10 |
|---|---|---|---|---|---|---|
| Single best head | 0.5356 | 0.8378 | 0.3428 | 0.5664 | 0.8845 | 0.3521 |
| Uniform multi-head | 0.6244 | 0.7378 | 0.2128 | 0.6881 | 0.8028 | 0.2803 |
| RRF (k=60) | 0.6689 | 0.7800 | 0.2622 | 0.7294 | 0.8385 | 0.3114 |
| Trained gating | 0.9533 | 0.8489 | 0.5322 | 0.9700 | 0.8942 | 0.5742 |
| Trained QK | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) |
| Gating + QK | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) |
| Exact rerank (offline exact fusion, uniform)* | 0.6511 | 0.7133 | 0.2406 | 0.7129 | 0.7843 | 0.3445 |
| Gating + exact rerank* | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) | pending (Phase 2C) |
| Oracle head selection | 0.9533 | 0.8400 | 0.9933 | 0.9700 | 0.8858 | 0.9957 |

\* The offline exact-fusion row is the PH2C-RERANK study (uniform weighting over cached candidates), not the Phase 2 pipeline mode E; the frozen Phase 2 pipeline mode E measured 0.558 vs mode D 0.672 on its own (non-comparable) ground truth [PH2-ABLATION-001]. Pipeline-level re-weighting is untested (ledger N3).
