# Table: PH2C-QK-001 — QK sanity dataset (held-out test, 250 queries)

Caption draft: *Candidate-level QK attention vs the entire gating class on a dataset where ordering provably requires query–candidate interaction (head selection easy; relevant candidate last under the signal-head cosine by construction). Mean ± std over seeds 42/7/1.* [PH2C-QK-001]

| arm | R@1 | NDCG@10 | MRR |
|---|---|---|---|
| uniform | 0.0000 | 0.2914 | 0.1022 |
| global_best_head0 | 0.0000 | 0.2891 | 0.1000 |
| gating_qualityreg | 0.0027 ± 0.0038 | 0.3183 | 0.1306 |
| gating_infnce | 0.1000 ± 0.0000 | 0.4528 | 0.2909 |
| qk_untrained | 0.1160 | 0.4497 | 0.2890 |
| qk_trained | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |
| oracle | 1.0000 | 1.0000 | 1.0000 |

R@5/R@10 are 1.0 for every arm by construction (single relevant candidate per 10-candidate pool): the dataset isolates ORDERING. Source: results/qk-sanity.csv.
