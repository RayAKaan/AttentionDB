# Table: query-path latency — PH3B-COMP-001 (warm, release, 2-CPU sandbox, tmpfs engine dir)

Caption draft: *Stage decomposition of the frozen path on text. Per-head ANN dominates; the learned gating forward pass stays in the ~1–4% range of the path (105 µs p50 vs ~3.1 ms full-head ANN), replicating the Phase 3 image-corpus cost structure.* [PH3B-COMP-001]

| Stage | p50 µs | p95 µs | p99 µs | QPS |
|---|---|---|---|---|
| ann_head_body | 3002.9 | 3523.5 | 3825.9 | — |
| ann_head_full | 3067.6 | 3568.0 | 3766.3 | — |
| ann_head_title | 3000.1 | 3492.6 | 3628.0 | — |
| bm25_query | 413.8 | 789.0 | 912.5 | — |
| exact_bruteforce | 6969.8 | 7165.3 | 7216.4 | — |
| fusion_rank | 39.2 | 58.8 | 73.3 | — |
| gating_forward | 104.8 | 136.5 | 148.6 | — |
| gating_path_total | 9231.3 | 10287.8 | 11060.8 | 109 |
| hybrid_path_total | 3441.9 | 4075.9 | 4411.5 | 290 |
| rrf_combine | 25.0 | 48.5 | 52.3 | — |
