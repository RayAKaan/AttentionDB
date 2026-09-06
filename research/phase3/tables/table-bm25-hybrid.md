# Table: sparse, dense, hybrid, learned — PH3B-COMP-001

Caption draft: *Against this benchmark's semantic-space ground truth, BM25 and BM25-hybrid trail the dense views; the trained gate outperforms every static combination. BM25 remains a verified-correct channel (bm25_verify.json) — the gap is a property of the relevance definition, reported without tuning.* [PH3B-COMP-001, PH3B-BM25-001]

| System | R@10 | NDCG@10 | MRR |
|---|---|---|---|
| Best single view (dense) | 0.4033 | 0.4397 | 0.6080 |
| BM25 (sparse) | 0.1787 | 0.2245 | 0.4533 |
| Hybrid RRF(BM25+full) k=60 | 0.3131 | 0.3590 | 0.6088 |
| Hybrid (engine channel) | 0.3197 | 0.3706 | 0.6220 |
| Trained gating (frozen arch.) | 0.6448 | 0.6752 | 0.8189 |
