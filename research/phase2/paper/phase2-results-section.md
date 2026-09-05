# Results (paper section draft) — measured only (§17)

All values are held-out test-set measurements; experiment IDs in brackets
trace to `raw/experiment-index.json`. Single-run values are marked (1);
multi-seed values are mean ± std over training seeds 42/7/1 with a fixed
split.

### 4.1 Experimental Setup

Three synthetic corpora (controlled: 384 docs, 4 heads; noise ladder:
10 000 docs, 8 heads; multiview: 3 000 docs, 3 views) with ground truth
defined by exact cosine over data-generating vectors (§ground-truth).
Candidates: per-head HNSW top-100. Metrics: Recall@{1,5,10,50}, NDCG@10,
MRR. Protocol: 70/15/15 split; objective/hyperparameter selection and
temperature calibration on validation only; one final test evaluation.

### 4.2 Baselines

| approach | controlled | noise | multiview |
|---|---|---|---|
| uniform multi-head R@10 | 0.6244 | 0.7378 | 0.2128 |
| global best single head R@10 | 0.5378 | 0.8378 | 0.3428 |
| RRF (k=60) R@10 | 0.6689 | 0.7800 | 0.2622 |

On the controlled corpus, uniform fusion already exceeds the global best
single head (union effect); on the noise corpus it does not (0.7378 vs
0.8378); on multiview the global best view dominates both.
[PH2B-GATING-004, PH2B-NOISE-003, PH2B-MULTIVIEW-005]

### 4.3 Learned Gating

Trained gating achieved R@10 = 0.9533 on controlled, identical to the
per-query oracle (0.9533) and to NDCG@10 (0.9700 both), recovering 100%
of the uniform→oracle gap; across training seeds it measured
0.9556 ± 0.0000 [PH2B-GATING-004, PH2B-MULTISEED-001]. On the noise
ladder the trained gate concentrated on head 0 (mean weight 0.894,
selection frequency 1.000) and achieved 0.8489, marginally above both the
global best single head (0.8378) and the per-query oracle (0.8400)
[PH2B-NOISE-003]. Predicted weights correlated with per-head recall at
Pearson r = 0.903 (controlled) and r = 0.609 (noise).

### 4.4 Multiview Retrieval

On the multiview corpus, trained gating achieved R@10 = 0.5322 versus
0.2128 for uniform fusion, 0.2622 for RRF, and 0.3428 for the global best
view (single run) [PH2B-MULTIVIEW-005]. Across seeds: 0.4365 ± 0.0361
[PH2B-MULTISEED-002] — still above every non-oracle baseline, but
recovering only 28–41% of the uniform→oracle gap (oracle 0.9933). Per
query type, all three types benefited (by-query-type.csv); selection
frequency across views was [0.311, 0.489, 0.200] against a uniform ⅓–⅓–⅓
ground-truth distribution.

### 4.5 Exact Reranking

The Phase 2 pipeline's equal-weight exact rerank measured 0.558 R@10
against 0.672 for the same pipeline without it [PH2-ABLATION-001]. The
offline weighting study on cached candidates attributed the regression to
head weighting rather than to exact scoring: uniform exact fusion measured
0.6511 vs 0.6244 (normalized, uniform) on controlled, 0.7133 vs 0.7378 on
noise, and oracle-weighted exact fusion attained 0.9533 / 0.8400 / 0.9933
— the oracle bound on all three corpora [PH2C-RERANK-001/002/003]. The
pipeline-level correction has not yet been implemented or evaluated.

### 4.6 QK Attention

No trained QK attention results exist. The identity-initialized scorer was
measured indistinguishable from untrained gating (R@10 0.672 both
[PH2-ABLATION-001]); this supports only the conclusion that the
untrained mechanism adds nothing. Trained-QK evaluation is future work
(`results/qk-attention.csv` is intentionally empty).

### 4.7 Sample Efficiency

Varying multiview training size (105/210/420/840) produced test R@10
0.2565 / 0.2422 / 0.2533 / 0.4928 [PH2B-SAMPLE-001]: a plateau through
420 followed by a transition at 840. At fixed data, the architecture grid
(hidden 32→64, lr 0.01→0.003, batch 32→16) changed validation R@10 by
less than 2 percentage points; a 4× increase in training data changed
test R@10 by more than 24 points.

### 4.8 Latency and Scalability

The gating network measures 0.87 µs/query on CPU (1188 parameters, 20 288
bytes serialized) [PH2B-LATENCY-001]. For reference, end-to-end pipeline
latency on the Phase 2 corpus (100 queries, one machine, release build)
was p50 186 µs (single head) to 1892 µs (8 heads, parallel), with exact
rerank adding ~180–300 µs [PH2-ABLATION-001]; parallel head execution
preserved recall exactly at 1/2/4/8 heads with 1.39×/1.72× p50 speedup at
4/8 heads. Absolute latencies are machine-specific.

### 4.9 Ablation Summary

Within the frozen Phase 2 protocol: single best head 0.763 > multi-head
fixed 0.672 = untrained gating 0.672 = identity-QK 0.672 > exact rerank
0.558; oracle head selection was not measured in that protocol. In Phase
2B (properly defined ground truth): trained gating 0.9533 (controlled) =
oracle; trained gating 0.8489 ≈ oracle 0.8400 (noise); trained gating
0.5322 < oracle 0.9933 (multiview). The two phases' absolute numbers are
not comparable (different ground-truth constructions; see
harness-corrections.md HC-4).
