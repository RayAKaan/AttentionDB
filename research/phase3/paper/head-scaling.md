# Paper section: head-count scaling (PH3C)

## Setup
AG News multi-field ladder (identical corpus/queries/GT; only the engine
head set varies): 1 (full) / 2 (title,body) / 4 (+body halves) / 8
(+quarters, cross views), size-matched at 5K docs, seeds 42/7/1, K=100
pools; 10K points for 1–4 heads as scale extras; 8×10K quality run OOM
(preserved). Caveat: at 4/8 heads, view granularity co-varies with head
count by construction (sub-field views).

## Findings (canonical: results/head-scaling-*.csv; figures 4–6)
- Quality: gating 0.519 → 0.658 → 0.715 → 0.709 (1/2/4/8 heads):
  large gains to 4 heads, saturation at 8 (within seed noise ±0.023).
  Uniform fusion never beats the best single view at any head count;
  gating beats both static baselines everywhere except degenerate H1.
- Latency (serial p50): 1.7 / 4.2 / 6.7 / 22.1 ms — super-linear at 8
  heads (pool fusion over 8×200 candidates dominates). Parallel-2:
  ≈ 2× at ≥2 heads on this 2-CPU sandbox; HURTS at 1 head (overhead).
- Memory: 178 / 269 / 464 / 873 MB peak — linear in heads.
- Practical region (measured, not labeled "best"): 2–4 heads on this
  sandbox; 8 heads trades linear memory and super-linear latency for no
  measured quality gain.
