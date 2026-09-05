# Phase 3 negative results (preserved per §36)

1. **Memory scaling wall** [PH3-QUAL-FM-T30, PH3-QUAL-FM-M]: index build
   OOM-kills between 10K and 30K docs (5 heads × 196-d, M=16, store_vectors)
   on a 1984 MB sandbox. 10K-doc peak = 571 MB (≈8.6× raw vector bytes).
   Consequence: Tier M/L quality evaluation is NOT possible on this
   hardware in this configuration; scaling claims are limited accordingly.
2. **Gating adds nothing over single-head on PH3-DS-FM** [PH3-QUAL-FM-S]:
   the corpus's full-image view uniformly dominates the quadrant views;
   trained gating collapses to it (w(full)=1.0). Honest reading: on
   single-modality corpora with one dominant view, the multi-head path pays
   3.3× latency for zero quality gain (F-P3-3). Head-complementarity
   corpora are required to demonstrate gating's value in Phase 3 — pending
   dataset family, not a failure of the architecture.
3. **Uniform multi-head fusion is harmful on PH3-DS-FM** (0.629 vs 0.986):
   dominated views act as noise under equal weighting. Consistent with the
   Phase 2 controlled-corpus result (uniform < best head).
