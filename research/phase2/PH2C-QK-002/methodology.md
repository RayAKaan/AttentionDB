# Methodology (PH2C-QK-002)

## Protocol
Paired evaluation on the canonical Phase 2B multiview cache
(dataset sha 6f41ab7b…, 1200 queries, 840/180/180, 3 heads × 64 dims,
per-head pools of 100). Every arm sees IDENTICAL pools/splits/GT (§6).
Content sidecar (qk-cache) passed ALL binding checks: fresh engine-id
mapping reproduces cached GT 1200/1200; cached exact scores reproduce from
cached-query × mapped-doc cosines (14,400 checked); query vectors identical.
controlled/noise legs REFUSED pre-training (HC-6: rebuilt doc content does
not correspond to frozen pools).

## Arms
uniform · global-best (train-split selection) · trained gating (shipped 2B
protocol: objective×lr×hidden grid, val-only selection, val temperature)
· trained QK (PH2C-QK-001-validated: Q=W_Q·q, K=W_K·x, s=Q·K/√8,
multi-positive InfoNCE at T; lr×T grid val-only, seed 42, frozen) ·
gating+QK (RRF k=60 of the two rankings — no new learned machinery) ·
RRF k=60 · oracle (per-query best head by cached head-recall; analysis-only).
Seeds 42/7/1; test evaluated once per seed after freezing.

## QK objective (exact)
p = softmax(s/T) over the candidate union; L = −ln Σ_{i∈GT∩pool} p_i;
dL/ds_j = p_j − [j∈GT]·p_j/Z. Multi-positive generalization of the
sanity single-positive InfoNCE.
