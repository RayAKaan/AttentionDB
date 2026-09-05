# Gating Findings (detail for F1–F4, F6, F7)

## What the gate actually learned

Controlled (PH2B-GATING-004): average weights [0.267, 0.244, 0.355, 0.133]
look near-uniform, but per-query the distribution is one-hot (entropy
0.003): the argmax follows the query's group (selection frequency
[0.267, 0.244, 0.356, 0.133] = the group distribution). Temperature
calibration (val-fitted T) sharpens the softmax; the PRE-temperature model
already ranked heads correctly (corr 0.903) — calibration changes the
FUSION mix, not the ranking. Quality correlation: predicted weight vs
head recall r = 0.903 (pooled test examples).

Noise (PH2B-NOISE-003): average weight on head0 0.894; per-query selection
100% head0. The gate recovers "head 0 is always best" from data — a
sensible local optimum indistinguishable from per-query oracle quality
(0.8489 vs 0.8400; the +0.9pp edge over oracle comes from residual
non-zero weights on boundary queries, INTERPRETATION).

Multiview (PH2B-MULTIVIEW-005): average weights [0.334, 0.465, 0.201],
selection frequency [0.311, 0.489, 0.200] across test queries vs true
1/3–1/3–1/3 — all three views are selected, lexical skew; pooled corr 0.269
(positive but weak: the fine/coarse contrast the model must read is subtle
in a 192-dim concatenation; INTERPRETATION). R@10 0.5322 single-run,
0.4365 ± 0.0361 over seeds — well above every non-oracle baseline, well
below oracle.

## Loss-function comparison (validation-selected)

At the final protocol, validation R@10 of the temperature-calibrated model
was effectively tied across objectives on controlled (0.9556 selected:
soft_target h64 / pairwise h32 / pairwise h64 all reach it) — on a task
this learnable, objective choice does not matter. On multiview at 300
queries, pairwise had a visible edge (val R@10@T 0.289 vs 0.213/0.216);
at 1200 queries soft_target h64 and pairwise h64 converged (0.5689/
0.5783 val). No claim of a universally best objective is made.

## Sample efficiency (PH2B-SAMPLE-001, multiview)

train 105 → test R@10 0.2565 · 210 → 0.2422 · 420 → 0.2533 · 840 → 0.4928.
A plateau (≈ uniform+ε: uniform is 0.2128 on the 1200-point corpus) and a
transition between 420 and 840 training queries. Two independent 840-train
runs landed at 0.4928 and 0.5322 (fresh datasets each time) — treat
multiview gating as "learns at ≈840 queries" not as a point value.

## Calibration behavior

On learnable tasks validation always picks sharpening temperatures
(0.25–0.5). On a failed model (multiview 300q) it picks T = 4 — the
safest near-uniform fallback. Calibration error in the probability sense
was not studied; per §19 we only need useful ORDERING, which corr
measures.
