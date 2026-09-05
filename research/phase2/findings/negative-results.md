# Negative Results (§23) — preserved, not deleted

1. **Fixed fusion loses to the best single head** (Phase 2, frozen):
   8-head fixed fusion R@10 0.672 vs single best head 0.763 at ~9× p50
   latency. Multi-head union is robustness, not accuracy. (`results/ablation.csv`)

2. **Untrained gating matches fixed fusion** (Phase 2): C = B = 0.672 to
   three decimals. An untrained gate is uniform weighting with extra
   latency.

3. **Identity-initialized QK attention matches untrained gating** (Phase 2):
   D = C = 0.672. Identity init was designed to be a no-op; it was.

4. **Equal-head exact rerank regressed** (Phase 2): mode E 0.558 < D 0.672.
   Investigated in Phase 2C (PH2C-RERANK-*): exact scores are not the
   problem; equal weighting of unequal heads is. Pipeline-level fix still
   unvalidated (ledger N3).

5. **Initial multiview design failed twice**:
   a. Tie-lottery GT (PH2B-MULTIVIEW-001): 75 same-topic docs tied exactly;
      oracle itself scored 0.26 — meaningless.
   b. Full-fidelity queries in all views (PH2B-MULTIVIEW-002): the best
      view became statistically undetectable; the model correctly refused
      to learn (corr −0.34 with uniform weights).

6. **Ground-truth id-mapping bug** (PH2B-GATING-001/002, PH2B-NOISE-001):
   hint-id vs engine-id shift produced recall ≈ chance (controlled) and an
   impossible all-zeros table (noise, oracle included). Detected BECAUSE
   the oracle scored zero — a useful diagnostic pattern.

7. **Gating failed to learn at 210 training queries** (PH2B-MULTIVIEW-003,
   004): weights pinned at uniform, corr ≈ 0 or negative. Preserved as the
   before-picture for F6.

8. **Seed sensitivity on multiview** (PH2B-MULTISEED-002): R@10 varies
   0.394–0.482 across training seeds; the single-run 0.5322 was optimistic.
   Reported in the ledger rather than quoting only the best run (§12).

9. **Hyperparameter grid was nearly useless** (vs data quantity): hidden/
   lr/batch changes moved validation R@10 < 2pp at fixed data size, while
   ×4 data moved test R@10 +25–30pp. Recorded to prevent over-claiming
   architecture contributions.
