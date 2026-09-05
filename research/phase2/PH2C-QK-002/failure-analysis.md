# Failure analysis: why QK lost here after winning the sanity set

1. **Not a training artifact** (§14 checks all pass): gradients flowed
   (‖ΔW‖=106.2), loss decreased, scores non-degenerate (std 8.26), untrained
   0.0339 → trained 0.1113.
2. **Not pool-limited**: 0.9975 GT-in-union. Purely an ordering failure.
3. **Harmful reordering**: τ(QK, uniform fusion) = 0.459 — QK substantially
   rewrites the ordering, and the rewrite is worse than the baseline it
   replaces (R@10 0.1113 < uniform 0.2128).
4. **Corpus structure**: multiview relevance is defined by the query's
   VIEW; per-head selection + cosine captures it (gating → 0.4983,
   oracle-view structure). Query–candidate bilinear interaction adds no
   separable signal in these pools — the capability demonstrated on the
   synthetic anti-cosine set has no corresponding structure to exploit here.
5. **Blend damages**: gating+QK (0.2769) < gating — mixing a worse ranking
   into a better one via RRF degrades it; QK is not a complementary signal.
6. **Group-uniform loss**: QK loses in every query group (by_query_type.csv)
   — no niche where candidate-level interaction pays.
