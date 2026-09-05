# Discussion (paper section draft) (§18)

Labels: [MEASURED] = number exists in raw results; [INTERPRETATION] = our
reading of measured results; [HYPOTHESIS] = testable, untested.

1. **Why gating succeeds on controlled data.** The query representation
   contains a block-structured signal that linearly identifies the query's
   group, and head quality is a deterministic function of that group.
   A linear-softmax gate over such a representation is logistically
   sufficient; the measured one-hot post-calibration distributions and
   r = 0.903 weight–quality correlation are consistent with this reading.
   [MEASURED: PH2B-GATING-004] [INTERPRETATION: the mechanism explanation]

2. **Why gating collapses on the noise ladder.** Head quality is constant
   across queries (global σ ladder), so the Bayes-optimal gate is
   constant. The trained gate's collapse (mean weight 0.894, selection
   100%) is the correct solution, and its quality (0.8489) matches both
   the global best head (0.8378) and the oracle (0.8400) — the oracle
   itself has almost nothing to add. [MEASURED] Forcing selection
   diversity here would have been fitting noise (§7 of the spec).

3. **Why multiview is only partial.** The gate must distinguish
   "this query carries fine structure in view V" from "coarse topic
   prototype" across a 192-dim concatenation; pooled correlation is weak
   (r = 0.269) and per-query routing imperfect ([0.311, 0.489, 0.200]).
   [MEASURED] Whether the limitation is representational (concatenation
   vs view-specific encoders) or statistical (sample efficiency, F6) is
   unresolved. [HYPOTHESIS: view-specific encoders would sharpen the
   signal]

4. **Why training-data quantity matters more than architecture.** At
   ≤ 420 training queries the gate stayed at the uniform solution; at 840
   it learned decisive routing, while capacity/lr/batch changes moved
   validation R@10 < 2pp. [MEASURED: PH2B-SAMPLE-001] [INTERPRETATION:
   the uniform solution is a flat region the optimizer must escape via
   gradient signal that only accumulates with data; we have not measured
   gradient magnitudes, so this remains interpretation.]

5. **Why head diversity does not imply utility.** The noise corpus's six
   weak heads are diverse (distinct noise draws) but individually
   near-useless; the trained gate discounts them to ~0. [MEASURED]
   Diversity raises the value of the UNION only when at least one head is
   locally best — the controlled corpus shows that case (uniform union
   beats every single head there). [MEASURED]

6. **Why exact reranking regressed.** The offline decomposition shows the
   regression direction appears only when unequal heads are weighted
   equally over exact scores (noise: 0.7133 exact-uniform < 0.7378
   norm-uniform), while oracle-weighted exact fusion attains the oracle
   bound everywhere. [MEASURED: PH2C-RERANK-*] Attribution to the exact
   pipeline arithmetic is partial: mode E mixes exact scores with the
   α/β/γ fusion, so this is attribution, not exact replication. [stated]

7. **Does QK attention add value?** Unknown. The only measured QK point
   is the identity initialization (a designed no-op). We decline to
   extrapolate from it in either direction. [F8]

8. **Latency tradeoffs.** The gate costs 0.87 µs/query (1188 params);
   HNSW per-head search costs hundreds of µs and grows with head count.
   [MEASURED] The design conclusion is that gating is effectively free
   relative to candidate generation; attention training will be judged
   against the same budget. [INTERPRETATION]

9. **Candidate-generation limitations.** HNSW layer RNG is OS-seeded, so
   candidate pools vary between processes (two 840-train runs differed by
   0.04 R@10 with identical code and corpus seed). [MEASURED] Cached
   datasets are therefore the reproducibility unit. Absolute recall also
   depends on ef_search 64; no sweep was performed in this phase.

10. **Generalization limits.** All corpora are synthetic; ground truth is
   computable only because generator vectors exist. Nothing here measures
   real-world distribution shift, adversarial queries, or drift after
   ingestion. Claims are scoped to the evaluated corpora throughout.
