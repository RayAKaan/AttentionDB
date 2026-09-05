# Ground Truth Definition (§8)

## Formal definition

For corpus C with data-generating (true) document vectors {t_d} and query q:

    ground_truth(q) = top-10 argsort_d cos(t_d, q)   (descending)

with ties broken by ascending document index. Relevance for a query is
binary over exactly these 10 ids (Recall@k counts hits / 10; NDCG uses rank
position with the standard log2 discount and IDCG over the same 10).

Per-view corpora (multiview): the query declares ONE modality V; ground
truth uses the view-V document vectors only:

    ground_truth(q of type V) = top-10 argsort_d cos(v_d^{(V)}, q^{(V)})

## Why true (generator) vectors

Heads observe noisy copies of the documents; ranking against a noisy copy
would measure agreement with a particular noise draw. The generator vector
is the ground-truth object the corpus defines. This is only possible in
synthetic corpora — stated as a limitation, not hidden (see
`paper/phase2-limitations.md`).

## Why the original same-centroid construction was invalid

The first Phase 2 harness assigned every doc in a cluster the SAME vector
(the centroid). All ~100 same-cluster docs then tied exactly under cosine,
so "top-10" was decided by the engine's insertion/tie-break order — an
index lottery. Measured consequence: every method scored recall ≈ 0.10
(= expected overlap of two independent random 10-subsets of a 100-doc tie
pool). The benchmark was measuring RNG agreement, not retrieval. Detected
by the sanity probe `benchmarks/phase2/src/bin/probe.rs`: retrieval itself
was healthy (99/100 same-cluster hits) while scored recall was chance —
proving the defect was in ground truth, not the engine.

## The corrected construction

Each doc's true vector is normalize(centroid + 0.35·u_d) with u_d a
distinct unit Gaussian direction. Intra-cluster cosine to the query now
varies continuously (centroid alignment + unique-direction alignment), so
the top-10 is a well-defined, noise-stable set. Inter-cluster separation is
preserved (0.35·u on a unit sphere does not collapse cluster structure).

## Ground truth in ENGINE ids

A second, subtler failure: ground truth was first recorded in corpus hint
ids (0-based), while the engine's numeric ids start at 1
(`IdMapper` INV-6). Every recall computation silently compared shifted id
sets. Detected when the noise corpus produced R@10 = 0.000 for EVERY
approach INCLUDING the oracle — an impossible pattern that flags harness
failure, not model failure. Correction: `generate_dataset` now maps
hint → engine id through the document store + id mapper and recomputes all
stored per-head quality metrics against the mapped ids.

Full record: `findings/harness-corrections.md`.
