# Methodology (paper section draft) — Phase 2/2B

Notation: query q; heads h ∈ {1..H}; candidate c. Every equation below is
transcribed from the implementation (file/function cited); nothing is
idealized.

## 9.1 Multi-head candidate generation

Each head h is an independent HNSW index [CITE: HNSW] over that view's
vectors (ef_search 64, ef_construction 400, M 16, cosine metric). A query
for head h returns the ranked candidate list

  C_h(q) = top-K_h HNSW_h(q),  K_h = 100 in all Phase 2B datasets
           (Phase 2 pipeline: K_h = max(k·m, min_per_head), m = 5, min 20)

with raw similarity scores s_h(c) ∈ [-1, 1] (cosine).

## 9.2 Candidate union

  C(q) = ⋃_h C_h(q)

with first-seen order and per-head provenance retained. The pipeline
variant clamps |C(q)| to a budget by best head score (ties: smaller id);
the Phase 2B offline studies union the cached pools without a budget.

## 9.3 Head scoring (normalized per-head score)

Per head, scores are min–max normalized over C_h(q):

  s̄_h(c) = (s_h(c) − min_h) / (max_h − min_h)  if max_h − min_h ≥ 1e-6
  s̄_h(c) = 0                                    otherwise

(`core/src/retrieval.rs::normalize_scores`, MinMax path). A candidate
absent from head h contributes 0 — there is no renormalization over
present heads, so being perfect in one channel cannot tie with being
perfect in all channels.

## 9.4 Learned gating

The gate is a one-hidden-layer MLP over the query representation q̃ (the
query vector, or the concatenation of per-view query vectors):

  g(q) = softmax( W₂ · ReLU(W₁ q̃ + b₁) + b₂ )  ∈ Δ^{H−1}

(`attentiondb_learned::gating_v2::GatingMlp::predict`). W₁ ∈
R^{32×dim} (or 64×dim), W₂ ∈ R^{H×hidden}. Seeded Gaussian init
(σ = 1/√fan_in), trained with Adam (lr 0.01 or 0.003, batch 16 or 32, L2
1e-4, early stopping on validation loss) against per-head Recall@10
targets, under three objectives: quality regression (MSE on logits),
soft-target cross-entropy (targets softmax(Q_h/τ), τ = 0.1), pairwise
logistic on quality gaps ≥ 0.05. At inference the logits may be
temperature-scaled with a validation-fitted T:

  g^T(q) = softmax( logits(q) / T ),  T* = argmax_T val-R@10.

## 9.5 Final head fusion (exact implemented equation)

The pipeline's multi-head-similarity component is the gated weighted sum

  mhs(c|q) = Σ_h  g_h(q) · s̄_h(c)        (absent h ⇒ term 0, no renorm)

and the final candidate score fuses three components with validated
weights α + β + γ (α = attention, β = multi-head similarity, γ = BM25;
defaults 0.3/0.5/0.2), each renormalized over the components present:

  S(c|q) = (α·a(c|q) + β·mhs(c|q) + γ·b(c|q)) / (α·[a present] + β + γ·[b present])

(`core/src/retrieval.rs::fuse_candidate`). Ordering is score descending,
engine-id ascending for ties (deterministic top-k).

## 9.6 Candidate-level QK attention

The implemented scorer (`AttentionScorer::score`) is NOT softmax
attention; it is a tanh-bounded bilinear map. With the gate profile p =
g(q) as the "query" object and per-candidate features
x_c = [ s̄_1(c)..s̄_H(c), r_1(c)..r_H(c) ] (normalized scores then rank
features):

  Q = W_Q p,  K_c = W_K x_c,  W_Q ∈ R^{d×H}, W_K ∈ R^{d×2H}
  a(c|q) = ½( tanh( Q·K_c / √d ) + 1 )   ∈ (0, 1)

(non-finite logits → 0.5, i.e. neutral). Identity initialization (W_Q, W_K
selecting the score block, rank features weighted 0) makes mode D exactly
mode C — measured, not assumed. We report this precisely because the term
"attention" commonly implies softmax(QK/√d) [CITE: attention]; our
candidate-level mechanism is a learned bilinear gate, and the paper will
not call it transformer attention.

## 9.7 Exact reranking

Mode E replaces normalized scores with exact similarities under the same
gated-sum rule, per candidate:

  mhs_exact(c|q) = Σ_{h: c ∈ C_h} g_h(q) · cos(v_h(c), q)
  mhs(c|q)       = mhs_exact(c|q)  if ∃h with c ∈ C_h, else mhs(c|q) as §9.5

(`core/src/collection.rs`, "Exact rerank (MODE E)" block; the fallback
preserves candidates that no head scored exactly).

## 9.8 RRF baseline

  S_rrf(c|q) = Σ_h  1 / (60 + rank_h(c))

with rank_h the 0-based position of c in C_h(q) [CITE: RRF]. Identical
candidates as the fusion arms; ties by smaller id.

## 9.9 Datasets, splits, seeds

Corpora, splits (70/15/15, split seed 42), training protocol, and
temperature calibration are specified in `methodology/*.md`. Ground truth:
`ground-truth.md`. All numbers in the Results section trace to experiment
IDs in `raw/experiment-index.json`.
