# Limitations (paper section draft) (§19)

- **Synthetic data only.** All corpora are generated Gaussians/hashed
  features. Ground truth exists because generator vectors exist; no real
  embeddings, real text, or real user queries were evaluated.
- **Corpus size.** 384–10 000 documents. No 100k/1M-scale evaluation in
  Phase 2B (the Phase 2 scale roadmap — including the known hardcoded
  hnsw `max_elements` 100_000 limit — remains open).
- **Dimensionality.** 32 (single-view) and 64 per view (192 concatenated).
  Real embedding dimensions (768–1536) are untested; gating input scaling
  is unmeasured.
- **Engineered noise.** Head quality was designed (σ ladders, group
  conditional). Real head-quality structure may be noisier, correlated, or
  drifting; the σ ladder makes head 0 globally best by construction in the
  noise corpus.
- **Multiview construction.** Views share topic structure by construction
  (keywords/categories derive from the topic). A "coarse projection" is a
  simulation of a cross-modal extractor, not one. Only 3 views.
- **Query counts.** 45–180 test queries per corpus; recall point estimates
  carry non-trivial sampling error at these sizes (visible as the ±0.036
  seed spread). No confidence intervals beyond the multi-seed spread.
- **Seeds.** Three training seeds on two corpora; one split seed. The
  split itself is unreplicated (sample-size caveat above).
- **Hardware.** A shared CI-class VM; latency numbers are indicative
  ratios, not production measurements.
- **HNSW configuration.** ef_search 64 / efc 400 / M 16 throughout; no
  parameter sweeps; candidate pools are top-100 per head.
- **Candidate budget.** Offline studies union the full cached pools; the
  live pipeline's budget arithmetic differs (documented, §9.2 of the
  methodology section).
- **Distribution shift.** No experiment inserts documents after training
  the gate; gating under corpus drift is unmeasured.
- **No large-scale or production evaluation.** No concurrency, no
  durability-path latency, no real deployment.
- **Model training limitations.** Linear-softmax gate only (one hidden
  layer); targets are Recall@10-derived only (NDCG/MRR targets untested
  as targets); no learning-rate schedules; no ensembling; temperature
  calibration is a single global scalar.
