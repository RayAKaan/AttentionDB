# Phase 3 datasets (spec §3/§4)

## PH3-DS-FM — Fashion-MNIST multi-view retrieval (REAL images)

- **Source**: Fashion-MNIST (Zalando SE), official distribution
  `http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/`
  (train/t10k IDX-ubyte). Retrieved 2026-09-05.
- **License**: MIT (dataset wrapper); images © Zalando SE, published for
  research use. Redistribution of derived vectors for research is permitted;
  the builder script re-downloads originals and verifies IDX magic/counts.
- **Preprocessing** (deterministic, in `datasets/build_fashion.py`):
  28×28 uint8 → f32/255 → 2×2 block-mean downscale to 14×14.
  Head layout: `full` = the whole 14×14 image (196-d); `q0..q3` = 7×7
  quadrants (49-d, zero-padded to 196 for the engine's uniform collection
  dimension — zero-padding adds nothing to dot products or norms, so
  per-head cosine rankings are mathematically identical to unpadded).
- **Corpora**: Tier S = first 10,000 train images, 1,000 test-image queries;
  Tier T30 = first 30,000; Tier M = all 60,000, 2,000 queries (even test
  indices). Tiers are prefixes of the same ordering — quality numbers are
  comparable across tiers up to pool effects.
- **Ground truth**: EXACT cosine ranking over full-view vectors, brute
  force (chunked argpartition; verified against full argsort on Tier S).
  Top-10 corpus doc indices per query. Self-retrieval impossible (train/test
  split is disjoint).
- **Hashes**: every tier writes `meta.json` with sha256 of the downloaded
  source files and the derived tensors (in `/tmp/phase3/` — regenerable;
  the BUILDER + hashes are the persisted provenance because the raw tensors
  exceed the workspace snapshot budget). Tier-S determinism was verified by
  rebuilding and matching hashes byte-for-byte.

## Known limitations of this suite (honest scope statement)

1. **Head complementarity**: with a full-image view present, the quadrant
   views are uniformly dominated — the corpus measures gating's ability to
   find that out, but cannot measure per-query complementarity gains. A
   second dataset family with semantically complementary views (e.g., real
   text documents with title/body/image views via hashed-bag-of-words
   embeddings + engine BM25) is the pending extension (§5 Baselines D/E).
2. **Scale**: the 2 GB sandbox OOMs above ~10K docs in this configuration
   (see PH3-QUAL-FM-M/T30 runs) — Tier M/L quality runs are blocked by
   memory, recorded as the §19 finding.
3. **Modality**: single modality (images). Text/BM25 baselines (D/E) are
   not applicable to PH3-DS-FM and are deferred to the text dataset family.
