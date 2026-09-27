# C5 Dataset Report

**Datasets**: BEIR SciFact, BEIR NFCorpus  
**Embeddings**: Frozen from C2/C3 (intfloat/e5-small-v2, dim=384)

---

## Corpus Statistics (TEST split)

| Metric | SciFact | NFCorpus |
|--------|---------|----------|
| Documents (corpus.jsonl) | 5,183 | 3,633 |
| Test queries (qrels/test.tsv) | 300 | 323 |
| Total qrels (test) | 1,247 | 1,402 |
| Mean relevant/test query | 4.16 | 4.34 |

---

## Embedding Artifacts (Frozen, Immutable)

| Dataset | Embedding Run | Heads | Path |
|---------|--------------|-------|------|
| SciFact | C3-EMBED-SCIFACT-001 | TITLE, BODY, CITE, CANONICAL, QUERIES | `raw/C3-EMBED-SCIFACT-001/artifacts/` |
| NFCorpus | C2-EMBED-NFCORPUS-003 | TITLE, BODY, CITE, CANONICAL, QUERIES | `raw/C2-EMBED-NFCORPUS-003/artifacts/` |

All `.npy` files verified against C4 run hashes (SHA256 in `RUN-INDEX.yaml`).

---

## Splits (Per Protocol §8)

| Split | Purpose | SciFact | NFCorpus |
|-------|---------|---------|----------|
| **TEST** | Frozen primary evaluation | 300 qrels (test.tsv) | 323 qrels (test.tsv) |
| **VALID** | λ tuning gate | 200 seeded train-sample | 324 dev.tsv (official) |
| **PROBE/SMOKE** | Safety/sanity | 20 subsample from VAL | 20 subsample from VAL |

**VALID split derivation**:
- NFCorpus: Official BEIR `dev.tsv` (324 qids) — **frozen**
- SciFact: Seeded sample from `train.tsv` — sort qids, shuffle with `rng(seed+0x5EED)`, take first 200 — **frozen**

No data leakage: TEST qids never used for tuning. VAL and TEST disjoint.

---

## Vector Verification (TEST)

| Dataset | Head | Shape (n, 384) | SHA256 (artifact .f32) |
|---------|------|----------------|------------------------|
| SciFact | CANONICAL | (5183, 384) | recorded per-run |
| SciFact | HEAD-TITLE | (5183, 384) | recorded per-run |
| SciFact | HEAD-BODY | (5183, 384) | recorded per-run |
| SciFact | HEAD-CITE | (5183, 384) | recorded per-run |
| SciFact | QUERIES | (1109, 384) | recorded per-run |
| NFCorpus | CANONICAL | (3633, 384) | recorded per-run |
| NFCorpus | HEAD-TITLE | (3633, 384) | recorded per-run |
| NFCorpus | HEAD-BODY | (3633, 384) | recorded per-run |
| NFCorpus | HEAD-CITE | (3633, 384) | recorded per-run |
| NFCorpus | QUERIES | (323, 384) | recorded per-run |

All vectors loaded via `mmap` → `f32` copy; no in-place modification.

---

## Qrels Format

TSV: `query-id  doc-id  grade` (grade ∈ {1,2} for BEIR). Only grade ≥1 treated as relevant.

---

## Split Integrity Checks

- TEST qids ⊂ corpus qids: ✅
- VAL qids ⊂ corpus qids: ✅  
- TEST ∩ VAL = ∅: ✅
- PROBE/SMOKE ⊂ VAL: ✅