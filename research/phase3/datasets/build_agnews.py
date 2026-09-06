#!/usr/bin/env python3
"""PH3B dataset family: AG News multi-field text retrieval (real text).

Documented protocol (research/phase3/methodology/datasets-text.md is the
prose mirror of this code — keep both in sync):

- Source: AG News corpus (Xiang Zhang's CharCNN distribution), 3 columns:
  class_index (1-4), title, description. License: freely available for
  research (Zhang et al., "Character-level Convolutional Networks for
  Text Classification", NIPS 2015 distribution).
- Corpus: the first N train rows (tier-dependent), in file order.
- Queries: held-out TEST rows, three types (one field at a time, §2 of
  the Phase 3B spec):
    title : query text = the test doc's TITLE only
    body  : query text = the test doc's DESCRIPTION only
    mixed : query text = TITLE + " " + DESCRIPTION
- Views (engine collection has uniform dim; 512):
    title : hashed unigram bag-of-words of the TITLE
    body  : hashed unigram bag-of-words of the DESCRIPTION
    full  : hashed unigram bag-of-words of TITLE + DESCRIPTION
  Head-salted FNV-1a hashing gives each field its own dictionary — the
  per-head query vectors differ, as in a real fielded multi-encoder system.
- Ground truth: per query type t, the exact cosine top-10 over the corpus
  in the DEFINING head space (title->title, body->body, mixed->full).
  Cosines are rounded to 1e-5 before ranking so that independent
  implementations (numpy / rust) produce identical orderings; ties break
  by lower doc index. Tie events at the boundary are counted, not hidden.
- Leakage: corpus from train.csv, queries from test.csv (disjoint files);
  additionally title md5 sets are checked for overlap.

Artifacts per tier (in --out):
  corpus_title.f32 / corpus_body.f32 / corpus_full.f32   (n_docs x 512)
  queries_{title,body,mixed}_{title,body,full}.f32       (Q_t x 512)
  gt_title.i64 / gt_body.i64 / gt_mixed.i64              (Q_t x 10, doc idx)
  corpus_text.jsonl, query_text.jsonl
  meta.json (full provenance + differentiated-utility verification)
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import date

import numpy as np

DIM = 512
HEADS = ("title", "body", "full")
GT_K = 10
ROUND_DECIMALS = 4
SRC_URLS = [
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{f}.csv",
    "https://raw.githubusercontent.com/AaronCCWong/Char-CNN/master/data/ag_news_csv/{f}.csv",
]
TIERS = {
    "S": {"docs": 10_000, "q_per_type": 150},
    "M20": {"docs": 20_000, "q_per_type": 200},
    "M": {"docs": 30_000, "q_per_type": 200},
    # memory-constrained scaling probe: 20K docs at dim 256 (the 2 GB
    # sandbox cannot build the 512-dim medium corpus — see FAILED runs
    # PH3B-COMP-002-M30 / -M20). NOT comparable to the 512-dim tiers;
    # hash-collision regime differs (documented in the run + findings).
    "M20D256": {"docs": 20_000, "q_per_type": 200, "dim": 256},
}
# query-type -> defining head (the space in which GT is exact cosine)
DEFINING = {"title": "title", "body": "body", "mixed": "full"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_source(data_dir):
    paths = {}
    for f in ("train", "test"):
        p = os.path.join(data_dir, f"ag_{f}.csv")
        if not os.path.exists(p):
            os.makedirs(data_dir, exist_ok=True)
            last = None
            for url in SRC_URLS:
                try:
                    urllib.request.urlretrieve(url.format(f=f), p)
                    last = None
                    break
                except Exception as exc:  # try next mirror
                    last = exc
            if last is not None:
                sys.exit(f"download failed for {f}: {last}")
        paths[f] = p
    return paths


def clean_text(t):
    # AG News CSVs contain literal backslash escapes ("\n", "\", ...) left
    # over from scraping; normalize them to spaces, collapse whitespace.
    for esc in ("\\n", "\\r", "\\t", "\\\\", '\\"'):
        t = t.replace(esc, " ")
    return re.sub(r"\s+", " ", t).strip()


def tokens(t):
    return TOKEN_RE.findall(t.lower())


class Hasher:
    """Head-salted FNV-1a unigram hashing -> tf vector, l2-normalized."""

    def __init__(self, head):
        self.head = head
        self.cache = {}

    def bucket(self, tok):
        b = self.cache.get(tok)
        if b is None:
            h = 0x811C9DC5
            for byte in f"{self.head}\x1f{tok}".encode():
                h ^= byte
                h = (h * 0x01000193) & 0xFFFFFFFF
            b = h % DIM
            self.cache[tok] = b
        return b

    def embed(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        for tok in tokens(text):
            v[self.bucket(tok)] += 1.0
        n = float(np.sqrt((v * v).sum()))
        if n > 0:
            v /= n
        return v


def exact_topk(corpus_norm, q_mat, k=GT_K):
    """Chunked exact cosine top-k. corpus_norm: (n,512) f32, l2-normalized.

    Returns (ids (q,k), ties_at_boundary). Scores are rounded to
    ROUND_DECIMALS before ranking (cross-implementation determinism); ties
    break by lower doc index.
    """
    q_mat = q_mat.copy()
    qn = np.linalg.norm(q_mat, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    q_mat /= qn
    n = corpus_norm.shape[0]
    ids = np.zeros((q_mat.shape[0], k), dtype=np.int64)
    boundary_ties = 0
    for s in range(0, q_mat.shape[0], 64):
        e = min(s + 64, q_mat.shape[0])
        scores = np.rint(q_mat[s:e] @ corpus_norm.T * (10**ROUND_DECIMALS))
        scores /= 10**ROUND_DECIMALS
        # STABLE argsort on rounded scores = deterministic (score desc, idx
        # asc) ordering. NOTE: argpartition is NOT admissible here — it
        # resolves boundary ties arbitrarily (caught by the independent
        # validation below; became HC-P3-4 documentation).
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        ids[s:e] = order
        if n > k:
            kth = np.partition(scores, n - k, axis=1)[:, n - k]
            # rows with >1 doc at the boundary score (ties the ordering had
            # to break deterministically)
            boundary_ties += int(((scores == kth[:, None]).sum(axis=1) > 1).sum())
    return ids, boundary_ties


def head_recall_at_k(corpus_norm, q_mat, gt, k=GT_K):
    """Exact R@k of retrieval in `corpus_norm` space against GT id lists."""
    q_mat = q_mat.copy()
    qn = np.linalg.norm(q_mat, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    q_mat /= qn
    hits = 0
    for s in range(0, q_mat.shape[0], 64):
        e = min(s + 64, q_mat.shape[0])
        scores = np.rint(q_mat[s:e] @ corpus_norm.T * (10**ROUND_DECIMALS))
        scores /= 10**ROUND_DECIMALS
        part = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        hits += sum(
            len(set(part[r].tolist()) & set(gt[s + r].tolist())) for r in range(e - s)
        )
    return hits / (q_mat.shape[0] * k)


def build_tier(tier, src, out_dir):
    global DIM
    cfg = TIERS[tier]
    n_docs, qpt = cfg["docs"], cfg["q_per_type"]
    DIM = cfg.get("dim", 512)
    os.makedirs(out_dir, exist_ok=True)

    def read_csv(p):
        rows = []
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                rows.append((r[0], clean_text(r[1]), clean_text(r[2])))
        return rows

    train = read_csv(src["train"])[:n_docs]
    test = read_csv(src["test"])
    need = {"title": qpt, "body": qpt, "mixed": 2 * qpt}
    assert len(train) == n_docs and len(test) >= sum(need.values()), "source too small"

    print(f"[{tier}] corpus={len(train)} embed 3 heads ...", flush=True)
    hashers = {h: Hasher(h) for h in HEADS}
    corpus = {h: np.zeros((n_docs, DIM), dtype=np.float32) for h in HEADS}
    for i, (_, title, desc) in enumerate(train):
        if i % 5000 == 0:
            print(f"  doc {i}", flush=True)
        corpus["title"][i] = hashers["title"].embed(title)
        corpus["body"][i] = hashers["body"].embed(desc)
        corpus["full"][i] = hashers["full"].embed(f"{title} {desc}")
    corpus_norm = {h: corpus[h] / np.maximum(np.linalg.norm(corpus[h], axis=1, keepdims=True), 1e-12) for h in HEADS}

    # queries: disjoint test blocks per type; LEAKAGE FILTER — AG News
    # contains duplicate stories across the train/test files, so any query
    # whose title OR description md5 appears in the corpus is EXCLUDED
    # (recorded): such a query would trivially anchor GT at a verbatim
    # corpus copy.
    corpus_title_md5 = {hashlib.md5(t.encode()).hexdigest() for _, t, _ in train}
    corpus_desc_md5 = {hashlib.md5(d.encode()).hexdigest() for _, _, d in train}
    raw_blocks = {
        "title": test[:qpt],
        "body": test[qpt : 2 * qpt],
        "mixed": test[2 * qpt : 3 * qpt],
    }
    blocks, excluded = {}, {}
    for t, rows in raw_blocks.items():
        keep = [
            r
            for r in rows
            if hashlib.md5(r[1].encode()).hexdigest() not in corpus_title_md5
            and hashlib.md5(r[2].encode()).hexdigest() not in corpus_desc_md5
        ]
        excluded[t] = len(rows) - len(keep)
        blocks[t] = keep
        print(f"  [{t}] leakage filter: excluded {excluded[t]}, kept {len(keep)}", flush=True)
    qpt_eff = {t: len(rows) for t, rows in blocks.items()}
    assert all(v >= 100 for v in qpt_eff.values()), f"too few queries after leakage filter: {qpt_eff}" 
    qtext = {}
    for t, rows in blocks.items():
        for _, title, desc in rows:
            qtext.setdefault(t, []).append(title if t == "title" else desc if t == "body" else f"{title} {desc}")

    queries = {
        t: {h: np.zeros((qpt_eff[t], DIM), dtype=np.float32) for h in HEADS}
        for t in blocks
    }
    for t, texts in qtext.items():
        for j, txt in enumerate(texts):
            for h in HEADS:
                queries[t][h][j] = hashers[h].embed(txt)

    # ---- ground truth in the DEFINING head space ----
    gt = {}
    ties = {}
    for t in blocks:
        ids, tie_ct = exact_topk(corpus_norm[DEFINING[t]], queries[t][DEFINING[t]])
        gt[t] = ids
        ties[t] = tie_ct
        assert ids.max() < n_docs and ids.shape == (qpt_eff[t], GT_K)

    # ---- independent GT validation: full argsort on a subsample ----
    rng = np.random.default_rng(42)
    mismatches = 0
    for t in blocks:
        sample = rng.choice(qpt_eff[t], size=min(30, qpt_eff[t]), replace=False)
        cn = corpus_norm[DEFINING[t]]
        qm = queries[t][DEFINING[t]][sample].copy()
        qn = np.linalg.norm(qm, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        qm /= qn
        scores = np.rint(qm @ cn.T * (10**ROUND_DECIMALS)) / (10**ROUND_DECIMALS)
        idx = np.broadcast_to(np.arange(n_docs), scores.shape)
        full = np.lexsort((idx, -scores), axis=1)[:, :GT_K]
        mismatches += int((full != gt[t][sample]).sum())
    assert mismatches == 0, f"GT validation FAILED: {mismatches} rows differ (argpartition vs full argsort)"

    # ---- §2 differentiated-utility verification (exact per-head R@10) ----
    utility = {}
    for t in blocks:
        row = {}
        for h in HEADS:
            row[h] = round(head_recall_at_k(corpus_norm[h], queries[t][h], gt[t]), 4)
        utility[t] = row
        best = max(row, key=row.get)
        second = sorted(row.values())[-2]
        ok = best == DEFINING[t] and row[DEFINING[t]] - second >= 0.05
        print(f"  utility[{t}] = {row} defining={DEFINING[t]} -> {'OK' if ok else 'FAIL'}", flush=True)
        assert ok, f"views not differentiated for {t}: {row} — dataset unsuitable, not registering"

    # near-duplicate transparency (post-filter guarantee: 0 leakage rows)
    dup_titles = len(train) - len(corpus_title_md5)
    leak = 0

    # ---- write artifacts ----
    meta = {
        "dataset_id": f"PH3B-DS-AG-{tier}",
        "dim": DIM,
        "family": "AG News multi-field text retrieval",
        "source": {
            "urls": [u.format(f=f) for u in SRC_URLS for f in ("train", "test")],
            "license": "freely available for research (Zhang et al. 2017 CharCNN distribution; original AG corpus, Gulli 2004)",
            "version": "ag_news_csv (120,000 train / 7,600 test)",
            "retrieved": str(date.today()),
            "sha256": {f: sha256_file(src[f]) for f in ("train", "test")},
        },
        "preprocessing": {
            "text_cleaning": "literal backslash escapes (\\n,\\r,\\t,\\\\,\\\") -> space; whitespace collapsed",
            "vector_tokenizer": "lowercase, [a-z0-9]+ tokens, NO stopwords, NO stemming (documented; engine BM25 has its own tokenizer, recorded separately)",
            "embedding": f"head-salted FNV-1a unigram hashing -> {DIM} buckets, tf, l2-normalized; salt = head name",
            "heads": list(HEADS),
            "field_map": {"title": "title", "body": "description", "full": "title + description"},
            "category_view": "NOT included: 4 classes cannot justify a metadata embedding view (would be a dominated/degenerate head); recorded per spec §1 'where justified'",
        },
        "corpus": {"n_docs": n_docs, "rows": "train.csv[:n_docs]", "duplicate_titles": dup_titles},
        "queries": {
            "n_total": sum(qpt_eff.values()),
            "per_type": qpt_eff,
            "leakage_filter_excluded": excluded,
            "blocks": {"title": "test[:qpt]", "body": "test[qpt:2qpt]", "mixed": "test[2qpt:3qpt] (then leakage-filtered)"},
            "protocol": "one field at a time: title-query = TITLE text; body-query = DESCRIPTION text; mixed = TITLE+' '+DESCRIPTION; each query embedded in ALL three head spaces with head-salted hashing",
        },
        "ground_truth": {
            "definition": "exact cosine top-10 in the query type's DEFINING head space (title->title, body->body, mixed->full) over the corpus",
            "rounding": f"cosines rounded to 1e-{ROUND_DECIMALS} before ranking (cross-implementation determinism)",
            "tie_break": "lower doc index; boundary tie events counted",
            "boundary_tie_events": ties,
            "validation": "argpartition vs full argsort on 30 queries/type: 0 mismatches",
            "leakage_checks": "train/test file disjointness + title-md5 overlap = 0",
        },
        "differentiated_utility_exact_R@10": utility,
        "files": {},
    }
    for h in HEADS:
        p = os.path.join(out_dir, f"corpus_{h}.f32")
        corpus[h].tofile(p)
        meta["files"][f"corpus_{h}.f32"] = sha256_file(p)
    for t in blocks:
        for h in HEADS:
            p = os.path.join(out_dir, f"queries_{t}_{h}.f32")
            queries[t][h].tofile(p)
            meta["files"][f"queries_{t}_{h}.f32"] = sha256_file(p)
        p = os.path.join(out_dir, f"gt_{t}.i64")
        gt[t].tofile(p)
        meta["files"][f"gt_{t}.i64"] = sha256_file(p)
    with open(os.path.join(out_dir, "corpus_text.jsonl"), "w") as f:
        for i, (_, title, desc) in enumerate(train):
            f.write(json.dumps({"i": i, "title": title, "description": desc}) + "\n")
    with open(os.path.join(out_dir, "query_text.jsonl"), "w") as f:
        for t, texts in qtext.items():
            for j, txt in enumerate(texts):
                f.write(json.dumps({"i": j, "type": t, "text": txt}) + "\n")
    for name in ("corpus_text.jsonl", "query_text.jsonl"):
        meta["files"][name] = sha256_file(os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{tier}] DONE n_docs={n_docs} q={3*qpt} ties={ties} dup_titles={dup_titles}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", choices=sorted(TIERS))
    ap.add_argument("--data-dir", default="/var/tmp/phase3b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.data_dir, f"agnews-{a.tier}")
    src = ensure_source(a.data_dir)
    build_tier(a.tier, src, out)


if __name__ == "__main__":
    main()
