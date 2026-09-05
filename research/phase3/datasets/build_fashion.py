#!/usr/bin/env python3
"""Phase 3 dataset builder — Fashion-MNIST multi-view retrieval (PH3-DS-FM).

Source: Fashion-MNIST (Zalando), MIT-licensed dataset wrapper of Zalando
product images; official distribution at the fashion-mnist S3 endpoint.
Retrieved 2026-09-05. License permits research use and redistribution.

Preprocessing (deterministic, documented):
  1. IDX files are parsed (magic 2051), images 28x28 uint8 -> f32 /255.
  2. Downscale 28x28 -> 14x14 by exact 2x2 block mean (no interpolation).
  3. Head layout (all zero-padded to 196 dims for the engine's uniform
     collection dim; zero-padding scales every vector's cosine by the same
     constant, so per-head rankings are unchanged — documented in
     methodology/datasets.md):
       full : the whole 14x14 image (196 dims, no padding)
       q0..q3: 7x7 quadrants of the 14x14 image (49 dims + 147 zeros)
  4. Corpus = train split (Tier S: first 10,000; Tier M: all 60,000).
     Queries = test split (Tier S: 1,000; Tier M: 2,000, even indices).
  5. Ground truth: EXACT cosine ranking over full-view vectors, brute
     force in numpy; top-10 ids (corpus index, 0-based) per query.
     (Self-retrieval impossible: queries come from the disjoint test set.)

Outputs under /tmp/phase3/fashion-<tier>/:
  corpus_heads.f32  (n_docs, 5, 196) row-major
  query_heads.f32   (n_q, 5, 196)
  gt.u32            (n_q, 10) corpus doc indices, best first
  meta.json         sizes, hashes, provenance
All downstream consumers must verify meta.json sha256 fields.
"""
import gzip
import hashlib
import json
import os
import struct
import sys
import urllib.request

import numpy as np

SOURCES = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
}
TIERS = {
    "S": {"docs": 10_000, "queries": 1_000},
    "T30": {"docs": 30_000, "queries": 1_000},
    "M": {"docs": 60_000, "queries": 2_000},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    assert magic == 2051, f"bad IDX magic {magic}"
    imgs = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    return imgs


def downscale(imgs):
    """28x28 -> 14x14 by exact 2x2 block mean."""
    n = imgs.shape[0]
    return imgs.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4)).astype(np.float32)


def head_views(img14):
    """(5, 196) float32: full + 4 zero-padded quadrants."""
    flat = img14.reshape(-1)  # 196
    q = img14.reshape(2, 7, 2, 7).swapaxes(1, 2).reshape(4, 49)  # q0..q3 row-major
    padded = np.zeros((4, 196), dtype=np.float32)
    padded[:, :49] = q
    return np.vstack([flat, padded])


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "S"
    n_docs, n_q = TIERS[tier]["docs"], TIERS[tier]["queries"]
    cache = "/tmp/phase3/src"
    os.makedirs(cache, exist_ok=True)
    src_hashes = {}
    for name, url in SOURCES.items():
        p = os.path.join(cache, os.path.basename(url))
        if not os.path.exists(p):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, p)
        # integrity: IDX magic + counts (verified on parse below)
        src_hashes[name] = {"path": p, "sha256": sha256(p), "url": url}
    train = downscale(read_idx(src_hashes["train_images"]["path"]))
    test = downscale(read_idx(src_hashes["test_images"]["path"]))
    assert train.shape[0] == 60_000 and test.shape[0] == 10_000

    corpus = train[:n_docs]
    queries = test[: n_q * 2 : 2][:n_q] if n_q > 1_000 else test[:n_q]
    # normalize per head for exact cosine GT (engine receives raw f32; its
    # pipeline normalizes internally — verify equivalence in the harness)
    def unit(v):
        return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)

    # Preallocate and fill chunk-wise (2 GB sandbox: np.stack over 60K small
    # arrays OOM-kills; chunked filling keeps peak RSS ~400 MB)
    corpus_h = np.empty((n_docs, 5, 196), dtype=np.float32)
    for i in range(0, n_docs, 512):
        corpus_h[i : i + 512] = np.stack([head_views(im) for im in corpus[i : i + 512]])
    query_h = np.empty((queries.shape[0], 5, 196), dtype=np.float32)
    for i in range(0, queries.shape[0], 512):
        query_h[i : i + 512] = np.stack([head_views(im) for im in queries[i : i + 512]])
    del train, test
    # full-view exact cosine, top-10 via argpartition in small chunks
    # (2 GB sandbox: 256-row chunks + full argsort grow RSS past the wall)
    qn = unit(query_h[:, 0])
    cn = np.ascontiguousarray(unit(corpus_h[:, 0]).T)  # (196, n) packed once
    gt_rows = []
    for i in range(0, qn.shape[0], 64):
        chunk = qn[i : i + 64] @ cn  # (64, n)
        part = np.argpartition(-chunk, 10, axis=1)[:, :10]
        rows = np.take_along_axis(chunk, part, axis=1)
        order = np.argsort(-rows, axis=1)
        gt_rows.append(np.take_along_axis(part, order, axis=1))
        del chunk, part, rows
    gt = np.vstack(gt_rows).astype(np.uint32)
    del cn

    out = f"/tmp/phase3/fashion-{tier}"
    os.makedirs(out, exist_ok=True)
    for name, arr in [
        ("corpus_heads.f32", corpus_h.astype(np.float32)),
        ("query_heads.f32", query_h.astype(np.float32)),
        ("gt.u32", gt),
    ]:
        arr.tofile(os.path.join(out, name))
    meta = {
        "dataset_id": f"PH3-DS-FM-{tier}",
        "name": f"Fashion-MNIST multi-view retrieval (tier {tier})",
        "source": "Fashion-MNIST (Zalando SE), MIT-licensed dataset wrapper; official S3 distribution",
        "license": "MIT (dataset wrapper); images (c) Zalando SE, research use",
        "retrieved": "2026-09-05",
        "tier": tier,
        "n_docs": int(n_docs),
        "n_queries": int(queries.shape[0]),
        "heads": ["full", "q0", "q1", "q2", "q3"],
        "head_dims": {"full": 196, "q*": 49},
        "engine_dim": 196,
        "preprocessing": "28x28 -> 14x14 (2x2 block mean); full view = 196-d; quadrants 49-d zero-padded to 196",
        "ground_truth": "exact cosine over full-view vectors, brute force, top-10",
        "source_files": {k: {"url": v["url"], "sha256": v["sha256"]} for k, v in src_hashes.items()},
        "files": {f: sha256(os.path.join(out, f)) for f in ["corpus_heads.f32", "query_heads.f32", "gt.u32"]},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("dataset_id", "n_docs", "n_queries", "files")}, indent=1))


if __name__ == "__main__":
    main()
