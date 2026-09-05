#!/usr/bin/env python3
"""PH2C-QK-002 §4: independent validation of cached Phase 2B datasets.

Checks dataset-schema invariants that are verifiable WITHOUT engine/doc
vectors: split disjointness (HC/leakage §13), norm=minmax(raw) consistency,
per-head quality metrics vs recomputed-with-sort (HC-5 class: slice-order
ranking), GT validity vs candidate pools, candidate pool structure.
Exit 1 on any violation."""
import json, sys
from bisect import bisect_left

def minmax(v):
    lo, hi = min(v), max(v)
    span = max(hi - lo, 1e-12)
    return [(x - lo) / span for x in v]

def metrics_from_sorted(ids, scores, gt, k=10):
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    ranked = [ids[i] for i in order]
    rel = set(gt[:k]); idcg = sum(1/math.log2(i+2) for i in range(k))
    dcg = sum(1/math.log2(pos+2) for pos, i in enumerate(ranked[:10]) if i in rel)
    mrr = next((1/(pos+1) for pos, i in enumerate(ranked) if i in rel), 0.0)
    r1 = 1.0 if ranked and ranked[0] in rel else 0.0
    r10 = len(set(ranked[:10]) & rel)/len(rel) if rel else 0.0
    return r1, r10, dcg/idcg, mrr

import math
fails = []
for corpus, path in [("controlled","benchmarks/phase2b/controlled/dataset.json"),
                     ("noise","benchmarks/phase2b/noise/dataset.json"),
                     ("multiview","benchmarks/phase2b/multiview/dataset.json")]:
    d = json.load(open(path))
    qs = d["queries"]
    # 1) split disjointness (§13)
    ids = {s: {q["query_id"] for q in qs if q["split"] == s} for s in ("Train","Val","Test")}
    for a in ("Train","Val"):
        for b in ("Val","Test"):
            if a != b and ids[a] & ids[b]:
                fails.append(f"{corpus}: split overlap {a}∩{b} = {len(ids[a]&ids[b])}")
    # 2) per-query checks
    n_gt_out, n_norm_bad, n_metric_bad, n_order_bad = 0, 0, 0, 0
    pool_sizes, gt_in_union = [], []
    engine_ids_ok = all(c >= 1 for q in qs for h in q["heads"] for c in h["candidates"])
    if not engine_ids_ok:
        fails.append(f"{corpus}: candidate id < 1 (HC-1 signature: engine ids are 1-based)")
    for q in qs:
        gt = q["ground_truth"]
        if not gt or len(set(gt)) != len(gt):
            fails.append(f"{corpus} q{q['query_id']}: GT empty/duplicated"); break
        union = set()
        for h in q["heads"]:
            if len(h["candidates"]) != len(h["raw_scores"]):
                fails.append(f"{corpus} q{q['query_id']}: candidate/score length mismatch"); break
            pool_sizes.append(len(h["candidates"]))
            # norm == minmax(raw) (tolerance for f32 serialization)
            nm = minmax(h["raw_scores"])
            if any(abs(a-b) > 2e-3 for a, b in zip(nm, h["norm_scores"])):
                n_norm_bad += 1
            # HC-5 class: do cached head-quality metrics match the SORTED ranking?
            r1, r10, nd, mrr = metrics_from_sorted(h["candidates"], h["raw_scores"], gt)
            if abs(r10 - h["recall_at_k"]) > 0.02 or abs(mrr - h["mrr"]) > 0.02:
                n_metric_bad += 1
            # is pool order already score-sorted (engine rank order)?
            rs = h["raw_scores"]
            if any(rs[i] < rs[i+1] - 1e-6 for i in range(len(rs)-1)):
                n_order_bad += 1
            union.update(h["candidates"])
        gt_in_union.append(sum(1 for g in gt if g in union)/len(gt))
        if not any(g in union for g in gt):
            n_gt_out += 1
    print(f"[{corpus}] queries={len(qs)} heads={d['num_heads']} dim={d['input_dim']} "
          f"pool(min/med/max)={min(pool_sizes)}/{sorted(pool_sizes)[len(pool_sizes)//2]}/{max(pool_sizes)}")
    print(f"  queries with ZERO gt-in-union: {n_gt_out} | mean gt-fraction-in-union: {sum(gt_in_union)/len(gt_in_union):.4f}")
    print(f"  norm!=minmax(raw): {n_norm_bad} | head-metric mismatch vs sorted: {n_metric_bad} | unsorted pools: {n_order_bad}")
    if n_norm_bad or n_metric_bad:
        fails.append(f"{corpus}: score/metric consistency failures (norm={n_norm_bad}, metrics={n_metric_bad})")
    if n_gt_out > len(qs)*0.5:
        fails.append(f"{corpus}: >50% of queries have no GT in union — GT/candidate link broken (HC-1/HC-2 signature)")
    # multiview: one-modality marker (HC-3 guard): oracle-capable structure per cached head metrics
    if corpus == "multiview":
        best_head_exists = any(any(h["recall_at_k"] > 0.5 for h in q["heads"]) for q in qs)
        print(f"  multiview: some head reaches R@10>0.5 on some query: {best_head_exists} (HC-3: GT view-linked)")
if fails:
    print("VALIDATION FAILED:"); [print("  ✗", f) for f in fails]; sys.exit(1)
print("dataset validation: PASS")
