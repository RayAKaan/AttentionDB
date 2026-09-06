#!/usr/bin/env python3
"""Sync PH3B canonical results CSVs from raw run dirs (§16).

Every canonical number originates from a raw run artifact; this script is
the only bridge. Re-running it is idempotent.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "raw", "runs"))

RUNS_SRC = [
    ("PH3B-COMP-001", "S-512"),
    ("PH3B-COMP-002-D256", "M20-256"),
    ("PH3B-COMP-003", "S-512-rerun"),
]


def read(run, name):
    p = os.path.join(RUNS, run, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def w(path, rows, fields):
    with open(os.path.join(HERE, path), "w") as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for r in rows:
            cw.writerow(r)


# ---- complementary-retrieval.csv (ALL-subset, main table source) ----
rows = []
for run, tier in RUNS_SRC:
    for r in read(run, "results.csv"):
        rows.append({"experiment": run, "tier": tier, "arm": r["arm"], "seed": r["seed"],
                     "R@1": r["R@1"], "R@5": r["R@5"], "R@10": r["R@10"],
                     "NDCG@10": r["NDCG@10"], "MRR": r["MRR"], "R@10_std": r["R@10_std"]})
w("complementary-retrieval.csv", rows,
  ["experiment", "tier", "arm", "seed", "R@1", "R@5", "R@10", "NDCG@10", "MRR", "R@10_std"])

# ---- gating-by-query-type.csv ----
rows = []
for run, tier in RUNS_SRC:
    for r in read(run, "query_groups.csv"):
        rows.append({"experiment": run, "tier": tier, "arm": r["arm"], "seed": r["seed"],
                     "qtype": r["qtype"], "n": r["n"], "R@1": r["R@1"], "R@5": r["R@5"],
                     "R@10": r["R@10"], "NDCG@10": r["NDCG@10"], "MRR": r["MRR"]})
w("gating-by-query-type.csv", rows,
  ["experiment", "tier", "arm", "seed", "qtype", "n", "R@1", "R@5", "R@10", "NDCG@10", "MRR"])

# ---- bm25.csv (BM25 diagnostics from the primary run) ----
run = "PH3B-COMP-001"
lat = {r["stage"]: r for r in read(run, "latency.csv")}
mem = {r["metric"]: r["value"] for r in read(run, "memory.csv")}
ver = json.load(open(os.path.join(RUNS, run, "bm25_verify.json")))
rows = [{
    "experiment": run,
    "R@10": next(r["R@10"] for r in read(run, "results.csv") if r["arm"] == "bm25" and r["seed"] == "agg"),
    "NDCG@10": next(r["NDCG@10"] for r in read(run, "results.csv") if r["arm"] == "bm25" and r["seed"] == "agg"),
    "MRR": next(r["MRR"] for r in read(run, "results.csv") if r["arm"] == "bm25" and r["seed"] == "agg"),
    "p50_us": lat["bm25_query"]["p50_us"], "p95_us": lat["bm25_query"]["p95_us"], "p99_us": lat["bm25_query"]["p99_us"],
    "indexing": f"embedded in insert_document ({mem['build_seconds']} s total build)",
    "tokenizer": "engine-internal: lowercase, punctuation-trim, stopwords, porter stem",
    "verified_containment": ver["term_containment_top10"],
    "verified_rare_token_hit_rate": ver["rare_token_known_answer_hit_rate"],
    "verified_overlap_indep_bm25": ver["overlap_at_10_engine_vs_independent"],
}]
w("bm25.csv", rows, ["experiment", "R@10", "NDCG@10", "MRR", "p50_us", "p95_us", "p99_us",
                     "indexing", "tokenizer", "verified_containment",
                     "verified_rare_token_hit_rate", "verified_overlap_indep_bm25"])

# ---- hybrid.csv (arms + val k-sensitivity) ----
rows = []
for arm in ("hybrid_rrf_k60", "hybrid_engine"):
    r = next(r for r in read(run, "results.csv") if r["arm"] == arm and r["seed"] == "agg")
    rows.append({"experiment": run, "arm": arm, "seed": "agg", "R@10": r["R@10"],
                 "NDCG@10": r["NDCG@10"], "MRR": r["MRR"], "k": "60 (pre-registered)"})
for r in read(run, "hybrid_k_val.csv"):
    rows.append({"experiment": run, "arm": "hybrid_rrf_VAL_ONLY", "seed": "-", "R@10": r["val_r10"],
                 "NDCG@10": r["val_ndcg"], "MRR": r["val_mrr"], "k": r["k"]})
w("hybrid.csv", rows, ["experiment", "arm", "seed", "R@10", "NDCG@10", "MRR", "k"])

# ---- complementary-latency.csv (stage percentiles, primary run) ----
rows = []
for r in read(run, "latency.csv"):
    rows.append({"experiment": run, "stage": r["stage"], "n": r["n"],
                 "p50_us": r["p50_us"], "p95_us": r["p95_us"], "p99_us": r["p99_us"], "qps": r["qps"]})
w("complementary-latency.csv", rows,
  ["experiment", "stage", "n", "p50_us", "p95_us", "p99_us", "qps"])

# ---- complementary-candidate-recall.csv ----
rows = []
for run2, tier in RUNS_SRC:
    for r in read(run2, "candidate_recall.csv"):
        rows.append({"experiment": run2, "tier": tier, "subset": r["subset"],
                     "gt_covered_frac": r["gt_covered_frac"],
                     "queries_full_coverage": r["queries_full_coverage"],
                     "avg_pool_candidates": r["avg_pool_candidates"]})
w("complementary-candidate-recall.csv", rows,
  ["experiment", "tier", "subset", "gt_covered_frac", "queries_full_coverage", "avg_pool_candidates"])

# ---- complementary-memory.csv ----
rows = []
for run2, tier in RUNS_SRC:
    for r in read(run2, "memory.csv"):
        rows.append({"experiment": run2, "tier": tier, "metric": r["metric"], "value": r["value"]})
for run2, note in (("PH3B-COMP-002-M30", "OOM exit 137 ~343s"),
                   ("PH3B-COMP-002-M20", "OOM exit 137 ~253s"),
                   ("PH3B-COMP-002-D256-FAILED-TMPFS", "OOM exit 137 (tmpfs pressure from leaked engine dirs)")):
    rows.append({"experiment": run2, "tier": "M-512" if "D256" not in run2 else "M20-256",
                 "metric": "status", "value": f"FAILED ({note})"})
w("complementary-memory.csv", rows, ["experiment", "tier", "metric", "value"])

# ---- gating weights by type + oracle agreement (primary run, seed-42 model) ----
gw = read(run, "gating_weights_test.csv")
rows = []
acc = {}
for r in gw:
    if r["qtype"].startswith("#") or r.get("w_title") is None:
        continue
    t = r["qtype"]
    a = acc.setdefault(t, {"n": 0, "w": [0.0, 0.0, 0.0], "agree": 0})
    a["n"] += 1
    a["w"][0] += float(r["w_title"]); a["w"][1] += float(r["w_body"]); a["w"][2] += float(r["w_full"])
    a["agree"] += int(r["agree"])
for t, a in sorted(acc.items()):
    rows.append({"experiment": run, "qtype": t, "n": a["n"],
                 "mean_w_title": round(a["w"][0] / a["n"], 4),
                 "mean_w_body": round(a["w"][1] / a["n"], 4),
                 "mean_w_full": round(a["w"][2] / a["n"], 4),
                 "oracle_agreement_rate": round(a["agree"] / a["n"], 4)})
w("gating-weights-by-type.csv", rows,
  ["experiment", "qtype", "n", "mean_w_title", "mean_w_body", "mean_w_full", "oracle_agreement_rate"])

print("canonical PH3B results synced:",
      sorted(f for f in os.listdir(HERE) if f.endswith(".csv")))
