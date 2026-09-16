#!/usr/bin/env python3
"""Sync PH3C canonical results CSVs from raw run dirs (§21).

Memory scaling, head scaling, candidate budgets, decomposition, and
reproduction — every number read from raw run artifacts.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "raw", "runs"))


def rd(run, name):
    p = os.path.join(RUNS, run, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def rdmem(run):
    p = os.path.join(RUNS, run, "memory.csv")
    return {r["metric"]: r["value"] for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


def w(path, rows, fields):
    with open(os.path.join(HERE, path), "w") as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for r in rows:
            cw.writerow(r)


MEM_RUNS = [
    ("PH3C-MEM-001", "3", "512", "10000", "COMPLETED"),
    ("PH3C-MEM-002-C5K", "3", "512", "5000", "COMPLETED"),
    ("PH3C-MEM-002-C15K", "3", "512", "15000", "COMPLETED"),
    ("PH3C-MEM-002-C20K", "3", "512", "20000", "FAILED-OOM"),
    ("PH3C-MEM-002-H1-512", "1", "512", "10000", "COMPLETED"),
    ("PH3C-MEM-002-H2-512", "2", "512", "10000", "COMPLETED"),
    ("PH3C-MEM-002-H4-512", "4", "512", "10000", "COMPLETED"),
    ("PH3C-MEM-002-H8-512", "8", "512", "10000", "FAILED-OOM"),
    ("PH3C-MEM-002-H8-512-5K", "8", "512", "5000", "COMPLETED"),
    ("PH3C-MEM-002-H1-256", "1", "256", "10000", "COMPLETED"),
    ("PH3C-MEM-002-H2-256", "2", "256", "10000", "COMPLETED"),
    ("PH3C-MEM-002-H4-256", "4", "256", "10000", "COMPLETED"),
    ("PH3C-MEM-002-H8-256", "8", "256", "10000", "COMPLETED"),
    ("PH3C-MEM-002-D128", "3", "128", "10000", "COMPLETED"),
    ("PH3C-MEM-002-D256", "3", "256", "10000", "COMPLETED"),
    ("PH3C-MEM-002-BUDGET-1H-30K", "1", "512", "30000", "COMPLETED"),
    ("PH3C-MEM-002-BUDGET-2H-20K", "2", "512", "20000", "COMPLETED"),
]

# ---- memory-scaling.csv ----
rows = []
for run, heads, dim, docs, status in MEM_RUNS:
    m = rdmem(run)
    rows.append({
        "experiment": run, "heads": heads, "dim": dim, "n_docs": docs, "status": status,
        "raw_vector_mb": m.get("raw_vector_mb", ""),
        "peak_rss_mb": m.get("peak_rss_mb", ""),
        "steady_rss_mb": m.get("steady_rss_mb", ""),
        "engine_multiplier": m.get("engine_multiplier_peak_over_raw", ""),
        "build_seconds": m.get("build_seconds", ""),
        "engine_dir_mb": "", "last_rss_mb": "", "last_peak_mb": "",
    })
    if status == "FAILED-OOM":
        ck = rd(run, "memory-checkpoints.csv")
        if ck:
            last = ck[-1]
            rows[-1].update({"last_rss_mb": last["rss_mb"], "last_peak_mb": last["peak_rss_mb"],
                             "engine_dir_mb": last["engine_dir_bytes"]})
    else:
        disk = sum(float(v) for k, v in m.items() if k.startswith("disk_"))
        rows[-1]["engine_dir_mb"] = f"{disk / 1048576:.1f}" if disk else ""
w("memory-scaling.csv", rows,
  ["experiment", "heads", "dim", "n_docs", "status", "raw_vector_mb", "peak_rss_mb",
   "steady_rss_mb", "engine_multiplier", "build_seconds", "engine_dir_mb",
   "last_rss_mb", "last_peak_mb"])

# ---- memory-components.csv (checkpoint table of the reference run) ----
rows = []
for r in rd("PH3C-MEM-001", "memory-checkpoints.csv"):
    rows.append({"experiment": "PH3C-MEM-001", **r})
w("memory-components.csv", rows,
  ["experiment", "checkpoint", "elapsed_s", "rss_mb", "peak_rss_mb", "anon_mb", "file_mb",
   "engine_dir_bytes", "wal_bytes", "data_bytes", "meta_bytes", "index_bytes", "other_bytes",
   "docs_inserted"])

# ---- head-scaling-quality.csv (size-matched 5K ladder @K=100 + 10K extras + H3 ref) ----
LADDER = [
    ("PH3C-HEAD-001-H1-S5K", "1", "5000"),
    ("PH3C-HEAD-001-H2-S5K", "2", "5000"),
    ("PH3C-HEAD-001-H4-S5K", "4", "5000"),
    ("PH3C-HEAD-001-H8-S5K", "8", "5000"),
    ("PH3C-HEAD-001-H1", "1", "10000"),
    ("PH3C-HEAD-001-H2", "2", "10000"),
]
rows = []
for run, heads, docs in LADDER:
    for r in rd(run, "results.csv"):
        if r["budget"] != "100":
            continue
        rows.append({"experiment": run, "heads": heads, "n_docs": docs, "arm": r["arm"],
                     "seed": r["seed"], "R@1": r["R@1"], "R@5": r["R@5"], "R@10": r["R@10"],
                     "NDCG@10": r["NDCG@10"], "MRR": r["MRR"], "R@10_std": r["R@10_std"],
                     "candidate_recall": r["candidate_recall"]})
# 3-head reference points at K=100 (textquality runs)
for run, docs in (("PH3B-COMP-001", "10000"), ("PH3C-REPRO-001", "10000")):
    for r in rd(run, "results.csv"):
        if r["arm"] in ("global_best_single", "uniform_multihead", "trained_gating",
                        "oracle_head_empirical"):
            if r["seed"] not in ("agg",) and r["arm"] != "trained_gating":
                continue
            rows.append({"experiment": run, "heads": "3", "n_docs": docs, "arm": r["arm"],
                         "seed": r["seed"], "R@1": r["R@1"], "R@5": r["R@5"], "R@10": r["R@10"],
                         "NDCG@10": r["NDCG@10"], "MRR": r["MRR"], "R@10_std": r["R@10_std"],
                         "candidate_recall": ""})
w("head-scaling-quality.csv", rows,
  ["experiment", "heads", "n_docs", "arm", "seed", "R@1", "R@5", "R@10", "NDCG@10", "MRR",
   "R@10_std", "candidate_recall"])

# ---- head-scaling-latency.csv (serial vs parallel, 5K ladder) ----
rows = []
for run, heads, docs in LADDER[:4]:
    for r in rd(run, "latency.csv"):
        rows.append({"experiment": run, "heads": heads, "n_docs": docs, "mode": r["mode"],
                     "p50_us": r["p50_us"], "p95_us": r["p95_us"], "p99_us": r["p99_us"],
                     "qps": r["qps"], "cpu_s_total": r["cpu_s_total"]})
w("head-scaling-latency.csv", rows,
  ["experiment", "heads", "n_docs", "mode", "p50_us", "p95_us", "p99_us", "qps", "cpu_s_total"])

# ---- head-scaling-memory.csv ----
rows = []
for run, heads, docs in LADDER[:4]:
    m = rdmem(run)
    rows.append({"experiment": run, "heads": heads, "n_docs": docs,
                 "raw_vector_mb": m.get("raw_vector_mb", ""), "peak_rss_mb": m.get("peak_rss_mb", ""),
                 "build_seconds": m.get("build_seconds", "")})
w("head-scaling-memory.csv", rows,
  ["experiment", "heads", "n_docs", "raw_vector_mb", "peak_rss_mb", "build_seconds"])

# ---- candidate-budget.csv (primary 3-head 10K config: PH3B-COMP-001 has
# budgets only via headsqual; the budget ladder comes from the H4 10K run
# AND the size-matched runs. Use PH3C-HEAD-001-H4 (10K, budget sweep) as
# the budget study + 3-head proxy from its sibling. We publish ALL.) ----
rows = []
for run in ("PH3C-HEAD-001-H4", "PH3C-HEAD-001-H1-S5K", "PH3C-HEAD-001-H2-S5K",
            "PH3C-HEAD-001-H4-S5K", "PH3C-HEAD-001-H8-S5K"):
    for r in rd(run, "results.csv"):
        rows.append({"experiment": run, "budget": r["budget"], "arm": r["arm"], "seed": r["seed"],
                     "R@1": r["R@1"], "R@5": r["R@5"], "R@10": r["R@10"], "NDCG@10": r["NDCG@10"],
                     "MRR": r["MRR"], "R@10_std": r["R@10_std"], "candidate_recall": r["candidate_recall"]})
w("candidate-budget.csv", rows,
  ["experiment", "budget", "arm", "seed", "R@1", "R@5", "R@10", "NDCG@10", "MRR", "R@10_std",
   "candidate_recall"])

# ---- candidate-decomposition.csv ----
rows = []
for run in ("PH3C-HEAD-001-H4", "PH3C-HEAD-001-H1-S5K", "PH3C-HEAD-001-H2-S5K",
            "PH3C-HEAD-001-H4-S5K", "PH3C-HEAD-001-H8-S5K"):
    for r in rd(run, "candidate_decomposition.csv"):
        rows.append({"experiment": run, "disposition": r["disposition"],
                     "count": r["count"], "frac_of_gt_misses": r["frac_of_gt_misses"]})
w("candidate-decomposition.csv", rows,
  ["experiment", "disposition", "count", "frac_of_gt_misses"])

# ---- head-query-type.csv (by-type gating at K=100 across ladder) ----
rows = []
for run, heads, docs in LADDER[:4]:
    for r in rd(run, "query_groups.csv"):
        rows.append({"experiment": run, "heads": heads, "arm": r["arm"], "seed": r["seed"],
                     "qtype": r["qtype"], "n": r["n"], "R@10": r["R@10"], "NDCG@10": r["NDCG@10"],
                     "MRR": r["MRR"]})
w("head-query-type.csv", rows,
  ["experiment", "heads", "arm", "seed", "qtype", "n", "R@10", "NDCG@10", "MRR"])

# ---- reproduction.csv ----
rows = []
base = {r["arm"]: r for r in rd("PH3B-COMP-001", "results.csv") if r["seed"] == "agg"}
rep = {r["arm"]: r for r in rd("PH3C-REPRO-001", "results.csv") if r["seed"] == "agg"}
for arm in ("global_best_single", "uniform_multihead", "trained_gating", "bm25",
            "hybrid_rrf_k60", "hybrid_engine", "oracle_head_empirical", "exact_reference"):
    a, b = base.get(arm), rep.get(arm)
    if a and b:
        rows.append({"arm": arm, "PH3B-COMP-001_R@10": a["R@10"], "PH3C-REPRO-001_R@10": b["R@10"],
                     "delta": f"{float(b['R@10']) - float(a['R@10']):+.4f}",
                     "PH3B-COMP-001_NDCG": a["NDCG@10"], "PH3C-REPRO-001_NDCG": b["NDCG@10"],
                     "PH3B-COMP-001_MRR": a["MRR"], "PH3C-REPRO-001_MRR": b["MRR"]})
w("reproduction.csv", rows,
  ["arm", "PH3B-COMP-001_R@10", "PH3C-REPRO-001_R@10", "delta",
   "PH3B-COMP-001_NDCG", "PH3C-REPRO-001_NDCG", "PH3B-COMP-001_MRR", "PH3C-REPRO-001_MRR"])

print("PH3C canonical CSVs synced:",
      sorted(f for f in os.listdir(HERE) if f.endswith(".csv")))
