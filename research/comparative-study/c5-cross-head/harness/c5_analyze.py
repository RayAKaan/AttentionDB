#!/usr/bin/env python3
"""C5 statistical analysis — paired bootstrap, Wilcoxon, Holm, Cohen's dz, candidate analysis.

Per protocol: unit=query; paired across 5 reps per query; contrasts A↔B, B↔C × 2 datasets.
"""

import json
import os
import sys
import random
import numpy as np
from scipy import stats

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "raw")
SEED = 20260925
N_BOOT = 10000
CONTRASTS = [
    ("SCI", "A", "B"),
    ("SCI", "B", "C"),
    ("NFC", "A", "B"),
    ("NFC", "B", "C"),
]

def load_rep_data(run_id, rep):
    art = os.path.join(RAW, run_id, "artifacts")
    path = os.path.join(art, f"RUN-rep{rep}.json")
    with open(path, encoding="utf8") as f:
        return json.load(f)

def get_per_query_recall(run_id, rep):
    arr = load_rep_data(run_id, rep)
    # per_query list: each has recall10_qrels
    return [pq["recall10_qrels"] for pq in arr["per_query"]]

def get_per_query_candidate_change(run_id, rep):
    arr = load_rep_data(run_id, rep)
    pq = arr["per_query"]
    return [1 if pq.get("ledger", {}).get("round2_added_ids_total", 0) > 0 else 0 for pq in pq]

def paired_bootstrap_diff(rep_data1, rep_data2, n_boot=N_BOOT, seed=SEED):
    """rep_dataX: list of 5 arrays (one per rep), each length n_queries.
    Returns bootstrap distribution of mean difference (1-2)."""
    rnd = random.Random(seed)
    nq = len(rep_data1[0])
    # average across 5 reps per query first (protocol: per-query recall = mean of 5 reps)
    mean1 = np.mean(np.array(rep_data1), axis=0)
    mean2 = np.mean(np.array(rep_data2), axis=0)
    obs = np.mean(mean1 - mean2)
    boots = []
    for _ in range(n_boot):
        idx = [rnd.randrange(nq) for _ in range(nq)]
        boots.append(np.mean(mean1[idx] - mean2[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(obs), float(lo), float(hi), np.array(boots)

def wilcoxon(rep_data1, rep_data2):
    mean1 = np.mean(np.array(rep_data1), axis=0)
    mean2 = np.mean(np.array(rep_data2), axis=0)
    # Wilcoxon signed-rank on paired differences
    diff = mean1 - mean2
    # scipy wilcoxon drops zero differences; use zero_method='wilcox'
    stat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)

def cohens_dz(rep_data1, rep_data2):
    mean1 = np.mean(np.array(rep_data1), axis=0)
    mean2 = np.mean(np.array(rep_data2), axis=0)
    diff = mean1 - mean2
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0

def main():
    # Resolve run IDs from RUN-INDEX for TEST cells
    # We know the 6 TEST run_ids:
    test_runs = {
        "SCI": {"A": "C5-TEST-SCI-A-001", "B": "C5-TEST-SCI-B-001", "C": "C5-TEST-SCI-C-001"},
        "NFC": {"A": "C5-TEST-NFC-A-001", "B": "C5-TEST-NFC-B-001", "C": "C5-TEST-NFC-C-001"},
    }

    # Load per-query per-rep data
    data = {}
    for ds in ("SCI", "NFC"):
        for arm in ("A", "B", "C"):
            run = test_runs[ds][arm]
            reps = [get_per_query_recall(run, r) for r in range(1, 6)]
            data[(ds, arm)] = reps

    # Primary contrasts
    results = {}
    for ds, a, b in CONTRASTS:
        d1 = data[(ds, a)]
        d2 = data[(ds, b)]
        obs, lo, hi, boots = paired_bootstrap_diff(d1, d2)
        w_stat, w_p = wilcoxon(d1, d2)
        dz = cohens_dz(d1, d2)
        key = f"{ds}_{a}{b}"
        results[key] = {
            "dataset": ds, "contrast": f"{a}vs{b}", "mean_diff": obs,
            "ci95": [lo, hi], "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "cohens_dz": dz,
            "negligible": abs(obs) < 0.01,
        }
        print(f"{key}: diff={obs:.4f} CI95=[{lo:.4f},{hi:.4f}] W={w_stat:.1f} p={w_p:.4f} dz={dz:.3f} neglig={abs(obs)<0.01}")

    # Holm correction across 4 primary contrasts
    pvals = [(k, v["wilcoxon_p"]) for k, v in results.items()]
    pvals.sort(key=lambda x: x[1])
    m = len(pvals)
    for i, (k, p) in enumerate(pvals):
        alpha = 0.05 / (m - i)
        results[k]["holm_alpha"] = alpha
        results[k]["holm_reject"] = p < alpha

    print("\nHolm-corrected:")
    for k, v in results.items():
        print(f"  {k}: p={v['wilcoxon_p']:.4f} alpha={v['holm_alpha']:.4f} reject={v['holm_reject']}")

    # Candidate-set analysis: per-query binary change (C arm only)
    cand_results = {}
    for ds in ("SCI", "NFC"):
        runC = test_runs[ds]["C"]
        # per-query change aggregated across 5 reps (mode C)
        change_reps = [get_per_query_candidate_change(runC, r) for r in range(1, 6)]
        # majority vote per query
        change_majority = [1 if sum(change_reps[r][q] for r in range(5)) >= 3 else 0
                           for q in range(len(change_reps[0]))]
        pct = sum(change_majority) / len(change_majority)
        # Exact binomial test vs null p=0 (any change)
        from scipy.stats import binomtest
        bt = binomtest(sum(change_majority), n=len(change_majority), p=0.01, alternative="greater")
        cand_results[ds] = {
            "n_queries": len(change_majority),
            "changed_count": sum(change_majority),
            "pct_changed": pct,
            "binom_p": float(bt.pvalue),
        }
        print(f"\nCandidate change {ds}: {sum(change_majority)}/{len(change_majority)} ({pct:.1%}) binom_p={bt.pvalue:.2e}")

    # Save full results
    out = {
        "primary_contrasts": results,
        "candidate_analysis": cand_results,
        "metadata": {
            "seed": SEED, "n_boot": N_BOOT, "n_reps": 5,
            "test_queries": {"SCI": 300, "NFC": 323},
            "protocol": "C5 cross-head candidate-generation",
        },
    }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "c5-cross-head", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "statistical_results.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir}/statistical_results.json")

if __name__ == "__main__":
    main()