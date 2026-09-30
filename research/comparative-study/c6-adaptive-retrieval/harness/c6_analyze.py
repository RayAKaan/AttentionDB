#!/usr/bin/env python3
"""C6 statistical analysis — paired bootstrap, Wilcoxon, Holm, Cohen's dz, candidate analysis.

Per protocol: unit=query; paired across 5 reps per query; contrasts A↔B, B↔C/D/E/F/RE × 2 datasets.
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

# Primary contrasts: B vs each adaptive method
CONTRASTS = [
    ("SCI", "B", "C"), ("SCI", "B", "D"), ("SCI", "B", "E"), ("SCI", "B", "F"), ("SCI", "B", "RE"),
    ("NFC", "B", "C"), ("NFC", "B", "D"), ("NFC", "B", "E"), ("NFC", "B", "F"), ("NFC", "B", "RE"),
]

def load_rep_data(run_id, rep):
    art = os.path.join(RAW, run_id, "artifacts")
    path = os.path.join(art, f"RUN-rep{rep}.json")
    with open(path, encoding="utf8") as f:
        return json.load(f)

def get_per_query_recall(run_id, rep):
    arr = load_rep_data(run_id, rep)
    return [pq["recall10_qrels"] for pq in arr["per_query"]]

def paired_bootstrap_diff(rep_data1, rep_data2, n_boot=N_BOOT, seed=SEED):
    rnd = random.Random(seed)
    nq = len(rep_data1[0])
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
    diff = mean1 - mean2
    stat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)

def cohens_dz(rep_data1, rep_data2):
    mean1 = np.mean(np.array(rep_data1), axis=0)
    mean2 = np.mean(np.array(rep_data2), axis=0)
    diff = mean1 - mean2
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0

def main():
    # TEST run IDs
    test_runs = {
        "SCI": {
            "A": "C6-TEST-SCI-A-001", "B": "C6-TEST-SCI-B-001", "C": "C6-TEST-SCI-C-001",
            "D": "C6-TEST-SCI-D-001", "E": "C6-TEST-SCI-E-001", "F": "C6-TEST-SCI-F-001",
            "RE": "C6-TEST-SCI-RE-001",
        },
        "NFC": {
            "A": "C6-TEST-NFC-A-001", "B": "C6-TEST-NFC-B-001", "C": "C6-TEST-NFC-C-001",
            "D": "C6-TEST-NFC-D-001", "E": "C6-TEST-NFC-E-001", "F": "C6-TEST-NFC-F-001",
            "RE": "C6-TEST-NFC-RE-001",
        },
    }

    # Load per-query per-rep data
    data = {}
    for ds in ("SCI", "NFC"):
        for arm in ("A", "B", "C", "D", "E", "F", "RE"):
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

    # Holm correction across 10 primary contrasts
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

    # Candidate-set analysis: adaptive changed queries
    cand_results = {}
    for ds in ("SCI", "NFC"):
        for arm in ("C", "D", "E", "F", "RE"):
            run = test_runs[ds][arm]
            # Load metrics for aggregate stats
            metrics_path = os.path.join(RAW, run, "metrics.json")
            if os.path.exists(metrics_path):
                m = json.load(open(metrics_path))
                changed = m.get("adaptive_changed_queries", [0])[0]
                redist = m.get("adaptive_redistributed_queries", [0])[0]
                nq = m.get("test_queries", 300 if ds == "SCI" else 323)
                cand_results[f"{ds}_{arm}"] = {
                    "n_queries": nq,
                    "changed_count": changed,
                    "pct_changed": changed / nq,
                    "redistributed_count": redist,
                    "pct_redistributed": redist / nq,
                }
                print(f"Candidate change {ds} {arm}: {changed}/{nq} ({changed/nq:.1%}) redist={redist}")

    # Save full results
    out = {
        "primary_contrasts": results,
        "candidate_analysis": cand_results,
        "metadata": {
            "seed": SEED, "n_boot": N_BOOT, "n_reps": 5,
            "test_queries": {"SCI": 300, "NFC": 323},
            "protocol": "C6 adaptive retrieval allocation",
        },
    }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "c6-adaptive-retrieval", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "statistical_results.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir}/statistical_results.json")

if __name__ == "__main__":
    main()