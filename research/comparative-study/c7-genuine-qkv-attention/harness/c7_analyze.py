#!/usr/bin/env python3
"""C7 statistical analysis — paired bootstrap, Wilcoxon, Holm, Cohen's dz.

Per protocol §9: unit = query; per-query metric = mean of 5 reps; primary
metric nDCG@10 (secondary: recall@10, MRR); paired bootstrap 10k (seed
20260925); Wilcoxon signed-rank; Holm across B vs C/D/E/F x 2 datasets
= 8 primary contrasts; Cohen's dz; |Δ| < 0.01 -> negligible.

Candidate/attention observables (union identity, mean entropy, Spearman
attention<->final) are summarized for the reports.
"""

import json
import os
import random

import numpy as np
from scipy import stats

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw"))
SEED = 20260925
N_BOOT = 10000
K = 10

CONTRASTS = [
    ("SCI", "B", "C"), ("SCI", "B", "D"), ("SCI", "B", "E"), ("SCI", "B", "F"),
    ("NFC", "B", "C"), ("NFC", "B", "D"), ("NFC", "B", "E"), ("NFC", "B", "F"),
]

TEST_RUNS = {
    "SCI": {"A": "C7-TEST-SCI-A-001", "B": "C7-TEST-SCI-B-001", "C": "C7-TEST-SCI-C-001",
            "D": "C7-TEST-SCI-D-001", "E": "C7-TEST-SCI-E-001", "F": "C7-TEST-SCI-F-001"},
    "NFC": {"A": "C7-TEST-NFC-A-001", "B": "C7-TEST-NFC-B-001", "C": "C7-TEST-NFC-C-001",
            "D": "C7-TEST-NFC-D-001", "E": "C7-TEST-NFC-E-001", "F": "C7-TEST-NFC-F-001"},
}


def load_arm(run_id, rep):
    art = os.path.join(RAW, run_id, "artifacts")
    arm = run_id.split("-")[3]  # C7-TEST-<DS>-<ARM>-001 -> <ARM>
    with open(os.path.join(art, f"ARM-{arm}-rep{rep}.json"), encoding="utf8") as f:
        return json.load(f)


def rep_count_for(run_id):
    metrics = os.path.join(RAW, run_id, "metrics.json")
    with open(metrics, encoding="utf8") as f:
        return int(json.load(f)["rep_count"])


def per_query_metrics(arm):
    out = []
    for pq in arm["per_query"]:
        rel = set(pq.get("relevant_ids") or [])
        top = [(r["row"], r["final"]) for r in sorted(
            pq["union_ledger"], key=lambda r: (-r["final"], r["rank"]))][:K]
        if not top:
            top = []
        n_rel = len(rel)
        hits = [row for row, _ in top if row in rel]
        recall = len(hits) / n_rel if n_rel else 0.0
        mrr = 0.0
        for i, (row, _) in enumerate(top):
            if row in rel:
                mrr = 1.0 / (i + 1)
                break
        out.append({
            "query_row": pq["query_row"],
            "recall10_qrels": pq["recall10_qrels"],
            "ndcg10_qrels": pq["ndcg10_qrels"],
            "mrr": mrr,
        })
    return out


def paired_bootstrap_diff(d1, d2, n_boot=N_BOOT, seed=SEED, metric="ndcg10_qrels"):
    a1 = np.array([[q[metric] for q in rep] for rep in d1])
    a2 = np.array([[q[metric] for q in rep] for rep in d2])
    m1 = a1.mean(axis=0)
    m2 = a2.mean(axis=0)
    obs = float(np.mean(m1 - m2))
    rnd = random.Random(seed)
    nq = len(m1)
    boots = []
    for _ in range(n_boot):
        idx = [rnd.randrange(nq) for _ in range(nq)]
        boots.append(float(np.mean(m1[idx] - m2[idx])))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return obs, lo, hi, np.array(boots)


def wilcoxon(d1, d2, metric="ndcg10_qrels"):
    m1 = np.mean([[q[metric] for q in rep] for rep in d1], axis=0)
    m2 = np.mean([[q[metric] for q in rep] for rep in d2], axis=0)
    stat, p = stats.wilcoxon(m1 - m2, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


def cohens_dz(d1, d2, metric="ndcg10_qrels"):
    m1 = np.mean([[q[metric] for q in rep] for rep in d1], axis=0)
    m2 = np.mean([[q[metric] for q in rep] for rep in d2], axis=0)
    diff = m1 - m2
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def main():
    data = {}
    n_reps = 5
    for ds in ("SCI", "NFC"):
        for arm in ("A", "B", "C", "D", "E", "F"):
            run = TEST_RUNS[ds][arm]
            n_reps = rep_count_for(run)
            reps = [per_query_metrics(load_arm(run, r)) for r in range(1, n_reps + 1)]
            data[(ds, arm)] = reps

    results = {}
    for ds, a, b in CONTRASTS:
        for metric in ("ndcg10_qrels", "recall10_qrels", "mrr"):
            d1, d2 = data[(ds, a)], data[(ds, b)]
            obs, lo, hi, boots = paired_bootstrap_diff(d1, d2, metric=metric)
            w_stat, w_p = wilcoxon(d1, d2, metric=metric)
            dz = cohens_dz(d1, d2, metric=metric)
            key = f"{ds}_{a}{b}"
            row = results.setdefault(key, {"dataset": ds, "contrast": f"{a}vs{b}",
                                           "base": a, "target": b})
            row[metric] = {"mean_diff": obs, "ci95": [lo, hi],
                           "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                           "cohens_dz": dz, "negligible": abs(obs) < 0.01}
            if metric == "ndcg10_qrels":
                print(f"{key} nDCG@10: mean={obs:.5f} CI95=[{lo:.5f},{hi:.5f}] "
                      f"p={w_p:.5f} dz={dz:.3f} negligible={abs(obs)<0.01}")

    pvals = sorted(((k, row["ndcg10_qrels"]["wilcoxon_p"]) for k, row in results.items()),
                   key=lambda x: x[1])
    m = len(pvals)
    for i, (k, p) in enumerate(pvals):
        alpha = 0.05 / (m - i)
        results[k]["ndcg10_qrels"]["holm_alpha"] = alpha
        results[k]["ndcg10_qrels"]["holm_reject"] = p < alpha

    # attention/candidate observables per arm (mean over reps)
    obs_table = {}
    for ds in ("SCI", "NFC"):
        for arm in ("B", "C", "D", "E", "F"):
            run = TEST_RUNS[ds][arm]
            ents, sps = [], []
            for r in range(1, n_reps + 1):
                arm_data = load_arm(run, r)
                agg = arm_data.get("c7_aggregate")
                if agg:
                    ents.append(agg.get("mean_entropy_over_queries"))
                    sps.append(agg.get("mean_spearman_attention_final"))
            obs_table[f"{ds}_{arm}"] = {
                "mean_entropy": float(np.mean([e for e in ents if e is not None])) if ents else None,
                "mean_spearman_attention_final": float(np.mean([s for s in sps if s is not None])) if sps else None,
            }

    # summary means for reports
    means = {}
    for ds in ("SCI", "NFC"):
        for arm in ("A", "B", "C", "D", "E", "F"):
            reps = data[(ds, arm)]
            for metric in ("ndcg10_qrels", "recall10_qrels", "mrr"):
                means.setdefault(f"{ds}_{arm}", {})[metric] = float(
                    np.mean([[q[metric] for q in rep] for rep in reps]))

    out = {
        "primary_contrasts": results,
        "means": means,
        "observables": obs_table,
        "metadata": {"seed": SEED, "n_boot": N_BOOT, "n_reps": n_reps, "k": K,
                     "primary_metric": "ndcg10_qrels", "secondary": ["recall10_qrels", "mrr"],
                     "protocol": "C7 candidate-level QKV attention",
                     "units": "queries", "contrast_set": "B vs C/D/E/F x 2 datasets (8)"}
    }
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "statistical_results.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir}/statistical_results.json")


if __name__ == "__main__":
    main()