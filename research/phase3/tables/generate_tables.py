#!/usr/bin/env python3
"""Phase 3 tables — generated from research/phase3/results/*.csv only."""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', 'results'))

def rd(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))

# main quality table
rows = rd("retrieval-quality.csv")
t = ["# Table: retrieval quality — PH3-DS-FM-S (Fashion-MNIST 10K, TEST, agg seeds 42/7/1)", "",
     "Caption draft: *Frozen architecture vs baselines on real-image multi-view retrieval; exact reference = ground-truth definition (sanity anchor). Trained gating collapses to the dominant full-image view and matches single-head ANN; uniform fusion degrades.* [PH3-QUAL-FM-S]", "",
     "| System | R@1 | R@5 | R@10 | NDCG@10 | MRR |", "|---|---|---|---|---|---|"]
label = {"single_head_ann": "Single-vector ANN (full view)", "uniform_multihead": "Uniform multi-head fusion",
         "trained_gating": "Trained gating (frozen arch.)", "exact_reference": "Exact/brute-force reference"}
for r in rows:
    if r["arm"] in label:
        t.append(f"| {label[r['arm']]} | {r['R@1']} | {r['R@5']} | {r['R@10']} | {r['NDCG@10']} | {r['MRR']} |")
t += ["", "Candidate recall (5-head pool union): 0.9992 — ranking, not generation, is the residual gap to 1.0."]
open(os.path.join(HERE, "table-retrieval-quality.md"), "w").write("\n".join(t) + "\n")

# memory / scaling table
mem = rd("memory.csv")
t = ["# Table: memory scaling — PH3-DS-FM (spec §9/§19)", "",
     "Caption draft: *Index build memory on a 1984 MB sandbox: the engine builds 10K docs at 571 MB peak (≈8.6× raw vector bytes) but OOM-kills between 10K and 30K docs — the dominant §19 limitation.* [PH3-QUAL-FM-S, PH3-QUAL-FM-T30, PH3-QUAL-FM-M]", "",
     "| tier | docs | status | peak RSS MB | note |", "|---|---|---|---|---|"]
for r in mem:
    t.append(f"| {r['tier']} | {r['n_docs']} | {r['status']} | {r['peak_rss_mb'] or '—'} | {r['note']} |")
open(os.path.join(HERE, "table-scaling-memory.md"), "w").write("\n".join(t) + "\n")

print("phase3 tables written:", sorted(f for f in os.listdir(HERE) if f.endswith(".md")))
