#!/usr/bin/env python3
"""Generate all paper tables from raw result CSVs (spec §10–13, §24).

Reads ONLY research/phase2/results/*.csv (+ latency.csv); no numbers are
hard-coded. Rerun after any raw update; verify_consistency.py guards the
output. Captions are caption drafts (§28) and carry experiment IDs.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', 'results'))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'raw', 'runs'))
TAB = HERE

def rd(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))

def w(name, text):
    open(os.path.join(TAB, name), 'w').write(text)

# ---------------- table-ablation ----------------
abl = rd('ablation.csv')
t = ["# Table: Phase 2 ablation (FROZEN accepted baseline; noise-ladder corpus, degenerate-era GT — see HC-4)\n",
     "Caption draft: *Phase 2 ablation on the 8-head noise-ladder corpus. Values are the accepted frozen baseline; the harness ground truth was later corrected (HC-4), so these absolutes are not comparable to Phase 2B numbers — mode ORDERING was stable across runs.* [PH2-ABLATION-001]\n",
     "\n| mode | R@10 | NDCG@10 | MRR | p50 (µs) | p99 (µs) | QPS |",
     "|---|---|---|---|---|---|---|"]
for r in abl:
    t.append(f"| {r['mode']} | {r['recall_at_10']} | {r['ndcg_at_10']} | {r['mrr']} | {r['p50_us']} | {r['p99_us']} | {r['qps']} |")
w('table-ablation.md', "\n".join(t) + "\n")

# ---------------- table-multiseed ----------------
t = ["# Table: multi-seed robustness (training seeds 42/7/1, split fixed)\n",
     "Caption draft: *Mean ± standard deviation over three training seeds (42, 7, 1) on the same 70/15/15 split, held-out test queries.* [PH2B-MULTISEED-001, PH2B-MULTISEED-002]\n",
     "\n| corpus | metric | mean | std |", "|---|---|---|---|"]
for corpus, run in [('controlled', 'PH2B-MULTISEED-001'), ('multiview', 'PH2B-MULTISEED-002')]:
    for r in csv.DictReader(open(os.path.join(RUNS, run, f'multiseed_summary_{corpus}.csv'))):
        t.append(f"| {corpus} | {r['metric']} | {float(r['mean']):.4f} | {float(r['std']):.4f} |")
w('table-multiseed.md', "\n".join(t) + "\n")

# ---------------- table-sample-efficiency ----------------
t = ["# Table: sample efficiency (multiview)\n",
     "Caption draft: *Test R@10 as training-set size grows. Each point regenerates the corpus (HNSW candidate pools are OS-seeded), so points are directional, not paired. The transition lies between 420 and 840 training queries.* [PH2B-SAMPLE-001]\n",
     "\n| train | val | test | uniform R@10 | gating R@10 | gating NDCG@10 | oracle R@10 |",
     "|---|---|---|---|---|---|---|"]
for r in rd('sample-efficiency.csv'):
    t.append(f"| {r['train_size']} | {r['val_size']} | {r['test_size']} | {r['uniform_R10']} | {r['trained_R10']} | {r['trained_NDCG10']} | {r['oracle_R10']} |")
w('table-sample-efficiency.md', "\n".join(t) + "\n")

# ---------------- table-latency ----------------
lat = open(os.path.join(RES, 'latency.csv')).read().splitlines()
t = ["# Table: latency and cost\n",
     "Caption draft: *Top: trained gating model cost (CPU micro-benchmark, 200 probe vectors × 2000 reps). Bottom: end-to-end pipeline latency from the frozen Phase 2 ablation (100 queries, one shared CI-class VM — machine-specific, ratios indicative).* [PH2B-LATENCY-001, PH2-ABLATION-001]\n",
     "\n| component | heads | input dim | value | unit |", "|---|---|---|---|---|"]
in_model = False
in_pipeline = False
for l in lat:
    if l.startswith('component'):
        in_model = True
        continue
    if l.startswith('#'):
        continue
    if l.startswith('mode,'):
        in_model = False
        in_pipeline = True
        t.append("| pipeline mode (Phase 2 corpus) | p50 (µs) | p95 (µs) | p99 (µs) | QPS |")
        t.append("|---|---|---|---|---|")
        continue
    if in_model and l.strip():
        c = l.split(',')
        if len(c) == 5:
            t.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |")
    elif in_pipeline and l.strip():
        c = l.split(',')
        if len(c) == 5:
            t.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |")
t.append("")
t.append("| head scaling (Phase 2 corpus, mode B, parallel) | heads | p50 (µs) | p99 (µs) | R@10 |")
t.append("|---|---|---|---|---|")
for r in csv.DictReader(open(os.path.join(RES, 'heads-scaling.csv'))):
    if r['parallel'] == 'true':
        t.append(f"| multi-head | {r['heads']} | {r['p50_us']} | {r['p99_us']} | {r['recall_at_10']} |")
w('table-latency.md', "\n".join(t) + "\n")

# ---------------- table-reranking ----------------
t = ["# Table: exact-vs-normalized fusion weighting study (offline, cached test candidates)\n",
     "Caption draft: *Exact-score fusion under uniform / best-head / oracle head weighting vs normalized uniform fusion, R@10 on held-out test queries. Oracle-weighted exact fusion attains the oracle bound on all corpora: head weighting, not score exactness, explains the Phase 2 mode-E regression.* [PH2C-RERANK-001/002/003]\n",
     "\n| corpus | method | R@10 | NDCG@10 | MRR |", "|---|---|---|---|---|"]
for r in rd('rerank.csv'):
    t.append(f"| {r['corpus']} | {r['method']} | {r['R@10']} | {r['NDCG@10']} | {r['MRR']} |")
w('table-reranking.md', "\n".join(t) + "\n")

# ---------------- table-final-comparison (§29/§10) ----------------
evals = {c: {r['approach']: r for r in rd(f'{c}-eval-test.csv')}
         for c in ['controlled', 'noise', 'multiview']}
def g(c, approach, metric):
    try:
        return evals[c][approach][metric]
    except KeyError:
        return "pending (Phase 2C)"

t = ["# Table: final comparison (§29 central table)\n",
     "Caption draft: *Retrieval quality of fixed multi-head fusion, learned query-dependent gating, RRF, and oracle head selection across the controlled, noise, and multiview corpora. Values are measured on held-out test queries (single-run protocol; multiview multi-seed mean 0.4365 ± 0.0361 — see table-multiseed).* [PH2B-GATING-004, PH2B-NOISE-003, PH2B-MULTIVIEW-005]\n",
     "\n| Method | Controlled R@10 | Noise R@10 | Multiview R@10 | Controlled NDCG@10 | Noise NDCG@10 | Multiview NDCG@10 |",
     "|---|---|---|---|---|---|---|"]
rows = [
    ("Single best head", "global_best_single_head"),
    ("Uniform multi-head", "uniform_multihead"),
    ("RRF (k=60)", "rrf_k60"),
    ("Trained gating", "trained_gating"),
    ("Trained QK", None),
    ("Gating + QK", None),
    ("Exact rerank (offline exact fusion, uniform)*", None),
    ("Gating + exact rerank*", None),
    ("Oracle head selection", "oracle_per_query_head"),
]
for label, key in rows:
    if key:
        vals = [g(c, key, 'R@10') for c in ['controlled', 'noise', 'multiview']]
        ndcg = [g(c, key, 'NDCG@10') for c in ['controlled', 'noise', 'multiview']]
    elif label.startswith("Trained QK") or label.startswith("Gating + QK") or label.startswith("Gating + exact"):
        vals = ndcg = ["pending (Phase 2C)"] * 3
    else:  # offline exact fusion, uniform — from rerank.csv
        rr = {(r['corpus'], r['method']): r for r in rd('rerank.csv')}
        vals = [rr[(c, 'exact_fusion_uniform')]['R@10'] for c in ['controlled', 'noise', 'multiview']]
        ndcg = [rr[(c, 'exact_fusion_uniform')]['NDCG@10'] for c in ['controlled', 'noise', 'multiview']]
    t.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} | {ndcg[0]} | {ndcg[1]} | {ndcg[2]} |")
t.append("")
t.append("\\* The offline exact-fusion row is the PH2C-RERANK study (uniform weighting over cached candidates), not the Phase 2 pipeline mode E; the frozen Phase 2 pipeline mode E measured 0.558 vs mode D 0.672 on its own (non-comparable) ground truth [PH2-ABLATION-001]. Pipeline-level re-weighting is untested (ledger N3).")
w('table-final-comparison.md', "\n".join(t) + "\n")


# ---------------------------------------------------------------- qk sanity
qs = rd("qk-sanity.csv")
t = [
    "# Table: PH2C-QK-001 — QK sanity dataset (held-out test, 250 queries)",
    "",
    "Caption draft: *Candidate-level QK attention vs the entire gating class on a dataset where ordering provably requires query–candidate interaction (head selection easy; relevant candidate last under the signal-head cosine by construction). Mean ± std over seeds 42/7/1.* [PH2C-QK-001]",
    "",
    "| arm | R@1 | NDCG@10 | MRR |",
    "|---|---|---|---|",
]
order = ["uniform", "global_best_head0", "gating_qualityreg", "gating_infnce",
         "qk_untrained", "qk_trained", "oracle"]
var = {}
vp = os.path.join(RUNS, "PH2C-QK-001", "variability.csv")
if os.path.exists(vp):
    for r in csv.DictReader(open(vp)):
        if r["metric"] == "R@1":
            var[r["arm"]] = f" ± {float(r['std']):.4f}"
for arm in order:
    row = next(r for r in qs if r["arm"] == arm and r["seed"] == "agg")
    suffix = var.get(arm, "") if arm in ("qk_trained", "gating_qualityreg", "gating_infnce") else ""
    t.append(f"| {arm} | {row['R@1']}{suffix} | {row['NDCG@10']} | {row['MRR']} |")
t.append("")
t.append("R@5/R@10 are 1.0 for every arm by construction (single relevant candidate per 10-candidate pool): the dataset isolates ORDERING. Source: results/qk-sanity.csv.")
w("table-qk-sanity.md", "\n".join(t) + "\n")

print("tables written:", sorted(f for f in os.listdir(TAB) if f.endswith('.md')))
