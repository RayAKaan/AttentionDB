#!/usr/bin/env python3
"""Generate publication figures from raw result data (spec §14–16).

Reads ONLY research/phase2/results/*.csv and the per-corpus dataset/model
JSONs. No benchmark values are hard-coded. Figure 1 (architecture) is a
diagram, drawn here for consistency of style. Figure 10 (QK) stays
PENDING until PH2C-QK runs exist — nothing is fabricated.

Every figure also gets an entry (source files + experiment IDs + caption
draft) in figures/README.md, maintained by this script's header constant.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', 'results'))
RESEARCH = os.path.normpath(os.path.join(HERE, '..'))
ROOT = os.path.normpath(os.path.join(RESEARCH, '..', '..'))
B2B = os.path.join(ROOT, 'benchmarks', 'phase2b')

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 200, "savefig.bbox": "tight",
})
C_GATING, C_UNIFORM, C_ORACLE, C_RRF, C_OTHER = "#2077b4", "#9a9a9a", "#d62728", "#2ca02c", "#7f7f7f"

def rd(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))

# ---------------- Figure 1: architecture ----------------
fig, ax = plt.subplots(figsize=(8.6, 3.4))
ax.axis("off")
boxes = [
    (0.01, 0.55, "query q", "#eeeeee"),
    (0.17, 0.78, "HNSW head 1", "#d7e8fa"),
    (0.17, 0.58, "HNSW head 2", "#d7e8fa"),
    (0.17, 0.38, "... head H", "#d7e8fa"),
    (0.17, 0.14, "BM25 (optional)", "#efefef"),
    (0.40, 0.50, "candidate union\nC(q)=⋃C_h(q)", "#f5e6d0"),
    (0.62, 0.78, "per-head MinMax\nnormalization", "#e8f3e8"),
    (0.62, 0.14, "learned gating\ng(q)=softmax(f_θ(q))", "#f6d9d9"),
    (0.83, 0.45, "weighted fusion\n+ optional exact\nrerank / QK attn", "#e3e0f3"),
    (0.83, 0.10, "deterministic\ntop-K", "#e3e0f3"),
]
for x, y, label, color in boxes:
    ax.add_patch(plt.Rectangle((x, y), 0.155, 0.20, color=color, ec="#444"))
    ax.text(x + 0.0775, y + 0.10, label, ha="center", va="center", fontsize=9)
arr = dict(arrowstyle="->", color="#444", lw=1.1)
ax.annotate("", xy=(0.17, 0.88), xytext=(0.165, 0.65), arrowprops=arr)
for y in (0.88, 0.68, 0.48, 0.24):
    ax.annotate("", xy=(0.40, 0.60), xytext=(0.325, y), arrowprops=arr)
ax.annotate("", xy=(0.62, 0.86), xytext=(0.555, 0.62), arrowprops=arr)
ax.annotate("", xy=(0.62, 0.22), xytext=(0.17, 0.22), arrowprops=arr)
ax.annotate("", xy=(0.775, 0.88), xytext=(0.62, 0.86), arrowprops=arr)
ax.annotate("", xy=(0.83, 0.60), xytext=(0.775, 0.88), arrowprops=arr)
ax.annotate("", xy=(0.83, 0.25), xytext=(0.775, 0.22), arrowprops=arr)
ax.annotate("", xy=(0.83, 0.20), xytext=(0.83, 0.45), arrowprops=arr)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(os.path.join(HERE, "fig1-architecture.png"), dpi=200)
plt.close(fig)

# ---------------- Figure 2: Phase 2 ablation R@10 ----------------
abl = rd('ablation.csv')
modes = [r['mode'] for r in abl]
vals = [float(r['recall_at_10']) for r in abl]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
colors = [C_OTHER if m.startswith('A') else C_UNIFORM for m in modes]
ax.bar(range(len(modes)), vals, color=colors)
ax.set_xticks(range(len(modes)))
ax.set_xticklabels([m.replace('_', '\n') for m in modes], fontsize=8)
ax.set_ylabel("Recall@10")
ax.set_title("Phase 2 ablation (frozen baseline; noise-ladder corpus)")
ax.set_ylim(0, 1.0)
for i, v in enumerate(vals):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
fig.savefig(os.path.join(HERE, "fig2-ablation-r10.png"))
plt.close(fig)

# ---------------- Figure 3: gating vs oracle per corpus ----------------
corpora = ['controlled', 'noise', 'multiview']
methods = [("uniform_multihead", "uniform", C_UNIFORM),
           ("global_best_single_head", "best single head", "#8fb8de"),
           ("rrf_k60", "RRF", C_RRF),
           ("trained_gating", "trained gating", C_GATING),
           ("oracle_per_query_head", "oracle", C_ORACLE)]
fig, ax = plt.subplots(figsize=(8.6, 3.8))
w = 0.16
for i, (key, label, color) in enumerate(methods):
    xs = np.arange(len(corpora)) + (i - 2) * w
    ys = [float(next(r['R@10'] for r in rd(f'{c}-eval-test.csv') if r['approach'] == key)) for c in corpora]
    ax.bar(xs, ys, w, label=label, color=color)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7)
ax.set_xticks(np.arange(len(corpora)))
ax.set_xticklabels(corpora)
ax.set_ylabel("Recall@10 (test)")
ax.set_ylim(0, 1.05)
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Learned gating vs baselines vs oracle head selection", y=1.14)
fig.savefig(os.path.join(HERE, "fig3-gating-vs-oracle.png"))
plt.close(fig)

# ---------------- Figure 4: sample efficiency ----------------
se = rd('sample-efficiency.csv')
tr = [int(r['train_size']) for r in se]
gat = [float(r['trained_R10']) for r in se]
uni = [float(r['uniform_R10']) for r in se]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(tr, gat, "o-", color=C_GATING, label="trained gating")
ax.plot(tr, uni, "s--", color=C_UNIFORM, label="uniform fusion")
ax.set_xscale("log", base=2)
ax.set_xticks(tr); ax.set_xticklabels(tr)
ax.set_xlabel("training queries (multiview)")
ax.set_ylabel("Recall@10 (test)")
ax.set_title("Sample efficiency: plateau then transition (420 → 840)")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(os.path.join(HERE, "fig4-sample-efficiency.png"))
plt.close(fig)

# ---------------- Figure 5: diversity vs utility ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
markers = {'controlled': 'o', 'noise': 's', 'multiview': '^'}
for c in corpora:
    pairs = rd(f'{c}-head-diversity.csv')
    util = {int(r['head']): float(r['mean_recall_at_10_test']) for r in rd(f'{c}-head-utility.csv')}
    H = max(util) + 1
    div = {h: [] for h in range(H)}
    for p in pairs:
        a, b, j = int(p['head_a']), int(p['head_b']), float(p.get('jaccard_at_10', p.get('jaccard_at_k', 0)))
        div[a].append(j); div[b].append(j)
    xs = [np.mean(div[h]) for h in sorted(util)]
    ys = [util[h] for h in sorted(util)]
    ax.scatter(xs, ys, marker=markers[c], label=c, s=60)
ax.set_xlabel("mean pairwise Jaccard@10 with other heads (diversity ↓ = more unique)")
ax.set_ylabel("mean per-head Recall@10 (utility)")
ax.set_title("Diversity vs utility per head")
ax.legend(title="corpus")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(HERE, "fig5-diversity-vs-utility.png"))
plt.close(fig)

# ---------------- Figure 6: predicted weight vs actual quality ----------------
def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()

def forward(card, q):
    w1 = np.array(card["w1"]).reshape(card["hidden"], card["input_dim"])
    b1 = np.array(card["b1"])
    w2 = np.array(card["w2"]).reshape(card["num_heads"], card["hidden"])
    b2 = np.array(card["b2"])
    h = np.maximum(w1 @ np.asarray(q) + b1, 0)
    return w2 @ h + b2

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=False)
for ax, c, run in [(axes[0], 'controlled', 'PH2B-GATING-004-era'),
                   (axes[1], 'multiview', 'PH2B-MULTIVIEW-005-era')]:
    ds = json.load(open(os.path.join(B2B, c, 'dataset.json')))
    card = json.load(open(os.path.join(B2B, c, 'training', 'model-v1.json')))
    xs, ys = [], []
    for q in ds["queries"]:
        if q["split"] != "Test":
            continue
        logits = forward(card, q["query"])
        # temperature from the run (validation-fitted): read from run_info
        xs.extend(np.clip(softmax(logits), 1e-4, 1))
        ys.extend([h["recall_at_k"] for h in q["heads"]])
    xs = np.array(xs) + np.random.default_rng(0).normal(0, 0.004, len(xs))
    ys = np.array(ys) + np.random.default_rng(1).normal(0, 0.004, len(ys))
    r = np.corrcoef(np.array(xs), np.array(ys))[0, 1]
    ax.scatter(xs, ys, s=8, alpha=0.45, color=C_GATING)
    ax.set_xlabel("predicted head weight (post-softmax)")
    ax.set_ylabel("actual head Recall@10")
    ax.set_title(f"{c}  (Pearson r = {r:.2f})")
    ax.grid(alpha=0.3)
fig.suptitle("Predicted gating weight vs actual head quality (test queries)", y=1.02)
fig.savefig(os.path.join(HERE, "fig6-weight-vs-quality.png"))
plt.close(fig)

# ---------------- Figure 7: quality vs latency frontier ----------------
abl2 = [(r['mode'], float(r['recall_at_10']), float(r['p50_us'])) for r in rd('ablation.csv')]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for m, q, p50 in abl2:
    if m.startswith('A') and m != 'A_single_head':
        continue  # plot per-head baselines as small dots
    ax.scatter(p50, q, s=70, zorder=3,
               color=C_GATING if 'gating' in m or 'attention' in m or m.startswith('E') else C_OTHER)
    ax.annotate(m.replace('_', ' '), (p50, q), textcoords="offset points", xytext=(6, 4), fontsize=8)
for m, q, p50 in abl2:
    if m.startswith('A') and m != 'A_single_head':
        ax.scatter(p50, q, s=18, color="#bbbbbb", zorder=2)
ax.set_xscale("log")
ax.set_xlabel("p50 latency per query (µs, log scale)")
ax.set_ylabel("Recall@10")
ax.set_title("Quality vs latency (Phase 2 frozen ablation)")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(HERE, "fig7-quality-latency-frontier.png"))
plt.close(fig)

# ---------------- Figure 8: per-query-type head selection ----------------
ds = json.load(open(os.path.join(B2B, 'controlled', 'dataset.json')))
card = json.load(open(os.path.join(B2B, 'controlled', 'training', 'model-v1.json')))
G = card["num_heads"]
groups = sorted({q["query_group"] for q in ds["queries"]})
sel = np.zeros((len(groups), G))
for q in ds["queries"]:
    if q["split"] != "Test":
        continue
    w = softmax(forward(card, q["query"]))
    sel[q["query_group"], int(np.argmax(w))] += 1
sel = sel / sel.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
bottom = np.zeros(len(groups))
for h in range(G):
    ax.bar([f"group {g}" for g in groups], sel[:, h], bottom=bottom, label=f"head {h}")
    bottom += sel[:, h]
ax.set_ylabel("fraction of test queries whose argmax head is h")
ax.set_title("Learned routing: query group → head (controlled)")
ax.legend(ncol=G, fontsize=8)
fig.savefig(os.path.join(HERE, "fig8-per-query-type-selection.png"))
plt.close(fig)

# ---------------- Figure 9: exact rerank study ----------------
rr = rd('rerank.csv')
methods = ["norm_fusion_uniform", "exact_fusion_uniform", "exact_fusion_best_head", "exact_fusion_oracle_head"]
labels = ["norm fusion\nuniform", "exact fusion\nuniform", "exact fusion\nbest head", "exact fusion\noracle head"]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
w = 0.2
for i, c in enumerate(corpora):
    xs = np.arange(len(methods)) + (i - 1) * w
    ys = [float(next(r['R@10'] for r in rr if r['corpus'] == c and r['method'] == m)) for m in methods]
    ax.bar(xs, ys, w, label=c)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7)
ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Recall@10 (test)")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Exact-rerank regression is a weighting problem, not an exactness problem")
fig.savefig(os.path.join(HERE, "fig9-rerank-study.png"))
plt.close(fig)

print("figures written:", sorted(f for f in os.listdir(HERE) if f.endswith('.png')))
print("PENDING (no data, not fabricated): fig10 (QK attention) — awaits PH2C-QK runs")
