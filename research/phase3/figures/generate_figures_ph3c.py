#!/usr/bin/env python3
"""PH3C figures 1–9 — pure-stdlib SVG from canonical results CSVs."""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))


def rd(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(name, title, body, caption, source, exps, w=880, h=560):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
             f'<rect width="{w}" height="{h}" fill="white"/>',
             f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{esc(title)}</text>']
    parts += body
    for i in range(0, len(caption), 108):
        parts.append(f'<text x="60" y="{h-64+i*16}" font-size="11.5" fill="#555" font-style="italic">{esc(caption[i:i+108])}</text>')
    parts.append(f'<text x="60" y="{h-14}" font-size="10.5" fill="#888">Source: {esc(source)} | Experiments: {esc(exps)}</text></svg>')
    open(os.path.join(HERE, name), "w").write("\n".join(parts) + "\n")


def axes(x0, y0, w, h, ymax, ylabel, xlabels, yticks=5):
    b = [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#333"/>',
         f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#333"/>']
    for i in range(yticks + 1):
        y = y0 + h * i / yticks
        b.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#eee"/>')
        b.append(f'<text x="{x0-6}" y="{y+4}" text-anchor="end" font-size="11" fill="#666">{ymax*(1-i/yticks):.0f}</text>')
    b.append(f'<text x="{x0-34}" y="{y0+h/2}" font-size="12" fill="#333" transform="rotate(-90 {x0-34} {y0+h/2})">{esc(ylabel)}</text>')
    for i, xl in enumerate(xlabels):
        b.append(f'<text x="{x0+w*(i+0.5)/len(xlabels):.0f}" y="{y0+h+18}" text-anchor="middle" font-size="12">{esc(xl)}</text>')
    return b


def bars(x0, y0, w, h, series, ymax, ngroups):
    palette = ["#4056b0", "#b04040", "#2e8b57", "#b08840", "#7040a0", "#555"]
    out = []
    ns = len(series)
    bw = w / (ngroups * ns + ngroups) * 0.8
    for gi in range(ngroups):
        for si, (sname, vals) in enumerate(series.items()):
            v = vals[gi]
            if v is None:
                continue
            x = x0 + (w / ngroups) * gi + (w / ngroups) * (si + 0.5) / ns - bw / 2
            bh = h * v / ymax
            out.append(f'<rect x="{x:.1f}" y="{y0+h-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{palette[si%6]}"/>')
            out.append(f'<text x="{x+bw/2:.1f}" y="{y0+h-bh-4:.1f}" text-anchor="middle" font-size="9.5" fill="#333">{v:.3f}</text>')
    return out


def legend(x, y, names, palette_offset=0):
    palette = ["#4056b0", "#b04040", "#2e8b57", "#b08840", "#7040a0", "#555"]
    out = []
    for i, n in enumerate(names):
        out.append(f'<rect x="{x}" y="{y+i*17}" width="12" height="12" fill="{palette[(i+palette_offset)%6]}"/>')
        out.append(f'<text x="{x+17}" y="{y+10+i*17}" font-size="12">{esc(n)}</text>')
    return out


mem = rd("memory-scaling.csv")
comp = rd("memory-components.csv")
hq = rd("head-scaling-quality.csv")
hl = rd("head-scaling-latency.csv")
hm = rd("head-scaling-memory.csv")
bud = rd("candidate-budget.csv")
dec = rd("candidate-decomposition.csv")
repro = rd("reproduction.csv")

# Fig 1: memory vs corpus size (3 heads, dim 512)
def _peak(r):
    return float(r["peak_rss_mb"] or r["last_peak_mb"] or 0)
pts = [(int(r["n_docs"]), _peak(r), r["status"]) for r in mem
       if r["heads"] == "3" and r["dim"] == "512" and r["experiment"] != "PH3C-MEM-002-BUDGET-2H-20K"]
pts.sort()
W, H, x0, y0, pw, ph = 880, 560, 90, 70, 700, 330
ymax = 1600
b = axes(x0, y0, pw, ph, ymax, "peak RSS MB", [f"{p[0]}K" for p in pts])
for i, (docs, rss, st) in enumerate(pts):
    cx = x0 + pw * (i + 0.5) / len(pts)
    if st == "COMPLETED":
        bh = ph * rss / ymax
        b.append(f'<rect x="{cx-28:.0f}" y="{y0+ph-bh:.1f}" width="56" height="{bh:.1f}" fill="#4056b0"/>')
        b.append(f'<text x="{cx:.0f}" y="{y0+ph-bh-5:.1f}" text-anchor="middle" font-size="11">{rss:.0f}</text>')
    else:
        lr = float(next(r["last_peak_mb"] for r in mem if r["experiment"] == "PH3C-MEM-002-C20K"))
        bh = ph * lr / ymax
        b.append(f'<rect x="{cx-28:.0f}" y="{y0+ph-bh:.1f}" width="56" height="{bh:.1f}" fill="#b04040" fill-opacity="0.55" stroke="#b04040" stroke-dasharray="4,3"/>')
        b.append(f'<text x="{cx:.0f}" y="{y0+ph-bh-5:.1f}" text-anchor="middle" font-size="11" fill="#b04040">OOM≥{lr:.0f}</text>')
b += legend(x0 + 420, y0 + 10, ["peak RSS (completed)", "last peak before OOM"])
svg("figure-1-memory-vs-corpus.svg", "Peak build memory vs corpus size (3 heads, dim 512)", b,
    "Caption (auto): RSS grows ~64 MB per 1K docs (≈10.9x raw vector rate); the 2 GB wall sits between 15K "
    "(1182 MB) and 20K docs (OOM, last peak 1410 MB). Dashed red = killed run (§7 preserved).",
    "results/memory-scaling.csv", "PH3C-MEM-001, PH3C-MEM-002-C*")

# Fig 2: memory vs head count (10K, dim 512 + 256 side)
series = {"dim 512": [], "dim 256": []}
for d, key in (("512", "dim 512"), ("256", "dim 256")):
    for hcount in ("1", "2", "3", "4", "8"):
        row = next((r for r in mem if r["heads"] == hcount and r["dim"] == d
                    and r["n_docs"] == "10000" and r["status"] == "COMPLETED"), None)
        series[key].append(float(row["peak_rss_mb"]) if row else None)
b = axes(x0, y0, pw, ph, 1400, "peak RSS MB", ["1 head", "2", "3", "4", "8"])
b += bars(x0, y0, pw, ph, series, 1400, 5)
b += legend(x0 + 500, y0 + 10, list(series))
svg("figure-2-memory-vs-heads.svg", "Peak memory vs head count (10K docs)", b,
    "Caption (auto): memory is approximately LINEAR in head count (~245-260 MB/head at dim 512, ~125-155 "
    "MB/head at dim 256); 8x10Kx512 OOMs (preserved run) while 8x5K completes at 960 MB.",
    "results/memory-scaling.csv", "PH3C-MEM-002-H*")

# Fig 3: memory vs dimension (3 heads, 10K)
series = {"peak RSS": [], "raw vectors": []}
dims = ["128", "256", "512"]
for d in dims:
    row = next(r for r in mem if r["dim"] == d and r["heads"] == "3" and r["n_docs"] == "10000")
    series["peak RSS"].append(float(row["peak_rss_mb"]))
    series["raw vectors"].append(float(row["raw_vector_mb"]))
b = axes(x0, y0, pw, ph, 1000, "MB", ["dim 128", "dim 256", "dim 512"])
b += bars(x0, y0, pw, ph, series, 1000, 3)
b += legend(x0 + 560, y0 + 10, list(series))
svg("figure-3-memory-vs-dim.svg", "Memory vs embedding dimension (3 heads, 10K docs)", b,
    "Caption (auto): peak RSS scales roughly linearly with dimension (marginal ~1.3-2.6 MB per dim-step per "
    "10K docs) on top of a ~130-160 MB fixed process/engine base — the multiplier falls as dim grows.",
    "results/memory-scaling.csv", "PH3C-MEM-002-D128, -D256, PH3C-MEM-001")

# Fig 4: R@10 vs head count (5K ladder, K=100)
series = {"global-best": [], "uniform": [], "gating": [], "oracle": []}
ladder = [("1", "PH3C-HEAD-001-H1-S5K"), ("2", "PH3C-HEAD-001-H2-S5K"),
          ("4", "PH3C-HEAD-001-H4-S5K"), ("8", "PH3C-HEAD-001-H8-S5K")]
for _, run in ladder:
    for arm, key in (("global_best_single", "global-best"), ("uniform_multihead", "uniform"),
                     ("trained_gating", "gating"), ("oracle_head_empirical", "oracle")):
        row = next(r for r in hq if r["experiment"] == run and r["arm"] == arm and r["seed"] == "agg")
        series[key].append(float(row["R@10"]))
b = axes(x0, y0, pw, ph, 1.0, "R@10 (K=100)", [f"{h} heads" for h, _ in ladder])
b += bars(x0, y0, pw, ph, series, 1.0, 4)
b += legend(x0 + 560, y0 + 10, list(series))
svg("figure-4-quality-vs-heads.svg", "Retrieval quality vs head count (5K docs, size-matched)", b,
    "Caption (auto): gating gains are large to 4 heads (0.519→0.658→0.715) and SATURATE at 8 (0.709, within "
    "seed noise ±0.023); single-view cannot exploit views (H1 arms identical by construction).",
    "results/head-scaling-quality.csv", "PH3C-HEAD-001-*-S5K")

# Fig 5: latency vs head count
series = {"serial": [], "parallel (2 workers)": []}
for _, run in ladder:
    for mode, key in (("serial", "serial"), ("parallel2", "parallel (2 workers)")):
        row = next(r for r in hl if r["experiment"] == run and r["mode"] == mode)
        series[key].append(float(row["p50_us"]) / 1000.0)
b = axes(x0, y0, pw, ph, 25.0, "p50 latency ms", [f"{h} heads" for h, _ in ladder])
b += bars(x0, y0, pw, ph, series, 25.0, 4)
b += legend(x0 + 560, y0 + 10, list(series))
svg("figure-5-latency-vs-heads.svg", "Query latency vs head count (p50, K=100 pools)", b,
    "Caption (auto): serial p50 grows super-linearly at 8 heads (22.1 ms); 2-worker parallelism halves it "
    "(10.9 ms) on the 2-CPU sandbox; parallelism HURTS at 1 head (thread overhead).",
    "results/head-scaling-latency.csv", "PH3C-HEAD-001-*-S5K")

# Fig 6: quality vs latency (ladder, gating, serial)
b = [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#333"/>',
     f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#333"/>']
lat = {h: next(float(r["p50_us"]) for r in hl if r["experiment"] == run and r["mode"] == "serial")
       for h, run in ladder}
qual = {h: next(float(r["R@10"]) for r in hq if r["experiment"] == run and r["arm"] == "trained_gating"
                and r["seed"] == "agg") for h, run in ladder}
xmax = max(lat.values()) * 1.1
palette = ["#4056b0", "#b04040", "#2e8b57", "#b08840"]
for i, (h, _) in enumerate(ladder):
    px = x0 + pw * lat[h] / xmax
    py = y0 + ph * (1 - qual[h])
    b.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="7" fill="{palette[i]}"/>')
    b.append(f'<text x="{px+10:.0f}" y="{py-8:.0f}" font-size="12">{h} head(s): {qual[h]:.3f} @ {lat[h]/1000:.1f} ms</text>')
for i in range(6):
    yy = y0 + ph * i / 5
    b.append(f'<text x="{x0-8}" y="{yy+4}" text-anchor="end" font-size="11" fill="#666">{1-i/5:.1f}</text>')
svg("figure-6-quality-vs-latency.svg", "Gating quality vs serial query latency (head ladder)", b,
    "Caption (auto): the practical frontier on this sandbox: 4 heads buys +0.20 R@10 over 1 head for ~3.9x "
    "latency; 8 heads adds no quality and ~2.4x more latency (super-linear pool fusion).",
    "results/head-scaling-quality.csv + results/head-scaling-latency.csv", "PH3C-HEAD-001-*-S5K")

# Fig 7: candidate budget vs recall/quality (3-head proxy = H4 10K? use H4-S5K + note)
runs_budget = "PH3C-HEAD-001-H4-S5K"
bs = ["10", "25", "50", "100", "200"]
series = {"candidate recall": [], "gating R@10": [], "uniform R@10": [], "global-best R@10": [], "oracle R@10": []}
for k in bs:
    rows_k = [r for r in bud if r["experiment"] == runs_budget and r["budget"] == k]
    series["candidate recall"].append(float(next(r["candidate_recall"] for r in rows_k)))
    for arm, key in (("trained_gating", "gating R@10"), ("uniform_multihead", "uniform R@10"),
                     ("global_best_single", "global-best R@10"), ("oracle_head_empirical", "oracle R@10")):
        series[key].append(float(next(r["R@10"] for r in rows_k if r["arm"] == arm and r["seed"] == "agg")))
b = axes(x0, y0, pw, ph, 1.0, "metric (TEST agg)", [f"K={k}" for k in bs])
b += bars(x0, y0, pw, ph, series, 1.0, 5)
b += legend(x0 + 560, y0 + 6, list(series))
svg("figure-7-candidate-budget.svg", "Candidate budget vs quality (4 heads, 5K docs; gate trained @K=100)", b,
    "Caption (auto): gating quality is FLAT across K=10..200 (view selection, not pool depth, binds); "
    "candidate recall still climbs 0.79→0.89 — plateau supports a small default budget, val-confirmed.",
    "results/candidate-budget.csv", runs_budget)

# Fig 8: candidate recall vs final recall (ladder @K=100)
series = {"candidate recall": [], "gating R@10": []}
labels = []
for h, run in ladder:
    row = next(r for r in hq if r["experiment"] == run and r["arm"] == "trained_gating" and r["seed"] == "agg")
    labels.append(f"{h}h@5K")
    series["candidate recall"].append(float(row["candidate_recall"]))
    series["gating R@10"].append(float(row["R@10"]))
b = axes(x0, y0, pw, ph, 1.0, "fraction", labels)
b += bars(x0, y0, pw, ph, series, 1.0, 4)
b += legend(x0 + 560, y0 + 10, list(series))
svg("figure-8-candrecall-vs-final.svg", "Candidate recall vs final gating recall (K=100)", b,
    "Caption (auto): the gap between pool coverage and final recall (0.09-0.23) is ranking/view-selection, "
    "not generation — consistent with the decomposition (34% of misses absent from pools at H4/10K).",
    "results/head-scaling-quality.csv + results/candidate-decomposition.csv", "PH3C-HEAD-001-*")

# Fig 9: component attribution (checkpoints of MEM-001)
cks = comp
labels = []
vals = []
for r in cks:
    if r["checkpoint"].startswith(("A_", "B_", "C_", "D_", "F_", "G2_", "J_", "K_")):
        labels.append(r["checkpoint"].split("_", 1)[1][:14])
        vals.append(float(r["rss_mb"]))
b = axes(x0, y0, pw, ph, 900, "RSS MB", labels)
bw = pw / len(vals) * 0.6
for i, v in enumerate(vals):
    x = x0 + pw * (i + 0.25) / len(vals)
    bh = ph * v / 900.0
    b.append(f'<rect x="{x:.0f}" y="{y0+ph-bh:.1f}" width="{bw:.0f}" height="{bh:.1f}" fill="#4056b0"/>')
    b.append(f'<text x="{x+bw/2:.0f}" y="{y0+ph-bh-4:.0f}" text-anchor="middle" font-size="9.5">{v:.0f}</text>')
b.append(f'<text x="{x0+pw/2:.0f}" y="{y0+ph+38}" text-anchor="middle" font-size="12" fill="#b04040">'
         f'+98 MB dataset | insert curve ~+64 MB/1K docs | peak 860 at inserts-done | drop releases only ~82 MB</text>')
svg("figure-9-component-attribution.svg", "Lifecycle RSS checkpoints — 3 heads x 512 x 10K (PH3C-MEM-001)", b,
    "Caption (auto): the jump is the INSERTION loop (inline HNSW build): ~10.9x raw rate in RSS; engine init, "
    "collection creation, flush, BM25 and the gating model are negligible; post-drop RSS stays elevated "
    "(allocator retention) — duplication + retained workspace dominate (§10).",
    "results/memory-components.csv", "PH3C-MEM-001")

print("PH3C figures:", sorted(f for f in os.listdir(HERE) if f.startswith("figure-")))
