#!/usr/bin/env python3
"""Generate Phase 3D figures (§42) — ONLY from canonical result CSVs.
Figures without backing data (memory/scaling after optimization) are
deliberately NOT produced: PH3D-MEM-OPT-001 was not run.

  figure-ph3d-1-crash-recovery.svg    : verdict per (durability, crash point)
  figure-ph3d-2-concurrent-throughput.svg   : QPS vs readers (pure read) + mixed combos
  figure-ph3d-3-mixed-latency.svg     : p50/p95/p99 latency under mixed r/w

Run from repo root: python3 research/phase3/figures/generate_figures_ph3d.py
"""
import csv, os

RES = "research/phase3/results"
FIG = "research/phase3/figures"
os.makedirs(FIG, exist_ok=True)


def rd(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">')


# ------------------------------------------------ figure 10: crash verdict grid
dur = rd("durability.csv")
points = ["after_acks", "after_flush", "after_checkpoint", "mid_inserts",
          "mid_flush", "after_compact"]
COLOR = {"ALL_ACKED": "#2e7d32", "PREFIX": "#f9a825", "OTHER": "#c62828"}
cellw, cellh, lw = 150, 42, 170
W = lw + cellw * len(points) + 40
H = 90 + cellh * len(dur) + 104
s = svg_open(W, H) + f'<rect width="{W}" height="{H}" fill="white"/>'
s += ('<text x="20" y="30" font-size="17" font-weight="bold">'
      'Crash-recovery verdicts: 7 crash points × 3 durability modes</text>'
      '<text x="20" y="50" font-size="11" fill="#555">'
      'PH3D-CRASH-001..003 · SIGKILL/exit(137) after acked inserts; verdicts from crash-recovery.csv</text>')
for j, p in enumerate(points):
    x = lw + j * cellw + 10
    s += f'<text x="{x + cellw/2 - 40}" y="{86}" font-size="12" fill="#333">{esc(p)}</text>'
txn_row = {"group": "txn COMMITTED_DURABLE (10/10)", "async": "txn COMMITTED_NOT_DURABLE_ASYNC (0/10)",
           "sync": "txn COMMITTED_DURABLE (10/10)"}
for i, r in enumerate(dur):
    y = 100 + i * cellh
    s += f'<text x="{lw - 12}" y="{y + 27}" font-size="13" text-anchor="end" fill="#333">{esc(r["durability"])}</text>'
    for j, p in enumerate(points):
        v = r[p]
        c = COLOR.get(v, COLOR["OTHER"])
        x = lw + j * cellw + 10
        s += (f'<rect x="{x}" y="{y}" width="{cellw-18}" height="{cellh-10}" rx="6" fill="{c}" opacity="0.88"/>'
              f'<text x="{x + (cellw-18)/2}" y="{y + (cellh-10)/2 + 4}" font-size="12" fill="white" '
              f'text-anchor="middle" font-weight="bold">{esc(v)}</text>')
y = 100 + len(dur) * cellh + 6
s += (f'<rect x="{lw+10}" y="{y}" width="{cellw*3-44}" height="30" rx="6" fill="#1565c0" opacity="0.88"/>'
      f'<text x="{lw+10+(cellw*3-44)/2}" y="{y+19}" font-size="12" fill="white" text-anchor="middle">'
      f'{esc(txn_row["group"])} — group &amp; sync</text>')
s += (f'<rect x="{lw+cellw*3+20}" y="{y}" width="{cellw*3-54}" height="30" rx="6" fill="#6a1b9a" opacity="0.9"/>'
      f'<text x="{lw+cellw*3+20+(cellw*3-54)/2}" y="{y+19}" font-size="12" fill="white" text-anchor="middle">'
      f'{esc(txn_row["async"])}</text>')
s += (f'<text x="20" y="{y+58}" font-size="11" fill="#555">During-commit txn is ALL-or-NOTHING at every point; under Async a committed txn may vanish (documented ack semantics) — never partial.</text></svg>')
open(os.path.join(FIG, "figure-ph3d-1-crash-recovery.svg"), "w").write(s)
print("wrote figure-ph3d-1-crash-recovery.svg")

# ------------------------------------------------ figures 11 + 12: concurrency
rows = rd("concurrency.csv")
pure = [r for r in rows if r["run"].endswith("CONC-001")]
mixed = [r for r in rows if r["run"].endswith("CONC-002")]

# figure 11: QPS vs readers (pure) + QPS vs combo (mixed)
W, H, ml, mb, mt = 760, 340, 60, 46, 40
pw, ph = W - ml - 30, H - mb - mt
allq = [float(r["qps"]) for r in pure + mixed]
qmax = max(allq) * 1.15
s = svg_open(W, H) + f'<rect width="{W}" height="{H}" fill="white"/>'
s += ('<text x="20" y="26" font-size="16" font-weight="bold">'
      'Concurrent throughput: pure-read ladder and mixed read/write matrix</text>'
      '<text x="20" y="44" font-size="11" fill="#555">PH3D-CONC-001/002 · 4 s windows · queries/s from concurrency.csv</text>')
for gy in range(5):
    v = qmax * gy / 4
    y = mt + ph - ph * gy / 4
    s += (f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e0e0e0"/>'
          f'<text x="{ml-8}" y="{y+4:.1f}" font-size="10" text-anchor="end" fill="#666">{v:,.0f}</text>')
bw = pw / (len(pure) + len(mixed)) - 8
for i, r in enumerate(pure):
    x = ml + i * (pw / (len(pure) + len(mixed))) + 4
    h = ph * float(r["qps"]) / qmax
    s += (f'<rect x="{x:.1f}" y="{mt+ph-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#1976d2"/>'
          f'<text x="{x+bw/2:.1f}" y="{mt+ph-h-4:.1f}" font-size="9" text-anchor="middle" fill="#1976d2">{int(float(r["qps"])):,}</text>'
          f'<text x="{x+bw/2:.1f}" y="{mt+ph+14:.1f}" font-size="10" text-anchor="middle" fill="#333">r{r["readers"]}</text>')
off = len(pure) * (pw / (len(pure) + len(mixed)))
for i, r in enumerate(mixed):
    x = ml + off + i * (pw / (len(pure) + len(mixed))) + 4
    h = ph * float(r["qps"]) / qmax
    s += (f'<rect x="{x:.1f}" y="{mt+ph-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#7b1fa2"/>'
          f'<text x="{x+bw/2:.1f}" y="{mt+ph-h-4:.1f}" font-size="9" text-anchor="middle" fill="#7b1fa2">{int(float(r["qps"])):,}</text>'
          f'<text x="{x+bw/2:.1f}" y="{mt+ph+14:.1f}" font-size="10" text-anchor="middle" fill="#333">r{r["readers"]}w{r["writers"]}</text>')
s += (f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>'
      f'<rect x="{ml}" y="{H-16}" width="12" height="12" fill="#1976d2"/>'
      f'<text x="{ml+17}" y="{H-6}" font-size="10" fill="#333">pure read (w=0)</text>'
      f'<rect x="{ml+130}" y="{H-16}" width="12" height="12" fill="#7b1fa2"/>'
      f'<text x="{ml+147}" y="{H-6}" font-size="10" fill="#333">mixed r/w (ins/upd/del + flush cycle)</text></svg>')
open(os.path.join(FIG, "figure-ph3d-2-concurrent-throughput.svg"), "w").write(s)
print("wrote figure-ph3d-2-concurrent-throughput.svg")

# figure 12: p50/p95/p99 for mixed combos
combos = [f'r{r["readers"]}w{r["writers"]}' for r in mixed]
pcts = [("p50_us", "#2e7d32"), ("p95_us", "#f9a825"), ("p99_us", "#c62828")]
lmax = max(float(r[p]) for r in mixed for p, _ in pcts) * 1.2
W, H, ml, mb, mt = 720, 330, 70, 46, 40
pw, ph = W - ml - 30, H - mb - mt
s = svg_open(W, H) + f'<rect width="{W}" height="{H}" fill="white"/>'
s += ('<text x="20" y="26" font-size="16" font-weight="bold">'
      'Query latency under mixed read/write (log scale)</text>'
      '<text x="20" y="44" font-size="11" fill="#555">PH3D-CONC-002 · p50/p95/p99 in µs per combo, from concurrency.csv</text>')
import math
lo = math.log10(min(float(r["p50_us"]) for r in mixed))
hi = math.log10(lmax)
for gy in range(5):
    v = 10 ** (lo + (hi - lo) * gy / 4)
    y = mt + ph - ph * gy / 4
    s += (f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e0e0e0"/>'
          f'<text x="{ml-8}" y="{y+4:.1f}" font-size="10" text-anchor="end" fill="#666">{v:,.0f}</text>')
gw = pw / len(mixed)
for i, r in enumerate(mixed):
    for k, (p, color) in enumerate(pcts):
        v = float(r[p])
        h = ph * (math.log10(max(v, 1)) - lo) / (hi - lo)
        x = ml + i * gw + gw * 0.18 + k * gw * 0.2
        s += (f'<rect x="{x:.1f}" y="{mt+ph-h:.1f}" width="{gw*0.16:.1f}" height="{h:.1f}" fill="{color}"/>'
              f'<text x="{x+gw*0.08:.1f}" y="{mt+ph-h-3:.1f}" font-size="8" text-anchor="middle" fill="{color}">{v:,.0f}</text>')
    s += f'<text x="{ml + i*gw + gw/2:.1f}" y="{mt+ph+16}" font-size="11" text-anchor="middle" fill="#333">{combos[i]}</text>'
s += f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>'
for k, (p, color) in enumerate(pcts):
    s += (f'<rect x="{ml+k*110}" y="{H-16}" width="12" height="12" fill="{color}"/>'
          f'<text x="{ml+k*110+17}" y="{H-6}" font-size="10" fill="#333">{p.split("_")[0]}</text>')
s += '</svg>'
open(os.path.join(FIG, "figure-ph3d-3-mixed-latency.svg"), "w").write(s)
print("wrote figure-ph3d-3-mixed-latency.svg")
print("note: figures 4/5 slots of §42 (memory/scaling after optimization) not produced — PH3D-MEM-OPT-001 not run")
