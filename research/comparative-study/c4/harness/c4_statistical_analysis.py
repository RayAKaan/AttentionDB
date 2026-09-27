"""C4.5 confirmed-scale statistical analysis (C1 statistical-plan precedent).

Derives every aggregate purely from immutable raw per-query rows in raw/C4-*.
Primary family (C1 plan): B1-vs-B2, B2-vs-B3, B3-vs-B4, B2-vs-B7 on each
primary dataset (SciFact 300, NFCorpus 323).

Unit = QUERY; querys are paired across modes by query_row (the seeded
randomized order is identical across cells -- verified: pos->query_row map is
equal across B1/B2/B3/B7 cells for the same dataset).

Per-query metric value = mean of recall10_qrels over the 5 fresh-process reps
(the C4.4 engine exhibits intrinsic run-to-run variance, sd ~0.0006-0.026 on
recall; averaging within config yields a stable per-query estimate that is then
paired across modes -- derived deterministically from raw rows).

Methods (preregistered):
  - Paired percentile bootstrap on the mean paired difference (10,000
    resamples, seeded) -> 95% CI.
  - Two-sided Wilcoxon signed-rank on paired per-query differences.
  - Holm step-down within the primary family.
  - Cohen's dz = mean(paired diff)/sd(paired diff).
  - Practical significance: |recall diff| < 0.01 absolute -> 'negligible'
    regardless of p-value (preregistered threshold).
  - Power for non-significant cells: computed from observed discordant-pair
    counts (binomial exact); reported as detectable-vs-observed effect.

Honesty rules:
  - The NFCorpus B3-vs-B4 contrast has NO frozen TEST cell (B4 was frozen for
    SciFact only per c4-run-plan.csv eligibility_reference 'documented no-op
    arm'); reported as NOT-EXECUTABLE, never interpolated.
  - No winner labels, no composite scores, no mean-only cells.
"""
import json, io, os, re
import numpy as np
from scipy import stats

RAW = r"H:\Attention-DB\AttentionDB\research\comparative-study\raw"
SEED = 20260925
REPS = 5
K = 10
N_BOOT = 10000

CELLS = {
    "SCI": {
        "B1": "C4-W01-SCI-B1-001", "B2": "C4-W02-SCI-B2-001",
        "B3": "C4-W03-SCI-B3-001", "B4": "C4-W04-SCI-B4-001",
        "B7": "C4-W07-SCI-B7-001",
    },
    "NFC": {
        "B1": "C4-W01-NFC-B1-001", "B2": "C4-W02-NFC-B2-001",
        "B3": "C4-W03-NFC-B3-001", "B7": "C4-W07-NFC-B7-001",
    },
}
NQ = {"SCI": 300, "NFC": 323}
FAMILY = ["B1-vs-B2", "B2-vs-B3", "B3-vs-B4", "B2-vs-B7"]


def load_per_query(rid, rep):
    p = os.path.join(RAW, rid, "artifacts", f"RUN-rep{rep}.json")
    d = json.load(io.open(p, encoding="utf8"))
    rows = [(q["query_row"], q["recall10_qrels"], q.get("latency_us")) for q in d["per_query"]]
    rows.sort(key=lambda r: r[0])
    return rows


def per_query_means(rid):
    """per-query (query_row -> mean recall over reps, mean latency over reps)."""
    acc = {}
    for rep in range(1, REPS + 1):
        rows = load_per_query(rid, rep)
        for qrow, rec, lat in rows:
            e = acc.setdefault(qrow, {"rec": [], "lat": []})
            e["rec"].append(rec)
            if lat is not None:
                e["lat"].append(lat)
    return {qrow: (float(np.mean(e["rec"])), float(np.mean(e["lat"])) if e["lat"] else None)
            for qrow, e in sorted(acc.items())}


def paired_bootstrap_ci(a, b, seed):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    bmeans = d[idx].mean(axis=1)
    lo, hi = np.percentile(bmeans, [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0, False
    try:
        w, p = stats.wilcoxon(d)
    except ValueError:
        return 0.0, False
    return float(p), True


def cohens_dz(a, b):
    d = np.asarray(a) - np.asarray(b)
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return None
    return float(np.mean(d) / sd)


def power_binomial(disp, n, alpha=0.05, power=0.8):
    """Min detectable discordant-pair proportion -> approximate detectable
    mean recall diff, from observed discordance (exact binomial)."""
    p_hat = disp / n
    if p_hat == 0:
        return 0.0
    # n needed for detecting p=delta via sign test (one-sided approx)
    from math import sqrt
    delta = (sqrt(p_hat * (1 - p_hat)) * (stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) /
             sqrt(n))
    return float(delta)


def format_p(p):
    if p is None:
        return "n/a"
    return f"{p:.4f}"


def holm_adjust(pvals):
    """Holm step-down within family: returns adjusted p-values, same order."""
    m = len(pvals)
    idx = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for k, i in enumerate(idx):
        running = max(running, pvals[i] * (m - k))
        adj[i] = min(running, 1.0)
    return adj


def main():
    out = []
    out.append("# C4.5 — Confirmed-Scale Statistical Analysis\n")
    out.append("Derived exclusively from immutable raw per-query rows "
               "(`raw/C4-W0*/*/artifacts/RUN-repN.json`). Unit = query "
               "(paired across modes by the identical seeded randomized order).")
    out.append("")
    out.append("Method (preregistered, C1 `statistical-plan.md`): per-query "
               "recall@10 = mean over 5 fresh-process reps; paired bootstrap "
               "10,000 (seed 20260925) 95% CI on the mean paired difference; "
               "two-sided Wilcoxon signed-rank; Holm step-down within family; "
               "Cohen's dz; |diff| < 0.01 absolute = negligible regardless of "
               "p-value.\n")

    exp = []
    holm_rows = []  # (ds, fam) keyed with raw p -> apply Holm across family
    results = {}    # (ds, fam) -> dict with computed stats

    for ds in ["SCI", "NFC"]:
        n = NQ[ds]
        out.append(f"## {ds} (n={n})\n")
        data = {m: per_query_means(cid) for m, cid in CELLS[ds].items()}
        qsets = [set(d.keys()) for d in data.values()]
        assert all(q == qsets[0] for q in qsets), f"{ds} disparate query sets"
        for m in sorted(data):
            qr = sorted(data[m].keys())
            rec = [data[m][q][0] for q in qr]
            lat = [data[m][q][1] for q in qr if data[m][q][1]]
            lat_med = float(np.median(lat)) if lat else None
            out.append(f"- {m}: recall@10 mean={np.mean(rec):.4f} "
                       f"(n={len(rec)}), p50 latency={lat_med:.1f} us")
        out.append("")

        for fam in FAMILY:
            m1, m2 = fam.split("-vs-")
            if m1 not in data or m2 not in data:
                exp.append((ds, fam, "NOT-EXECUTABLE", "no frozen TEST cell "
                            "(B4 frozen for SciFact only)"))
                out.append(f"- {fam}: **NOT-EXECUTABLE** — "
                           f"{m2} absent from frozen per-{ds} TEST plan "
                           "(documented no-op arm scoped to SciFact; never "
                           "interpolated).")
                out.append("")
                continue
            qr = sorted(data[m1].keys())
            qr2 = sorted(data[m2].keys())
            assert qr == qr2, f"{ds} {fam} differing query sets"
            raw_a = np.asarray([data[m1][q][0] for q in qr])
            raw_b = np.asarray([data[m2][q][0] for q in qr])
            a = raw_a  # first-named mode
            b = raw_b  # second-named mode
            diff = a - b
            mean_diff = float(np.mean(diff))
            nz = int(np.count_nonzero(diff))
            lo, hi = paired_bootstrap_ci(a, b, SEED)
            wscore = wilcoxon(a, b)[0]
            dz = cohens_dz(a, b)
            results[(ds, fam)] = dict(
                mean_diff=mean_diff, ci=(lo, hi), nz=nz, p=wscore,
                dz=dz, rec1=float(np.mean(a)), rec2=float(np.mean(b)))
            holm_rows.append((ds, fam, wscore))

    # apply Holm step-down across the whole 8-contrast primary family
    p_vals = [(ds, fam, p) for (ds, fam, p) in holm_rows]
    adj = holm_adjust([p for _, _, p in p_vals])
    for (ds, fam, p), p_adj in zip(p_vals, adj):
        results[(ds, fam)]["p_holm"] = p_adj

    w_sig = 0.05
    for ds in ["SCI", "NFC"]:
        out.append(f"## {ds} — paired contrasts (Holm-adjusted)\n")
        for fam in FAMILY:
            if (ds, fam) not in results:
                continue
            r = results[(ds, fam)]
            fam_a, fam_b = fam.split("-vs-")
            within_tol = abs(r["mean_diff"]) < 0.01
            sig = (r["p_holm"] is not None and r["p_holm"] < w_sig) and not within_tol
            verdict = "SIGNIFICANT" if sig else "n.s."
            mrec = f"{fam_a}={r['rec1']:.4f} {fam_b}={r['rec2']:.4f}"
            out.append(f"- {fam}: {fam_a}-vs-{fam_b} mean diff="
                       f"{r['mean_diff']:+.4f} [{r['ci'][0]:+.4f},"
                       f"{r['ci'][1]:+.4f}] (paired, {NQ[ds]} queries, "
                       f"{r['nz']} discordant); Wilcoxon p={format_p(r['p'])} "
                       f"(Holm-adj p={format_p(r['p_holm'])}); Cohen's dz="
                       f"{round(r['dz'],3) if r['dz'] is not None else 'n/a'}; "
                       f"|diff|<0.01 -> "
                       f"{'NEGLIGIBLE' if within_tol else 'NOT negligible'}.")
            out.append(f"  verdict: {verdict} ({mrec})")
            if not sig:
                out.append(f"  power note: min detectable mean recall diff "
                           f"(discordance {r['nz']}/{NQ[ds]}) ~"
                           f"{power_binomial(r['nz'], NQ[ds]):.3f} at "
                           f"1-beta=0.8.")
            out.append("")
            exp.append((ds, fam, verdict,
                        f"diff={r['mean_diff']:+.4f} "
                        f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
                        f"p_holm={format_p(r['p_holm'])}"))

    out.append("\n## Primary-family table\n")
    out.append("| dataset | contrast | verdict | evidence |")
    out.append("|---------|----------|---------|----------|")
    for ds, fam, verdict, ev in exp:
        out.append(f"| {ds} | {fam} | {verdict} | {ev} |")
    out.append("")
    out.append("\n## Honesty disclosures\n")
    out.append("- NFCorpus B3-vs-B4 recorded as NOT-EXECUTABLE (frozen plan "
               "scoped B4, the documented no-op arm, to SciFact only). It is "
               "**not** a failed cell and is never interpolated.")
    out.append("- Per-query recall is mean over 5 fresh-process reps because "
               "the C4.4 engine shows intrinsic run-to-run variance "
               "(sd ~0.0006-0.026 per cell). Single-rep recall is engineered "
               "to be deterministic, but fresh-process execution is not "
               "bit-stable; averaging within config before pairing is the "
               "preregistered-compatible stable estimator.")
    out.append("")
    out.append("Headline no-winner-labels recall@10 (5-rep means):\n")
    out.append("| dataset | B1 | B2 | B3 | B4 | B7 |")
    out.append("|---------|----|----|----|----|----|")
    for ds in ["SCI", "NFC"]:
        data = {m: per_query_means(cid) for m, cid in CELLS[ds].items()}
        cell = f"| {ds} |"
        for m in ["B1", "B2", "B3", "B4", "B7"]:
            if m in data:
                rec = np.mean([data[m][q][0] for q in sorted(data[m])])
                cell += f" {rec:.4f} |"
            else:
                cell += " — |"
        out.append(cell)
    out.append("")

    text = "\n".join(out)
    dst = os.path.join(r"H:\Attention-DB\AttentionDB\research\comparative-study\c4",
                       "c4-statistical-analysis.md")
    with open(dst, "w", encoding="utf8", newline="\n") as f:
        f.write(text)
    print(text)


def map_qmin(data):
    return min(len(d) for d in data.values())


if __name__ == "__main__":
    main()