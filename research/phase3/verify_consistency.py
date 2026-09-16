#!/usr/bin/env python3
"""Phase 3 consistency check (spec §38).

Verifies tables ↔ canonical results CSVs ↔ registry ↔ raw run dirs.
Exits non-zero on any mismatch — run before quoting phase 3 numbers.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "tables")
ERR = []


def rd(p):
    return list(csv.DictReader(open(p)))


def main():
    # 1. every canonical results file maps to a registered experiment
    idx = json.load(open(os.path.join(HERE, "raw", "experiment-index.json")))
    ids = {e["experiment_id"] for e in idx["experiments"]}
    run = os.path.join(HERE, "raw", "runs")
    for eid in sorted(os.listdir(run)):
        if eid.startswith("PH3-") and eid not in ids:
            ERR.append(f"run dir {eid} missing from registry")

    # 2. retrieval-quality.csv ↔ table-retrieval-quality.md ↔ registry metrics
    qpath = os.path.join(RES, "retrieval-quality.csv")
    if os.path.exists(qpath):
        tab = open(os.path.join(TAB, "table-retrieval-quality.md")).read()
        for row in rd(qpath):
            if row["dataset"] != "PH3-DS-FM-S":
                continue
            for metric in ("R@1", "R@5", "R@10", "NDCG@10", "MRR"):
                if row[metric] not in tab:
                    ERR.append(f"quality table missing {row['arm']}/{metric}={row[metric]}")
        exp = next((e for e in idx["experiments"] if e["experiment_id"] == "PH3-QUAL-FM-S"), None)
        if exp is None:
            ERR.append("retrieval-quality.csv exists but PH3-QUAL-FM-S not in registry")
        else:
            for row in rd(qpath):
                m = exp["metrics"].get(row["arm"], {})
                if m and m.get("R@10") != row["R@10"]:
                    ERR.append(f"registry drift {row['arm']}: {m.get('R@10')} vs results {row['R@10']}")

        # 3. harness anchor: exact reference must be R@10 = 1.0
        for row in rd(qpath):
            if row["arm"] == "exact_reference" and row["R@10"] != "1.0000":
                ERR.append(f"exact reference R@10 != 1.0 ({row['R@10']}) — GT/harness broken (HC-class)")

    # 4. memory.csv ↔ table-scaling-memory.md + failed runs registered
    mpath = os.path.join(RES, "memory.csv")
    if os.path.exists(mpath):
        tab = open(os.path.join(TAB, "table-scaling-memory.md")).read()
        for row in rd(mpath):
            if row["status"] == "OOM_KILLED":
                eid = "PH3-QUAL-FM-M" if row["tier"] == "M" else "PH3-QUAL-FM-T30"
                if eid not in ids:
                    ERR.append(f"OOM tier {row['tier']} not registered ({eid})")
            if row["peak_rss_mb"] and row["peak_rss_mb"] not in tab:
                ERR.append(f"memory table missing {row['peak_rss_mb']}")

    # 5. every COMPLETED experiment has config + dataset hash + commit
    #    (analysis runs PH3B-BM25-001 / PH3B-HYBRID-001 carry run_info +
    #     copied analysis artifacts instead of config.json)
    ANALYSIS = {"PH3B-BM25-001", "PH3B-HYBRID-001", "PH3C-MEM-003", "PH3C-HEAD-002"}
    for e in idx["experiments"]:
        if e.get("status", "").startswith("COMPLETED"):
            d = os.path.join(run, e["experiment_id"])
            if e["experiment_id"] not in ANALYSIS and not os.path.exists(os.path.join(d, "config.json")):
                ERR.append(f"{e['experiment_id']}: COMPLETED but no config.json")
            if e.get("git_commit") in ("", None):
                ERR.append(f"{e['experiment_id']}: no code commit")

    # 6. PH3B: canonical CSVs ↔ registry ↔ exact anchor ↔ gating numbers
    comp = os.path.join(RES, "complementary-retrieval.csv")
    if os.path.exists(comp):
        for row in rd(comp):
            eid = row["experiment"]
            exp = next((x for x in idx["experiments"] if x["experiment_id"] == eid), None)
            if exp is None:
                ERR.append(f"complementary-retrieval.csv row for unregistered {eid}")
                continue
            key = row["arm"] + ("@" + row["seed"] if row["seed"] != "agg" else "")
            m = exp["metrics"].get(key, {})
            if m and m.get("R@10") != row["R@10"]:
                ERR.append(f"registry drift {eid}/{key}: {m.get('R@10')} vs results {row['R@10']}")
            if row["arm"] == "exact_reference" and row["R@10"] != "1.0000":
                ERR.append(f"{eid}: exact-reference anchor != 1.0 ({row['R@10']}) — run invalid (HC-class)")
        # raw run dir must exist for every experiment referenced in canonical CSVs
        for row in rd(comp):
            if not os.path.isdir(os.path.join(run, row["experiment"])):
                ERR.append(f"canonical CSV references missing raw run {row['experiment']}")
        # gating headline must match between canonical + registry (agg row)
        canon = next((r for r in rd(comp) if r["experiment"] == "PH3B-COMP-001"
                      and r["arm"] == "trained_gating" and r["seed"] == "agg"), None)
        tab = os.path.join(TAB, "table-complementary-main.md")
        if canon and os.path.exists(tab):
            if canon["R@10"] not in open(tab).read():
                ERR.append("complementary main table missing gating R@10 " + canon["R@10"])

    # 7. figures exist for the PH3B (A-F) and PH3C (1-9) sets, with provenance
    for f in ("figure-A-comparisons", "figure-B-gating-weights", "figure-C-gating-vs-oracle",
              "figure-D-sparse-dense-hybrid", "figure-E-quality-vs-latency", "figure-F-candidate-recall"):
        pth = os.path.join(HERE, "figures", f + ".svg")
        if not os.path.exists(pth):
            ERR.append(f"missing PH3B figure {f}.svg")
        elif "PH3B-COMP-001" not in open(pth).read():
            ERR.append(f"{f}.svg missing experiment-ID provenance")
    for i, f in enumerate(("figure-1-memory-vs-corpus", "figure-2-memory-vs-heads",
                           "figure-3-memory-vs-dim", "figure-4-quality-vs-heads",
                           "figure-5-latency-vs-heads", "figure-6-quality-vs-latency",
                           "figure-7-candidate-budget", "figure-8-candrecall-vs-final",
                           "figure-9-component-attribution"), 1):
        pth = os.path.join(HERE, "figures", f + ".svg")
        if not os.path.exists(pth):
            ERR.append(f"missing PH3C figure {f}.svg")
        elif "PH3C-" not in open(pth).read():
            ERR.append(f"{f}.svg missing PH3C experiment-ID provenance")

    # 8. PH3C: memory-scaling ↔ registry (OOM rows preserved w/ failure.txt)
    mpath2 = os.path.join(RES, "memory-scaling.csv")
    if os.path.exists(mpath2):
        for row in rd(mpath2):
            eid = row["experiment"]
            if eid not in ids:
                ERR.append(f"memory-scaling.csv references unregistered {eid}")
            if row["status"] == "FAILED-OOM":
                if not os.path.exists(os.path.join(run, eid, "failure.txt")):
                    ERR.append(f"{eid}: FAILED-OOM without failure.txt")
            elif row["peak_rss_mb"]:
                exp = next((x for x in idx["experiments"] if x["experiment_id"] == eid), None)
                if exp and exp["metrics"].get("peak_rss_mb") and \
                        abs(float(exp["metrics"]["peak_rss_mb"]) - float(row["peak_rss_mb"])) > 0.05:
                    ERR.append(f"registry drift {eid} peak_rss_mb")
            if row["status"] == "COMPLETED" and (not row["peak_rss_mb"] or not row["build_seconds"]):
                ERR.append(f"{eid}: COMPLETED memory row missing metrics")

    # 9. head ladder K=100 ↔ registry drift (gating agg rows)
    hp = os.path.join(RES, "head-scaling-quality.csv")
    if os.path.exists(hp):
        for row in rd(hp):
            if row["arm"] == "trained_gating" and row["seed"] == "agg":
                exp = next((x for x in idx["experiments"] if x["experiment_id"] == row["experiment"]), None)
                key = f"K100/trained_gating"
                m = (exp or {}).get("metrics", {}).get(key, {})
                if m and m.get("R@10") != row["R@10"]:
                    ERR.append(f"registry drift {row['experiment']} gating {m.get('R@10')} vs {row['R@10']}")

    # 10. Phase 3B headline preserved (gating 0.6448 agg row present + anchor)
    comp2 = os.path.join(RES, "complementary-retrieval.csv")
    if os.path.exists(comp2):
        g = [r for r in rd(comp2) if r["experiment"] == "PH3B-COMP-001"
             and r["arm"] == "trained_gating" and r["seed"] == "agg"]
        if not g or g[0]["R@10"] != "0.6448":
            ERR.append("PH3B-COMP-001 gating headline row altered or missing (Phase 3B must remain unchanged)")
    if ERR:
        print("PHASE 3 CONSISTENCY CHECK FAILED:")
        for e in ERR:
            print("  ✗", e)
        sys.exit(1)
    print("phase3 consistency check: PASS (tables ↔ results ↔ registry; exact-reference anchor holds)")


if __name__ == "__main__":
    main()
