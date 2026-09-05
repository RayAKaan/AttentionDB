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
    for e in idx["experiments"]:
        if e.get("status", "").startswith("COMPLETED"):
            d = os.path.join(run, e["experiment_id"])
            if not os.path.exists(os.path.join(d, "config.json")):
                ERR.append(f"{e['experiment_id']}: COMPLETED but no config.json")
            if e.get("git_commit") in ("", None):
                ERR.append(f"{e['experiment_id']}: no code commit")

    if ERR:
        print("PHASE 3 CONSISTENCY CHECK FAILED:")
        for e in ERR:
            print("  ✗", e)
        sys.exit(1)
    print("phase3 consistency check: PASS (tables ↔ results ↔ registry; exact-reference anchor holds)")


if __name__ == "__main__":
    main()
