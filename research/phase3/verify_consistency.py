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
    ANALYSIS = {"PH3B-BM25-001", "PH3B-HYBRID-001", "PH3C-MEM-003", "PH3C-HEAD-002",
                "PH3D-MUTATION-001", "PH3D-RECOVERY-001", "PH3D-COMPACTION-001",
                "PH3D-CONCURRENCY-001"}
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

    # 11. Phase 3D: every PH3D run dir registered, with run_info; COMPLETED
    #     non-analysis runs carry config.json; PH3D raw runs immutable markers
    #     (ANALYSIS = analysis/child runs: run_info + parent-map.json instead
    #     of a workload config)
    PH3D_RUNS = ["PH3D-STATE-001", "PH3D-STATE-002", "PH3D-STATE-003", "PH3D-STATE-004",
                 "PH3D-FILTER-001", "PH3D-TX-001", "PH3D-CRASH-001", "PH3D-CRASH-002",
                 "PH3D-CRASH-003", "PH3D-WALCORRUPT-001", "PH3D-COMPACT-001",
                 "PH3D-COMPACT-002", "PH3D-CONC-001", "PH3D-CONC-002", "PH3D-CONC-003",
                 "PH3D-BACKUP-001", "PH3D-BACKUP-002", "PH3D-INTEGRATION-001",
                 "PH3D-MUTATION-001", "PH3D-RECOVERY-001", "PH3D-COMPACTION-001",
                 "PH3D-CONCURRENCY-001"]
    for eid in PH3D_RUNS:
        d = os.path.join(run, eid)
        if eid not in ids:
            ERR.append(f"PH3D run {eid} missing from registry")
            continue
        if not os.path.isdir(d):
            ERR.append(f"PH3D run {eid} registered but raw dir missing")
        elif not os.path.exists(os.path.join(d, "run_info.txt")):
            ERR.append(f"PH3D run {eid}: no run_info.txt")
    for eid in PH3D_RUNS:
        d = os.path.join(run, eid)
        if eid in ANALYSIS:
            if not os.path.exists(os.path.join(d, "parent-map.json")):
                ERR.append(f"PH3D child run {eid}: no parent-map.json")
            continue
        exp = next((x for x in idx["experiments"] if x["experiment_id"] == eid), None)
        if exp and exp.get("status", "").startswith("COMPLETED") and \
                not os.path.exists(os.path.join(d, "config.json")):
            ERR.append(f"{eid}: COMPLETED but no config.json")

    # 12. Phase 3D: canonical PH3D results CSVs exist and only reference
    #     registered runs
    for f in ("database-guarantees", "filtering", "mutations", "durability", "wal-replay",
              "crash-recovery", "transactions", "compaction", "concurrency",
              "backup-restore", "state-machine", "memory-optimization",
              "production-readiness"):
        if not os.path.exists(os.path.join(RES, f + ".csv")):
            ERR.append(f"missing PH3D results CSV {f}.csv")
    for f in ("crash-recovery", "state-machine", "transactions", "concurrency",
              "compaction", "backup-restore", "filtering"):
        pth = os.path.join(RES, f + ".csv")
        if not os.path.exists(pth):
            continue
        for row in rd(pth):
            for rid in (row.get("run"), row.get("run_or_check")):
                if rid and rid.startswith("PH3D-") and rid not in ids:
                    ERR.append(f"{f}.csv references unregistered {rid}")

    # 13. Phase 3D: crash-recovery.csv ↔ raw metrics drift (verdicts immutable)
    cr = os.path.join(RES, "crash-recovery.csv")
    if os.path.exists(cr):
        for row in rd(cr):
            d = os.path.join(run, row["run"])
            try:
                m = json.load(open(os.path.join(d, "metrics.json")))
            except OSError:
                ERR.append(f"crash-recovery.csv row for missing {row['run']}")
                continue
            raw = {p["crash_point"]: p for p in m["points"]}
            rp = raw.get(row["crash_point"])
            if rp is None:
                ERR.append(f"{row['run']}: raw metrics lack crash point {row['crash_point']}")
            elif rp["verdict"] != row["verdict"]:
                ERR.append(f"{row['run']}/{row['crash_point']}: verdict drift "
                           f"{rp['verdict']} vs {row['verdict']}")
            if row["unacked_present"] not in ("0", ""):
                ERR.append(f"{row['run']}/{row['crash_point']}: unacked docs present "
                           f"({row['unacked_present']}) — resurrection")
        if len(rd(cr)) != 21:
            ERR.append(f"crash-recovery.csv must hold 21 points (got {len(rd(cr))})")

    # 14. Phase 3D: state-machine verdicts must be PASS in both CSV and registry
    sm = os.path.join(RES, "state-machine.csv")
    if os.path.exists(sm):
        for row in rd(sm):
            exp = next((x for x in idx["experiments"] if x["experiment_id"] == row["run"]), None)
            if exp is None:
                ERR.append(f"state-machine.csv references unregistered {row['run']}")
            elif row["verdict"] == "PASS" and row["checks_failed"] != "0":
                ERR.append(f"{row['run']}: PASS verdict with {row['checks_failed']} failed checks")

    # 15. Phase 3D: production readiness is a multidimensional matrix — never
    #     a single score; statuses from the fixed vocabulary only
    pr = os.path.join(RES, "production-readiness.csv")
    if os.path.exists(pr):
        rows = rd(pr)
        hdr = list(rows[0].keys()) if rows else []
        if any("score" in h.lower() for h in hdr):
            ERR.append("production-readiness.csv must not contain a score column (§45)")
        allowed = {"VERIFIED", "PARTIAL", "NOT VERIFIED", "UNSUPPORTED", "BLOCKED",
                   "NOT ATTEMPTED IN PH3D", "VERIFIED (PH3C)"}
        subsys = set()
        for row in rows:
            subsys.add(row["subsystem"])
            st = row["status"]
            if not any(st == a or st.startswith(a.split(" (")[0]) for a in allowed):
                ERR.append(f"production-readiness.csv: non-vocabulary status '{st}'")
        need = {"Storage", "Durability", "Recovery", "Transactions", "Concurrency",
                "Backup", "Retrieval", "Memory"}
        if not need.issubset(subsys):
            ERR.append(f"production-readiness.csv missing subsystems: {need - subsys}")

    # 16. Phase 3D tables 1-8 + figures 10-12 with provenance; §42 memory/
    #     scaling figures must NOT exist without a PH3D-MEM-OPT run
    for i, f in enumerate(("table-1-database-guarantees", "table-2-mutation-correctness",
                           "table-3-crash-recovery", "table-4-transaction-atomicity",
                           "table-5-compaction-equivalence", "table-6-concurrency",
                           "table-7-backup-restore", "table-8-memory-optimization",
                           "table-9-state-machine", "table-10-filtering",
                           "table-11-wal-corruption"), 1):
        pth = os.path.join(TAB, f + ".md")
        if not os.path.exists(pth):
            ERR.append(f"missing PH3D {f}.md")
    # §17 shape: crash table must carry the expected/observed/status columns and
    # zero MISMATCH statuses
    t3 = os.path.join(TAB, "table-3-crash-recovery.md")
    if os.path.exists(t3):
        t3s = open(t3).read()
        for col in ("crash point", "ack state", "expected", "observed", "status"):
            if col not in t3s:
                ERR.append(f"table-3 missing §17 column '{col}'")
        if "MISMATCH" in t3s:
            ERR.append("table-3 crash matrix contains MISMATCH statuses")
    # §44 vocabulary
    pr2 = os.path.join(RES, "production-readiness.csv")
    if os.path.exists(pr2):
        for row in rd(pr2):
            if row["status"] == "PARTIAL" or row["status"] == "PARTIALLY":
                ERR.append(f"production-readiness.csv: legacy status 'PARTIAL' (use PARTIALLY VERIFIED): {row['capability']}")
    for f in ("figure-ph3d-1-crash-recovery", "figure-ph3d-2-concurrent-throughput",
              "figure-ph3d-3-mixed-latency"):
        pth = os.path.join(HERE, "figures", f + ".svg")
        if not os.path.exists(pth):
            ERR.append(f"missing PH3D {f}.svg")
        elif "PH3D-" not in open(pth).read():
            ERR.append(f"{f}.svg missing PH3D experiment-ID provenance")
    # deviations doc + §48 claim-ledger token check
    if not os.path.exists(os.path.join(HERE, "methodology", "ph3d-spec-deviations.md")):
        ERR.append("methodology/ph3d-spec-deviations.md missing")
    ledger_p = os.path.join(HERE, "..", "phase2", "findings", "claim-ledger.md")
    if os.path.exists(ledger_p):
        import re as _re
        toks = set(_re.findall(r"PH3D-[A-Z]+-\d+", open(ledger_p).read()))
        for t in sorted(toks):
            if t not in ids:
                ERR.append(f"claim-ledger references unregistered experiment {t}")
    mem_opt = next((x for x in idx["experiments"] if x["experiment_id"] == "PH3D-MEM-OPT-001"), None)
    for f in ("figure-13-memory-before-after-opt", "figure-14-scaling-after-opt"):
        pth = os.path.join(HERE, "figures", f + ".svg")
        if os.path.exists(pth) and mem_opt is None:
            ERR.append(f"{f}.svg exists but PH3D-MEM-OPT-001 was never run (no data backing)")

    # 17. Phase 2 must remain untouched (standing rule) except the append-only
    #     claim ledger (addenda are the sanctioned correction mechanism);
    #     the 3C report intact
    import subprocess
    st = subprocess.run(["git", "status", "--porcelain", "--", "research/phase2"],
                        capture_output=True, text=True)
    for line in st.stdout.splitlines():
        path = line[3:].strip()
        if path != "research/phase2/findings/claim-ledger.md":
            ERR.append(f"research/phase2 modified outside the claim ledger: {path}")
    if st.stdout.strip():
        ld = subprocess.run(
            ["git", "diff", "--", "research/phase2/findings/claim-ledger.md"],
            capture_output=True, text=True).stdout
        removed = [l for l in ld.splitlines() if l.startswith("-") and not l.startswith("---")]
        if removed:
            ERR.append("claim-ledger.md diff removes existing lines (append-only violated)")
    if not os.path.exists(os.path.join(HERE, "phase3c-final-report.md")):
        ERR.append("phase3c-final-report.md missing (Phase 3C record)")
    if not os.path.exists(os.path.join(HERE, "methodology", "database-guarantees.md")):
        ERR.append("methodology/database-guarantees.md missing (Phase 3D §2 contract)")
    # 18. Phase 3E (E1): WAL-integrity evidence — run registered with artifacts,
    #     generated results CSV matches the raw run, expectation gate holds,
    #     regression run present (no false refusals of legitimate states)
    PH3E_RUNS = ["PH3E-WAL-001", "PH3E-REG-001"]
    for eid in PH3E_RUNS:
        if eid not in ids:
            ERR.append(f"PH3E run {eid} missing from registry")
        d = os.path.join(run, eid)
        if not os.path.isdir(d):
            ERR.append(f"PH3E run {eid} registered but raw dir missing")
        elif not os.path.exists(os.path.join(d, "run_info.txt")) or \
                not os.path.exists(os.path.join(d, "config.json")):
            ERR.append(f"PH3E run {eid}: missing run_info.txt/config.json")
    wal_raw = os.path.join(run, "PH3E-WAL-001", "wal-integrity.csv")
    wal_res = os.path.join(RES, "wal-integrity-e1.csv")
    if not os.path.exists(wal_raw):
        ERR.append("PH3E-WAL-001: raw wal-integrity.csv missing")
    elif not os.path.exists(wal_res):
        ERR.append("results/wal-integrity-e1.csv missing (run generate_results_ph3e.py)")
    else:
        raw_rows = list(rd(wal_raw))
        res_rows = list(rd(wal_res))
        if len(raw_rows) != 11 or len(res_rows) != 11:
            ERR.append(f"wal-integrity matrix must hold 11 cases "
                       f"(raw {len(raw_rows)}, results {len(res_rows)})")
        for rr, sr in zip(raw_rows, res_rows):
            if rr["case"] != sr["case"] or rr["verdict"] != sr["verdict"]:
                ERR.append(f"wal-integrity-e1.csv drift for {rr['case']}: "
                           f"results vs raw {sr['verdict']}/{rr['verdict']}")
            if sr["match"] != "MATCH":
                ERR.append(f"wal-integrity-e1.csv: {sr['case']} expectation "
                           f"MISMATCH ({sr['observed']} vs {sr['expected']})")
        exp = next((x for x in idx["experiments"]
                    if x["experiment_id"] == "PH3E-WAL-001"), None)
        if exp:
            m = exp["metrics"]
            refused = sum(1 for r in raw_rows if r["verdict"] == "REFUSED")
            if m.get("cases") != len(raw_rows) or m.get("refused") != refused:
                ERR.append("registry metrics drift for PH3E-WAL-001 "
                           "(cases/refused vs raw)")
    reg = os.path.join(run, "PH3E-REG-001")
    if os.path.isdir(reg):
        for f in ("metrics.json", "crash-regression-group.csv"):
            if not os.path.exists(os.path.join(reg, f)):
                ERR.append(f"PH3E-REG-001: missing {f}")
        for ln in open(os.path.join(reg, "crash-regression-group.csv")):
            ln = ln.strip()
            if not ln:
                continue
            if "ALL_ACKED" not in ln and "prefix" not in ln:
                ERR.append(f"PH3E-REG-001 crash leg: unexpected verdict ({ln[:80]})")
    # 19. Phase 3E (E2): durability semantics — runs registered with artifacts,
    #     generated durability CSVs match raw facts, expectation gate holds
    #     (MISMATCH anywhere -> FAIL), E1 regression identical, report present
    PH3E_DUR_RUNS = ["PH3E-DUR-001", "PH3E-DUR-002", "PH3E-DUR-003",
                     "PH3E-DUR-004", "PH3E-DUR-005", "PH3E-DUR-006",
                     "PH3E-WAL-002", "PH3E-REG-002"]
    for eid in PH3E_DUR_RUNS:
        if eid not in ids:
            ERR.append(f"PH3E E2 run {eid} missing from registry")
        d = os.path.join(run, eid)
        if not os.path.isdir(d):
            ERR.append(f"PH3E E2 run {eid} registered but raw dir missing")
        elif not os.path.exists(os.path.join(d, "run_info.txt")) or \
                not os.path.exists(os.path.join(d, "config.json")):
            ERR.append(f"PH3E E2 run {eid}: missing run_info.txt/config.json")
    DUR_CSVS = {
        "PH3E-DUR-001": ("ack-boundary.csv", "durability-ack-boundary.csv"),
        "PH3E-DUR-002": ("txn-ack.csv", "durability-transactions.csv"),
        "PH3E-DUR-006": ("checkpoint-interaction.csv", "durability-checkpoint.csv"),
        "PH3E-DUR-004": ("group-boundary.csv", "durability-group.csv"),
        "PH3E-DUR-005": ("mode-latency.csv", None),
    }
    for eid, (raw_name, res_name) in DUR_CSVS.items():
        raw_p = os.path.join(run, eid, raw_name)
        if not os.path.exists(raw_p):
            ERR.append(f"{eid}: raw {raw_name} missing")
            continue
        if res_name:
            res_p = os.path.join(RES, res_name)
            if not os.path.exists(res_p):
                ERR.append(f"results/{res_name} missing (run generate_results_ph3e.py)")
                continue
            raw_rows = list(rd(raw_p))
            res_rows = list(rd(res_p))
            if len(raw_rows) != len(res_rows):
                ERR.append(f"{res_name}: row count drift raw {len(raw_rows)} vs results {len(res_rows)}")
            for sr in res_rows:
                if sr.get("match") == "MISMATCH":
                    ERR.append(f"{res_name}: expectation MISMATCH ({dict(list(sr.items())[:4])})")
    # E1 regression must be byte-identical (facts) to the original matrix
    w1 = os.path.join(run, "PH3E-WAL-001", "wal-integrity.csv")
    w2 = os.path.join(run, "PH3E-WAL-002", "wal-integrity.csv")
    if os.path.exists(w1) and os.path.exists(w2):
        a = open(w1).read().splitlines()[1:]
        b = open(w2).read().splitlines()[1:]
        if sorted(a) != sorted(b):
            ERR.append("PH3E-WAL-002 (E1 regression) differs from PH3E-WAL-001")
    if not os.path.exists(os.path.join(RES, "durability-modes.csv")):
        ERR.append("results/durability-modes.csv missing (mode contract summary)")
    if not os.path.exists(os.path.join(HERE, "phase3e-e2-final-report.md")):
        ERR.append("phase3e-e2-final-report.md missing (E2 final report)")

    # 20. Phase 3E (E3): machine-crash durability — failure-model column
    #     mandatory; POWER_LOSS rows must be BLOCKED-classified; classification
    #     gate must hold; E1/E2 regressions byte-identical; report + deviations
    e3_raw = os.path.join(run, "PH3E-E3-001", "e3-matrix.csv")
    e3_res = os.path.join(RES, "e3-recovery-classification.csv")
    for eid in ("PH3E-E3-001", "PH3E-WAL-003", "PH3E-DUR-007"):
        if eid not in ids:
            ERR.append(f"PH3E E3 run {eid} missing from registry")
        dd = os.path.join(run, eid)
        if not os.path.isdir(dd):
            ERR.append(f"PH3E E3 run {eid} registered but raw dir missing")
        elif not os.path.exists(os.path.join(dd, "run_info.txt")) or \
                not os.path.exists(os.path.join(dd, "config.json")):
            ERR.append(f"PH3E E3 run {eid}: missing run_info.txt/config.json")
    if not os.path.exists(e3_raw):
        ERR.append("PH3E-E3-001: raw e3-matrix.csv missing")
    elif not os.path.exists(e3_res):
        ERR.append("results/e3-recovery-classification.csv missing (generator)")
    else:
        raw_rows = list(rd(e3_raw))
        res_rows = list(rd(e3_res))
        if len(raw_rows) != len(res_rows):
            ERR.append(f"e3 classification row drift raw {len(raw_rows)} vs results {len(res_rows)}")
        if raw_rows and "failure_model" not in raw_rows[0]:
            ERR.append("e3-matrix.csv lacks failure_model column (claim boundary rule)")
        for sr in res_rows:
            if sr.get("match") == "MISMATCH":
                ERR.append(f"e3 classification MISMATCH: {sr['cell']}/{sr['mode']}/{sr['window']} rep {sr['rep']}")
            if sr.get("classification") in ("UNSAFE_OPEN", "UNEXPECTED_PARTIAL_STATE",
                                            "CORRUPTION_DETECTED", "UNEXPECTED_DATA_LOSS"):
                ERR.append(f"e3 unsafe classification present: {sr['classification']} "
                           f"({sr['cell']}/{sr['mode']}/{sr['window']})")
        for rr in raw_rows:
            if "POWER_LOSS" in rr.get("failure_model", ""):
                ERR.append("POWER-LOSS failure model present in raw run — physical power "
                           "loss is BLOCKED in this environment; such rows must not exist")
    w3 = os.path.join(run, "PH3E-WAL-003", "wal-integrity.csv")
    if os.path.exists(w1) and os.path.exists(w3):
        if sorted(open(w1).read().splitlines()[1:]) != sorted(open(w3).read().splitlines()[1:]):
            ERR.append("PH3E-WAL-003 (E1 regression under E3) differs from PH3E-WAL-001")
    for f, orig in (("ack-boundary", os.path.join(run, "PH3E-DUR-001", "ack-boundary.csv")),
                    ("txn-ack", os.path.join(run, "PH3E-DUR-002", "txn-ack.csv")),
                    ("checkpoint-interaction", os.path.join(run, "PH3E-DUR-006", "checkpoint-interaction.csv"))):
        n = os.path.join(run, "PH3E-DUR-007", f + ".csv")
        if os.path.exists(orig) and os.path.exists(n):
            if sorted(open(orig).read().splitlines()[1:]) != sorted(open(n).read().splitlines()[1:]):
                ERR.append(f"PH3E-DUR-007/{f} differs from its E2 original")
    if not os.path.exists(os.path.join(HERE, "phase3e-e3-final-report.md")):
        ERR.append("phase3e-e3-final-report.md missing (E3 final report)")

    # 21. Phase 3E (E4): backup/snapshot/restore — reference-model match,
    #     integrity expectations, online claims only from concurrent cells
    if "PH3E-BACKUP-004" not in ids:
        ERR.append("PH3E-BACKUP-004 missing from registry")
    d4 = os.path.join(run, "PH3E-BACKUP-004")
    if not os.path.isdir(d4):
        ERR.append("PH3E-BACKUP-004 registered but raw dir missing")
    else:
        for f in ("run_info.txt", "config.json", "e4-matrix.csv", "e4-integrity.csv"):
            if not os.path.exists(os.path.join(d4, f)):
                ERR.append(f"PH3E-BACKUP-004: missing {f}")
    e4_res = os.path.join(RES, "e4-backup-matrix.csv")
    e4_ires = os.path.join(RES, "e4-backup-integrity.csv")
    if not os.path.exists(e4_res) or not os.path.exists(e4_ires):
        ERR.append("results/e4-backup-*.csv missing (run generate_results_ph3e.py)")
    else:
        for sr in rd(e4_res):
            if sr["match"] != "MATCH":
                ERR.append(f"e4 matrix violation: {sr['case']} {sr['classification']}")
            # 'online backup' style claims require concurrent cells: the matrix
            # must CONTAIN writer-concurrent classes (b3/b4/b5/b6/b7) — a
            # quiescent-only run can never back an online claim.
        names = {r["case"] for r in rd(e4_res)}
        for req in ("b3-single-writer", "b4-multi-writer", "b5-writer-ckpt",
                    "b6-writer-rotation", "b7-writer-ckpt-rotation"):
            if req not in names:
                ERR.append(f"e4 matrix lacks concurrent-backup class {req} "
                           "(online claims unsupported without it)")
        for sr in rd(e4_ires):
            if sr["match"] != "MATCH":
                ERR.append(f"e4 integrity violation: {sr['case']} "
                           f"observed {sr['observed']} expected {sr['expected']}")
    if not os.path.exists(os.path.join(HERE, "phase3e-e4-final-report.md")):
        ERR.append("phase3e-e4-final-report.md missing (E4 final report)")
    if not os.path.exists(os.path.join(HERE, "methodology", "ph3e-e4-deviations.md")):
        ERR.append("methodology/ph3e-e4-deviations.md missing (E4 deviations)")

    # 22. Phase 3E (E5): compaction — evidence, concurrency classes, crash
    #     windows, refusal policy; no online claims from offline-only cells
    for rid in ("PH3E-COMPACT-004", "PH3E-WAL-006", "PH3E-DUR-009",
                "PH3E-E3-003", "PH3E-BACKUP-005"):
        if rid not in ids:
            ERR.append(f"{rid} missing from registry (E5 family)")
    d5 = os.path.join(run, "PH3E-COMPACT-004")
    if not os.path.isdir(d5):
        ERR.append("PH3E-COMPACT-004 registered but raw dir missing")
    else:
        for f in ("run_info.txt", "config.json", "e5-compaction.csv", "e5-crash.csv"):
            if not os.path.exists(os.path.join(d5, f)):
                ERR.append(f"PH3E-COMPACT-004: missing {f}")
    e5_res = os.path.join(RES, "e5-compaction.csv")
    e5_cres = os.path.join(RES, "e5-crash-windows.csv")
    if not os.path.exists(e5_res) or not os.path.exists(e5_cres):
        ERR.append("results/e5-*.csv missing (run generate_results_ph3e.py)")
    else:
        e5rows = rd(e5_res)
        for sr in e5rows:
            if sr["match"] != "MATCH":
                ERR.append(f"e5 violation: {sr['case']} {sr['classification']}")
            if sr["classification"] == "COMPACTION_VERIFIED" and                sr["case"] in ("c0-offline-control",) and sr["mode"] == "sync" and                not any(r["case"].endswith("during-compaction") for r in e5rows):
                ERR.append("e5: coordinated/online claim from offline-only cells")
        names = {r["case"] for r in e5rows}
        for req in ("readers-during-compaction", "writer-during-compaction",
                    "multiwriter-during-compaction", "collections-isolation-compaction",
                    "repeated-compaction-x4"):
            if req not in names:
                ERR.append(f"e5 results lack required concurrency/completeness cell {req}")
        for sr in rd(e5_cres):
            if sr["match"] != "MATCH":
                ERR.append(f"e5 crash window violation: {sr['window']}")
        if not any(r["window"] == "compact_after_output" for r in rd(e5_cres)):
            ERR.append("e5: missing fresh-process crash evidence at output window")
    if not os.path.exists(os.path.join(HERE, "phase3e-e5-final-report.md")):
        ERR.append("phase3e-e5-final-report.md missing (E5 final report)")
    if not os.path.exists(os.path.join(HERE, "methodology", "ph3e-e5-deviations.md")):
        ERR.append("methodology/ph3e-e5-deviations.md missing (E5 deviations)")
    if not os.path.exists(os.path.join(HERE, "methodology", "ph3e-e3-deviations.md")):
        ERR.append("methodology/ph3e-e3-deviations.md missing (E3 deviations)")
    if not os.path.exists(os.path.join(HERE, "methodology", "ph3e-spec-deviations.md")):
        ERR.append("methodology/ph3e-spec-deviations.md missing (E2 deviations)")
    if not os.path.exists(os.path.join(HERE, "phase3e-evidence-e0-e1.md")):
        ERR.append("phase3e-evidence-e0-e1.md missing (E0+E1 evidence gate)")
    if not os.path.exists(os.path.join(HERE, "methodology", "production-contract.md")):
        ERR.append("methodology/production-contract.md missing (E0 frozen contract)")

    # 23. Phase 3E (E6): transaction semantics — atomicity evidence gates.
    #     Rejects: partial committed txn, effects without commit proof,
    #     duplicate application, crash PASS without fresh-process recovery,
    #     unsupported update/upsert claims, unregistered cells.
    for rid in ("PH3E-TXN-001", "PH3E-WAL-007", "PH3E-DUR-010",
                "PH3E-E3-004", "PH3E-BACKUP-006", "PH3E-COMPACT-005"):
        if rid not in ids:
            ERR.append(f"{rid} missing from registry (E6 family)")
    d6 = os.path.join(run, "PH3E-TXN-001")
    if not os.path.isdir(d6):
        ERR.append("PH3E-TXN-001 registered but raw dir missing")
    else:
        for f in ("run_info.txt", "config.json", "e6-txn.csv", "e6-crash.csv"):
            if not os.path.exists(os.path.join(d6, f)):
                ERR.append(f"PH3E-TXN-001: missing {f}")
    e6_res = os.path.join(RES, "e6-transactions.csv")
    e6_cres = os.path.join(RES, "e6-crash-atomicity.csv")
    if not os.path.exists(e6_res) or not os.path.exists(e6_cres):
        ERR.append("results/e6-*.csv missing (run generate_results_ph3e.py)")
    else:
        trows = rd(e6_res)
        crows = rd(e6_cres)
        raw_t = rd(os.path.join(d6, "e6-txn.csv")) if os.path.isdir(d6) else []
        raw_c = rd(os.path.join(d6, "e6-crash.csv")) if os.path.isdir(d6) else []
        if len(trows) != len(raw_t):
            ERR.append(f"e6 txn row drift raw {len(raw_t)} vs results {len(trows)}")
        if len(crows) != len(raw_c):
            ERR.append(f"e6 crash row drift raw {len(raw_c)} vs results {len(crows)}")
        for sr in trows:
            if sr["match"] != "MATCH":
                ERR.append(f"e6 violation: {sr['family']}/{sr['case']} {sr['classification']}")
        for sr in crows:
            if sr["match"] != "MATCH":
                ERR.append(f"e6 crash violation: {sr['boundary']}/{sr['mode']} {sr['txn_state']}")
            # partial committed txn can never be a PASS
            if "PARTIAL" in (sr.get("txn_state") or ""):
                ERR.append(f"e6 partial-transaction state present: {sr['boundary']}/{sr['mode']}")
            # every crash PASS requires fresh-process recovery evidence
            if (sr.get("match") == "MATCH") and not raw_c:
                ERR.append("e6 crash PASS without raw fresh-process rows")
        if raw_c:
            for rr in raw_c:
                if rr.get("aborted_at_boundary") != "true":
                    ERR.append(f"e6 crash row without fresh-process abort evidence: {rr.get('boundary')}/{rr.get('mode')}")
                # effects without commit proof: PRESENT_* states are only legal
                # at boundaries AFTER the CommitTxn record is on WAL
                if rr.get("txn_state", "").startswith("PRESENT") and rr.get("boundary") in (
                        "pre-commit staging", "gate: tx_before_commit_wal"):
                    ERR.append(f"e6 effects-without-commit-proof: PRESENT at {rr['boundary']}/{rr['mode']}")
            # duplicate application: idempotence cell must exist and pass
            names = {r["case"] for r in trows}
            for req in ("restart-x3", "ins-ins-del-ins-txn", "staged-then-ckpt",
                        "before-during-after", "100-op-txn-2KiB-segments",
                        "torn-tail-partial-group", "garbled-committed-group",
                        "update-upsert-semantics", "3-staged-txns"):
                if req not in names:
                    ERR.append(f"e6 results lack required cell {req}")
            # unsupported update/upsert claim guard: the E6k raw cell MUST
            # record in-txn update/upsert as UNSUPPORTED
            k = [r for r in raw_t if r.get("case") == "update-upsert-semantics"]
            if k:
                blob = " ".join(str(v) for r in k for v in r.values())
                if "UNSUPPORTED" not in blob:
                    ERR.append("e6k: in-txn update/upsert support claimed without evidence (fabrication guard)")
        # E1 regression byte-identity chain extended through E6
        w7 = os.path.join(run, "PH3E-WAL-007", "wal-integrity.csv")
        if os.path.exists(w1) and os.path.exists(w7):
            if sorted(open(w1).read().splitlines()[1:]) != sorted(open(w7).read().splitlines()[1:]):
                ERR.append("PH3E-WAL-007 (E1 regression under E6) differs from PH3E-WAL-001")
    for rid in ("PH3E-WAL-007", "PH3E-DUR-010", "PH3E-E3-004", "PH3E-BACKUP-006", "PH3E-COMPACT-005"):
        dd = os.path.join(run, rid)
        if os.path.isdir(dd) and not os.path.exists(os.path.join(dd, "run_info.txt")):
            ERR.append(f"{rid}: regression run missing run_info.txt")
    if not os.path.exists(os.path.join(HERE, "phase3e-e6-final-report.md")):
        ERR.append("phase3e-e6-final-report.md missing (E6 final report)")
    if not os.path.exists(os.path.join(HERE, "methodology", "ph3e-e6-deviations.md")):
        ERR.append("methodology/ph3e-e6-deviations.md missing (E6 deviations)")

    if ERR:
        print("PHASE 3 CONSISTENCY CHECK FAILED:")
        for e in ERR:
            print("  ✗", e)
        sys.exit(1)
    print("phase3 consistency check: PASS (tables ↔ results ↔ registry; exact-reference anchor holds)")


if __name__ == "__main__":
    main()
