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



# Artifact chain: PH3E-SOAK-002/003/006/008/011 top-level evidence was lost to a
# platform snapshot cap (see raw/runs/TRIAGE-2026-09-22.md); deterministic
# same-seed restoration executions under NEW IDs carry the artifact set. Original
# dirs and their surviving files are never modified. PH3E-SOAK-010 (INVALIDATED)
# is documented-loss (its harness bugs were fixed; not re-creatable).
RESTORATION = {"PH3E-SOAK-002": "PH3E-SOAK-012", "PH3E-SOAK-003": "PH3E-SOAK-013",
               "PH3E-SOAK-006": "PH3E-SOAK-017", "PH3E-SOAK-011": "PH3E-SOAK-015",
               "PH3E-SOAK-008": "PH3E-SOAK-016"}
def e8_art_dir(rid, base):
    import os as _os
    if _os.path.exists(f"{base}/{rid}/counts.json"):
        return f"{base}/{rid}"
    return RESTORATION.get(rid, rid) and (f"{base}/{RESTORATION[rid]}" if rid in RESTORATION else f"{base}/{rid}")

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

    # 24. Phase 3E (E7): concurrency & isolation — evidence gates.
    #     Rejects: claimed isolation w/o evidence, serializability beyond the
    #     blind-write subset, linearizability w/o full real-time history,
    #     dirty-read contradiction, partial-txn visibility denial, lost
    #     acknowledged write, reference-model mismatch, event-log corruption,
    #     missing raw run, unregistered experiment.
    for rid in ("PH3E-CONC-001", "PH3E-CONC-002", "PH3E-WAL-008", "PH3E-DUR-011",
                "PH3E-E3-005", "PH3E-BACKUP-007", "PH3E-COMPACT-006", "PH3E-TXN-002"):
        if rid not in ids:
            ERR.append(f"{rid} missing from registry (E7 family)")
    d7 = os.path.join(run, "PH3E-CONC-002")
    if not os.path.isdir(d7):
        ERR.append("PH3E-CONC-002 registered but raw dir missing")
    else:
        for f in ("run_info.txt", "config.json", "e7-conc.csv", "e7-crash.csv",
                  "e7-events.csv", "e7-visibility.csv"):
            if not os.path.exists(os.path.join(d7, f)):
                ERR.append(f"PH3E-CONC-002: missing {f}")
    e7_res = os.path.join(RES, "e7-concurrency.csv")
    e7_cres = os.path.join(RES, "e7-crash-atomicity.csv")
    e7_vres = os.path.join(RES, "e7-visibility-matrix.csv")
    if not (os.path.exists(e7_res) and os.path.exists(e7_cres) and os.path.exists(e7_vres)):
        ERR.append("results/e7-*.csv missing (run generate_results_ph3e.py)")
    else:
        c7 = rd(e7_res)
        raw7 = rd(os.path.join(d7, "e7-conc.csv")) if os.path.isdir(d7) else []
        if len(c7) != len(raw7):
            ERR.append(f"e7 row drift raw {len(raw7)} vs results {len(c7)}")
        for sr in c7:
            if sr["match"] not in ("MATCH", "-"):
                ERR.append(f"e7 violation: {sr['family']}/{sr['case']}")
            if sr["match"] == "-":
                blob = " ".join(str(v) for v in sr.values())
                if "UNSUPPORTED" not in blob:
                    ERR.append(f"e7 '-' row without UNSUPPORTED justification: {sr['family']}")
        c7c = rd(e7_cres)
        raw7c = rd(os.path.join(d7, "e7-crash.csv")) if os.path.isdir(d7) else []
        if len(c7c) != len(raw7c):
            ERR.append(f"e7 crash row drift raw {len(raw7c)} vs results {len(c7c)}")
        for sr in c7c:
            if sr["match"] != "MATCH" or sr["atomicity"] != "ATOMIC" or sr["aborted_at_gate"] != "true":
                ERR.append(f"e7 crash row without fresh-process ATOMIC evidence: {sr['case']}/{sr['mode']}")
        # dirty-read contradiction guard: any claimed dirty-read observation?
        for sr in c7:
            blob = " ".join(str(v) for v in sr.values()).lower()
            if "dirty read observed" in blob or "uncommitted value observed" in blob:
                ERR.append(f"e7 dirty-read contradiction: {sr['family']}/{sr['case']}")
        # event-log sanity: monotonic seq, non-empty
        evp = os.path.join(d7, "e7-events.csv")
        if os.path.exists(evp):
            ev = rd(evp)
            if len(ev) < 1000:
                ERR.append(f"e7 event log suspiciously small ({len(ev)} rows)")
            seqs = [int(r["seq"]) for r in ev]
            if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
                ERR.append("e7 event log seq corruption")
        # full history guard: E7u must have UNSAMPLED reads (more u-read rows
        # than a 1/256 sample could produce for its runtime)
        u_reads = [r for r in ev if r["op"] == "u-read"] if os.path.exists(evp) else []
        if len(u_reads) < 200:
            ERR.append("e7u linearizability history not FULL (sampled) — claim would be unsupported")
    # superseded first run preserved
    d71 = os.path.join(run, "PH3E-CONC-001")
    if not os.path.isdir(d71):
        ERR.append("PH3E-CONC-001 missing (superseded runs are never deleted)")
    bugdir = os.path.join(d71, "bug-MISSING_MAPPING-preserved-dir")
    if not os.path.isdir(bugdir):
        ERR.append("defect-#1 preserved failing dir missing (bug protocol)")
    # E1-E6 regression equivalence chains extended through E7
    for base, reg, f in (
        ("PH3E-WAL-007", "PH3E-WAL-008", "wal-integrity.csv"),
        ("PH3E-E3-004", "PH3E-E3-005", "e3-matrix.csv"),
        ("PH3E-TXN-001", "PH3E-TXN-002", "e6-txn.csv"),
        ("PH3E-TXN-001", "PH3E-TXN-002", "e6-crash.csv"),
        ("PH3E-COMPACT-005", "PH3E-COMPACT-006", "e5-crash.csv"),
        ("PH3E-BACKUP-006", "PH3E-BACKUP-007", "e4-integrity.csv"),
    ):
        fb = os.path.join(run, base, f)
        fr = os.path.join(run, reg, f)
        if os.path.exists(fb) and os.path.exists(fr):
            if sorted(open(fb).read().splitlines()[1:]) != sorted(open(fr).read().splitlines()[1:]):
                ERR.append(f"{reg}/{f} differs from {base} (byte-identity chain broken)")
        else:
            ERR.append(f"regression pair missing: {reg}/{f}")
    # durability byte-identity (four classification CSVs)
    for f in ("ack-boundary.csv", "txn-ack.csv", "checkpoint-interaction.csv", "group-boundary.csv"):
        fb = os.path.join(run, "PH3E-DUR-010", f)
        fr = os.path.join(run, "PH3E-DUR-011", f)
        if os.path.exists(fb) and os.path.exists(fr):
            if sorted(open(fb).read().splitlines()[1:]) != sorted(open(fr).read().splitlines()[1:]):
                ERR.append(f"PH3E-DUR-011/{f} differs from PH3E-DUR-010")
        else:
            ERR.append(f"regression pair missing: PH3E-DUR-011/{f}")
    # MVCC / isolation overclaim guards in results
    if os.path.exists(e7_res):
        for sr in rd(e7_res):
            blob = " ".join(str(v) for v in sr.values()).lower()
            for banned in ("mvcc verified", "serializable level", "snapshot isolation provided"):
                if banned in blob:
                    ERR.append(f"e7 overclaim '{banned}': {sr['family']}/{sr['case']}")
    if not os.path.exists(os.path.join(HERE, "phase3e-e7-final-report.md")):
        ERR.append("phase3e-e7-final-report.md missing (E7 final report)")
    if not os.path.exists(os.path.join(HERE, "methodology", "phase3e-e7-deviations.md")):
        ERR.append("methodology/phase3e-e7-deviations.md missing (E7 deviations)")
    rep = os.path.join(HERE, "phase3e-e7-final-report.md")
    if os.path.exists(rep):
        rtext = open(rep).read()
        for h in range(1, 36):
            if f"## §{h} " not in rtext:
                ERR.append(f"E7 final report missing heading §{h}")
    contract = os.path.join(HERE, "methodology", "production-contract.md")
    if os.path.exists(contract) and "A7 — Concurrency & Isolation Guarantee" not in open(contract).read():
        ERR.append("production contract lacks A7 (E7 evidence-warranted amendment)")

    # 25. Phase 3E (E8): soak/stability/reliability — evidence gates.
    #     Rejects: any official soak run with verification failures or reader
    #     violations, INVALIDATED runs without preserved INVALIDATED.md,
    #     missing telemetry, registry drift, report/heading/numerical drift,
    #     memory-optimization contamination (E8 measures, E9 optimizes),
    #     production-ready claims from soak, missing final report.
    e8_runs = [("PH3E-SOAK-001", "COMPLETED"), ("PH3E-SOAK-002", "COMPLETED"),
               ("PH3E-SOAK-003", "COMPLETED"), ("PH3E-SOAK-004", "COMPLETED"),
               ("PH3E-SOAK-005", "COMPLETED"), ("PH3E-SOAK-006", "COMPLETED"),
               ("PH3E-SOAK-007", "COMPLETED"), ("PH3E-SOAK-008", "COMPLETED"),
               ("PH3E-SOAK-009", "INVALIDATED"), ("PH3E-SOAK-010", "INVALIDATED"),
               ("PH3E-SOAK-011", "COMPLETED")]
    triage = os.path.join(run, "TRIAGE-2026-09-22.md")
    if not os.path.exists(triage):
        ERR.append("raw/runs/TRIAGE-2026-09-22.md missing (platform-loss record)")
    else:
        ttext = open(triage).read()
        for rid in ("PH3E-SOAK-002", "PH3E-SOAK-003", "PH3E-SOAK-006",
                    "PH3E-SOAK-008", "PH3E-SOAK-010", "PH3E-SOAK-011"):
            if rid not in ttext:
                ERR.append(f"TRIAGE does not document {rid}")
    for rid, status in e8_runs:
        ent = [e for e in idx["experiments"] if e["experiment_id"] == rid]
        if not ent:
            ERR.append(f"{rid} missing from registry (E8 family)")
            continue
        if ent[0]["status"] != status:
            ERR.append(f"{rid} registry status {ent[0]['status']} != {status}")
        d8 = os.path.join(run, rid)
        if not os.path.isdir(d8):
            ERR.append(f"{rid} registered but raw dir missing")
            continue
        # artifact chain: counts/resource/verif may live in the restoration run
        adir = d8
        if not os.path.exists(os.path.join(d8, "counts.json")) and rid in RESTORATION:
            adir = os.path.join(run, RESTORATION[rid])
            rent = [e for e in idx["experiments"] if e["experiment_id"] == RESTORATION[rid]]
            if not rent or rent[0].get("restoration_for") != rid:
                ERR.append(f"{RESTORATION[rid]} not registered as restoration_for {rid}")
            else:
                orig_ops = ent[0].get("metrics", {}).get("ops")
                new_ops = rent[0].get("metrics", {}).get("ops")
                if orig_ops != new_ops:
                    ERR.append(f"restoration {RESTORATION[rid]} ops {new_ops} != original {rid} ops {orig_ops}")
        if rid == "PH3E-SOAK-010":
            # the one accepted documented-loss run: INVALIDATED, not re-creatable
            # (its harness bugs were fixed); loss recorded in TRIAGE + report s26.
            # A COMPLETED run must NEVER take this path.
            continue
        for f in ("counts.json", "resource.csv", "verif.csv"):
            if not os.path.exists(os.path.join(adir, f)):
                ERR.append(f"{rid}: missing {f} (chain dir {os.path.basename(adir)})")
        if not (os.path.exists(os.path.join(adir, "oplog.csv")) or
                os.path.exists(os.path.join(adir, "oplog.csv.gz"))):
            ERR.append(f"{rid}: missing oplog.csv[.gz] (chain dir {os.path.basename(adir)})")
        if status == "INVALIDATED" and rid == "PH3E-SOAK-009" and \
                not os.path.exists(os.path.join(d8, "INVALIDATED.md")):
            ERR.append(f"{rid} INVALIDATED without preserved INVALIDATED.md")
    res8 = os.path.join(RES, "e8-soak.csv")
    mem8 = os.path.join(RES, "e8-memory.csv")
    tab8 = os.path.join(TAB, "table-e8-soak.md")
    if not (os.path.exists(res8) and os.path.exists(mem8) and os.path.exists(tab8)):
        ERR.append("results/e8-*.csv or tables/table-e8-soak.md missing (run generator)")
    else:
        s8 = rd(res8)
        if len(s8) != 9:
            ERR.append(f"e8-soak.csv must hold exactly the 9 official families, got {len(s8)}")
        for sr in s8:
            if sr["status"] != "VERIFIED":
                ERR.append(f"e8 official {sr['run']} not VERIFIED")
            if int(sr["verification_failures"]) or int(sr["reader_violations"]):
                ERR.append(f"e8 official {sr['run']} carries verification/reader failures")
            a8 = e8_art_dir(sr["run"], run)
            c8 = json.load(open(os.path.join(a8, "counts.json")))
            if int(sr["ops"]) != c8["ops"] or int(sr["verifications"]) != c8.get("verifications", 0):
                ERR.append(f"e8 {sr['run']} counts drift vs raw")
            if sr["run"] not in open(tab8).read():
                ERR.append(f"e8 table missing {sr['run']}")
        m8 = rd(mem8)
        if len(m8) != 9:
            ERR.append(f"e8-memory.csv must hold 9 families, got {len(m8)}")
        # S16 honesty gate: memory rows must be a classification, not a leak claim
        for mr_ in m8:
            if mr_["rss_class"] not in ("linear-with-ops", "bounded-plateau", "stepwise", "unexplained", "mixed"):
                ERR.append(f"e8 memory class illegal for {mr_['run']}: {mr_['rss_class']}")
    rep8 = os.path.join(HERE, "phase3e-e8-final-report.md")
    if not os.path.exists(rep8):
        ERR.append("phase3e-e8-final-report.md missing (E8 final report)")
    else:
        rtext8 = open(rep8).read()
        if "## §1 " not in rtext8 or "## §31 " not in rtext8:
            ERR.append("E8 final report must use exactly 31 fixed headings §1..§31")
        for h in range(1, 32):
            if f"## §{h} " not in rtext8:
                ERR.append(f"E8 final report missing heading §{h}")
        if "E8 COMPLETE; E9 NOT STARTED" not in rtext8:
            ERR.append("E8 final report missing the HARD STOP line")
        low8 = rtext8.lower()
        if "production-ready" in low8 and "not production-ready" not in low8 and "≠ production-ready" not in rtext8 and "is not production" not in low8:
            ERR.append("E8 report must not imply production-readiness from soak")
        dev8 = os.path.join(HERE, "methodology", "phase3e-e8-deviations.md")
        if not os.path.exists(dev8):
            ERR.append("phase3e-e8-deviations.md missing (E8 deviations gate)")
        else:
            dtext8 = open(dev8).read()
            for token in ("PH3E-SOAK-009", "PH3E-SOAK-010", "E8i", "E8d", "E8f",
                          "linear-with-ops", "E9"):
                if token not in dtext8:
                    ERR.append(f"E8 deviations missing token {token}")
        if "e8-soak.csv" not in rtext8:
            ERR.append("E8 final report does not cite results/e8-soak.csv")

    # 26. Phase 3E (E9): memory root-cause & optimization — evidence gates.
    #     Rejects: missing registry entries for PH3E-MEM-001..029, missing
    #     before/after artifacts, optimization that fails its measured floor,
    #     missing INV-E9-HYGIENE mechanism or its regression test, missing
    #     A9 amendment, report heading/verdict drift, leak-claim wording.
    for n in list(range(1, 30)):
        rid = f"PH3E-MEM-{n:03d}"
        if not any(e["experiment_id"] == rid for e in idx["experiments"]):
            ERR.append(f"{rid} missing from registry (E9 family)")
        if not os.path.isdir(os.path.join(run, rid)):
            ERR.append(f"{rid} registered but raw dir missing")
    ba = os.path.join(RES, "e9-before-after.csv")
    ct = os.path.join(RES, "e9-controls.csv")
    t9 = os.path.join(TAB, "table-e9-memory.md")
    f9 = os.path.join(HERE, "figures", "e9-before-after.svg")
    if not all(os.path.exists(p) for p in (ba, ct, t9, f9)):
        ERR.append("results/e9-*.csv, tables/table-e9-memory.md or figures/e9-before-after.svg missing")
    else:
        for r in rd(ba):
            red = float(r["rss_reduction_pct"])
            if r["experiment"].startswith("repro") and red < 55:
                ERR.append(f"E9 repro reduction {red}% below the 55% measured floor")
            if r["experiment"].startswith("restart") and red < 30:
                ERR.append(f"E9 restart-reset reduction {red}% below floor")
            if r["experiment"].startswith("growing") and abs(red) > 5:
                ERR.append("E9 growing-dataset workload must be unchanged within 5% (no dead nodes)")
    if not os.path.exists("/home/user/AttentionDB/core/tests/regression_e9_index_hygiene.rs"):
        ERR.append("regression_e9_index_hygiene.rs missing (E9 seal)")
    eng = open("/home/user/AttentionDB/core/src/engine.rs").read()
    if "INV-E9-HYGIENE" not in eng:
        ERR.append("INV-E9-HYGIENE missing from engine.rs")
    contract9 = os.path.join(HERE, "methodology", "production-contract.md")
    if not os.path.exists(contract9) or "A9 — Resource Hygiene Guarantee" not in open(contract9).read():
        ERR.append("production contract lacks A9 (E9 evidence-warranted amendment)")
    rep9 = os.path.join(HERE, "phase3e-e9-final-report.md")
    if not os.path.exists(rep9):
        ERR.append("phase3e-e9-final-report.md missing")
    else:
        rtext9 = open(rep9).read()
        for h in range(1, 26):
            if f"## {h}. " not in rtext9:
                ERR.append(f"E9 final report missing heading {h}.")
        if "E9 COMPLETE; E10 NOT STARTED" not in rtext9:
            ERR.append("E9 final report missing the HARD STOP line")
        low9 = rtext9.lower()
        if "no memory leak" in low9 or "not a leak}" in low9 or "no leak" in low9:
            ERR.append("E9 report must not make leak-freedom claims (A9 wording binds)")
    dev9 = os.path.join(HERE, "methodology", "phase3e-e9-deviations.md")
    if not os.path.exists(dev9):
        ERR.append("phase3e-e9-deviations.md missing")
    else:
        dtext9 = open(dev9).read()
        for token in ("D30", "D31", "D32", "D33", "D34", "D35", "D36", "mallinfo", "PH3E-SOAK-017"):
            if token not in dtext9:
                ERR.append(f"E9 deviations missing token {token}")

    # 27. Phase 3E (E10): scale-envelope — evidence gates.
    cap = os.path.join(RES, "e10-capacity.csv")
    tabc = os.path.join(TAB, "table-e10-capacity.md")
    figc = os.path.join(HERE, "figures", "e10-capacity.svg")
    if not all(os.path.exists(p) for p in (cap, tabc, figc)):
        ERR.append("results/e10-capacity.csv, tables/table-e10-capacity.md or figures/e10-capacity.svg missing")
    else:
        crows = rd(cap)
        if len(crows) < 21:
            ERR.append(f"e10-capacity.csv must hold the 21 registered runs, got {len(crows)}")
        reg10 = [e for e in idx["experiments"] if e["experiment_id"].startswith("PH3E-SCALE-")]
        if len(reg10) != 21:
            ERR.append(f"registry must hold 21 PH3E-SCALE runs, got {len(reg10)}")
        for cr in crows:
            if not any(e["experiment_id"] == cr["run"] for e in idx["experiments"]):
                ERR.append(f"{cr['run']} in capacity table but missing from registry")
        for n in range(1, 22):
            rid = f"PH3E-SCALE-{n:03d}"
            if not os.path.isdir(os.path.join(run, rid)):
                ERR.append(f"{rid} registered but raw dir missing")
    rep10 = os.path.join(HERE, "phase3e-e10-final-report.md")
    if not os.path.exists(rep10):
        ERR.append("phase3e-e10-final-report.md missing")
    else:
        rtext10 = open(rep10).read()
        for h in range(1, 33):
            if f"## {h}. " not in rtext10:
                ERR.append(f"E10 final report missing heading {h}.")
        if "E10 COMPLETE; E11 NOT STARTED" not in rtext10:
            ERR.append("E10 final report missing the HARD STOP line")
        if "million" in rtext10.lower():
            ERR.append("E10 report must not contain million-document claims")
    contract10 = os.path.join(HERE, "methodology", "production-contract.md")
    if not os.path.exists(contract10) or "A10 — Scale Envelope" not in open(contract10).read():
        ERR.append("production contract lacks A10 (E10 evidence-warranted amendment)")
    eng10 = open("/home/user/AttentionDB/core/src/engine.rs").read()
    if "E10 SCALE-DEFECT #1" not in eng10:
        ERR.append("SCALE-DEFECT #1 fix marker missing from engine.rs")
    if not os.path.exists("/home/user/AttentionDB/core/tests/regression_e9_index_hygiene.rs"):
        ERR.append("regression_e9_index_hygiene.rs missing (E10 seal)")
    dev10 = os.path.join(HERE, "methodology", "phase3e-e10-deviations.md")
    if not os.path.exists(dev10):
        ERR.append("phase3e-e10-deviations.md missing")
    else:
        dtext10 = open(dev10).read()
        for token in ("D38", "D39", "D40", "D41", "D42", "D43", "D44", "D45", "D46", "D47"):
            if token not in dtext10:
                ERR.append(f"E10 deviations missing token {token}")

    # 28. Phase 3E (E11): fault-injection evidence gates.
    rep11 = os.path.join(HERE, "phase3e-e11-final-report.md")
    verd = os.path.join(HERE, "phase3e-final-reliability-verdict.md")
    dev11 = os.path.join(HERE, "methodology", "phase3e-e11-deviations.md")
    spec11 = os.path.join(HERE, "methodology", "phase3e-e11-spec.md")
    cmat = os.path.join(HERE, "methodology", "phase3e-e11-contract-matrix.md")
    for need in (rep11, verd, dev11, spec11, cmat):
        if not os.path.exists(need):
            ERR.append(f"E11 required document missing: {os.path.basename(need)}")
    if os.path.exists(dev11):
        dt = open(dev11).read()
        for tok in ("D49", "D50", "D51", "D52", "D53", "D54", "D55"):
            if tok not in dt:
                ERR.append(f"E11 deviations missing token {tok}")
    runroot = os.path.join(HERE, "raw", "runs")
    fault_dirs = sorted(d for d in os.listdir(runroot) if d.startswith("PH3E-FAULT-") and os.path.isdir(os.path.join(runroot, d)))
    reg_ids = [e["experiment_id"] for e in idx["experiments"]]
    if len(reg_ids) != len(set(reg_ids)):
        ERR.append("duplicate experiment_ids in registry")
    for rid in fault_dirs:
        if rid not in reg_ids:
            ERR.append(f"{rid} raw dir present but not registered")
    fault_reg = [r for r in reg_ids if r.startswith("PH3E-FAULT-")]
    for rid in fault_reg:
        if rid not in fault_dirs:
            ERR.append(f"{rid} registered but raw dir missing")
        else:
            rd_ = os.path.join(runroot, rid)
            for art in ("config.json", "fault-plan.json", "exit-status.json", "summary.json", "checksums.sha256"):
                if not os.path.exists(os.path.join(rd_, art)):
                    ERR.append(f"{rid} missing required artifact {art}")
            ssum = json.load(open(os.path.join(rd_, "summary.json")))
            corr = os.path.join(rd_, "classification-correction.json")
            cls = ssum.get("classification", "?")
            if os.path.exists(corr):
                cls = json.load(open(corr))["artifact_derived_classification"]
            if cls not in ("VERIFIED", "SUPPORTED", "OBSERVED_LIMIT", "FAILED", "INVALIDATED", "BLOCKED", "UNSUPPORTED"):
                ERR.append(f"{rid} has out-of-vocabulary classification {cls!r}")
            if cls in ("VERIFIED", "SUPPORTED") and not os.path.exists(os.path.join(rd_, "recovery-verification.json")):
                ERR.append(f"{rid} claims {cls} without recovery-verification.json")
    corrective = {"PH3E-FAULT-007": "PH3E-FAULT-038", "PH3E-FAULT-022": "PH3E-FAULT-039",
                  "PH3E-FAULT-024": "PH3E-FAULT-040", "PH3E-FAULT-032": "PH3E-FAULT-042",
                  "PH3E-FAULT-034": "PH3E-FAULT-041"}
    reg_set = set(reg_ids)
    for orig, corr in corrective.items():
        if orig in reg_set:
            if corr not in reg_set:
                ERR.append(f"INVALIDATED run {orig} lacks registered corrective {corr}")
            else:
                ccorr = os.path.join(runroot, corr, "classification-correction.json")
                ccls = json.load(open(os.path.join(runroot, corr, "summary.json"))).get("classification")
                if os.path.exists(ccorr):
                    ccls = json.load(open(ccorr))["artifact_derived_classification"]
                if ccls not in ("VERIFIED", "SUPPORTED"):
                    ERR.append(f"corrective run {corr} did not succeed ({cs.get('classification')})")
    e11res = os.path.join(HERE, "results", "e11-runs.csv")
    if not os.path.exists(e11res):
        ERR.append("results/e11-runs.csv missing (gen_e11 not run)")
    else:
        for t in ("table-e11-fault-coverage", "table-e11-ack-recovery", "table-e11-txn-atomicity",
                  "table-e11-backup-restore", "table-e11-ckpt-compact", "table-e11-integrated",
                  "table-e11-failure-register", "table-e11-evidence-coverage"):
            if not os.path.exists(os.path.join(HERE, "tables", t + ".md")):
                ERR.append(f"tables/{t}.md missing")
        if not os.path.exists(os.path.join(HERE, "figures", "e11-ack-recovery.svg")):
            ERR.append("figures/e11-ack-recovery.svg missing")
    if os.path.exists(rep11):
        r11 = open(rep11).read()
        for h in range(1, 22):
            if f"## {h}. " not in r11:
                ERR.append(f"E11 final report missing heading {h}.")
        if "PHASE 3E E11 COMPLETE — FINAL RELIABILITY VERDICT ISSUED" not in r11 and            "PHASE 3E E11 INCOMPLETE — BLOCKERS AND UNVERIFIED CONTRACTS DOCUMENTED" not in r11:
            ERR.append("E11 final report missing the terminal status line")
        if "power-loss" in r11.lower() and "UNSUPPORTED" not in r11:
            ERR.append("E11 report mentions power-loss without the UNSUPPORTED qualifier")
    contract11 = os.path.join(HERE, "methodology", "production-contract.md")
    if not os.path.exists(contract11) or "A11 —" not in open(contract11).read():
        ERR.append("production contract lacks A11 (E11 fault-model qualification)")
    if not os.path.exists("/home/user/AttentionDB/core/tests/regression_e11_faults.rs"):
        ERR.append("regression_e11_faults.rs missing (E11 seal)")

    if ERR:
        print("PHASE 3 CONSISTENCY CHECK FAILED:")
        for e in ERR:
            print("  ✗", e)
        sys.exit(1)
    print("phase3 consistency check: PASS (tables ↔ results ↔ registry; exact-reference anchor holds)")


if __name__ == "__main__":
    main()
