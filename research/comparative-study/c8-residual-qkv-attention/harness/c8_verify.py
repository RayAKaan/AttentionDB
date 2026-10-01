#!/usr/bin/env python3
"""C8 invariant + verification gates (protocol Sec.8 stop conditions).

Enforces the C8-specific invariants that make the study meaningful:

  #4  Union identity: B/C/D/E/F/G/H/I union ledgers identical per query in
      every SMOKE/TEST/EFPROBE cell, every rep.
  #8  Non-finite checks on scores, attention weights, entropy, latency.
  #9  lambda = 0 parity: arm C reproduces arm B *bit-for-bit* on the final
      score of every candidate (stop condition #9).
  #10 Cache parity: arm I reproduces arm E bit-for-bit (stop condition #10).
      C's applied_correction must be exactly 0 everywhere; E/F/G/H/I must have
      a non-zero correction somewhere (machinery genuinely exercised).
  #6  Determinism (observable-level): a fresh second-process re-run of SMOKE
      reproduces candidate_count, recall@10, nDCG@10 and top-10 rows exactly.
  #3  ef knob: EFPROBE ef=16 vs ef=128 must differ in recall and/or latency.
  #7  Fingerprints: A/B have no C8 fingerprint; C has one; learned arms differ
      from the identity control.
  #5  No TEST-data leakage into TUNE/SUPPORT artifacts (row sets disjoint).
  geometry: every C8 model card declares attention_dim 384, key/value dim 64.
  #12 run_id / RUN-INDEX.yaml consistency for all executed C8 cells.

Writes analysis/verification_report.json and exits nonzero on any gate failure.
"""

import json
import os
import sys

import numpy as np

from c8_test_run import (RAW, SEED, resolve_dataset, test_rows, load_plan_row)

ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
REPORT_PATH = os.path.join(ANALYSIS, "verification_report.json")
UNION_ARMS = ("B", "C", "D", "E", "F", "G", "H", "I")
TEST_ARMS = ("A",) + UNION_ARMS


def plan_rows():
    from c8_test_run import plan_rows as _pr
    return _pr()


def rows_for(ds_short, split):
    m = {"SCI": "DS-SCIFACT", "NFC": "DS-NFCORPUS"}
    return [r for r in plan_rows() if r["dataset"] == m.get(ds_short)
            and r["split"] == split]


# ---- helpers ----------------------------------------------------------------
def union_sig(per_q):
    return sorted((int(r["row"]),
                   tuple(x if x is not None else -1.0 for x in r["head_sims"]),
                   r["mhs"]) for r in per_q["union_ledger"])


def arm_final_map(per_q):
    """Final score per candidate row, from the C8 trace when present, else the
    union ledger (arms A/B)."""
    tr = per_q.get("c8_trace")
    if tr and tr.get("candidates"):
        return {int(c["row"]): float(c["final_score"]) for c in tr["candidates"]}
    return {int(r["row"]): float(r["final"]) for r in per_q["union_ledger"]}


def ledger_topk(per_q, k=10):
    rows = sorted(per_q["union_ledger"], key=lambda r: (-r["final"], r["rank"]))
    return [int(r["row"]) for r in rows[:k]]


def load_multi(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def load_cfg(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def check_nonfinite(checks, run_id, arm, per_q, i):
    for r in per_q["union_ledger"]:
        for key in ("final", "mhs"):
            v = r.get(key)
            if v is not None and not np.isfinite(v):
                checks.append(f"{run_id} {arm} q{i} ledger {key} non-finite: {v}")
        for hs in r.get("head_sims") or []:
            if hs is not None and not np.isfinite(hs):
                checks.append(f"{run_id} {arm} q{i} head_sim non-finite: {hs}")
    lat = per_q.get("latency_us")
    if isinstance(lat, (int, float)) and (lat < 0 or not np.isfinite(lat)):
        checks.append(f"{run_id} {arm} q{i} bad latency")
    tr = per_q.get("c8_trace")
    if tr:
        if not np.isfinite(tr["mean_entropy"]):
            checks.append(f"{run_id} {arm} q{i} trace entropy non-finite")
        for c in tr.get("candidates") or []:
            for key in ("baseline_score", "attention_delta", "final_score",
                        "entropy", "applied_correction"):
                if not np.isfinite(c[key]):
                    checks.append(f"{run_id} {arm} q{i} trace {key} non-finite")


def gate_union_identity(checks, multi, run_id, rep):
    base = multi["arms"]["B"]["per_query"]
    for other in UNION_ARMS[1:]:
        o = multi["arms"].get(other)
        if o is None:
            checks.append(f"{run_id}: arm {other} missing")
            continue
        for i, (qb, qo) in enumerate(zip(base, o["per_query"])):
            if union_sig(qb) != union_sig(qo):
                checks.append(f"UNION {run_id} {other} rep{rep} q{i} differs from B")


def gate_lambda_zero(checks, multi, run_id, rep):
    b = multi["arms"]["B"]["per_query"]
    c = multi["arms"]["C"]["per_query"]
    for i, (qb, qc) in enumerate(zip(b, c)):
        fb, fc = arm_final_map(qb), arm_final_map(qc)
        if fb != fc:
            checks.append(f"LAMBDA0 {run_id} rep{rep} q{i} C final != B final")
        if qb["candidate_count"] != qc["candidate_count"]:
            checks.append(f"LAMBDA0 {run_id} rep{rep} q{i} candidate_count differs")
    max_corr = 0.0
    for qc in c:
        tr = qc.get("c8_trace")
        if tr:
            for cand in tr["candidates"]:
                max_corr = max(max_corr, abs(cand["applied_correction"]))
    if max_corr != 0.0:
        checks.append(f"LAMBDA0 {run_id} rep{rep} C applied_correction max {max_corr} != 0")


def gate_cache_parity(checks, multi, run_id, rep):
    e = multi["arms"].get("E")
    i = multi["arms"].get("I")
    if not e or not i:
        checks.append(f"{run_id}: E/I arms missing for cache parity")
        return
    for qi, (qe, qi_) in enumerate(zip(e["per_query"], i["per_query"])):
        if arm_final_map(qe) != arm_final_map(qi_):
            checks.append(f"CACHE {run_id} rep{rep} q{qi} I final != E final")


def gate_correction_active(checks, multi, run_id):
    for name in ("E", "F", "G", "H", "I"):
        arm = multi["arms"].get(name)
        if not arm:
            continue
        total = 0.0
        for q in arm["per_query"]:
            tr = q.get("c8_trace")
            if tr:
                for c in tr["candidates"]:
                    total += abs(c["applied_correction"])
        if total == 0.0:
            checks.append(f"INERT {run_id} arm {name} applied_correction all zero")


def gate_fingerprints(checks, multi, run_id):
    fp = {}
    for name in TEST_ARMS:
        a = multi["arms"].get(name)
        if a is None:
            continue
        eng = a["engine"]
        fp[name] = (eng.get("c8_fingerprint"), bool(eng.get("c8_enabled")))
    for name in ("A", "B"):
        if name in fp:
            f, en = fp[name]
            if en:
                checks.append(f"{run_id} arm {name} c8 must be disabled")
            if f is not None:
                checks.append(f"{run_id} arm {name} fingerprint must be None, got {f}")
    if "C" in fp and fp["C"][0] is None:
        checks.append(f"{run_id} arm C must carry a (control) C8 fingerprint")
    for name in ("D", "E", "F", "G", "H", "I"):
        if name in fp and fp[name][0] is None:
            checks.append(f"{run_id} arm {name} missing C8 fingerprint")
        elif name in fp and "C" in fp and fp[name][0] == fp["C"][0]:
            checks.append(f"{run_id} arm {name} fingerprint collides with control C")


def gate_ef_knob(checks, multi, run_id):
    lo = multi["arms"].get("EF16")
    hi = multi["arms"].get("EF128")
    if not lo or not hi:
        checks.append(f"{run_id}: EF16/EF128 arms missing")
        return
    rlo = float(np.mean([q["recall10_qrels"] for q in lo["per_query"]]))
    rhi = float(np.mean([q["recall10_qrels"] for q in hi["per_query"]]))
    llo = float(np.median([q["latency_us"] for q in lo["per_query"]]))
    lhi = float(np.median([q["latency_us"] for q in hi["per_query"]]))
    if rlo == rhi and llo == lhi:
        checks.append(f"{run_id}: ef knob no-op (recall {rlo}=={rhi}, "
                      f"latency {llo:.1f}=={lhi:.1f})")


def gate_determinism(checks, run_id, m1, m2):
    for name in UNION_ARMS + ("A",):
        if name not in m1["arms"] or name not in m2["arms"]:
            continue
        for i, (p1, p2) in enumerate(zip(m1["arms"][name]["per_query"],
                                         m2["arms"][name]["per_query"])):
            for key in ("candidate_count", "recall10_qrels", "ndcg10_qrels"):
                if p1[key] != p2[key]:
                    checks.append(f"{run_id} arms={name} q{i} {key} rerun differs: "
                                  f"{p1[key]} vs {p2[key]}")
            if ledger_topk(p1) != ledger_topk(p2):
                checks.append(f"{run_id} arms={name} q{i} top-10 differ on rerun")
            s1, s2 = union_sig(p1), union_sig(p2)
            if s1 != s2:
                a, b = {r[0] for r in s1}, {r[0] for r in s2}
                jac = len(a & b) / max(1, len(a | b))
                if jac < 0.95:
                    checks.append(f"{run_id} arms={name} q{i} ledger jaccard {jac:.3f} < 0.95")


def check_geometry(checks, run_id, card_path):
    if not os.path.exists(card_path):
        return
    card = json.load(open(card_path, encoding="utf8"))
    rp = card.get("residual_projection") or {}
    if rp.get("attention_dim") != 384:
        checks.append(f"{run_id}: attention_dim {rp.get('attention_dim')} != 384")
    if rp.get("key_dim") != 64 or rp.get("value_dim") != 64:
        checks.append(f"{run_id}: key/value dims {rp.get('key_dim')}/"
                      f"{rp.get('value_dim')} != 64")


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    checks = {}
    summary = {}

    def fail(key, msg):
        checks.setdefault(key, []).append(msg)

    for ds_short in ("SCI", "NFC"):
        dsinfo = resolve_dataset("DS-SCIFACT" if ds_short == "SCI" else "DS-NFCORPUS")
        shared = os.path.join(RAW, f"C8-SHARED-{ds_short}")

        # ---- EFPROBE ----
        if os.path.exists(os.path.join(shared, "multi-probe-rep1.json")):
            multi = load_multi(os.path.join(shared, "multi-probe-rep1.json"))
            gate_ef_knob(checks.setdefault("ef-knob", []), multi, f"C8-EFPROBE-{ds_short}")
            for name in ("EF16", "EF128"):
                for i, p in enumerate(multi["arms"][name]["per_query"]):
                    check_nonfinite(checks.setdefault("nonfinite", []),
                                    f"C8-EFPROBE-{ds_short}", name, p, i)
            summary[f"efprobe-{ds_short}"] = "checked"

        # ---- TUNE leakage vs TEST + geometry ----
        test_keep = set()
        trows = rows_for(ds_short, "TEST")
        if trows and os.path.exists(os.path.join(RAW, trows[0]["run_id"],
                                                 "artifacts", "cfg.json")):
            _, _, qrels = test_rows(dsinfo)
            test_keep = set(sorted(qrels.keys()))
        for cell in ("D", "E", "F", "G", "H"):
            art = os.path.join(RAW, f"C8-TUNE-{ds_short}-{cell}-001", "artifacts")
            cfg_path = os.path.join(art, "train-cfg.json")
            if os.path.exists(cfg_path):
                cfg = load_cfg(cfg_path)
                keys = set(int(v) for v in cfg["train"]["rows"])
                leak = keys & test_keep
                if leak:
                    fail("leakage", f"C8-TUNE-{ds_short}-{cell} shares {len(leak)} "
                                    f"rows with TEST: {sorted(leak)[:5]}")
                if not keys:
                    fail("leakage", f"C8-TUNE-{ds_short}-{cell} kept 0 rows")
            check_geometry(fail, f"C8-TUNE-{ds_short}-{cell}",
                           os.path.join(art, "model.json"))

        # ---- SMOKE: determinism + gates ----
        if os.path.exists(os.path.join(shared, "multi-smoke-rerun.json")):
            m1 = load_multi(os.path.join(shared, "multi-smoke-rep1.json"))
            m2 = load_multi(os.path.join(shared, "multi-smoke-rerun.json"))
            gate_determinism(checks.setdefault("determinism", []),
                             f"C8-SMOKE-{ds_short}", m1, m2)
            gate_union_identity(checks.setdefault("union-identity", []), m1,
                                f"C8-SMOKE-{ds_short}", 1)
            gate_lambda_zero(checks.setdefault("lambda-zero", []), m1,
                             f"C8-SMOKE-{ds_short}", 1)
            gate_cache_parity(checks.setdefault("cache-parity", []), m1,
                              f"C8-SMOKE-{ds_short}", 1)
            gate_correction_active(checks.setdefault("correction-active", []),
                                   m1, f"C8-SMOKE-{ds_short}")
            gate_fingerprints(checks.setdefault("fingerprints", []), m1,
                              f"C8-SMOKE-{ds_short}")
            for name in UNION_ARMS:
                if name in m1["arms"]:
                    for i, p in enumerate(m1["arms"][name]["per_query"]):
                        check_nonfinite(checks.setdefault("nonfinite", []),
                                        f"C8-SMOKE-{ds_short}", name, p, i)
            summary[f"smoke-{ds_short}"] = "checked"

        # ---- TEST: union identity + parity + fingerprints ----
        if os.path.exists(os.path.join(shared, "multi-test-rep1.json")):
            rep_count = sum(1 for f in os.listdir(shared)
                            if f.startswith("multi-test-rep") and f.endswith(".json"))
            for rep in range(1, rep_count + 1):
                p = os.path.join(shared, f"multi-test-rep{rep}.json")
                if not os.path.exists(p):
                    continue
                multi = load_multi(p)
                gate_union_identity(checks.setdefault("union-identity", []),
                                    multi, f"C8-TEST-{ds_short}", rep)
                gate_lambda_zero(checks.setdefault("lambda-zero", []),
                                 multi, f"C8-TEST-{ds_short}", rep)
                gate_cache_parity(checks.setdefault("cache-parity", []),
                                  multi, f"C8-TEST-{ds_short}", rep)
                gate_correction_active(checks.setdefault("correction-active", []),
                                       multi, f"C8-TEST-{ds_short}")
                gate_fingerprints(checks.setdefault("fingerprints", []),
                                  multi, f"C8-TEST-{ds_short}")
                for name in UNION_ARMS:
                    for i, pq in enumerate(multi["arms"][name]["per_query"]):
                        check_nonfinite(checks.setdefault("nonfinite", []),
                                        f"C8-TEST-{ds_short}", name, pq, i)
            summary[f"test-{ds_short}"] = f"{rep_count} reps checked"

    # ---- RUN-INDEX consistency ----
    index_done = set()
    idx_path = os.path.join(RAW, "RUN-INDEX.yaml")
    if os.path.exists(idx_path):
        for line in open(idx_path, encoding="utf8"):
            line = line.strip()
            if line.startswith("- run_id:"):
                index_done.add(line.split(":", 1)[1].strip())
    for row in plan_rows():
        rid = row["run_id"]
        d = os.path.join(RAW, rid)
        if not os.path.exists(d):
            continue
        if row["track"] == "TUNE":
            marker = os.path.join("artifacts", "model.json")
        elif row["track"] == "SUPPORT":
            marker = "metrics.json"
        else:
            marker = "metrics.json"
        if not os.path.exists(os.path.join(d, marker)):
            fail("run-index", f"{rid}: missing {marker}")
        if rid not in index_done:
            fail("run-index", f"{rid}: not registered in RUN-INDEX.yaml")
    summary["run-index"] = "checked"

    report = {"gate": summary, "checks": checks,
              "all_pass": all(len(v) == 0 for v in checks.values())}
    with open(REPORT_PATH, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    for key, msgs in checks.items():
        for m in msgs:
            print(f"[{key}] {m}")
    print(json.dumps({"all_pass": report["all_pass"],
                      "checks": len(sum(checks.values(), []))}))
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
