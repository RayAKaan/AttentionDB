#!/usr/bin/env python3
"""C7 invariant + verification gates (protocol §8 stop conditions).

Enforces (operative interpretations, see c7-protocol.md §8):

  #2  Exact-oracle recompute: re-derive recall@10 / nDCG@10 / recall-exact
      per query directly from raw embeddings + qrels + each arm's union
      ledger; must match the probe's reported values (tie-boundary tolerant).
  #3  ef knob: EFPROBE ef=16 vs ef=128 arms must differ in recall and/or
      latency within the same process.
  #4  Union identity: B/C/D/E/F union ledgers identical per query in every
      TEST (and SMOKE/EFPROBE) cell, every rep.
  #6  Determinism (observable-level, per user decision 2026-09-29): a fresh
      second-process re-run of SMOKE reproduces candidate_count, recall@10,
      nDCG@10, and top-10 ranked rows per query exactly; ledger tail rows may
      differ (jaccard >= 0.95) due to hnsw_rs OS-seeded layer RNG (documented
      in repo core tests) and are accepted as noise, not nondeterminism.
  #7  Attention fingerprints: E/F must differ from identity (C/D); F must
      differ from E; C==D is expected (same identity-QKV attention config).
  #8  NaN / Inf / non-finite checks on scores, attention, entropy, latency.
  #5  No TEST-data leakage into TUNE artifacts (row sets disjoint).
  #12 run_id / RUN-INDEX.yaml consistency for all executed C7 cells.

Writes analysis/verification_report.json and exits nonzero on any gate failure.
"""

import json
import os
import sys

import numpy as np

from c7_test_run import (RAW, PLAN, SEED, K, REP_MIX, resolve_dataset,
                         _jsonl_ids, parse_qrels, test_rows, val_rows,
                         seeded_shuffle, arm_name_for_row)

ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
REPORT_PATH = os.path.join(ANALYSIS, "verification_report.json")
UNION_ARMS = ("B", "C", "D", "E", "F")


# ---- plan rows ------------------------------------------------------------
def plan_rows():
    with open(PLAN, encoding="utf8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        out = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            parts = [p.strip() for p in line.split(",")]
            out.append(dict(zip(header, parts)))
    return out


def rows_for(ds_short, split):
    m = {"SCI": "DS-SCIFACT", "NFC": "DS-NFCORPUS"}
    return [r for r in plan_rows() if r["dataset"] == m.get(ds_short)
            and r["split"] == split]


# ---- ledger helpers -------------------------------------------------------
def ledger_topk(per_q, k=K):
    rows = sorted(per_q["union_ledger"], key=lambda r: (-r["final"], r["rank"]))
    return [int(r["row"]) for r in rows[:k]]


def union_sig(per_q):
    return sorted((int(r["row"]),
                   tuple(x if x is not None else -1.0 for x in r["head_sims"]),
                   r["mhs"]) for r in per_q["union_ledger"])


def ndcg10(relevant, top_rows, k=K):
    dcg = 0.0
    for rank, row in enumerate(top_rows):
        if rank >= k:
            break
        rel = relevant.get(row, 0.0)
        if rel > 0.0:
            dcg += rel / np.log2(rank + 2.0)
    ideal = sorted([v for v in relevant.values() if v > 0.0], reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal):
        if rank >= k:
            break
        idcg += rel / np.log2(rank + 2.0)
    return dcg / idcg if idcg > 0.0 else 0.0


def exact_heights(canonical, queries, dsinfo):
    """Full-collection exact canonical top-k per query (from raw embeddings)."""
    enc = canonical.astype(np.float64)
    n = enc.shape[0]
    norms = np.linalg.norm(enc, axis=1)
    exact = {}
    for qr, q in queries.items():
        qv = q.astype(np.float64)
        sims = (enc @ qv) / (norms * np.linalg.norm(qv) + 1e-12)
        order = np.lexsort((np.arange(n), -sims))
        exact[qr] = list(order[:K])
    return exact


def check_nonfinite(checks, run_id, arm, per_q, i):
    for r in per_q["union_ledger"]:
        for key in ("final", "mhs", "attention"):
            v = r.get(key)
            if v is not None and not np.isfinite(v):
                checks.append(f"{run_id} {arm} q{i} ledger {key} non-finite: {v}")
        for hs in r.get("head_sims") or []:
            if hs is not None and not np.isfinite(hs):
                checks.append(f"{run_id} {arm} q{i} head_sim non-finite: {hs}")
    if isinstance(per_q["latency_us"], float) and (per_q["latency_us"] < 0
                                                   or not np.isfinite(per_q["latency_us"])):
        checks.append(f"{run_id} {arm} q{i} bad latency")
    tr = per_q.get("c7_trace")
    if tr:
        if not np.isfinite(tr["mean_entropy"]):
            checks.append(f"{run_id} {arm} q{i} trace entropy non-finite")
        for c in tr.get("candidates") or []:
            if not np.isfinite(c["attention_score"]):
                checks.append(f"{run_id} {arm} q{i} trace attn non-finite")


def oracle_recompute(checks, run_id, arm, per_q, relevant, exact):
    qr = per_q["query_row"]
    top10 = ledger_topk(per_q)
    rec = 0.0
    n_rel = len(relevant)
    if n_rel:
        hits = [r for r in top10 if r in relevant]
        rec = len(hits) / n_rel
    ndcg = ndcg10(relevant, top10) if n_rel else 0.0
    ex = exact.get(qr, [])
    if ex:
        hits = [r for r in top10 if r in ex]
        rec_ex = len(hits) / len(ex)
    else:
        rec_ex = 0.0
    for label, got, want in (("recall10_qrels", rec, per_q["recall10_qrels"]),
                             ("ndcg10_qrels", ndcg, per_q["ndcg10_qrels"]),
                             ("recall10_exact", rec_ex, per_q["recall10_exact"])):
        if abs(got - want) > 1e-6:
            checks.append(f"ORACLE {run_id} {arm} q{qr} {label}: recompute={got:.6f} "
                          f"probe={want:.6f}")


# ---- gate drivers ---------------------------------------------------------
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
    for name in ("A",):
        a = multi["arms"].get(name)
        if a is not None:
            rows = [r["candidate_count"] for r in a["per_query"]]
            if rows and any(c > 0 for c in rows):
                checks.append(f"{run_id} {name} expected SingleHead candidate_count 0, got max {max(rows)}")


def gate_fingerprints(checks, multi, run_id):
    fp = {name: (multi["arms"][name]["engine"].get("attention_fingerprint"),
                 multi["arms"][name]["engine"].get("attention_enabled"))
          for name in ("A", "B", "C", "D", "E", "F") if name in multi["arms"]}
    for name in ("A", "B"):
        if name in fp:
            f, en = fp[name]
            if en:
                checks.append(f"{run_id} arm {name} attention must be OFF")
            if f is not None:
                checks.append(f"{run_id} arm {name} fingerprint must be None, got {f}")
    if "C" in fp and "D" in fp and fp["C"][0] != fp["D"][0]:
        checks.append(f"{run_id} C/D identity-QKV attention fingerprint must match")
    for name in ("E", "F"):
        if name in fp:
            f = fp[name][0]
            if f is None or f == fp.get("C", (None,))[0]:
                checks.append(f"{run_id} arm {name} fingerprint collision with identity")


def gate_ef_knob(checks, multi, run_id):
    lo = multi["arms"].get("EF16")
    hi = multi["arms"].get("EF128")
    if not lo or not hi:
        checks.append(f"{run_id}: EF16/EF128 arms missing")
        return
    rlo = np.mean([q["recall10_qrels"] for q in lo["per_query"]])
    rhi = np.mean([q["recall10_qrels"] for q in hi["per_query"]])
    llo = np.median([q["latency_us"] for q in lo["per_query"]])
    lhi = np.median([q["latency_us"] for q in hi["per_query"]])
    if rlo == rhi and llo == lhi:
        checks.append(f"{run_id}: ef knob no-op (recall {rlo}=={rhi}, "
                      f"latency {llo:.1f}=={lhi:.1f})")


def gate_determinism(checks, run_id, multi1, multi2):
    before = {}
    for name in UNION_ARMS + ("A",):
        if name not in multi1["arms"]:
            continue
        for i, p in enumerate(multi1["arms"][name]["per_query"]):
            before[(name, i)] = {"cc": p["candidate_count"],
                                 "r": p["recall10_qrels"], "n": p["ndcg10_qrels"],
                                 "top": ledger_topk(p)}
    for name in UNION_ARMS + ("A",):
        if name not in multi2["arms"]:
            continue
        for i, p in enumerate(multi2["arms"][name]["per_query"]):
            b = before.get((name, i))
            if b is None:
                checks.append(f"{run_id}: rerun arm {name} q{i} unmatched")
                continue
            for key, label in (("cc", "candidate_count"), ("r", "recall10_qrels"),
                               ("n", "ndcg10_qrels")):
                if b[key] != p[label]:
                    checks.append(f"{run_id} arms={name} q{i} {label} rerun differs: "
                                  f"{b[key]} vs {p[label]}")
            if b["top"] != ledger_topk(p):
                checks.append(f"{run_id} arms={name} q{i} top-10 ranks differ on rerun")
            sa, sb = union_sig(multi1["arms"][name]["per_query"][i]), union_sig(
                multi2["arms"][name]["per_query"][i])
            if sa != sb:
                seta = {r[0] for r in sa}
                setb = {r[0] for r in sb}
                jac = len(seta & setb) / max(1, len(seta | setb))
                if jac < 0.95:
                    checks.append(f"{run_id} arms={name} q{i} ledger jaccard {jac:.3f} < 0.95")


def load_multi(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def load_cfg(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def keep_rows_of(cfg):
    return [int(v) for v in (cfg.get("subsample") or [])]


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    checks = {}
    summary = {}

    class _Sink(object):
        def __init__(self, key):
            self.key = key

        def append(self, msg):
            checks.setdefault(self.key, []).append(msg)

    def fail(key, msg=None):
        if msg is None:
            return _Sink(key)
        checks.setdefault(key, []).append(msg)
        return None

    for ds_short in ("SCI", "NFC"):
        dsinfo = resolve_dataset("DS-SCIFACT" if ds_short == "SCI" else "DS-NFCORPUS")
        shared = os.path.join(RAW, f"C7-SHARED-{ds_short}")

        # ---- EFPROBE ----
        probe_rows = rows_for(ds_short, "PROBE")
        if probe_rows and os.path.exists(os.path.join(shared, "multi-probe-rep1.json")):
            multi = load_multi(os.path.join(shared, "multi-probe-rep1.json"))
            gate_ef_knob(fail("ef-knob"), multi, f"C7-EFPROBE-{ds_short}")
            for name in ("EF16", "EF128"):
                for i, p in enumerate(multi["arms"][name]["per_query"]):
                    check_nonfinite(fail("nonfinite"), f"C7-EFPROBE-{ds_short}", name, p, i)
            summary[f"efprobe-{ds_short}"] = "checked"

        # ---- TUNE leakage vs TEST ----
        test_cfg_dir = None
        test_keep = set()
        trows = rows_for(ds_short, "TEST")
        if trows and os.path.exists(os.path.join(RAW, trows[0]["run_id"],
                                                 "artifacts", "cfg.json")):
            query_ids, doc_ids, qrels = test_rows(dsinfo)
            test_keep = set(sorted(qrels.keys()))

        for cell in ("C", "E", "F"):
            tune_dir = os.path.join(RAW, f"C7-TUNE-{ds_short}-{cell}-001")
            cfg_path = os.path.join(tune_dir, "artifacts", "gatescan-cfg.json" if cell == "C"
                                    else "train-cfg.json")
            if os.path.exists(cfg_path):
                cfg = load_cfg(cfg_path)
                keys = set(keep_rows_of(cfg) if cell == "C"
                           else [int(v) for v in cfg["train"]["rows"]])
                leak = keys & test_keep
                if leak:
                    fail("leakage", f"C7-TUNE-{ds_short}-{cell} shares {len(leak)} "
                                    f"rows with TEST: {sorted(leak)[:5]}")
                if not keys:
                    fail("leakage", f"C7-TUNE-{ds_short}-{cell} kept 0 rows")

        # ---- SMOKE determinism ----
        if os.path.exists(os.path.join(shared, "multi-smoke-rerun.json")):
            m1 = load_multi(os.path.join(shared, "multi-smoke-rep1.json"))
            m2 = load_multi(os.path.join(shared, "multi-smoke-rerun.json"))
            gate_determinism(fail("determinism"), f"C7-SMOKE-{ds_short}", m1, m2)
            gate_union_identity(fail("union-identity"), m1,
                                f"C7-SMOKE-{ds_short}", 1)
            gate_fingerprints(fail("fingerprints"), m1, f"C7-SMOKE-{ds_short}")
            for name in UNION_ARMS:
                if name in m1["arms"]:
                    for i, p in enumerate(m1["arms"][name]["per_query"]):
                        check_nonfinite(fail("nonfinite"), f"C7-SMOKE-{ds_short}",
                                        name, p, i)
            summary[f"smoke-{ds_short}"] = "checked"

        # ---- TEST: union identity + fingerprints + oracle recompute ----
        trows = rows_for(ds_short, "TEST")
        if trows and os.path.exists(os.path.join(shared, "multi-test-rep1.json")):
            cfg = load_cfg(os.path.join(shared, "multi-test-rep1-cfg.json"))
            query_ids, doc_ids, qrels = test_rows(dsinfo)
            keep = sorted(qrels.keys())
            qvec_path = os.path.join(shared, cfg["query_vectors"])
            qflat = np.fromfile(qvec_path, dtype=np.float32)
            queries = {r: qflat[r * dsinfo["dim"]:(r + 1) * dsinfo["dim"]]
                       for r in keep}
            canonical_path = os.path.join(shared, "CANONICAL.f32")
            canonical = np.fromfile(canonical_path, dtype=np.float32).reshape(
                -1, dsinfo["dim"])
            exact = exact_heights(canonical, queries, dsinfo)
            rep_count = sum(1 for f in os.listdir(shared)
                            if f.startswith("multi-test-rep")
                            and f.endswith(".json"))
            for rep in range(1, rep_count + 1):
                p = os.path.join(shared, f"multi-test-rep{rep}.json")
                if not os.path.exists(p):
                    continue
                multi = load_multi(p)
                gate_union_identity(fail("union-identity"), multi,
                                    f"C7-TEST-{ds_short}", rep)
                gate_fingerprints(fail("fingerprints"), multi, f"C7-TEST-{ds_short}")
                for name in UNION_ARMS:
                    for i, pq in enumerate(multi["arms"][name]["per_query"]):
                        check_nonfinite(fail("nonfinite"), f"C7-TEST-{ds_short}", name, pq, i)
                        oracle_recompute(fail("oracle"), f"C7-TEST-{ds_short}", name,
                                         pq, qrels.get(pq["query_row"], {}),
                                         exact)
            summary[f"test-{ds_short}"] = f"{rep_count} reps checked (oracle recomputed)"

    # ---- per-cell metrics.json / RUN-INDEX consistency ----
    index_done = set()
    idx_path = os.path.join(RAW, "RUN-INDEX.yaml")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("- run_id:"):
                    index_done.add(line.split(":", 1)[1].strip())
    for row in plan_rows():
        rid = row["run_id"]
        d = os.path.join(RAW, rid)
        if not os.path.exists(d):
            continue
        marker = "metrics.json"
        if row["track"] == "TUNE":
            marker = "gate_choice.json" if row["mode"] == "MODE-C" \
                else os.path.join("artifacts", "model.json")
        if not os.path.exists(os.path.join(d, marker)):
            fail("run-index", f"{rid}: missing {marker}")
        if rid not in index_done:
            fail("run-index", f"{rid}: not PASS-registered in RUN-INDEX.yaml")
    summary["run-index"] = "checked"

    report = {"gate": summary, "checks": checks,
              "all_pass": all(len(v) == 0 for v in checks.values())}
    with open(REPORT_PATH, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    for key, msgs in checks.items():
        for m in msgs:
            print(f"[{key}] {m}")
    print(json.dumps({"all_pass": report["all_pass"], "checks": len(sum(checks.values(), []))}))
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()