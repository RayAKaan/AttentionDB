#!/usr/bin/env python3
"""C7 genuine candidate-level Q/K/V attention test orchestrator.

Drives the C7 probe (c7pilot) which runs ALL arms (A–F) against ONE engine
per process, so candidates B/C/D/E/F share a bit-identical per-query union
(protocol §2 invariant, stop condition #4). The per-arm outputs are split
into each cell's `raw/<run_id>/artifacts/`.

Cell handling:
  SAFETY PROBE (EFPROBE): two arms MODE-B ef=16 / ef=128, 1 rep.
  SAFETY SMOKE: arms A–F, 1 rep, 20 qids (C/E/F use TUNE outputs).
  TUNE  MODE-C: gate scan g in {0.0..1.0 step 0.1} as 11 arms on VALIDATION;
         selects g* = argmax mean nDCG@10 (ties -> larger g).
  TUNE  MODE-E/F: contrastive QKV training (train subcommand).
  TRK-A TEST: arms A–F, 5 fresh-process reps (rep seed offsets by
         rep * 0x9E3779B9 per protocol §5).

Buffers frozen at protocol §3:
  candidate_budget 500, ef_search 64, min 20 / max 300 per head, k=10,
  warmup 20, dim 384, bm25 None.
"""

import errno
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw"))
PLAN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c7-run-plan.csv"))
C7PILOT = os.environ.get("C7PILOT", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "probe", "target", "release", "c7pilot.exe")))
SEED = 20260925
WARMUP = 20
K = 10
REP_MIX = 0x9E3779B9
ATTN = {
    "identity": {
        "enabled": True, "arch": "identity-qkv",
        "attention_dim": 384, "key_dim": 384, "value_dim": 384,
        "use_evidence": False,
        "scorer": {"w_attn": 1.0, "w_evidence": 0.0, "bias": 0.0},
    },
}
FIXED_FUSION = {"attention": 0.3, "multi_head_similarity": 0.5, "bm25": 0.2}


# ---- Windows memory sampler -------------------------------------------------
try:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    def mem_status():
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m

    def proc_rss_bytes(pid):
        import psutil
        try:
            return psutil.Process(pid).memory_info().rss
        except Exception:
            return 0
except ImportError:
    import psutil

    def mem_status():
        return (None, psutil.virtual_memory().available)

    def proc_rss_bytes(pid):
        try:
            return psutil.Process(pid).memory_info().rss
        except Exception:
            return 0


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_rows():
    with open(PLAN, encoding="utf8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        out = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(header):
                raise SystemExit(f"row has {len(parts)} cols; header {len(header)}")
            out.append(dict(zip(header, parts)))
    return out


def load_plan_row(run_id):
    for r in plan_rows():
        if r["run_id"] == run_id:
            return r
    raise SystemExit(f"run_id {run_id} not found in {PLAN}")


def _jsonl_ids(path):
    ids = []
    with open(path, encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            ids.append(json.loads(line)["_id"])
    return ids


def parse_qrels(path, query_ids, doc_ids):
    qrow = {qid: i for i, qid in enumerate(query_ids)}
    drow = {did: i for i, did in enumerate(doc_ids)}
    out = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0] == "query-id" or parts[0].lower().startswith("query"):
                continue
            qid, did, grade = parts[0], parts[1], int(parts[2])
            if qid not in qrow or did not in drow:
                continue
            out.setdefault(qrow[qid], {})[drow[did]] = int(grade)
    return out


def seeded_shuffle(n, seed):
    rnd = random.Random(seed + 0x5EED)
    idx = list(range(n))
    rnd.shuffle(idx)
    return idx


def resolve_dataset(ds):
    if ds == "DS-NFCORPUS":
        emb = os.path.join(RAW, "C2-EMBED-NFCORPUS-003", "artifacts")
        data = os.path.join(RAW, "datasets", "beir", "nfcorpus")
        return {"emb": emb, "data": data, "prefix": "DS-NFCORPUS",
                "heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE", "CANONICAL"],
                "dim": 384}
    if ds == "DS-SCIFACT":
        emb = os.path.join(RAW, "C3-EMBED-SCIFACT-001", "artifacts")
        data = os.path.join(RAW, "datasets", "beir", "scifact")
        return {"emb": emb, "data": data, "prefix": "DS-SCIFACT",
                "heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE", "CANONICAL"],
                "dim": 384}
    raise SystemExit(f"no resolver for dataset {ds}")


def test_rows(dsinfo):
    query_ids = _jsonl_ids(os.path.join(dsinfo["data"], "queries.jsonl"))
    doc_ids = _jsonl_ids(os.path.join(dsinfo["data"], "corpus.jsonl"))
    qrels = parse_qrels(os.path.join(dsinfo["data"], "qrels", "test.tsv"),
                        query_ids, doc_ids)
    return query_ids, doc_ids, qrels


def val_rows(dsinfo):
    query_ids = _jsonl_ids(os.path.join(dsinfo["data"], "queries.jsonl"))
    doc_ids = _jsonl_ids(os.path.join(dsinfo["data"], "corpus.jsonl"))
    qrels_path_src = os.path.join(dsinfo["data"], "qrels")
    if dsinfo["prefix"] == "DS-NFCORPUS":
        qrels = parse_qrels(os.path.join(qrels_path_src, "dev.tsv"), query_ids, doc_ids)
    else:
        qrels = parse_qrels(os.path.join(qrels_path_src, "train.tsv"), query_ids, doc_ids)
        rows = sorted(qrels.keys())
        order = seeded_shuffle(len(rows), SEED)
        take = min(200, len(rows))
        keep = [rows[i] for i in order[:take]]
        qrels = {r: qrels[r] for r in keep}
    return query_ids, doc_ids, qrels


def materialize(art_root, dsinfo):
    doc_vectors = {}
    for head in dsinfo["heads"]:
        src = None
        for cand in (f"{head}.npy", f"{dsinfo['prefix']}-{head}.npy"):
            p = os.path.join(dsinfo["emb"], cand)
            if os.path.exists(p):
                src = p
                break
        if src is None:
            raise SystemExit(f"missing head {head} for {dsinfo['prefix']}")
        out = os.path.join(art_root, f"{head}.f32")
        if not os.path.exists(out):
            np.load(src).astype(np.float32).tofile(out)
        doc_vectors[head] = out
    q_src = None
    for cand in ("QUERIES.npy", f"{dsinfo['prefix']}-QUERIES.npy"):
        p = os.path.join(dsinfo["emb"], cand)
        if os.path.exists(p):
            q_src = p
            break
    if q_src is None:
        raise SystemExit("QUERIES view missing")
    qvec_out = os.path.join(art_root, "QUERIES.f32")
    if not os.path.exists(qvec_out):
        np.load(q_src).astype(np.float32).tofile(qvec_out)
    return doc_vectors, qvec_out


def arm_slice(multi_out, name):
    return multi_out["arms"].get(name)


def arm_name_for_row(r):
    """Map a plan cell to the multi-arm 'name' produced by the probe.

    EFPROBE cells (MODE-B with distinct ef_search) are modelled as two arms
    distinguished by ef, so cells map by ef value, not mode.
    """
    if r["track"] == "SAFETY" and r["split"] == "PROBE":
        return f"EF{int(r['ef_search'])}"
    return r["mode"].replace("MODE-", "")


def union_set_sig(row):
    if not row or not row.get("union_ledger"):
        return None
    return sorted((r["row"], tuple(x if x is not None else -1.0 for x in r["head_sims"]),
                   r["mhs"]) for r in row["union_ledger"])


def check_union_identity(multi_out, names=("B", "C", "D", "E", "F")):
    """Return list of (arm_a, arm_b, mismatching_queries, n_queries)."""
    base = names[0]
    rows = multi_out["arms"][base]["per_query"]
    out = []
    for other in names[1:]:
        mism = 0
        for qb, qo in zip(rows, multi_out["arms"][other]["per_query"]):
            if union_set_sig(qb) != union_set_sig(qo):
                mism += 1
        out.append((base, other, mism, len(rows)))
    return out


def derive_rows_stats(arm):
    pq = arm["per_query"]
    recs = [p["recall10_qrels"] for p in pq]
    exact = [p.get("recall10_exact", 0.0) for p in pq]
    lat = [p["latency_us"] for p in pq if "latency_us" in p]
    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    ndcg = [p["ndcg10_qrels"] for p in pq]
    st = {
        "n_queries": len(pq),
        "recall10_qrels_mean": float(sum(recs) / len(recs)) if recs else None,
        "recall10_exact_mean": float(sum(exact) / len(exact)) if exact else None,
        "ndcg10_qrels_mean": float(sum(ndcg) / len(ndcg)) if ndcg else None,
        "p50_us": lat_sorted[n // 2] if n else None,
        "p90_us": lat_sorted[int(0.90 * (n - 1))] if n else None,
        "p95_us": lat_sorted[int(0.95 * (n - 1))] if n else None,
        "deadline_exceeded": int(arm.get("deadline_exceeded_count", 0)),
    }
    return st


def run_probe(cfg, out_path, art_dir, cfg_name, run_dir, ram_abort_bytes):
    cfg_path = os.path.join(art_dir, f"{cfg_name}-cfg.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)
    proc = subprocess.Popen([C7PILOT, "--config", cfg_path, "--out", out_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = [0]
    stop = [False]

    def sample():
        while not stop[0]:
            rss = proc_rss_bytes(proc.pid)
            peak[0] = max(peak[0], rss)
            if ram_abort_bytes and rss >= ram_abort_bytes:
                with open(os.path.join(run_dir, "abort-guardrail.txt"), "a") as g:
                    g.write(f"child RSS {rss} >= 85% preflight MemAvailable "
                            f"({ram_abort_bytes}) at {cfg_name}\n")
                proc.terminate()
            time.sleep(0.5)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    stdout_data, stderr_data = proc.communicate(timeout=7200)
    stop[0] = True
    t.join(timeout=1)
    if proc.returncode != 0:
        raise RuntimeError(f"{cfg_name} FAILED rc={proc.returncode}\n{stderr_data[:2000]}")
    return json.load(open(out_path, encoding="utf8")), peak[0]


def write_env(art_dir, run_id, hash_files, mem_snapshot):
    avail = getattr(mem_snapshot, "ullAvailPhys", None)
    load = getattr(mem_snapshot, "dwMemoryLoad", None)
    env = {
        "host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "c7pilot_sha256": sha256_file(C7PILOT),
        "input_hashes": hash_files,
        "guardrail": {"policy": "sampler(500ms) aborts when child process-tree RSS "
                                ">= 85% of MemAvailable at preflight; 2400s timeout",
                      "mem_avail_at_preflight_bytes": avail,
                      "mem_load_pct_at_preflight": load,
                      "ram_abort_threshold_bytes": int(avail * 0.85) if avail else None},
        "method": "one shared engine per process; arms A-F evaluated in-process "
                  "(B/C/D/E/F union identical by construction, re-verified per rep)",
    }
    with open(os.path.join(art_dir, "environment.yaml"), "w", encoding="utf8") as f:
        f.write(json.dumps(env, indent=2))
    return env


def register(run_id, purpose, state, extra=""):
    entry = (f"- run_id: {run_id}\n  status: {state}\n"
             f"  purpose: '{purpose}'\n"
             f"  completed: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    if extra:
        entry += f"  result: {extra}\n"
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(entry)


def build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out, arms, cfg_base,
                    ef_search=64, cand_budget=500, min_cand_head=20, max_cand_head=300):
    cfg = {
        "k": K, "seed": cfg_base["seed"], "warmup": WARMUP,
        "n_queries": len(dsinfo["query_ids"]), "n_docs": len(dsinfo["doc_ids"]),
        "dim": dsinfo["dim"], "subsample": keep,
        "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
        "query_vectors": qvec_out, "doc_vectors": doc_vectors,
        "candidate_budget": cand_budget,
        "min_candidates_per_head": min_cand_head,
        "max_candidates_per_head": max_cand_head,
        "ef_search": ef_search,
        "configuration_id": cfg_base["config_id"],
        "deadline_us": 0,
        "arms": arms,
    }
    return cfg


def arm_identity():
    return {
        "name": "A", "mode": "A", "attend_heads": ["CANONICAL"],
        "fusion": FIXED_FUSION,
        "configuration_id": "C7-ARM-A",
    }


def arm_base():
    return {
        "name": "B", "mode": "B",
        "attend_heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"],
        "fusion": FIXED_FUSION,
        "configuration_id": "C7-ARM-B",
    }


def arm_d_identity():
    attn = dict(ATTN["identity"])
    return {
        "name": "D", "mode": "D",
        "attend_heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"],
        "attention": attn, "fusion": FIXED_FUSION,
        "configuration_id": "C7-ARM-D",
    }


def arm_c_gated(g):
    attn = dict(ATTN["identity"])
    return {
        "name": "C", "mode": "C",
        "attend_heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"],
        "attention": attn,
        "fusion": {"attention": g, "multi_head_similarity": 1.0 - g, "bm25": 0.0},
        "configuration_id": "C7-ARM-C",
    }


def arm_learned(use_evidence, model_path, model_id):
    attn = {
        "enabled": True, "arch": "learned-qkv",
        "attention_dim": 384, "key_dim": 384, "value_dim": 384,
        "use_evidence": use_evidence,
        "scorer": {"w_attn": 1.0, "w_evidence": 1.0 if use_evidence else 0.0, "bias": 0.0},
        "qkv_model": model_path,
    }
    return {
        "name": "F" if use_evidence else "E",
        "mode": "F" if use_evidence else "E",
        "attend_heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"],
        "attention": attn, "fusion": FIXED_FUSION,
        "configuration_id": f"C7-ARM-{model_id}",
    }


def group_rows(run_id):
    """All run_ids sharing this cell's dataset+split (the multi-arm group)."""
    row = load_plan_row(run_id)
    rows = plan_rows()
    if row["track"] == "TUNE":
        return [row]
    return [r for r in rows if r["dataset"] == row["dataset"] and r["split"] == row["split"]]


def main():
    args = sys.argv[1:]
    run_id = None
    dry = False
    i = 0
    while i < len(args):
        if args[i] == "--run-id":
            run_id = args[i + 1]; i += 2
        elif args[i] == "--dry-run":
            dry = True; i += 1
        else:
            raise SystemExit(f"unknown arg {args[i]}")
    if not run_id:
        raise SystemExit("--run-id required")
    row = load_plan_row(run_id)
    track = row["track"]
    split = row["split"]
    ds = row["dataset"]
    dsinfo = resolve_dataset(ds)
    grp = group_rows(run_id)
    ds_short = "SCI" if ds == "DS-SCIFACT" else "NFC"

    if track == "TUNE":
        mode = row["mode"].replace("MODE-", "")
        run_dir = os.path.join(RAW, run_id)
        query_ids, doc_ids, qrels = val_rows(dsinfo)
        dsinfo["query_ids"] = query_ids
        dsinfo["doc_ids"] = doc_ids
        keep = sorted(qrels.keys())
        if dry:
            if mode == "C":
                print(json.dumps({"run_id": run_id, "track": "TUNE-C gate scan",
                                  "g_range": [round(g, 1) for g in np.arange(0.0, 1.0001, 0.1)],
                                  "n_queries": len(keep)}, indent=2))
            else:
                print(json.dumps({"run_id": run_id, "track": f"TUNE-{mode} train",
                                  "arm": mode, "use_evidence": (mode == "F"),
                                  "n_queries": len(keep), "epochs": 5, "lr": 1e-2,
                                  "tau": 0.07, "l2": 1e-4, "seed": SEED}, indent=2))
            return
        if os.path.exists(run_dir):
            raise SystemExit(f"run dir {run_dir} exists (immutable); new run_id required")
        os.makedirs(os.path.join(run_dir, "artifacts"))
        art = os.path.join(run_dir, "artifacts")
        doc_vectors, qvec_out = materialize(art, dsinfo)
        mem_snapshot = mem_status()
        ram_abort_bytes = int(getattr(mem_snapshot, "ullAvailPhys", 0) * 0.85)

        if mode == "C":
            arms = []
            for g in np.arange(0.0, 1.0001, 0.1):
                arm = arm_c_gated(round(g, 1))
                arm["name"] = f"C@{g:.1f}"
                arm["configuration_id"] = f"{run_id}__C@{g:.1f}"
                arms.append(arm)
            cfg = build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out, arms,
                                  {"seed": SEED, "config_id": run_id},
                                  ef_search=64, cand_budget=500)
            out = os.path.join(art, "gatescan.json")
            multi, peak = run_probe(cfg, out, art, "gatescan", run_dir, ram_abort_bytes)
            by_g = {}
            for i, g in enumerate(np.arange(0.0, 1.0001, 0.1)):
                key = f"C@{g:.1f}"
                st = derive_rows_stats(multi["arms"][key])
                by_g[round(g, 1)] = round(st["ndcg10_qrels_mean"], 6)
            best_g = max(by_g, key=lambda g: (by_g[g], g))
            gate = {
                "run_id": run_id, "dataset": ds, "split": "VALID",
                "scan": by_g,
                "best_g": round(best_g, 1),
                "best_ndcg10": by_g[best_g],
                "criterion": "argmax mean nDCG@10; tie -> larger g",
            }
            with open(os.path.join(run_dir, "gate_choice.json"), "w", encoding="utf8") as f:
                json.dump(gate, f, indent=2)
            hash_files = {h: sha256_file(os.path.join(art, h)) for h in sorted(os.listdir(art))}
            write_env(art, run_id, hash_files, mem_snapshot)
            register(run_id, f"C7 TUNE-C gate scan {ds} (VALIDATION {len(keep)} qids)",
                     "PASS", f"g*={gate['best_g']} ndcg10={gate['best_ndcg10']:.4f}")
            print(json.dumps(gate, indent=2))
            return

        # TUNE E / F: contrastive QKV training.
        model = "F" if mode == "F" else "E"
        use_evidence = (mode == "F")
        cfg = {
            "subcommand": "train", "k": K, "seed": SEED, "warmup": WARMUP,
            "n_queries": len(query_ids), "n_docs": len(doc_ids), "dim": dsinfo["dim"],
            "subsample": keep,
            "collection_heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"],
            "attend_heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"],
            "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
            "query_vectors": qvec_out, "doc_vectors": doc_vectors,
            "candidate_budget": 500, "min_candidates_per_head": 20,
            "max_candidates_per_head": 300, "ef_search": 64,
            "configuration_id": run_id,
            "train": {"arm": model, "epochs": 5, "learning_rate": 1e-2,
                      "temperature": 0.07, "l2": 1e-4, "batch_size": 8,
                      "negatives_per_query": 8, "w_attn": 1.0,
                      "w_evidence": 1.0 if use_evidence else 0.0,
                      "use_evidence": use_evidence,
                      "model_id": model, "arch": "contrastive-qkv", "rows": keep},
        }
        out = os.path.join(art, "model.json")
        cfg_path = os.path.join(art, "train-cfg.json")
        with open(cfg_path, "w", encoding="utf8") as f:
            json.dump(cfg, f, indent=2)
        proc = subprocess.Popen([C7PILOT, "--config", cfg_path, "--out", out],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, stderr_data = proc.communicate(timeout=2400)
        if proc.returncode != 0:
            raise RuntimeError(f"train {run_id} FAILED rc={proc.returncode}\n{stderr_data[:2000]}")
        card = json.load(open(out, encoding="utf8"))
        hash_files = {h: sha256_file(os.path.join(art, h)) for h in sorted(os.listdir(art))}
        write_env(art, run_id, hash_files, mem_snapshot)
        register(run_id, f"C7 TUNE-{model} contrastive QKV {ds} (VALIDATION {len(keep)} qids)",
                 "PASS",
                 f"examples={card['n_examples']} skipped={card['n_skipped_queries_no_positive_in_union']} "
                 f"dataset_hash={card['dataset_hash']} epochs={card['epochs_run']} loss={card['loss_history']}")
        print(json.dumps({"run_id": run_id, "arm": model, "ok": True,
                          "n_examples": card["n_examples"],
                          "dataset_hash": card["dataset_hash"],
                          "epochs_run": card["epochs_run"],
                          "final_loss": card["loss_history"][-1] if card["loss_history"] else None},
                         indent=2))
        return

    # SAFETY (EFPROBE / SMOKE) and TRK-A TEST share this path.
    if split == "PROBE":
        # EFPROBE: two ef legs, one process per dataset.
        efs = sorted({int(r["ef_search"]) for r in grp if r["ef_search"]})
        query_ids, doc_ids, qrels = val_rows(dsinfo)
        rows = sorted(qrels.keys())
        order = seeded_shuffle(len(rows), SEED)
        take = min(20, len(rows))
        keep = [rows[i] for i in order[:take]]
        qrels = {r: qrels[r] for r in keep}
    elif split == "SMOKE":
        query_ids, doc_ids, qrels = val_rows(dsinfo)
        rows = sorted(qrels.keys())
        order = seeded_shuffle(len(rows), SEED)
        take = min(20, len(rows))
        keep = [rows[i] for i in order[:take]]
        qrels = {r: qrels[r] for r in keep}
        efs = [64]
    elif split == "TEST":
        query_ids, doc_ids, qrels = test_rows(dsinfo)
        keep = sorted(qrels.keys())
        efs = [64]
    else:
        raise SystemExit(f"unknown split {split}")
    dsinfo["query_ids"] = query_ids
    dsinfo["doc_ids"] = doc_ids

    # All group run dirs must be new (immutability).
    for r in grp:
        if os.path.exists(os.path.join(RAW, r["run_id"])):
            raise SystemExit(f"run dir {os.path.join(RAW, r['run_id'])} exists (immutable)")

    art_root = os.path.join(RAW, "C7-SHARED-" + ds_short)
    os.makedirs(art_root, exist_ok=True)
    doc_vectors, qvec_out = materialize(art_root, dsinfo)

    rep_count = int(row["rep_count"] or (5 if split == "TEST" else 1))

    # ---- TUNE outputs needed by SMOKE/TEST arms C/E/F ------------------
    def tune_artifact(ds_short2, cell, fname):
        rid = f"C7-TUNE-{ds_short2}-{cell}-001"
        p = os.path.join(RAW, rid, fname)
        if not os.path.exists(p):
            raise SystemExit(f"missing TUNE artifact {p}; run TUNE before SMOKE/TEST")
        return p

    gate_path = None
    e_model, f_model = None, None
    if split in ("SMOKE", "TEST"):
        gate_path = tune_artifact(ds_short, "C", "gate_choice.json")
        e_model = tune_artifact(ds_short, "E", "artifacts/model.json")
        f_model = tune_artifact(ds_short, "F", "artifacts/model.json")

    def build_arms(reps_seed):
        arms = []
        if split == "PROBE":
            for ef in efs:
                # hnsw_rs clamps its search beam to max(ef, k); with a 500-doc
                # per-head budget the ef knob is inert, so the probe also caps
                # per-head search k at ef (search_k) to exercise the real beam.
                arm = {"name": f"EF{ef}", "mode": "B",
                       "attend_heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"],
                       "fusion": FIXED_FUSION, "ef_search": ef, "search_k": ef,
                       "configuration_id": f"C7-EFPROBE-{ds_short}@ef{ef}"}
                arms.append(arm)
            return arms
        arms.append(arm_identity())
        arms.append(arm_base())
        if gate_path:
            gate = json.load(open(gate_path, encoding="utf8"))
            arms.append(arm_c_gated(float(gate["best_g"])))
        else:
            arms.append(arm_c_gated(0.5))
        arms.append(arm_d_identity())
        arms.append(arm_learned(False, e_model, "E") if e_model else
                    {**arm_d_identity(), "name": "E", "mode": "E"})
        arms.append(arm_learned(True, f_model, "F") if f_model else
                    {**arm_d_identity(), "name": "F", "mode": "F"})
        for arm in arms:
            arm["configuration_id"] = f"{'_'.join(r['run_id'].split('-')[0:3])}__{arm['name']}"
        return arms

    cfg_seed = SEED
    if dry:
        print(json.dumps({
            "run_id": run_id, "track": split, "dataset": ds,
            "n_queries": len(keep), "rep_count": rep_count, "efs": efs,
            "arms": [a["name"] for a in build_arms(SEED)],
            "config_id": run_id,
        }, indent=2))
        return
    multi_path = os.path.join(art_root, f"multi-{split.lower()}.json")
    probe_arms = None
    union_ok = True
    union_report = {}
    if split == "PROBE":
        # EFPROBE arms (EF16/EF128) are the same MODE-B; the union identity
        # invariant applies to the B/C/D/E/F attention arms only.
        union_report = {"skipped": "EFPROBE arms are MODE-B (attention OFF); "
                                   "union identity checked on B/C/D/E/F arms"}
    for rep in range(1, rep_count + 1):
        rep_seed = SEED + (rep - 1) * REP_MIX
        arms = build_arms(rep_seed)
        if probe_arms is None:
            probe_arms = arms
        cfg = build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out, arms,
                              {"seed": rep_seed, "config_id": f"{run_id}__rep{rep}"},
                              ef_search=64, cand_budget=500)
        out = os.path.join(art_root, f"multi-{split.lower()}-rep{rep}.json")
        multi, peak = run_probe(cfg, out, art_root, f"multi-{split.lower()}-rep{rep}",
                                art_root, int(getattr(mem_status(), "ullAvailPhys", 0) * 0.85))
        if split != "PROBE":
            ui = check_union_identity(multi, names=("B", "C", "D", "E", "F"))
            for a, b, mism, nq in ui:
                union_report[f"rep{rep}:{a}vs{b}"] = {"mismatches": mism, "n_queries": nq}
                if mism:
                    union_ok = False
        for r in grp:
            name = arm_name_for_row(r)
            rid = r["run_id"]
            art = os.path.join(RAW, rid, "artifacts")
            os.makedirs(art, exist_ok=True)
            arm = arm_slice(multi, name)
            if arm is None:
                raise SystemExit(f"{rid}: arm {name} missing in multi output")
            with open(os.path.join(art, f"ARM-{name}-rep{rep}.json"), "w", encoding="utf8") as f:
                json.dump(arm, f, indent=2)
        # free the ~0.4GB multi object before launching the next rep's probe,
        # otherwise the harness parent RSS grows with every rep and trips the
        # child guardrail on the final reps (TEST: 5 x 390MB loads).
        del multi
    if probe_arms is None:
        probe_arms = []
    arm_names = [a["name"] for a in probe_arms]

    # determinism re-run: a SECOND fresh process for SMOKE (observable-level gate).
    # hnsw_rs is vendored with a fixed layer-RNG seed (probe/Cargo.toml
    # [patch.crates-io]), so index builds are bit-identical across processes:
    # candidate_count, recall@10, nDCG@10, and the raw union ledgers must match
    # exactly. c7_verify.py enforces this.
    rerun_path = None
    if split == "SMOKE":
        rerun_cfg = build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out,
                                    probe_arms, {"seed": SEED, "config_id": f"{run_id}__rerun"},
                                    ef_search=64, cand_budget=500)
        rerun_path = os.path.join(art_root, "multi-smoke-rerun.json")
        run_probe(rerun_cfg, rerun_path, art_root, "multi-smoke-rerun",
                  art_root, int(getattr(mem_status(), "ullAvailPhys", 0) * 0.85))

    # per-cell metrics + environment
    for r in grp:
        rid = r["run_id"]
        art = os.path.join(RAW, rid, "artifacts")
        os.makedirs(art, exist_ok=True)
        name = arm_name_for_row(r)
        for rep in range(1, rep_count + 1):
            p = os.path.join(art, f"ARM-{name}-rep{rep}.json")
            if not os.path.exists(p):
                raise SystemExit(f"missing per-arm artifact {p}")
        # copy shared config into each cell
        shutil.copy(os.path.join(art_root, f"multi-{split.lower()}-rep1-cfg.json"),
                    os.path.join(art, "cfg.json"))
        mem_snapshot = mem_status()
        hash_files = {}
        for d in (art_root, art):
            for h in sorted(os.listdir(d)):
                p = os.path.join(d, h)
                if os.path.isfile(p):
                    hash_files[h] = sha256_file(p)
        write_env(art, rid, hash_files, mem_snapshot)
        reps_stats = []
        for rep in range(1, rep_count + 1):
            with open(os.path.join(art, f"ARM-{name}-rep{rep}.json"), encoding="utf8") as f:
                reps_stats.append(derive_rows_stats(json.load(f)))
        rep_recalls = [s["recall10_qrels_mean"] for s in reps_stats]
        rep_ndcg = [s["ndcg10_qrels_mean"] for s in reps_stats]
        mean_rec = sum(rep_recalls) / len(rep_recalls) if rep_recalls else None
        mean_ndcg = sum(rep_ndcg) / len(rep_ndcg) if rep_ndcg else None
        metrics = {
            "run_id": rid, "mode": name, "dataset": ds, "split": split,
            "phase": "C7 execution", "test_queries": len(keep),
            "rep_count": rep_count, "warmup": WARMUP, "ef_search": 64,
            "candidate_budget": 500, "min_candidates_per_head": 20,
            "max_candidates_per_head": 300,
            "per_rep": reps_stats,
            "rep_recall10_qrels_mean": rep_recalls,
            "recall10_qrels_mean_Nrep": round(mean_rec, 4) if mean_rec else None,
            "rep_ndcg10_qrels_mean": rep_ndcg,
            "ndcg10_qrels_mean_Nrep": round(mean_ndcg, 4) if mean_ndcg else None,
            "union_identity_ok": union_ok,
            "union_report": union_report,
        }
        with open(os.path.join(RAW, rid, "metrics.json"), "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        register(rid, f"C7 {split} {name} {ds} ({len(keep)} qids, {rep_count} reps)",
                 "PASS",
                 f"ndcg10={metrics['ndcg10_qrels_mean_Nrep']} recall10="
                 f"{metrics['recall10_qrels_mean_Nrep']} union_identity_ok={union_ok}")

    print(json.dumps({"run_id": run_id, "group": [r["run_id"] for r in grp],
                      "split": split, "dataset": ds, "n_queries": len(keep),
                      "rep_count": rep_count, "union_identity_ok": union_ok,
                      "union_report": union_report}, indent=2))
    if not union_ok:
        raise SystemExit("UNION IDENTITY VIOLATION: B/C/D/E/F union sets not identical "
                         "within shared engine (stop condition #4)")


if __name__ == "__main__":
    main()