#!/usr/bin/env python3
"""C8 residual candidate-level Q/K/V attention test orchestrator.

Drives the C8 probe (c8pilot) which runs ALL arms (A-I) against ONE engine per
process, so candidates B/C/D/E/F/G/H/I share a bit-identical per-query union
(protocol Sec.2 invariant, stop condition #4). Per-arm outputs are split into
each cell's `raw/<run_id>/artifacts/`.

Cell handling:
  SAFETY PROBE (EFPROBE): two arms MODE-B ef=16 / ef=128, 1 rep.
  SAFETY SMOKE: arms A-I, 1 rep, 20 qids (C/D/E/F/G/H/I use TUNE outputs).
  TUNE  D/E/F/G/H: residual QKV training (train subcommand) on VALIDATION.
  TRK-A TEST: arms A-I, 5 fresh-process reps (rep seed offset by
          rep * 0x9E3779B9 per protocol Sec.5).
  SUPPORT: dimension/depth sweep on E on VALIDATION (train + single-arm run).

Frozen buffers (protocol Sec.3):
  candidate_budget 500, ef_search 64, min 20 / max 300 per head, k=10,
  warmup 20, dim 384, d_k=d_v=64, bm25 None.
"""

import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw"))
PLAN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c8-run-plan.csv"))
C8PILOT = os.environ.get("C8PILOT", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "probe", "target", "release", "c8pilot.exe")))
SEED = 20260925
WARMUP = 20
K = 10
REP_MIX = 0x9E3779B9
HEADS3 = ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"]
FIXED_FUSION = {"attention": 0.3, "multi_head_similarity": 0.5, "bm25": 0.2}

# Frozen primary C8 geometry (protocol Sec.3/4).
ATTN_DIM = 384
KEY_DIM = 64
VALUE_DIM = 64
LAMBDA = 0.1
RESIDUAL_ALPHA = 0.1
EPOCHS = 5
LR = 1e-2
TAU = 0.07
L2 = 1e-4
BATCH = 8
NEG_PER_Q = 8
RESID_REG = 1e-3
DISTILL_TEMP = 0.5


# ---- Cross-platform memory sampler -----------------------------------------
# psutil exposes the required memory metrics on both Windows and Linux.
# Keeping the sampler platform-neutral is required because authoritative C8
# execution runs on Linux while local development may run on Windows.
import psutil


def mem_status():
    return psutil.virtual_memory()


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
    with open(PLAN, encoding="utf8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


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
    if r["track"] == "SAFETY" and r["split"] == "PROBE":
        return f"EF{int(r['ef_search'])}"
    if r["track"] == "SUPPORT":
        return _support_variant(r["run_id"])
    return r["mode"].replace("MODE-", "")


_SUP_RE = re.compile(r"^C8-SUPPORT-(SCI|NFC)-(DV\d+|DP\d+)-\d+$")


def _support_variant(run_id):
    m = _SUP_RE.match(run_id)
    if not m:
        raise SystemExit(f"unparseable SUPPORT run_id {run_id}")
    return m.group(2)


def union_set_sig(row):
    if not row or not row.get("union_ledger"):
        return None
    return sorted((r["row"], tuple(x if x is not None else -1.0 for x in r["head_sims"]),
                   r["mhs"]) for r in row["union_ledger"])


def check_union_identity(multi_out, names=("B", "C", "D", "E", "F", "G", "H", "I")):
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
    mrr = []
    for p in pq:
        rel = set(p.get("relevant_ids", []))
        rr = 0.0
        for rank, h in enumerate(p.get("hits", []), start=1):
            if h in rel:
                rr = 1.0 / rank
                break
        mrr.append(rr)
    st = {
        "n_queries": len(pq),
        "recall10_qrels_mean": float(sum(recs) / len(recs)) if recs else None,
        "recall10_exact_mean": float(sum(exact) / len(exact)) if exact else None,
        "ndcg10_qrels_mean": float(sum(ndcg) / len(ndcg)) if ndcg else None,
        "mrr10_qrels_mean": float(sum(mrr) / len(mrr)) if mrr else None,
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
    proc = subprocess.Popen([C8PILOT, "--config", cfg_path, "--out", out_path],
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
    with open(out_path, encoding="utf8") as f:
        return json.load(f), peak[0]


def write_env(art_dir, run_id, hash_files, mem_snapshot):
    avail = getattr(mem_snapshot, "ullAvailPhys", None)
    load = getattr(mem_snapshot, "dwMemoryLoad", None)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    try:
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit_sha = None
    try:
        rustc_version = subprocess.run(
            ["rustc", "--version"], cwd=repo_root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rustc_version = None
    env = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "rustc_version": rustc_version,
        "cpu_model": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "memory_total_bytes": getattr(mem_snapshot, "total", None),
        "build_profile": "release" if "release" in C8PILOT.lower() else "unknown",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "c8pilot_path": os.path.abspath(C8PILOT),
        "c8pilot_sha256": sha256_file(C8PILOT),
        "input_hashes": hash_files,
        "guardrail": {"policy": "sampler(500ms) aborts when child process-tree RSS "
                                ">= 85% of MemAvailable at preflight; 7200s timeout",
                      "mem_avail_at_preflight_bytes": avail,
                      "mem_load_pct_at_preflight": load,
                      "ram_abort_threshold_bytes": int(avail * 0.85) if avail else None},
        "method": "one shared engine per process; arms A-I evaluated in-process "
                  "(B/C/D/E/F/G/H/I union identical by construction, re-verified per rep)",
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


# ---- C8 config blocks -------------------------------------------------------

def c8_block(arch, model=None, scale=LAMBDA, use_evidence=False,
             w_attn=1.0, w_evidence=0.0, w_disagree=0.0, cache=False,
             d_k=KEY_DIM, d_v=VALUE_DIM):
    return {
        "enabled": True, "arch": arch,
        "attention_dim": ATTN_DIM, "key_dim": d_k, "value_dim": d_v,
        "residual_scale": scale, "use_evidence": use_evidence,
        "scorer": {"w_attn": w_attn, "w_evidence": w_evidence,
                   "w_disagree": w_disagree, "bias": 0.0},
        "cache": cache, "model": model or "",
    }


def arm_identity():
    return {"name": "A", "mode": "A", "attend_heads": ["CANONICAL"],
            "fusion": FIXED_FUSION, "configuration_id": "C8-ARM-A"}


def arm_base():
    return {"name": "B", "mode": "B", "attend_heads": list(HEADS3),
            "fusion": FIXED_FUSION, "configuration_id": "C8-ARM-B"}


def arm_c_control():
    return {"name": "C", "mode": "C", "attend_heads": list(HEADS3),
            "c8": c8_block("truncated-identity", scale=0.0, use_evidence=False),
            "fusion": FIXED_FUSION, "configuration_id": "C8-ARM-C"}


def arm_learned(name, model, use_evidence=False, w_disagree=0.0,
                arch="residual", cache=False, d_k=KEY_DIM, d_v=VALUE_DIM, scale=LAMBDA):
    return {"name": name, "mode": name, "attend_heads": list(HEADS3),
            "c8": c8_block(arch, model=model, scale=scale, use_evidence=use_evidence,
                           w_evidence=1.0 if use_evidence else 0.0,
                           w_disagree=w_disagree, cache=cache, d_k=d_k, d_v=d_v),
            "fusion": FIXED_FUSION, "configuration_id": f"C8-ARM-{name}"}


def build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out, arms, cfg_base,
                    ef_search=64, cand_budget=500, min_cand_head=20, max_cand_head=300):
    return {
        "subcommand": "run",
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


def group_rows(run_id):
    row = load_plan_row(run_id)
    rows = plan_rows()
    if row["track"] in ("TUNE", "SUPPORT"):
        return [row]
    return [r for r in rows if r["dataset"] == row["dataset"] and r["split"] == row["split"]]


# ---- TUNE -------------------------------------------------------------------

def tune_train_block(mode, model_id, rows, d_k=KEY_DIM, d_v=VALUE_DIM):
    use_evidence = mode in ("F", "G")
    use_distill = (mode == "H")
    w_disagree = 1.0 if mode == "G" else 0.0
    return {
        "arm": mode, "epochs": EPOCHS, "learning_rate": LR,
        "temperature": TAU, "l2": L2, "batch_size": BATCH,
        "negatives_per_query": NEG_PER_Q,
        "residual_scale": LAMBDA, "residual_alpha": RESIDUAL_ALPHA,
        "residual_regularization": RESID_REG,
        "distillation_weight": 1.0 if use_distill else 0.0,
        "distillation_temperature": DISTILL_TEMP,
        "use_distillation": use_distill, "use_evidence": use_evidence,
        "w_disagree": w_disagree, "key_dim": d_k, "value_dim": d_v,
        "model_id": model_id, "arch": "residual-qkv", "rows": rows,
    }


def run_train(run_id, row, dsinfo, art, query_ids, doc_ids, qrels, keep,
              doc_vectors, qvec_out, mem_snapshot, model_id,
              d_k=KEY_DIM, d_v=VALUE_DIM):
    mode = row["mode"].replace("MODE-", "")
    cfg = {
        "subcommand": "train", "k": K, "seed": SEED, "warmup": WARMUP,
        "n_queries": len(query_ids), "n_docs": len(doc_ids), "dim": dsinfo["dim"],
        "subsample": keep,
        "collection_heads": list(HEADS3), "attend_heads": list(HEADS3),
        "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
        "query_vectors": qvec_out, "doc_vectors": doc_vectors,
        "candidate_budget": 500, "min_candidates_per_head": 20,
        "max_candidates_per_head": 300, "ef_search": 64,
        "configuration_id": run_id,
        "train": tune_train_block(mode, model_id, keep, d_k=d_k, d_v=d_v),
    }
    out = os.path.join(art, "train_report.json")
    cfg_path = os.path.join(art, "train-cfg.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)
    proc = subprocess.Popen([C8PILOT, "--config", cfg_path, "--out", out],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, stderr_data = proc.communicate(timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"train {run_id} FAILED rc={proc.returncode}\n{stderr_data[:2000]}")
    report = json.load(open(out, encoding="utf8"))
    card = report["model_card"]
    # The run path's `load_projection` reads a bare C8ModelCard, so persist that
    # as `model.json` (the full train report stays alongside as evidence).
    with open(os.path.join(art, "model.json"), "w", encoding="utf8") as f:
        json.dump(card, f, indent=2)
    hash_files = {h: sha256_file(os.path.join(art, h)) for h in sorted(os.listdir(art))}
    write_env(art, run_id, hash_files, mem_snapshot)
    register(run_id, f"C8 TUNE-{mode} residual QKV {row['dataset']} "
                     f"(VALIDATION {len(keep)} qids)",
             "PASS",
             f"examples={report['n_examples']} "
             f"skipped={report['n_skipped_queries_no_positive_in_union']} "
             f"dataset_hash={report['dataset_hash']} epochs={report['epochs_run']} "
             f"steps={report['optimizer_steps']} loss={report['loss_history']}")
    return report


# ---- SUPPORT ----------------------------------------------------------------

def support_params(variant):
    if variant.startswith("DV"):
        d = int(variant[2:])
        return d, 500
    if variant.startswith("DP"):
        return KEY_DIM, int(variant[2:])
    raise SystemExit(f"bad support variant {variant}")


# ---- main -------------------------------------------------------------------

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
    ds_short = "SCI" if ds == "DS-SCIFACT" else "NFC"

    run_dir = os.path.join(RAW, run_id)

    # ---- TUNE / SUPPORT ----------------------------------------------------
    if track in ("TUNE", "SUPPORT"):
        query_ids, doc_ids, qrels = val_rows(dsinfo)
        dsinfo["query_ids"] = query_ids
        dsinfo["doc_ids"] = doc_ids
        keep = sorted(qrels.keys())
        if dry:
            print(json.dumps({"run_id": run_id, "track": track,
                              "mode": row["mode"], "n_queries": len(keep),
                              "lambda": LAMBDA, "d_k": KEY_DIM, "d_v": VALUE_DIM},
                             indent=2))
            return
        if os.path.exists(run_dir):
            raise SystemExit(f"run dir {run_dir} exists (immutable); new run_id required")
        os.makedirs(os.path.join(run_dir, "artifacts"))
        art = os.path.join(run_dir, "artifacts")
        doc_vectors, qvec_out = materialize(art, dsinfo)
        mem_snapshot = mem_status()

        if track == "SUPPORT":
            variant = _support_variant(run_id)
            d_kv, depth = support_params(variant)
            train_row = dict(row)
            train_row["mode"] = "MODE-E"
            card = run_train(run_id, train_row, dsinfo, art, query_ids, doc_ids,
                             qrels, keep, doc_vectors, qvec_out, mem_snapshot,
                             model_id=f"{ds_short.lower()}-c8-support-{variant.lower()}",
                             d_k=d_kv, d_v=d_kv)
            # single-arm run at the sweep point
            arm = arm_learned(variant, os.path.join(art, "model.json"),
                              use_evidence=False, arch="residual",
                              d_k=d_kv, d_v=d_kv)
            arm["configuration_id"] = run_id
            cfg = build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out, [arm],
                                  {"seed": SEED, "config_id": run_id},
                                  ef_search=64, cand_budget=depth)
            out = os.path.join(art, "sweep.json")
            multi, _ = run_probe(cfg, out, art, "sweep", run_dir,
                                 int(getattr(mem_snapshot, "ullAvailPhys", 0) * 0.85))
            a = arm_slice(multi, variant)
            st = derive_rows_stats(a)
            metrics = {
                "run_id": run_id, "track": "SUPPORT", "variant": variant,
                "dataset": ds, "split": "VALID", "d_k": d_kv, "d_v": d_kv,
                "candidate_budget": depth, "n_queries": len(keep),
                "ndcg10_qrels_mean": round(st["ndcg10_qrels_mean"], 4),
                "recall10_qrels_mean": round(st["recall10_qrels_mean"], 4),
                "mrr10_qrels_mean": round(st["mrr10_qrels_mean"], 4),
                "mean_abs_correction": (
                    a.get("c8_aggregate", {}).get("mean_abs_correction")
                    if isinstance(a.get("c8_aggregate"), dict)
                    else None
                ),
                "train_dataset_hash": card["dataset_hash"],
            }
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
                json.dump(metrics, f, indent=2)
            register(run_id, f"C8 SUPPORT {variant} {ds} (VALIDATION {len(keep)} qids)",
                     "PASS", f"ndcg10={metrics['ndcg10_qrels_mean']} d_kv={d_kv} "
                             f"depth={depth}")
            print(json.dumps(metrics, indent=2))
            return

        # TUNE D/E/F/G/H
        mode = row["mode"].replace("MODE-", "")
        card = run_train(run_id, row, dsinfo, art, query_ids, doc_ids, qrels, keep,
                         doc_vectors, qvec_out, mem_snapshot,
                         model_id=f"{ds_short.lower()}-c8-{mode.lower()}")
        print(json.dumps({"run_id": run_id, "arm": mode, "ok": True,
                          "n_examples": card["n_examples"],
                          "dataset_hash": card["dataset_hash"],
                          "epochs_run": card["epochs_run"],
                          "optimizer_steps": card["optimizer_steps"],
                          "final_loss": card["loss_history"][-1]
                          if card["loss_history"] else None}, indent=2))
        return

    # ---- SAFETY / TRK-A ----------------------------------------------------
    grp = group_rows(run_id)
    if split == "PROBE":
        query_ids, doc_ids, qrels = val_rows(dsinfo)
        rows = sorted(qrels.keys())
        order = seeded_shuffle(len(rows), SEED)
        take = min(20, len(rows))
        keep = [rows[i] for i in order[:take]]
        qrels = {r: qrels[r] for r in keep}
        efs = sorted({int(r["ef_search"]) for r in grp if r["ef_search"]})
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

    if dry:
        names = [f"EF{e}" for e in efs] if split == "PROBE" else list("ABCDEFGHI")
        print(json.dumps({"run_id": run_id, "track": split, "dataset": ds,
                          "n_queries": len(keep), "arms": names,
                          "config_id": run_id}, indent=2))
        return

    for r in grp:
        if os.path.exists(os.path.join(RAW, r["run_id"])):
            raise SystemExit(f"run dir {os.path.join(RAW, r['run_id'])} exists (immutable)")

    art_root = os.path.join(RAW, "C8-SHARED-" + ds_short)
    os.makedirs(art_root, exist_ok=True)
    doc_vectors, qvec_out = materialize(art_root, dsinfo)
    rep_count = int(row["rep_count"] or (5 if split == "TEST" else 1))

    def tune_artifact(cell):
        rid = f"C8-TUNE-{ds_short}-{cell}-001"
        p = os.path.join(RAW, rid, "artifacts", "model.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing TUNE artifact {p}; run TUNE before SMOKE/TEST")
        return p

    models = {}
    if split in ("SMOKE", "TEST"):
        for m in ("D", "E", "F", "G", "H"):
            models[m] = tune_artifact(m)

    def build_arms():
        arms = []
        if split == "PROBE":
            for ef in efs:
                arms.append({"name": f"EF{ef}", "mode": "B",
                             "attend_heads": list(HEADS3),
                             "fusion": FIXED_FUSION, "ef_search": ef,
                             "search_k": ef,
                             "configuration_id": f"C8-EFPROBE-{ds_short}@ef{ef}"})
            return arms
        arms.append(arm_identity())
        arms.append(arm_base())
        arms.append(arm_c_control())
        arms.append(arm_learned("D", models["D"], use_evidence=False,
                                arch="unrestricted"))
        arms.append(arm_learned("E", models["E"], use_evidence=False))
        arms.append(arm_learned("F", models["F"], use_evidence=True))
        arms.append(arm_learned("G", models["G"], use_evidence=True, w_disagree=1.0))
        arms.append(arm_learned("H", models["H"], use_evidence=False))
        arms.append(arm_learned("I", models["E"], use_evidence=False, cache=True))
        for arm in arms:
            arm["configuration_id"] = f"{run_id}__{arm['name']}"
        return arms

    arm_names = [a["name"] for a in build_arms()]
    multi_path = os.path.join(art_root, f"multi-{split.lower()}.json")
    probe_arms = None
    union_ok = True
    union_report = {}
    if split == "PROBE":
        union_report = {"skipped": "EFPROBE arms are MODE-B (attention OFF); "
                                   "union identity checked on B/C/D/E/F/G/H/I arms"}
    for rep in range(1, rep_count + 1):
        rep_seed = SEED + (rep - 1) * REP_MIX
        arms = build_arms()
        if probe_arms is None:
            probe_arms = arms
        cfg = build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out, arms,
                              {"seed": rep_seed, "config_id": f"{run_id}__rep{rep}"},
                              ef_search=64, cand_budget=500)
        out = os.path.join(art_root, f"multi-{split.lower()}-rep{rep}.json")
        multi, peak = run_probe(cfg, out, art_root, f"multi-{split.lower()}-rep{rep}",
                                art_root,
                                int(getattr(mem_status(), "ullAvailPhys", 0) * 0.85))
        if split != "PROBE":
            ui = check_union_identity(multi)
            for a, b, mism, nq in ui:
                union_report[f"rep{rep}:{a}vs{b}"] = {"mismatches": mism, "n_queries": nq}
                if mism:
                    union_ok = False
        for r in grp:
            name = arm_name_for_row(r)
            rid = r["run_id"]
            art = os.path.join(RAW, rid, "artifacts")
            os.makedirs(art, exist_ok=True)
            a = arm_slice(multi, name)
            if a is None:
                raise SystemExit(f"{rid}: arm {name} missing in multi output")
            with open(os.path.join(art, f"ARM-{name}-rep{rep}.json"), "w",
                      encoding="utf8") as f:
                json.dump(a, f, indent=2)
        del multi

    # determinism re-run on SMOKE (observable-level gate)
    rerun_path = None
    if split == "SMOKE":
        rerun_cfg = build_multi_cfg(dsinfo, keep, qrels, doc_vectors, qvec_out,
                                    probe_arms,
                                    {"seed": SEED, "config_id": f"{run_id}__rerun"},
                                    ef_search=64, cand_budget=500)
        rerun_path = os.path.join(art_root, "multi-smoke-rerun.json")
        run_probe(rerun_cfg, rerun_path, art_root, "multi-smoke-rerun", art_root,
                  int(getattr(mem_status(), "ullAvailPhys", 0) * 0.85))

    for r in grp:
        rid = r["run_id"]
        art = os.path.join(RAW, rid, "artifacts")
        os.makedirs(art, exist_ok=True)
        name = arm_name_for_row(r)
        for rep in range(1, rep_count + 1):
            p = os.path.join(art, f"ARM-{name}-rep{rep}.json")
            if not os.path.exists(p):
                raise SystemExit(f"missing per-arm artifact {p}")
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
        rep_ndcg = [s["ndcg10_qrels_mean"] for s in reps_stats]
        rep_rec = [s["recall10_qrels_mean"] for s in reps_stats]
        mean_ndcg = sum(rep_ndcg) / len(rep_ndcg) if rep_ndcg else None
        mean_rec = sum(rep_rec) / len(rep_rec) if rep_rec else None
        metrics = {
            "run_id": rid, "mode": name, "dataset": ds, "split": split,
            "phase": "C8 execution", "test_queries": len(keep),
            "rep_count": rep_count, "warmup": WARMUP, "ef_search": 64,
            "candidate_budget": 500, "min_candidates_per_head": 20,
            "max_candidates_per_head": 300,
            "attention_dim": ATTN_DIM, "key_dim": KEY_DIM, "value_dim": VALUE_DIM,
            "lambda": LAMBDA,
            "per_rep": reps_stats,
            "rep_recall10_qrels_mean": rep_rec,
            "recall10_qrels_mean_Nrep": round(mean_rec, 4) if mean_rec else None,
            "rep_ndcg10_qrels_mean": rep_ndcg,
            "ndcg10_qrels_mean_Nrep": round(mean_ndcg, 4) if mean_ndcg else None,
            "union_identity_ok": union_ok,
            "union_report": union_report,
        }
        with open(os.path.join(RAW, rid, "metrics.json"), "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        register(rid, f"C8 {split} {name} {ds} ({len(keep)} qids, {rep_count} reps)",
                 "PASS",
                 f"ndcg10={metrics['ndcg10_qrels_mean_Nrep']} "
                 f"recall10={metrics['recall10_qrels_mean_Nrep']} "
                 f"union_identity_ok={union_ok}")

    print(json.dumps({"run_id": run_id, "group": [r["run_id"] for r in grp],
                      "split": split, "dataset": ds, "n_queries": len(keep),
                      "rep_count": rep_count, "union_identity_ok": union_ok,
                      "union_report": union_report}, indent=2))
    if not union_ok:
        raise SystemExit("UNION IDENTITY VIOLATION: B/C/D/E/F/G/H/I union sets not "
                         "identical within shared engine (stop condition #4)")


if __name__ == "__main__":
    main()
