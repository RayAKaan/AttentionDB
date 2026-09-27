"""C5 cross-head interaction test orchestrator (Windows-safe, C1/C4 protocol).

Mirrors c4_test_run.py structure but drives the C5 probe (c5pilot) with the
frozen C5 fields (ef_search real knob, cross-refine lambda). Runs one plan row
(--run-id) from c5-run-plan.csv per invocation:

  - registers the run pre-execution as C5-... in append-only RUN-INDEX.yaml;
  - resolves the frozen dataset/vector artifacts (same embed files as C4);
  - builds the engine via c5pilot with per-head ef applied as a REAL knob
    (RetrievalConfig.ef_search) and, for MODE-C, one-step cross-head refinement
    at the configured lambda;
  - emits per-query rows incl. the C5 causal ledger for interaction cells;
  - records environment/status/metrics a la C4; raw evidence immutable.

Split handling is consistent with C4:
  - PROBE/SMOKE: tiny seeded subsample of the VALID pool (disjoint from TEST);
  - VALID       : NFC dev 324 / SciFact seeded train-sample 200 (tuning gate);
  - TEST        : frozen official test qrels (SCI 300 / NFC 323).

C5-EFPROBE-00X rows run MODE-B at ef 16 vs 128 on the same tiny pool; the
analysis layer decides whether ef is a real knob (stop condition: knoo inert).
"""

import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "raw")
RAW = os.path.abspath(RAW)
PLAN = os.path.join(os.path.dirname(__file__), "..", "c5-run-plan.csv")
PLAN = os.path.abspath(PLAN)
C5PILOT = os.path.join(os.path.dirname(__file__), "..", "probe",
                       "target", "release", "c5pilot.exe")
C5PILOT = os.path.abspath(C5PILOT)
SEED = 20260925
WARMUP = 20
K = 10


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
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_plan_row(run_id):
    with open(PLAN, encoding="utf8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == run_id:
                if len(parts) != len(header):
                    raise SystemExit(f"row {run_id} has {len(parts)} cols; header {len(header)}")
                return dict(zip(header, parts))
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
    """Attach (query_ids, doc_ids, qrels_by_qrow) restricted to the TEST split."""
    query_ids = _jsonl_ids(os.path.join(dsinfo["data"], "queries.jsonl"))
    doc_ids = _jsonl_ids(os.path.join(dsinfo["data"], "corpus.jsonl"))
    qrels = parse_qrels(os.path.join(dsinfo["data"], "qrels", "test.tsv"),
                        query_ids, doc_ids)
    return query_ids, doc_ids, qrels


def val_rows(dsinfo):
    """VALID pool: NFC dev qids (all 324); SciFact seeded train-sample (200)."""
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


def write_env(run_dir, hash_files, mem_snapshot):
    avail = getattr(mem_snapshot, "ullAvailPhys", None)
    load = getattr(mem_snapshot, "dwMemoryLoad", None)
    env = {
        "host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "c5pilot_sha256": sha256_file(C5PILOT),
        "input_hashes": hash_files,
        "guardrail": {"policy": "sampler(500ms) aborts when child process-tree RSS "
                                ">= 85% of MemAvailable at preflight; 1800s timeout",
                      "mem_avail_at_preflight_bytes": avail,
                      "mem_load_pct_at_preflight": load,
                      "ram_abort_threshold_bytes": int(avail * 0.85) if avail else None},
    }
    with open(os.path.join(run_dir, "environment.yaml"), "w", encoding="utf8") as f:
        f.write(json.dumps(env, indent=2))
    return env


def run_one(art, cfg, cfg_name, ram_abort_bytes, run_dir, rep):
    out = os.path.join(art, f"{cfg_name}-rep{rep}.json")
    cfg_path = os.path.join(art, f"{cfg_name}-cfg.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)
    proc = subprocess.Popen([C5PILOT, "--config", cfg_path, "--out", out],
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
                            f"({ram_abort_bytes}) at {cfg_name} rep {rep}\n")
                proc.terminate()
            time.sleep(0.5)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    stdout_data, stderr_data = proc.communicate(timeout=1800)
    stop[0] = True
    t.join(timeout=1)
    if proc.returncode != 0:
        raise RuntimeError(f"{cfg_name} rep{rep} FAILED rc={proc.returncode}\n"
                           f"{stderr_data[:2000]}")
    return json.load(open(out, encoding="utf8")), peak[0]


def derive_rows_stats(arr, query_rows):
    pq = arr["per_query"]
    recs = [p["recall10_qrels"] for p in pq]
    exact = [p.get("recall10_exact", 0.0) for p in pq]
    lat = [p["latency_us"] for p in pq if "latency_us" in p]
    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    p50 = lat_sorted[n // 2] if n else None
    st = {
        "n_queries": len(pq),
        "recall10_qrels_mean": float(sum(recs) / len(recs)) if recs else None,
        "recall10_exact_mean": float(sum(exact) / len(exact)) if exact else None,
        "p50_us": p50,
        "p95_us": lat_sorted[int(0.95 * (n - 1))] if n else None,
        "deadline_exceeded": int(arr.get("deadline_exceeded_count", 0)),
    }
    if arr.get("mode") == "C":
        st["candidate_change_count"] = int(arr.get("candidate_change_count", 0))
        st["surprise_empty_queries"] = int(arr.get("surprise_empty_queries", 0))
    return st


def build_engine_cfg(run_id, row, dsinfo, keep, qrels, qvec_out, doc_vectors,
                     coll_heads, attend, ef, lam, config_id, deadline_us=0):
    return {
        "run_id": run_id, "mode": row["mode"].replace("MODE-", ""), "k": K,
        "seed": SEED, "warmup": WARMUP,
        "n_queries": len(dsinfo["query_ids"]),
        "n_docs": len(dsinfo["doc_ids"]),
        "dim": dsinfo["dim"], "subsample": keep,
        "collection_heads": coll_heads, "attend_heads": attend,
        "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
        "query_vectors": qvec_out, "doc_vectors": doc_vectors,
        "candidate_budget": int(row["candidate_budget"]) if row["candidate_budget"].isdigit() else 500,
        "min_candidates_per_head": 20,
        "ef_search": ef,
        "lambda": lam,
        "configuration_id": config_id,
        "deadline_us": deadline_us,
    }


def write_status(run_dir, status, extra=""):
    with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf8") as f:
        f.write(status + ("\n" + extra if extra else ""))


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
    dsinfo = resolve_dataset(row["dataset"])
    mode = row["mode"].replace("MODE-", "")
    split = row["split"]
    rep_count = int(row["rep_count"] or 1)
    ef = int(row["ef_search"]) if row["ef_search"] else 64
    lam = float(row["lambda"]) if row["lambda"] else 0.0

    if split in ("PROBE", "SMOKE"):
        query_ids, doc_ids, qrels = val_rows(dsinfo)
        rows = sorted(qrels.keys())
        take = min(int(row["query_count"]), len(rows))
        order = seeded_shuffle(len(rows), SEED)
        keep = [rows[i] for i in order[:take]]
        qrels = {r: qrels[r] for r in keep}
        n_q = len(keep)
    elif split == "VALID":
        query_ids, doc_ids, qrels = val_rows(dsinfo)
        keep = sorted(qrels.keys())
        n_q = len(keep)
    elif split == "TEST":
        query_ids, doc_ids, qrels = test_rows(dsinfo)
        keep = sorted(qrels.keys())
        n_q = len(keep)
    else:
        raise SystemExit(f"unknown split {split}")
    dsinfo["query_ids"] = query_ids
    dsinfo["doc_ids"] = doc_ids

    if dry:
        print(json.dumps({
            "run_id": run_id, "mode": mode, "dataset": row["dataset"],
            "split": split, "n_queries": n_q,
            "rep_count": rep_count, "warmup": WARMUP,
            "ef": ef, "lambda": lam,
            "heads": list(dsinfo["heads"]),
        }, indent=2))
        return

    run_dir = os.path.join(RAW, run_id)
    if os.path.exists(run_dir):
        raise SystemExit(f"run dir {run_dir} exists (immutable); new run_id required")
    os.makedirs(os.path.join(run_dir, "artifacts"))
    art = os.path.join(run_dir, "artifacts")

    reg_pre = (f"- run_id: {run_id}\n  status: RUN\n"
               f"  purpose: 'C5 {split} {mode} {row['dataset']} "
               f"({n_q} qids, {rep_count} fresh-process reps, ef={ef}, lambda={lam})'\n"
               f"  started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
               f"  preregistration: 'frozen c5-run-plan.csv {run_id}; "
               f"pre-execution registration before measurement'\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg_pre)

    def head_src(head):
        prefix = dsinfo["prefix"]
        for cand in (f"{head}.npy", f"{prefix}-{head}.npy"):
            p = os.path.join(dsinfo["emb"], cand)
            if os.path.exists(p):
                return p
        return None

    doc_vectors = {}
    for head in dsinfo["heads"]:
        if head == "CANONICAL":
            continue
        src = head_src(head)
        if src is None:
            continue
        out = os.path.join(art, f"{head}.f32")
        np.load(src).astype(np.float32).tofile(out)
        doc_vectors[head] = out
    canonical_out = os.path.join(art, "CANONICAL.f32")
    if not os.path.exists(os.path.join(art, "CANONICAL.f32")):
        canon_src = head_src("CANONICAL")
        if canon_src is None:
            raise SystemExit("CANONICAL view missing")
        np.load(canon_src).astype(np.float32).tofile(canonical_out)
    q_src = head_src("QUERIES")
    if q_src is None:
        raise SystemExit("QUERIES view missing")
    qvec_out = os.path.join(art, "QUERIES.f32")
    np.load(q_src).astype(np.float32).tofile(qvec_out)

    mem_snapshot = mem_status()
    ram_abort_bytes = int(getattr(mem_snapshot, "ullAvailPhys", 0) * 0.85)

    if mode == "A":
        coll_heads = ["CANONICAL"]
        attend = ["CANONICAL"]
        grid_doc_vectors = {"CANONICAL": os.path.join(art, "CANONICAL.f32")}
    else:
        coll_heads = list(doc_vectors.keys())
        attend = list(doc_vectors.keys())
        grid_doc_vectors = dict(doc_vectors)
        grid_doc_vectors["CANONICAL"] = os.path.join(art, "CANONICAL.f32")
        if not coll_heads:
            raise SystemExit(f"no HEAD views materialized for {row['dataset']} multi-head mode")

    cfg = build_engine_cfg(
        run_id, row, dsinfo, keep, qrels, qvec_out, grid_doc_vectors,
        coll_heads, attend, ef, lam,
        f"{run_id}__{mode}")

    arrs, peak_rss = [], []
    for rep in range(1, rep_count + 1):
        arr, pr = run_one(art, cfg, "RUN", ram_abort_bytes, run_dir, rep)
        arrs.append(arr)
        peak_rss.append(pr)
    reps_stats = [derive_rows_stats(a, keep) for a in arrs]
    rep_recalls = [r["recall10_qrels_mean"] for r in reps_stats]
    rep_exacts = [r["recall10_exact_mean"] for r in reps_stats]
    mean_rec = sum(rep_recalls) / len(rep_recalls)
    sd_rec = (sum((x - mean_rec) ** 2 for x in rep_recalls) / len(rep_recalls)) ** 0.5
    p50s = [r["p50_us"] for r in reps_stats]

    metrics = {
        "run_id": run_id, "mode": mode, "dataset": row["dataset"],
        "split": split, "phase": "C5 execution",
        "test_queries": n_q, "rep_count": rep_count, "warmup": WARMUP,
        "ef_search": ef, "lambda": lam,
        "candidate_budget": int(row["candidate_budget"]) if row["candidate_budget"].isdigit() else 500,
        "per_rep": reps_stats,
        "rep_recall10_qrels_mean": rep_recalls,
        "recall10_qrels_mean_5rep": round(mean_rec, 4),
        "recall10_qrels_sd_5rep": round(sd_rec, 4),
        "rep_recall10_exact_mean": rep_exacts,
        "recall10_exact_mean_5rep": round(sum(rep_exacts) / len(rep_exacts), 4),
        "p50_us_per_rep": p50s,
        "peak_child_rss_MiB": [round(x / 1048576, 1) if x else None for x in peak_rss],
    }
    if mode == "C":
        metrics["candidate_change_count"] = [r.get("candidate_change_count", 0) for r in reps_stats]
        metrics["surprise_empty_queries"] = [r.get("surprise_empty_queries", 0) for r in reps_stats]

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    hash_files = {}
    for h in sorted(os.listdir(art)):
        p = os.path.join(art, h)
        if os.path.isfile(p):
            hash_files[h] = sha256_file(p)
    write_env(run_dir, hash_files, mem_snapshot)
    write_status(run_dir, "PASS",
                 f"recall10_qrels mean={metrics['recall10_qrels_mean_5rep']:.4f} "
                 f"sd={metrics['recall10_qrels_sd_5rep']:.4f} "
                 f"p50_us={round(float(p50s[0]), 1) if p50s else None}")
    reg = (f"- run_id: {run_id}\n  status: PASS\n"
           f"  purpose: 'C5 {split} {mode} {row['dataset']} "
           f"({n_q} qids, {rep_count} reps)'\n"
           f"  completed: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
           f"  result: recall10_qrels={metrics['recall10_qrels_mean_5rep']:.4f} "
           f"sd={metrics['recall10_qrels_sd_5rep']:.4f} p50_us="
           f"{round(float(p50s[0]), 1) if p50s else None}\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()