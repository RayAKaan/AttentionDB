"""C4.4 confirmed-scale TEST-cell orchestrator (Windows-safe, C1 protocol).

Runs ONE frozen c4-run-plan.csv TEST cell (split == TEST) on the FULL frozen
test set (SciFact qrels/test.tsv 300 qids; NFCorpus qrels/test.tsv 323 qids)
with 5 fresh-process reps (B0 oracle cells rep_count 1), warmup 20, seeded
paired order, per-config reflections from C4.2 validation tuning
(c4-validation-tuning-results.md), a 500 ms sampler-abort guardrail (>=85% of
preflight MemAvailable), and an append-only RUN-INDEX registration before and
after measurement.

Config reflection table (frozen in this file, keyed by run_id):
  - C4.2 matched configs (all recorded in c4-validation-tuning-results.md):
    NFC-B1 ef64 | SCI-B1 ef32 | NFC-B2 ef16 cand50 | NFC-B7 ef64.
  - TARGET-UNREACHABLE cells (SCI-B2, SCI-B7) run the PLAN DEFAULT config
    (ef 64, cand 500) and are reported with the honest quality-latency curve,
    never interpolated.
  - B3 cells mirror the B2 candidate generation: SCI-B3 inherits the SCI-B2
    plan default (B2 unreachable), NFC-B3 inherits NFC-B2 ef16 cand50, with
    the frozen modelcard b3-lodo-nf-v2-s20260925-h32-lr0.01.json.
  - B4 (C4-W04-SCI-B4-001) = B2 candidate gen + identity-init QK attention
    (engine default fusion attention 0.3), config mirrors the B2 row.
  - B7 TRKB cells run vendor-default Full with all defaults recorded.
  - MEM/TIME boundary cells run the same base config as their reference cell
    across budget legs: MEM {512MiB|1.0GiB|1.6GiB} via sampler-abort;
    TIME {1|10|100ms} via c2pilot deadline_us (deadline misses = FAILURES
    counted; latencies reported with exclusion counts).

Usage:
  python c4_test_run.py --run-id C4-W01-SCI-B1-001 [--dry-run]

  The runtime c2pilot sha256 is recorded in environment.yaml for every run.
"""
import ctypes, hashlib, json, os, platform, random, re, shutil, subprocess, sys, threading, time
from datetime import datetime, timezone

import numpy as np

CS = r"H:\Attention-DB\AttentionDB\research\comparative-study"
RAW = os.path.join(CS, "raw")
PLAN = os.path.join(CS, "c4", "c4-run-plan.csv")
C2PILOT = os.path.join(CS, "c2", "probe", "target", "release", "c2pilot.exe")
SEED = 20260925
K = 10
WARMUP = 20

MODELCARD = os.path.join(CS, "c2", "b3", "modelcards",
                         "b3-lodo-nf-v2-s20260925-h32-lr0.01.json")

# ---- frozen per-cell config reflections (C4.2 -> C4.4) -------------------
# ef=None -> engine default 64; cand=None -> engine default 500.
REFLECT = {
    "C4-W01-SCI-B1-001":   {"mode": "B1", "ef": 32, "cand": None, "note": "C4.2 matched SCI-B1 EF32 0.7775"},
    "C4-W01-NFC-B1-001":   {"mode": "B1", "ef": 64, "cand": None, "note": "C4.2 matched NFC-B1 EF64 0.1351"},
    "C4-W02-SCI-B2-001":   {"mode": "B2", "ef": 64, "cand": 500, "note": "TARGET-UNREACHABLE (best 0.7458 EF64-C50); plan default retained"},
    "C4-W02-NFC-B2-001":   {"mode": "B2", "ef": 16, "cand": 50, "note": "C4.2 matched NFC-B2 EF16-C50 0.1393"},
    "C4-W03-SCI-B3-001":   {"mode": "B3", "ef": 64, "cand": 500, "note": "B3 mirrors B2 candidate gen; SCI-B2 unreachable -> plan default"},
    "C4-W03-NFC-B3-001":   {"mode": "B3", "ef": 16, "cand": 50, "note": "B3 mirrors B2 candidate gen; NFC-B2 ef16 cand50"},
    "C4-W04-SCI-B4-001":   {"mode": "B4", "ef": 64, "cand": 500, "note": "B4 = B2 cand gen + identity-init QK (fusion 0.3); mirrors B2 row"},
    "C4-W07-SCI-B7-001":   {"mode": "B7", "ef": 64, "cand": 500, "note": "TARGET-UNREACHABLE (best 0.7021 EF32); Full default retained"},
    "C4-W07-NFC-B7-001":   {"mode": "B7", "ef": 64, "cand": 500, "note": "C4.2 matched NFC-B7 EF64 0.1372"},
    "C4-W07-SCI-B7-TRKB-001": {"mode": "B7", "ef": 64, "cand": None, "note": "Track-B practical default Full; no tuning"},
    "C4-W01-NFC-B1-TRKB-001": {"mode": "B1", "ef": 64, "cand": None, "note": "Track-B practical default single-head (ef 64 k10)"},
}
MEM_LEGS_MIB = [512, 1024, 1638]           # {512MiB | 1.0GiB | 1.6GiB}
TIME_LEGS_US = [1000, 10000, 100000]       # {1 | 10 | 100 ms}


# ---- Windows memory sampler (same as validated c3_pilot_run.py) ----------
if os.name == "nt":
    class MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcMemCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def mem_status():
        ms = MemStatus(); ms.dwLength = ctypes.sizeof(MemStatus)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms

    def proc_rss_bytes(pid):
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if not h:
            return 0
        try:
            pmc = ProcMemCounters(); pmc.cb = ctypes.sizeof(ProcMemCounters)
            ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            return int(pmc.WorkingSetSize) if ok else 0
        finally:
            kernel32.CloseHandle(h)
else:
    def mem_status():
        with open("/proc/meminfo") as f:
            d = dict(l.strip().split(":", 1) for l in f)
        total = int(d["MemTotal"].split()[0]) * 1024
        avail = int(d["MemAvailable"].split()[0]) * 1024
        return {"dwMemoryLoad": int(100 * (total - avail) / total),
                "ullAvailPhys": avail, "ullTotalPhys": total}

    def proc_rss_bytes(pid):
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0


def sha256_file(p):
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
    """Attach (query_ids, doc_ids, qrels_by_qrow) restricted to the frozen TEST split."""
    query_ids = _jsonl_ids(os.path.join(dsinfo["data"], "queries.jsonl"))
    doc_ids = _jsonl_ids(os.path.join(dsinfo["data"], "corpus.jsonl"))
    qrels = parse_qrels(os.path.join(dsinfo["data"], "qrels", "test.tsv"),
                        query_ids, doc_ids)
    dsinfo["query_ids"] = query_ids
    dsinfo["doc_ids"] = doc_ids
    return query_ids, doc_ids, qrels


def write_env(run_dir, hash_files, mem_snapshot):
    pyver = platform.python_version()
    avail = getattr(mem_snapshot, "ullAvailPhys", None)
    load = getattr(mem_snapshot, "dwMemoryLoad", None)
    env = {
        "host": platform.node(), "os": platform.platform(), "python": pyver,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "c2pilot_sha256": sha256_file(C2PILOT),
        "input_hashes": hash_files,
        "guardrail": {"policy": "sampler(500ms) aborts when child process-tree "
                                "RSS >= 85% of MemAvailable at preflight",
                      "mem_avail_at_preflight_bytes": avail,
                      "mem_load_pct_at_preflight": load,
                      "ram_abort_threshold_bytes": int(avail * 0.85) if avail else None},
    }
    with open(os.path.join(run_dir, "environment.yaml"), "w", encoding="utf8") as f:
        f.write(json.dumps(env, indent=2))
    return env


def run_one(art, cfg, cfg_name, ram_abort_bytes, run_dir, rep):
    """Run one c2pilot invocation (fresh process) -> (arr, peak_rss)."""
    out = os.path.join(art, f"{cfg_name}-rep{rep}.json")
    cfg_path = os.path.join(art, f"{cfg_name}-cfg.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)
    proc = subprocess.Popen([C2PILOT, "--config", cfg_path, "--out", out],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = [0]
    stop = [False]

    def sample():
        while not stop[0]:
            rss = proc_rss_bytes(proc.pid)
            peak[0] = max(peak[0], rss)
            if rss >= ram_abort_bytes:
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
    return {
        "n_queries": len(pq),
        "recall10_qrels_mean": float(sum(recs) / len(recs)) if recs else None,
        "recall10_exact_mean": float(sum(exact) / len(exact)) if exact else None,
        "p50_us": p50,
        "p95_us": lat_sorted[int(0.95 * (n - 1))] if n else None,
        "deadline_exceeded": int(arr.get("deadline_exceeded_count", 0)),
    }


def build_engine_cfg(run_id, row, dsinfo, keep, qrels, qvec_out, doc_vectors,
                     coll_heads, attend, ef, cand, modelcard, config_id, deadline_us=0):
    return {
        "run_id": run_id, "mode": row["mode"].replace("MODE-", ""), "k": K,
        "seed": SEED, "warmup": WARMUP,
        "n_queries": len(dsinfo["query_ids"]), "n_docs": len(dsinfo["doc_ids"]),
        "dim": dsinfo["dim"], "subsample": keep,
        "collection_heads": coll_heads, "attend_heads": attend,
        "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
        "query_vectors": qvec_out, "doc_vectors": doc_vectors,
        "candidate_budget": cand if cand is not None else 500,
        "min_candidates_per_head": 20,
        "ef_search": ef if ef is not None else 64,
        "ef_construction": 400, "m": 16,
        "modelcard": modelcard if row["mode"].replace("MODE-", "") == "B3" else "",
        "configuration_id": config_id,
        "deadline_us": deadline_us,
    }


def build_b0_cfg(run_id, row, dsinfo, keep, qrels, qvec_out, canonical_out):
    return {
        "run_id": run_id, "mode": "B0", "k": K, "seed": SEED, "warmup": WARMUP,
        "n_queries": len(dsinfo["query_ids"]), "n_docs": len(dsinfo["doc_ids"]),
        "dim": dsinfo["dim"], "subsample": keep,
        "collection_heads": [], "attend_heads": [],
        "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
        "query_vectors": qvec_out,
        "doc_vectors": {"CANONICAL": canonical_out},
        "candidate_budget": 500, "min_candidates_per_head": 20,
        "ef_search": 64, "configuration_id": f"{run_id}__B0-ORACLE",
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
    if row["split"] != "TEST":
        raise SystemExit(f"{run_id} is not a TEST cell (split={row['split']}); "
                         f"use c4_tune_run.py for VAL tuning")
    dsinfo = resolve_dataset(row["dataset"])
    query_ids, doc_ids, qrels = test_rows(dsinfo)
    keep = sorted(qrels.keys())
    n_test = len(keep)
    mode_full = row["mode"].replace("MODE-", "")
    rep_count = int(row["rep_count"] or 1)
    is_b0 = mode_full == "B0"
    is_mem = run_id == "C4-W19-SCI-B3-MEM-001"
    is_time = run_id == "C4-W19-SCI-B1-TIME-001"

    if dry:
        base_of = {
            "C4-W19-SCI-B3-MEM-001": "C4-W03-SCI-B3-001",
            "C4-W19-SCI-B1-TIME-001": "C4-W01-SCI-B1-001",
        }
        ref = REFLECT.get(base_of.get(run_id, run_id), {})
        print(json.dumps({
            "run_id": run_id, "mode": mode_full, "dataset": row["dataset"],
            "split": "TEST", "n_test_queries": n_test,
            "rep_count": rep_count, "warmup": WARMUP,
            "reflection": REFLECT.get(base_of.get(run_id, run_id), {}),
            "legs": ("MEM_MiB " + str(MEM_LEGS_MIB) if is_mem else
                     "TIME_us " + str(TIME_LEGS_US) if is_time else None),
        }, indent=2))
        return

    run_dir = os.path.join(RAW, run_id)
    if os.path.exists(run_dir):
        raise SystemExit(f"run dir {run_dir} exists (immutable); new run_id required")
    os.makedirs(os.path.join(run_dir, "artifacts"))
    art = os.path.join(run_dir, "artifacts")

    reg_pre = (f"- run_id: {run_id}\n  status: RUN\n"
               f"  purpose: 'C4.4 confirmed-scale TEST {mode_full} {row['dataset']} "
               f"on full test set ({n_test} qids), {rep_count} fresh-process reps'\n"
               f"  started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
               f"  preregistration: 'frozen c4-run-plan.csv {run_id}; "
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
    if not os.path.exists(canonical_out):
        np.load(head_src("CANONICAL")).astype(np.float32).tofile(canonical_out)
    q_src = head_src("QUERIES")
    if q_src is None:
        raise SystemExit("QUERIES view missing")
    qvec_out = os.path.join(art, "QUERIES.f32")
    np.load(q_src).astype(np.float32).tofile(qvec_out)

    mem_snapshot = mem_status()
    ram_abort_bytes = int(getattr(mem_snapshot, "ullAvailPhys", 0) * 0.85)

    # ---- B0 oracle TEST cell ----------------------------------------------
    if is_b0:
        b0_cfg = build_b0_cfg(run_id, row, dsinfo, keep, qrels, qvec_out, canonical_out)
        arr, pr = run_one(art, b0_cfg, "B0", ram_abort_bytes, run_dir, 1)
        st = derive_rows_stats(arr, keep)
        metrics = {
            "run_id": run_id, "mode": "B0", "dataset": row["dataset"],
            "split": "TEST", "phase": "C4.4 confirmed-scale oracle",
            "test_queries": n_test, "rep_count": 1,
            "recall10_qrels_mean": st["recall10_qrels_mean"],
            "recall10_exact_mean": st["recall10_exact_mean"],
            "exact_top10_hash": sha256_file(os.path.join(art, "B0-rep1.json")),
            "config": b0_cfg,
        }
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        hash_files = {}
        for h in sorted(os.listdir(art)):
            p = os.path.join(art, h)
            if os.path.isfile(p):
                hash_files[h] = sha256_file(p)
        write_env(run_dir, hash_files, mem_snapshot)
        write_status(run_dir, "PASS", f"recall10_qrels={metrics['recall10_qrels_mean']:.4f}")
        reg = (f"- run_id: {run_id}\n  status: PASS\n"
               f"  purpose: 'C4.4 confirmed-scale oracle {row['dataset']} "
               f"({n_test} test qids)'\n"
               f"  completed: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
               f"  result: recall10_qrels={metrics['recall10_qrels_mean']:.4f}\n")
        with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
            f.write(reg)
        print(json.dumps(metrics, indent=2))
        return

    # ---- engine cells -----------------------------------------------------
    # MEM/TIME boundary cells inherit the reflection of their reference cell.
    base_of = {
        "C4-W19-SCI-B3-MEM-001": "C4-W03-SCI-B3-001",
        "C4-W19-SCI-B1-TIME-001": "C4-W01-SCI-B1-001",
    }
    ref_key = base_of.get(run_id, run_id)
    ref = REFLECT.get(ref_key)
    if ref is None:
        raise SystemExit(f"no frozen reflection for engine TEST cell {run_id}")
    mode = ref["mode"]
    if mode == "B1":
        coll_heads = ["CANONICAL"]; attend = ["CANONICAL"]
        grid_doc_vectors = {"CANONICAL": canonical_out}
    else:
        coll_heads = list(doc_vectors.keys()); attend = list(doc_vectors.keys())
        grid_doc_vectors = dict(doc_vectors)
        grid_doc_vectors["CANONICAL"] = canonical_out
        if not coll_heads:
            raise SystemExit(f"no HEAD views materialized for {row['dataset']} multi-head mode")

    modelcard = MODELCARD if mode == "B3" else ""

    # ---- MEM boundary cell (legs = memory budgets via sampler) ------------
    if is_mem:
        legs = []
        for miB in MEM_LEGS_MIB:
            leg_bytes = int(miB * 1048576)
            guard = min(ram_abort_bytes, leg_bytes)
            cfg = build_engine_cfg(
                run_id, row, dsinfo, keep, qrels, qvec_out, grid_doc_vectors,
                coll_heads, attend, ref["ef"], ref["cand"], modelcard,
                f"{run_id}__MEM{miB}MiB")
            arrs, peak_rss = [], []
            for rep in range(1, rep_count + 1):
                arr, pr = run_one(art, cfg, f"MEM{miB}MiB", guard, run_dir, rep)
                arrs.append(arr); peak_rss.append(pr)
            st = derive_rows_stats(arrs[0], keep)
            st["leg_miB"] = miB
            st["leg_guard_bytes"] = guard
            st["peak_child_rss_MiB"] = [round(x / 1048576, 1) if x else None for x in peak_rss]
            st["within_leg_budget"] = all(
                (x or 0) <= leg_bytes for x in peak_rss)
            legs.append(st)
        metrics = {
            "run_id": run_id, "mode": mode, "dataset": row["dataset"],
            "split": "TEST", "phase": "C4.4 confirmed-scale MEM boundary",
            "test_queries": n_test, "rep_count": rep_count, "warmup": WARMUP,
            "legs_miB": MEM_LEGS_MIB,
            "legs": legs,
            "guardrail": "sampler-abort at min(85% MemAvailable, leg budget) applied per leg; within_leg_budget=False = budget crossed",
            "config_note": ref["note"],
        }
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        hash_files = {}
        for h in sorted(os.listdir(art)):
            p = os.path.join(art, h)
            if os.path.isfile(p):
                hash_files[h] = sha256_file(p)
        write_env(run_dir, hash_files, mem_snapshot)
        all_within = all(l["within_leg_budget"] for l in legs)
        write_status(run_dir, "PASS" if all_within else "FAILED",
                     f"legs_within_budget={[l['within_leg_budget'] for l in legs]} "
                     f"peak_rss_MiB={[l['peak_child_rss_MiB'] for l in legs]}")
        reg = (f"- run_id: {run_id}\n  status: {'PASS' if all_within else 'FAILED'}\n"
               f"  purpose: 'C4.4 confirmed-scale MEM boundary {mode} {row['dataset']} "
               f"legs {MEM_LEGS_MIB} MiB'\n"
               f"  completed: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
               f"  result: legs_within_budget="
               f"{[l['within_leg_budget'] for l in legs]} peak_rss_MiB="
               f"{[l['peak_child_rss_MiB'] for l in legs]}\n")
        with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
            f.write(reg)
        print(json.dumps(metrics, indent=2))
        return

    # ---- TIME boundary cell (legs = per-query deadlines) ------------------
    if is_time:
        legs = []
        for d_us in TIME_LEGS_US:
            cfg = build_engine_cfg(
                run_id, row, dsinfo, keep, qrels, qvec_out, grid_doc_vectors,
                coll_heads, attend, ref["ef"], ref["cand"], modelcard,
                f"{run_id}__TIME{d_us}us", deadline_us=d_us)
            arrs, peak_rss = [], []
            for rep in range(1, rep_count + 1):
                arr, pr = run_one(art, cfg, f"TIME{d_us}us", ram_abort_bytes, run_dir, rep)
                arrs.append(arr); peak_rss.append(pr)
            st = derive_rows_stats(arrs[0], keep)
            st["deadline_us"] = d_us
            misses = [a.get("deadline_exceeded_count", 0) for a in arrs]
            st["deadline_exceeded_per_rep"] = misses
            st["deadline_exceeded_total"] = sum(misses)
            st["peak_child_rss_MiB"] = [round(x / 1048576, 1) if x else None for x in peak_rss]
            legs.append(st)
        all_pass = all(l["deadline_exceeded_total"] == 0 for l in legs)
        metrics = {
            "run_id": run_id, "mode": mode, "dataset": row["dataset"],
            "split": "TEST", "phase": "C4.4 confirmed-scale TIME boundary",
            "test_queries": n_test, "rep_count": rep_count, "warmup": WARMUP,
            "legs_us": TIME_LEGS_US,
            "legs": legs,
            "deadline_policy": "deadline misses = FAILURES counted; latency excludes deadline-miss queries (they return no result)",
            "config_note": ref["note"],
        }
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        hash_files = {}
        for h in sorted(os.listdir(art)):
            p = os.path.join(art, h)
            if os.path.isfile(p):
                hash_files[h] = sha256_file(p)
        write_env(run_dir, hash_files, mem_snapshot)
        write_status(run_dir, "PASS" if all_pass else "FAILED",
                     f"legs={[{'us': l['deadline_us'], 'exceeded': l['deadline_exceeded_total']} for l in legs]}")
        reg = (f"- run_id: {run_id}\n  status: {'PASS' if all_pass else 'FAILED'}\n"
               f"  purpose: 'C4.4 confirmed-scale TIME boundary {mode} {row['dataset']} "
               f"legs {TIME_LEGS_US} us'\n"
               f"  completed: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
               f"  result: deadline_exceeded="
               f"{[{'us': l['deadline_us'], 'exceeded': l['deadline_exceeded_total']} for l in legs]}\n")
        with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
            f.write(reg)
        print(json.dumps(metrics, indent=2))
        return

    # ---- standard engine TEST cell ----------------------------------------
    cfg = build_engine_cfg(
        run_id, row, dsinfo, keep, qrels, qvec_out, grid_doc_vectors,
        coll_heads, attend, ref["ef"], ref["cand"], modelcard,
        f"{run_id}__{'UNREACHABLE' if 'UNREACHABLE' in ref.get('note','') else mode}")
    arrs, peak_rss = [], []
    for rep in range(1, rep_count + 1):
        arr, pr = run_one(art, cfg, "RUN", ram_abort_bytes, run_dir, rep)
        arrs.append(arr); peak_rss.append(pr)
    reps_stats = [derive_rows_stats(a, keep) for a in arrs]
    rep_recalls = [r["recall10_qrels_mean"] for r in reps_stats]
    rep_exacts = [r["recall10_exact_mean"] for r in reps_stats]
    mean_rec = sum(rep_recalls) / len(rep_recalls)
    sd_rec = (sum((x - mean_rec) ** 2 for x in rep_recalls) / len(rep_recalls)) ** 0.5
    p50s = [r["p50_us"] for r in reps_stats]
    metrics = {
        "run_id": run_id, "mode": mode, "dataset": row["dataset"],
        "split": "TEST", "phase": "C4.4 confirmed-scale execution",
        "test_queries": n_test, "rep_count": rep_count, "warmup": WARMUP,
        "ef_search": cfg["ef_search"], "candidate_budget": cfg["candidate_budget"],
        "modelcard": cfg["modelcard"],
        "config_note": ref["note"],
        "per_rep": reps_stats,
        "rep_recall10_qrels_mean": rep_recalls,
        "recall10_qrels_mean_5rep": round(mean_rec, 4),
        "recall10_qrels_sd_5rep": round(sd_rec, 4),
        "rep_recall10_exact_mean": rep_exacts,
        "recall10_exact_mean_5rep": round(sum(rep_exacts) / len(rep_exacts), 4),
        "p50_us_per_rep": p50s,
        "peak_child_rss_MiB": [round(x / 1048576, 1) if x else None for x in peak_rss],
        "conclusion_rule": ("matches C4-BINVERIFY-003 verdict (>=5 fresh-process reps, "
                            "mean+sd reported; tolerance comparisons use rep-mean, not single draws)"),
    }
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
           f"  purpose: 'C4.4 confirmed-scale TEST {mode} {row['dataset']} "
           f"({n_test} test qids, {rep_count} reps)'\n"
           f"  completed: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
           f"  result: recall10_qrels={metrics['recall10_qrels_mean_5rep']:.4f} "
           f"sd={metrics['recall10_qrels_sd_5rep']:.4f} p50_us="
           f"{round(float(p50s[0]), 1) if p50s else None}\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()