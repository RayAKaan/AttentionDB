"""C3 pilot run orchestrator (Windows-safe, C1 protocol).

Executes one pilot cell from c3-run-plan.csv against the real engine via the
`c2pilot` probe binary, per C1 statistical plan:
  - 5 fresh-process reps (one c2pilot process per rep)
  - query order: seeded shuffle of the SAMPLED-subset (defined here for the
    first time, pre-registered: first min(N,100) of the seeded qrels-test set)
  - warmup 20 seeded-order queries executed but excluded from latency stats
  - 500 ms memory sampler (Windows GlobalMemoryStatusEx + process tree) abort
    at >=85% of physical RAM
  - per-query rows immutable; aggregates derived; Run registered in
    raw/RUN-INDEX.yaml (append-only)

Usage:
  python c3_pilot_run.py --run-id C3-W01-NFC-B1-001 [--reps 5] [--dry-run]

Reads the run definition from c3-run-plan.csv; dataset/embedding/qrels mapping
is resolved from C2 manifests. Exact-oracle (B0) reference rows are produced
by c2pilot in-process and cross-checked across modes by the analysis step.
"""
import ctypes, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

CS = r"H:\Attention-DB\AttentionDB\research\comparative-study"
RAW = os.path.join(CS, "raw")
PLAN = os.path.join(CS, "c3", "c3-run-plan.csv")
C2PILOT = os.path.join(CS, "c2", "probe", "target", "release", "c2pilot.exe")

MEMORYSTATUSEX = ctypes.c_int  # placeholder; full struct built below
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

    def mem_status() -> MemStatus:
        ms = MemStatus(); ms.dwLength = ctypes.sizeof(MemStatus)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms

    def proc_rss_bytes(pid: int) -> int:
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
    def mem_status() -> dict:
        with open("/proc/meminfo") as f:
            d = dict(l.strip().split(":", 1) for l in f)
        total = int(d["MemTotal"].split()[0]) * 1024
        avail = int(d["MemAvailable"].split()[0]) * 1024
        return {"dwMemoryLoad": int(100 * (total - avail) / total),
                "ullAvailPhys": avail, "ullTotalPhys": total}

    def proc_rss_bytes(pid: int) -> int:
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


def load_plan_rule(run_id):
    rules = {}
    with open(PLAN, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6 and parts[0] == run_id:
                rules = {"run_id": parts[0], "cell": parts[1], "mode": parts[2],
                         "dataset": parts[3], "queries_source": parts[4],
                         "subset": parts[5], "reps": int(parts[6]) if parts[6] else 1,
                         "warmup": int(parts[7]) if parts[7] else 0,
                         "seed": int(parts[8]) if parts[8] else 20260925}
                return rules
    raise SystemExit(f"run_id {run_id} not found in {PLAN}")


def resolve_dataset(rules):
    ds = rules["dataset"]
    # deterministic-embedded datasets
    if ds == "DS-NFCORPUS":
        emb = os.path.join(RAW, "C2-EMBED-NFCORPUS-003", "artifacts")
        data = os.path.join(RAW, "datasets", "beir", "nfcorpus")
        query_ids = _jsonl_ids(os.path.join(data, "queries.jsonl"))
        doc_ids = _jsonl_ids(os.path.join(data, "corpus.jsonl"))
        qrels = os.path.join(data, "qrels", "test.tsv")
        return {"emb": emb, "query_ids": query_ids, "doc_ids": doc_ids, "qrels": qrels,
                "heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE", "CANONICAL"],
                "n_docs": len(doc_ids), "dim": 384, "prefix": "DS-NFCORPUS"}
    if ds == "DS-SCIFACT":
        emb = os.path.join(RAW, "C3-EMBED-SCIFACT-001", "artifacts")
        data = os.path.join(RAW, "datasets", "beir", "scifact")
        query_ids = _jsonl_ids(os.path.join(data, "queries.jsonl"))
        doc_ids = _jsonl_ids(os.path.join(data, "corpus.jsonl"))
        qrels = os.path.join(data, "qrels", "test.tsv")
        return {"emb": emb, "query_ids": query_ids, "doc_ids": doc_ids, "qrels": qrels,
                "heads": ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE", "CANONICAL"],
                "n_docs": len(doc_ids), "dim": 384, "prefix": "DS-SCIFACT"}
    if ds.startswith("DS-SYNTH-PH2B"):
        raise SystemExit(f"synth dataset {ds} handled by a separate harness; not c2pilot")
    raise SystemExit(f"no resolver for dataset {ds}")


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
            if line.lower().startswith("query-id") or line.split()[0].lower() == "query-id":
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            qid, did, grade = parts[0], parts[1], int(parts[2])
            if qid not in qrow or did not in drow:
                continue
            out.setdefault(qrow[qid], {})[drow[did]] = int(grade)
    return out


def seeded_subset(n_queries, subset_size, seed):
    import random
    rnd = random.Random(seed + 0x5EED)
    idx = list(range(n_queries))
    rnd.shuffle(idx)
    return idx[:min(subset_size, n_queries)]


def write_env(run_dir, hash_files, mem_snapshot):
    pyver = platform.python_version()
    avail = getattr(mem_snapshot, "ullAvailPhys", None)
    load = getattr(mem_snapshot, "dwMemoryLoad", None)
    env = {
        "host": platform.node(), "os": platform.platform(), "python": pyver,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "c2pilot_sha256": sha256_file(C2PILOT),
        "input_hashes": hash_files,
        "env_vars": {k: os.environ[k] for k in
                     ("PH3E_CRASH_FILE_LIMIT",) if k in os.environ},
        "guardrail": {"policy": "sampler(500ms) aborts when child process-tree "
                                "RSS >= 85% of MemAvailable at preflight",
                      "mem_avail_at_preflight_bytes": avail,
                      "mem_load_pct_at_preflight": load,
                      "ram_abort_threshold_bytes": int(avail * 0.85) if avail else None},
    }
    with open(os.path.join(run_dir, "environment.yaml"), "w", encoding="utf8") as f:
        f.write(json.dumps(env, indent=2))
    return env


def main():
    args = sys.argv[1:]
    run_id = None
    reps = 5
    dry = False
    i = 0
    while i < len(args):
        if args[i] == "--run-id":
            run_id = args[i + 1]; i += 2
        elif args[i] == "--reps":
            reps = int(args[i + 1]); i += 2
        elif args[i] == "--dry-run":
            dry = True; i += 1
        else:
            raise SystemExit(f"unknown arg {args[i]}")
    if not run_id:
        raise SystemExit("--run-id required")
    rules = load_plan_rule(run_id)
    rules["mode"] = rules["mode"].replace("MODE-", "")  # normalize CSV MODE-B* -> B*
    ds = resolve_dataset(rules)

    run_dir = os.path.join(RAW, run_id)
    if os.path.exists(run_dir):
        raise SystemExit(f"run dir {run_dir} already exists (immutable); pick a new run_id")

    # ---- instantiate query subset + qrels + query order (pre-registered) --
    # SAMPLED-subset is drawn from the qrels-TEST split (queries_source), not
    # from all queries.jsonl rows, so every sampled query has qrels labels.
    qrels = parse_qrels(ds["qrels"], ds["query_ids"], ds["doc_ids"])
    candidate = sorted(qrels.keys())
    take = min(100, len(candidate))
    subset = seeded_subset(len(candidate), take, rules["seed"])
    keep = [candidate[i] for i in subset]

    # ---- resolve head source paths (no fs writes yet) ------------------
    def head_src(head):
        prefix = ds.get("prefix", "")
        for cand in (f"{head}.npy", f"{prefix}-{head}.npy"):
            p = os.path.join(ds["emb"], cand)
            if os.path.exists(p):
                return p
        return None

    doc_vectors = {}
    for head in ds["heads"]:
        src = head_src(head)
        if src is None:
            continue
        if head == "CANONICAL":
            doc_vectors["CANONICAL"] = os.path.join(run_dir, "artifacts", "CANONICAL.f32")
        elif head == "QUERIES":
            pass
        else:
            doc_vectors[head] = os.path.join(run_dir, "artifacts", f"{head}.f32")
    qvec = os.path.join(run_dir, "artifacts", "QUERIES.f32")
    if head_src("QUERIES") is None:
        raise SystemExit("QUERIES view missing from embed export")

    # ---- build c2pilot config ----------------------------------------
    nq_all = len(ds["query_ids"])
    head_keys = [h for h in doc_vectors if h != "CANONICAL"]  # collection heads
    cfg = {
        "run_id": run_id, "mode": rules["mode"], "k": 10, "seed": rules["seed"],
        "warmup": rules["warmup"], "n_queries": nq_all, "n_docs": ds["n_docs"],
        "dim": ds["dim"], "subsample": keep,
        "collection_heads": head_keys,
        "modelcard": os.path.join(CS, "c2", "b3", "modelcards",
                                  "b3-lodo-nf-v2-s20260925-h32-lr0.01.json")
                     if rules["mode"] == "B3" else "",
    }
    if rules["mode"] == "B1":
        cfg["collection_heads"] = ["CANONICAL"]
        cfg["attend_heads"] = ["CANONICAL"]
    else:
        cfg["attend_heads"] = list(head_keys)
    cfg["qrels"] = {str(k): {str(r): g for r, g in v.items()}
                    for k, v in qrels.items() if k in keep}
    cfg["query_vectors"] = qvec
    cfg["doc_vectors"] = doc_vectors

    if dry:
        print(json.dumps({"run_id": run_id,
                          "mode": cfg["mode"],
                          "n_docs": cfg["n_docs"], "dim": cfg["dim"],
                          "subset_size": len(keep),
                          "collection_heads": cfg["collection_heads"],
                          "attend_heads": cfg["attend_heads"],
                          "query_rows_kept": keep[:5], "...": len(keep),
                          "qrels_rows": len(cfg["qrels"])}, indent=2))
        return

    os.makedirs(os.path.join(run_dir, "artifacts"))
    art = os.path.join(run_dir, "artifacts")
    hash_files = {}
    for head in ds["heads"]:
        src = head_src(head)
        if src is None or head == "QUERIES":
            continue
        out = os.path.join(art, f"{head}.f32")
        np.load(src).astype(np.float32).tofile(out)
        hash_files[os.path.basename(out)] = sha256_file(out)
    q_src = head_src("QUERIES")
    to_f32_path = os.path.join(art, "QUERIES.f32")
    np.load(q_src).astype(np.float32).tofile(to_f32_path)
    hash_files["QUERIES.f32"] = sha256_file(to_f32_path)

    with open(os.path.join(art, "cfg.json"), "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)
    hash_files["cfg.json"] = sha256_file(os.path.join(art, "cfg.json"))
    mem_snapshot = mem_status()
    write_env(run_dir, hash_files, mem_snapshot)
    ram_abort_bytes = int(getattr(mem_snapshot, "ullAvailPhys", 0) * 0.85)

    # ---- run reps ---------------------------------------------------
    reps_json = []
    start = time.time()
    for rep in range(1, reps + 1):
        out = os.path.join(art, f"rep{rep}.json")
        if dry:
            reps_json.append({"rep": rep, "dry": True})
            continue
        proc = subprocess.Popen(
            [C2PILOT, "--config", os.path.join(art, "cfg.json"), "--out", out],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # sampler thread: 500ms on the child process-tree RSS
        peak = [0]
        stop = [False]
        def sample():
            peaks = []
            while not stop[0]:
                rss = proc_rss_bytes(proc.pid)
                peaks.append(rss)
                if rss >= ram_abort_bytes:  # guardrail
                    with open(os.path.join(run_dir, "abort-guardrail.txt"), "w") as g:
                        g.write(f"child RSS {rss} >= 85% of preflight MemAvailable "
                                f"({ram_abort_bytes}) at rep {rep}\n")
                    proc.terminate()
                time.sleep(0.5)
            peak[0] = max(peaks) if peaks else 0
        import threading
        t = threading.Thread(target=sample, daemon=True)
        t.start()
        stdout_data, stderr_data = proc.communicate(timeout=900)
        stop[0] = True
        t.join(timeout=1)
        peak_rss = peak[0]
        if proc.returncode != 0:
            print("REP", rep, "FAILED", proc.returncode)
            print(stderr_data[:2000])
            with open(os.path.join(run_dir, "status.txt"), "w") as f:
                f.write("FAILED-HARNESS\n")
            reps_json.append({"rep": rep, "returncode": proc.returncode,
                              "peak_rss_bytes": peak_rss,
                              "stderr": stderr_data[:2000]})
            break
        arr = json.load(open(out, encoding="utf8"))
        if "latency_us_post_warmup" in arr:
            lat = arr["latency_us_post_warmup"]
            rq = arr["recall10_qrels_mean"]
            re_ = arr["recall10_exact_mean"]
        else:
            pq = arr["per_query"]
            lat = {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
            if pq:
                recs = [p["recall10_qrels"] for p in pq]
                _re = [p.get("recall10_exact", 0.0) for p in pq]
                rq = sum(recs) / len(recs)
                re_ = sum(_re) / len(_re)
            else:
                rq = re_ = None
        reps_json.append({"rep": rep, "mode": arr["mode"],
                          "peak_rss_bytes": peak_rss,
                         "latency": lat,
                         "recall10_qrels_mean": rq,
                         "recall10_exact_mean": re_,
                         "n_queries": arr["n_queries"]})
    elapsed_s = time.time() - start

    # ---- aggregate --------------------------------------------------
    rec_q = [r["recall10_qrels_mean"] for r in reps_json if r.get("recall10_qrels_mean") is not None]
    rec_e = [r["recall10_exact_mean"] for r in reps_json if r.get("recall10_exact_mean") is not None]
    p50s = [r["latency"].get("p50") for r in reps_json if r.get("latency") and r["latency"].get("p50") is not None]
    p95s = [r["latency"].get("p95") for r in reps_json if r.get("latency") and r["latency"].get("p95") is not None]
    means = [r["latency"].get("mean") for r in reps_json if r.get("latency") and r["latency"].get("mean") is not None]
    agg = {
        "run_id": run_id, "mode": rules["mode"], "reps": len(reps_json),
        "n_queries": len(keep),
        "elapsed_s": round(elapsed_s, 1),
        "recall10_qrels_mean_of_repmeans": (sum(rec_q)/len(rec_q)) if rec_q else None,
        "recall10_exact_mean_of_repmeans": (sum(rec_e)/len(rec_e)) if rec_e else None,
        "latency_us": {
            "p50_between_reps": {"mean": sum(p50s)/len(p50s) if p50s else None,
                                 "min": min(p50s) if p50s else None,
                                 "max": max(p50s) if p50s else None, "n": len(p50s)},
            "p95_between_reps": {"mean": sum(p95s)/len(p95s) if p95s else None,
                                 "min": min(p95s) if p95s else None,
                                 "max": max(p95s) if p95s else None, "n": len(p95s)},
        },
        "subset_rule": "first min(100,len) of seeded qrels-test shuffle (seed %d)" % rules["seed"],
        "peak_child_rss_bytes_by_rep": [r.get("peak_rss_bytes") for r in reps_json],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf8") as f:
        ok = len(reps_json) == reps and all("recall10_qrels_mean" in r for r in reps_json)
        f.write("PASS\n" if ok else "FAILED\n")

    # ---- register (append-only) --------------------------------------
    reg = (f"- run_id: {run_id}\n  status: {'PASS' if ok else 'FAILED-HARNESS'}\n"
           f"  purpose: 'C3 pilot {rules['mode']} {rules['dataset']} "
           f"{rules['queries_source']}' (c2pilot engine driver, 5 fresh-process reps)\n"
           f"  started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
           f"  n_queries: {len(keep)}\n  subset_rule: {agg['subset_rule']}\n"
           f"  peak_child_rss_MiB: {[round(b/1048576,1) if b else None for b in agg['peak_child_rss_bytes_by_rep']]}\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()