"""C4 validation-tuning orchestrator (Windows-safe, C1 protocol).

Runs ONE frozen c4-run-plan.csv W-19 tuning row over its preregistered grid
(ef_search x candidate_budget levels parsed from the row's plan columns) on
the VALIDATION split only, per c1/recall-and-budget-matching.md:

  - matching metric (semantic datasets, fixed pre-outcome): qrels-based
    Relevance-Recall@10 = fraction of graded (>=1) judged docs retrieved
    among the top-10 returned.
  - target: B0 exact-oracle Recall@10 on the SAME VAL queries (derived
    in-process from c2pilot B0 pass over the same row set -> exact_top10).
  - tolerance: |config - target| <= 0.01 INCLUSIVE (matches c1 wording
    "within +-0.01"). Comparison uses a 1e-9 epsilon so exact-boundary float
    artifacts (e.g. diff exactly 0.010) never falsely report
    TARGET-UNREACHABLE. FASTEST p50 post-warmup among the qualifying
    configs is selected; TARGET-UNREACHABLE only when NO config qualifies.
  - every config-level run uses a fresh c2pilot process; rep_count from the
    frozen row (1 for tuning rows).
  - 500 ms memory sampler on the child process-tree RSS; abort at >=85% of
    preflight MemAvailable (guardrail evidence preserved).

Split freeze (c4-dataset-validation.md Section 4):
  - NFCorpus VAL = the 324 official qrel-dev qids (disjoint from test).
  - SciFact VAL = seeded train-sample: first 200 of the seeded shuffle
    (seed 20260925) of sorted qrel-train qids (disjoint from test; 0 overlap
    -- recorded in the run). Preregistered here on the frozen plan branch.

Usage:
  python c4_tune_run.py --run-id C4-W19-SCI-B1-EF-001 [--dry-run] [--art-sub SUFFIX]

  --art-sub SUFFIX writes evidence under {run_id}_{SUFFIX} (e.g. a
  CONFIG-CORRECTION-N dir) while keeping the planned run_id in RUN-INDEX,
  mirroring the C3 _CONFIG-CORRECTION-* convention. The original run dir is
  left untouched and a CORRECTION-NOTE.txt is expected in the new dir.
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


def val_rows(dsinfo):
    """Return (query_ids, doc_ids, qrels_by_qrow) restricted to the frozen VAL split."""
    query_ids = _jsonl_ids(os.path.join(dsinfo["data"], "queries.jsonl"))
    doc_ids = _jsonl_ids(os.path.join(dsinfo["data"], "corpus.jsonl"))
    qrels_path_src = os.path.join(dsinfo["data"], "qrels")
    ds = dsinfo["prefix"]
    if ds == "DS-NFCORPUS":
        qrels = parse_qrels(os.path.join(qrels_path_src, "dev.tsv"), query_ids, doc_ids)
    else:  # SciFact: seeded train-sample
        qrels = parse_qrels(os.path.join(qrels_path_src, "train.tsv"), query_ids, doc_ids)
        rows = sorted(qrels.keys())
        order = seeded_shuffle(len(rows), SEED)
        take = min(200, len(rows))
        keep = [rows[i] for i in order[:take]]
        qrels = {r: qrels[r] for r in keep}
    return query_ids, doc_ids, qrels


def parse_grid(row):
    """Parse ef / candidate_budget grid levels from the frozen row columns.

    Level braces live in the row's `representation` / `query_parameters` /
    `index_parameters` / `candidate_budget` text (e.g. `{16|32|64|128}`).
    Grids are recognized by their frozen, preregistered level signatures:
      - EF   = the member-set {16, 32, 64, 128} (BUDGET-EF)
      - CAND = the member-set {50, 100, 200, 500} (BUDGET-CAND)
    A row may define either or both; absence -> that axis is [None].
    """
    def groups(txt):
        out = []
        for m in re.finditer(r"\{(.*?)\}", txt or ""):
            vals = [int(t) for t in re.split(r"[\|,]", m.group(1)) if t.strip()]
            if vals:
                out.append(vals)
        return out

    ef_sig = set([16, 32, 64, 128])
    cand_sig = set([50, 100, 200, 500])
    ef_levels, cand_levels = None, None
    for g in groups(row.get("representation", "")) + \
             groups(row.get("query_parameters", "")) + \
             groups(row.get("index_parameters", "")) + \
             groups(row.get("candidate_budget", "")):
        if set(g) == ef_sig and ef_levels is None:
            ef_levels = g
        elif set(g) == cand_sig and cand_levels is None:
            cand_levels = g
    # Rows that reference the frozen BUDGET-EF axis without repeating its
    # levels (e.g. B7 "default Full params; ef levels") default to the frozen
    # axis: EF {16|32|64|128} (c1/recall-and-budget-matching.md Section B).
    ef_hint = ("ef_search" in row.get("representation", "") or
               "ef levels" in row.get("query_parameters", "") or
               "BUDGET-EF" in row.get("representation", "") or
               "BUDGET-EF" in row.get("candidate_budget", ""))
    if ef_levels is None and (ef_hint or row.get("representation", "").startswith("ef_search")):
        ef_levels = list(sorted(ef_sig))
    return ef_levels, cand_levels


def mode_heads(row):
    ds = row["dataset"]
    if ds == "DS-NFCORPUS":
        return ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE", "CANONICAL"]
    return ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE", "CANONICAL"]


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


def run_one(art, cfg, level_key, ram_abort_bytes, run_dir, rep):
    """Run one config level (fresh process) -> (arr, peak_rss)."""
    out = os.path.join(art, f"{level_key}-rep{rep}.json")
    cfg_path = os.path.join(art, f"{level_key}-cfg.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)
    proc = subprocess.Popen([C2PILOT, "--config", cfg_path, "--out", out],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = [0]
    stop = [False]

    def sample():
        peaks = []
        while not stop[0]:
            rss = proc_rss_bytes(proc.pid)
            peaks.append(rss)
            if rss >= ram_abort_bytes:
                with open(os.path.join(run_dir, "abort-guardrail.txt"), "a") as g:
                    g.write(f"child RSS {rss} >= 85% of preflight MemAvailable "
                            f"({ram_abort_bytes}) at {level_key} rep {rep}\n")
                proc.terminate()
            time.sleep(0.5)
        peak[0] = max(peaks) if peaks else 0

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    stdout_data, stderr_data = proc.communicate(timeout=1800)
    stop[0] = True
    t.join(timeout=1)
    if proc.returncode != 0:
        raise RuntimeError(f"{level_key} rep{rep} FAILED rc={proc.returncode}\n"
                           f"{stderr_data[:2000]}")
    return json.load(open(out, encoding="utf8")), peak[0]


def derive_rows_stats(arr, query_rows):
    """mean recall10_qrels/exact + p50 latency over the run's VAL rows."""
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
    }


def write_status(run_dir, status, extra=""):
    with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf8") as f:
        f.write(status + ("\n" + extra if extra else ""))


def main():
    args = sys.argv[1:]
    run_id = None
    dry = False
    art_sub = None
    i = 0
    while i < len(args):
        if args[i] == "--run-id":
            run_id = args[i + 1]; i += 2
        elif args[i] == "--dry-run":
            dry = True; i += 1
        elif args[i] == "--art-sub":
            art_sub = args[i + 1]; i += 2
        else:
            raise SystemExit(f"unknown arg {args[i]}")
    if not run_id:
        raise SystemExit("--run-id required")
    row = load_plan_row(run_id)
    if row["split"] != "VAL":
        raise SystemExit(f"{run_id} is not a VAL tuning row (split={row['split']}); "
                         f"use c4_test_run.py for TEST cells")
    dsinfo = resolve_dataset(row["dataset"])
    query_ids, doc_ids, qrels = val_rows(dsinfo)
    keep = sorted(qrels.keys())
    n_docs = len(doc_ids)

    ef_levels, cand_levels = parse_grid(row)
    if ef_levels is None:
        ef_levels = [None]
    if cand_levels is None:
        cand_levels = [None]

    if dry:
        print(json.dumps({
            "run_id": run_id, "mode": row["mode"].replace("MODE-", ""),
            "dataset": row["dataset"], "split": "VAL",
            "n_val_queries": len(keep),
            "ef_levels": ef_levels, "cand_levels": cand_levels,
            "n_grid": len(ef_levels) * len(cand_levels),
            "rep_count": int(row["rep_count"] or 1)}, indent=2))
        return

    run_dir = os.path.join(RAW, run_id if not art_sub else f"{run_id}_{art_sub}")
    if os.path.exists(run_dir):
        raise SystemExit(f"run dir {run_dir} exists (immutable); new run_id required")
    os.makedirs(os.path.join(run_dir, "artifacts"))
    art = os.path.join(run_dir, "artifacts")

    # ---- pre-execution registration (append-only) -----------------------
    reg_pre = (f"- run_id: {run_id}\n  status: RUN\n"
               f"  purpose: 'C4 validation tuning {row['mode']} {row['dataset']} "
               f"VAL grid {ef_levels} x {cand_levels}'\n"
               f"  started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
               f"  n_val_queries: {len(keep)}\n"
               f"  preregistration: 'frozen c4-run-plan.csv {run_id}; "
               f"pre-execution registration before measurement'\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg_pre)

    # ---- resolve head sources + materialize .f32 (hashes recorded) ------
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
    nq_all = len(query_ids)

    mem_snapshot = mem_status()
    ram_abort_bytes = int(getattr(mem_snapshot, "ullAvailPhys", 0) * 0.85)

    # ---- B0 target pass on VAL (exact oracle reference) -----------------
    b0_cfg = {
        "run_id": run_id, "mode": "B0", "k": K, "seed": SEED, "warmup": 20,
        "n_queries": nq_all, "n_docs": n_docs, "dim": dsinfo["dim"],
        "subsample": keep,
        "collection_heads": [],
        "attend_heads": [],
        "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
        "query_vectors": qvec_out,
        "doc_vectors": {"CANONICAL": os.path.join(art, "CANONICAL.f32")},
        "candidate_budget": 500, "min_candidates_per_head": 20,
        "ef_search": 64, "configuration_id": f"{run_id}__B0-REF",
    }
    np.load(head_src("CANONICAL")).astype(np.float32).tofile(b0_cfg["doc_vectors"]["CANONICAL"])
    b0_arr, _b0_rss = run_one(art, b0_cfg, "B0", ram_abort_bytes, run_dir, 1)
    target = derive_rows_stats(b0_arr, keep)
    target_recall = target["recall10_qrels_mean"]

    # ---- grid passes ----------------------------------------------------
    mode = row["mode"].replace("MODE-", "").replace("B", "")
    mode_full = row["mode"].replace("MODE-", "")
    if mode_full == "B1":
        coll_heads = ["CANONICAL"]; attend = ["CANONICAL"]
        grid_doc_vectors = {"CANONICAL": canonical_out}
    else:
        coll_heads = list(doc_vectors.keys()); attend = list(doc_vectors.keys())
        grid_doc_vectors = dict(doc_vectors)
        grid_doc_vectors["CANONICAL"] = canonical_out
    modelcard = (os.path.join(CS, "c2", "b3", "modelcards",
                              "b3-lodo-nf-v2-s20260925-h32-lr0.01.json")
                 if mode_full == "B3" else "")
    candidate_default = int(row.get("candidate_budget", "500").split(":")[1]) \
        if row.get("candidate_budget", "").startswith("BUDGET-CAND:") \
        and "{" not in row["candidate_budget"] else 500
    min_cand = 20

    results = []
    for ef in ef_levels:
        for cand in cand_levels:
            level_key = (f"EF{ef}" if ef is not None else "EFdef") + \
                        (f"-C{cand}" if cand is not None else "")
            cfg = {
                "run_id": run_id, "mode": mode_full, "k": K, "seed": SEED,
                "warmup": 20, "n_queries": nq_all, "n_docs": n_docs,
                "dim": dsinfo["dim"], "subsample": keep,
                "collection_heads": coll_heads, "attend_heads": attend,
                "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
                "query_vectors": qvec_out, "doc_vectors": grid_doc_vectors,
                "candidate_budget": cand if cand is not None else candidate_default,
                "min_candidates_per_head": min_cand,
                "ef_search": ef if ef is not None else 64,
                "ef_construction": 400, "m": 16,
                "modelcard": modelcard,
                "configuration_id": f"{run_id}__{level_key}",
            }
            reps = int(row["rep_count"] or 1)
            arrs, peak_rss = [], []
            for rep in range(1, reps + 1):
                arr, pr = run_one(art, cfg, level_key, ram_abort_bytes, run_dir, rep)
                arrs.append(arr); peak_rss.append(pr)
            st = derive_rows_stats(arrs[0], keep)
            st["level"] = level_key
            st["ef_search"] = ef if ef is not None else 64
            st["candidate_budget"] = cand if cand is not None else candidate_default
            st["peak_child_rss_MiB"] = [round(x / 1048576, 1) if x else None for x in peak_rss]
            st["within_tolerance"] = (st["recall10_qrels_mean"] is not None and
                                      abs(st["recall10_qrels_mean"] - target_recall)
                                      <= 0.01 + 1e-9)
            results.append(st)

    # ---- selection: fastest p50 within tolerance ------------------------
    matched = [r for r in results if r["within_tolerance"]]
    selection = None
    if matched:
        selection = min(matched, key=lambda r: r["p50_us"])
    metrics = {
        "run_id": run_id, "mode": mode_full, "dataset": row["dataset"],
        "split": "VAL", "phase": "C4.2 validation tuning",
        "matching_metric": "qrels Relevance-Recall@10 (>=1 judged docs, top-10)",
        "target_recall10_qrels_B0_VAL": target_recall,
        "tolerance": 0.01,
        "rep_count": int(row["rep_count"] or 1),
        "grid": {"ef_search": ef_levels, "candidate_budget": cand_levels},
        "levels": results,
        "selection": selection,
        "selection_rule": "fastest p50 post-warmup within +-0.01 of B0 VAL target "
                          "(TARGET-UNREACHABLE never interpolated)",
        "val_split": {"dataset": row["dataset"],
                      "rule": ("NFCorpus official qrel-dev (324 qids)"
                               if dsinfo["prefix"] == "DS-NFCORPUS"
                               else "SciFact seeded train-sample (first 200 of seed 20260925 "
                                    "shuffle of sorted qrel-train qids)"),
                      "n_queries": len(keep)},
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    hash_files = {}
    for h in sorted(os.listdir(art)):
        p = os.path.join(art, h)
        if os.path.isfile(p):
            hash_files[h] = sha256_file(p)
    write_env(run_dir, hash_files, mem_snapshot)

    status = "PASS" if (selection is not None and any(r["within_tolerance"] for r in results)) \
        else "PASS-TARGET-UNREACHABLE" if results else "FAILED"
    write_status(run_dir, status,
                 f"target_recall={target_recall:.4f} "
                 f"selected={selection['level'] if selection else None} "
                 f"best_recall={max(r['recall10_qrels_mean'] for r in results) if results else None}")

    reg = (f"- run_id: {run_id}\n"
           f"  status: {status}\n"
           f"  purpose: 'C4 validation tuning completion {row['mode']} "
           f"{row['dataset']} VAL grid {ef_levels} x {cand_levels}'\n"
           f"  completed: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
           f"  result: target={target_recall:.4f} selected="
           f"{selection['level'] if selection else 'TARGET-UNREACHABLE'}\n"
           f"  recall_by_level: "
           f"{ {r['level']: round(r['recall10_qrels_mean'],4) for r in results} }\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()