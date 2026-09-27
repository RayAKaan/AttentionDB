"""C3 synthetic diagnostic cells (DIAGNOSTIC-labeled, harness validation).

W-08 (conflicting-signals, MODE-B1) and W-10 (duplicate-heavy, MODE-B0) run
the in-repo deterministic generator `phase2b_bench::corpora::multiview` via
the audited `c2probe corpus` export, convert doc/query vectors to raw f32,
build qrels from the generator's ground-truth (exact cosine ranking per query
view), and drive the same `c2pilot` engine driver under the frozen plan params
(reps=1, warmup=0, generator-validation query set).

Verification baked in: the exported generator ground truth must equal the
brute-force exact top-10 computed from the query's OWN view vectors (the
generator's own GT method, Phase 2 exact-cosine) - a soundness cross-check.

Usage:
  python c3_synth_pilot.py --run-id C3-W08-SYN-B1-001 [--nq 200]
"""
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c3_pilot_run import (C2PILOT, CS, PLAN, RAW, load_plan_rule, mem_status,
                          proc_rss_bytes, seeded_subset, sha256_file, write_env)

C2PROBE = os.path.join(CS, "c2", "probe", "target", "release", "c2probe.exe")


def cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def brute_topk(vecs, q, k):
    scored = [(i, cosine(v, q)) for i, v in enumerate(vecs)]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [i for i, _ in scored[:k]]


def main():
    args = sys.argv[1:]
    run_id = nq = None
    i = 0
    while i < len(args):
        if args[i] == "--run-id":
            run_id = args[i + 1]; i += 2
        elif args[i] == "--nq":
            nq = int(args[i + 1]); i += 2
        else:
            raise SystemExit(f"unknown arg {args[i]}")
    if not run_id:
        raise SystemExit("--run-id required")
    nq = nq or 200
    rules = load_plan_rule(run_id)
    mode = rules["mode"].replace("MODE-", "")
    seed = rules["seed"]

    run_dir = os.path.join(RAW, run_id)
    if os.path.exists(run_dir):
        raise SystemExit(f"run dir {run_dir} already exists (immutable)")
    os.makedirs(os.path.join(run_dir, "artifacts"))
    art = os.path.join(run_dir, "artifacts")

    # ---- 1. deterministic in-repo generator export -----------------------
    export = os.path.join(art, "corpora-multiview.json")
    proc = subprocess.run([C2PROBE, "corpus", str(nq), str(seed), export],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"c2probe corpus failed: {proc.stderr[:2000]}")

    data = json.load(open(export, encoding="utf8"))
    heads = data["head_names"]          # ["semantic","lexical","metadata"]
    dim = data["dim"]
    docs, queries = data["docs"], data["queries"]
    n_docs = len(docs)
    head_idx = {h: i for i, h in enumerate(heads)}

    # ---- 2. raw f32 materialization (engine rows = doc order) -----------
    doc_f32 = {}
    for h in heads:
        arr = np.stack([np.array(d["head_vecs"][head_idx[h]], dtype=np.float32)
                        for d in docs])
        p = os.path.join(art, f"{h}.f32")
        arr.tofile(p)
        doc_f32[h] = p
    q_arr = np.stack([np.array(q["vectors"][0], dtype=np.float32) for q in queries])
    q_f32 = os.path.join(art, "q_view0.f32")
    q_arr.tofile(q_f32)

    # ---- 3. soundness cross-check: GT-view oracle vs generator GT --------
    checked = 0
    for qi in range(nq):
        qv = np.array(queries[qi]["vectors"][queries[qi]["group"]])
        gt = queries[qi]["ground_truth"]
        view = heads[queries[qi]["group"]]
        vvecs = np.stack([np.array(d["head_vecs"][head_idx[view]], dtype=np.float32)
                          for d in docs])
        oracle = brute_topk(vvecs, qv, 10)
        if oracle != [int(x) for x in gt]:
            raise SystemExit(
                f"GEN-GT MISMATCH at qi={qi}: oracle {oracle[:5]} != GT {gt[:5]}")
        checked += 1

    # ---- 4. qrels from generator GT (doc rows, grade 1 over GT top-10) ---
    qrels = {}
    for qi in range(nq):
        qrels[qi] = {int(d): 1 for d in queries[qi]["ground_truth"]}

    # ---- 5. c2pilot config ----------------------------------------------
    if mode == "B0":
        # duplicate-heavy: replicate ~half the docs deterministically
        reps = [i % (n_docs // 2) for i in range(n_docs)]  # static dup: pattern
        arr = np.stack([np.array(docs[r]["head_vecs"][0], dtype=np.float32)
                        for r in reps])
        p = os.path.join(art, "dup.f32")
        arr.tofile(p)
        doc_vectors = {"CANONICAL": p}
        cfg = {"run_id": run_id, "mode": "B0", "k": 10, "seed": seed,
               "warmup": 0, "n_queries": nq, "n_docs": n_docs, "dim": dim,
               "subsample": list(range(nq)), "collection_heads": [],
               "attend_heads": [],
               "doc_vectors": doc_vectors, "query_vectors": q_f32,
               "qrels": {str(k): {str(r): g for r, g in v.items()}
                         for k, v in qrels.items()}}
    else:
        # B1: single-head approx on the semantic view (head 0 = CANONICAL)
        doc_vectors = {"CANONICAL": doc_f32[heads[0]]}
        cfg = {"run_id": run_id, "mode": "B1", "k": 10, "seed": seed,
               "warmup": 0, "n_queries": nq, "n_docs": n_docs, "dim": dim,
               "subsample": list(range(nq)),
               "collection_heads": ["CANONICAL"],
               "attend_heads": ["CANONICAL"],
               "doc_vectors": doc_vectors, "query_vectors": q_f32,
               "qrels": {str(k): {str(r): g for r, g in v.items()}
                         for k, v in qrels.items()}}
    cfg_path = os.path.join(art, "cfg.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)

    hash_files = {os.path.basename(p): sha256_file(p)
                  for p in [export, cfg_path] + list(doc_f32.values()) + [q_f32]}
    write_env(run_dir, hash_files, mem_status())

    # ---- 6. run (reps=1, warmup=0 per plan) -----------------------------
    out = os.path.join(art, "rep1.json")
    p = subprocess.Popen([C2PILOT, "--config", cfg_path, "--out", out],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = p.communicate(timeout=900)
    peak = 0
    if p.returncode != 0:
        print(err[:2000])
        raise SystemExit(f"c2pilot failed rc={p.returncode}")
    arr = json.load(open(out, encoding="utf8"))
    rq = arr.get("recall10_qrels_mean") or np.mean([r["recall10_qrels"] for r in arr["per_query"]])
    re_ = arr.get("recall10_exact_mean") if arr.get("recall10_exact_mean") is not None \
        else np.mean([r.get("recall10_exact", 0.0) for r in arr["per_query"]])

    agg = {"run_id": run_id, "mode": mode, "reps": 1, "n_queries": nq,
           "generator": "phase2b_bench::corpora::multiview",
           "gen_gt_checks_passed": checked,
           "recall10_qrels": rq, "recall10_exact": re_,
           "gen_gt_match_oracle": "verified-all"}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
        json.dump(agg, f, indent=2)
    ok = arr.get("ok", False)
    with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf8") as f:
        f.write(("PASS\n" if ok else "FAILED\n"))
    reg = (f"- run_id: {run_id}\n  status: {'PASS' if ok else 'FAILED'}\n"
           f"  purpose: 'C3 {mode} DIAGNOSTIC synthetic (harness validation) "
           f"on {data['name']} n={nq} seed={seed}'\n"
           f"  started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
           f"  gen_gt_checks: {checked}/{checked} pass\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()