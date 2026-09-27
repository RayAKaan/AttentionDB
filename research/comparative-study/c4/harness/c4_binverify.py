"""C4-BINVERIFY-001: behavioral equivalence of the new deterministic binary
(A8E0C0CB..., /Brepro + B4 arm) vs the C4.2 frozen binary (90249747...).

Motivation: the C4.2 tuning runs recorded c2pilot_sha256 = 90249747... in
their environment.yaml. Building a new binary (MSVC link.exe embeds a fresh
PE TimeDateStamp + PDB GUID per link, so identical source produced different
sha256) overwrote the original file. All code sections were verified
byte-identical across builds; /Brepro now makes clean rebuilds deterministic.
This script empirically re-runs the C4.2-selected configs on the SAME frozen
VAL rows with the NEW binary and compares recall to the recorded C4.2
metrics.json values (tolerance +-0.01, the C4 matching rule).

Artifacts: raw/C4-BINVERIFY-001/ + ledger entry appended to RUN-INDEX.
Does not modify any C4.2 evidence.
"""

import json, os, subprocess, sys

CS = r"H:\Attention-DB\AttentionDB\research\comparative-study"
RAW = os.path.join(CS, "raw")
OUT = os.path.join(RAW, "C4-BINVERIFY-001")
SEED = 20260925
K = 10

from c4_tune_run import (C2PILOT, sha256_file, load_plan_row, resolve_dataset,
                         val_rows, parse_qrels, seeded_shuffle)

BIN_NEW = sha256_file(C2PILOT)

# (run_id, level_key, expected_recall10_qrels, tolerance)
C4_2_SELECTED = [
    # mode, dataset, VAL run_id, selected level, recorded recall
    ("C4-W19-NFC-B1-EF-001", "DS-NFCORPUS", "B1", "EF64", 0.1351),
    ("C4-W19-NFC-B2-EF-001", "DS-NFCORPUS", "B2", "EF16-C50", 0.1393),
    ("C4-W19-NFC-B7-EF-001", "DS-NFCORPUS", "B7", "EF64", 0.1372),
    ("C4-W19-SCI-B1-EF-001", "DS-SCIFACT", "B1", "EF32", 0.7775),
]

def build_cfg(run_id, ds, mode, level_key, ef, cand):
    row = load_plan_row(run_id)
    dsinfo = resolve_dataset(ds)
    query_ids, doc_ids, qrels = val_rows(dsinfo)
    keep = sorted(qrels.keys())
    n_docs = len(doc_ids)
    dim = dsinfo["dim"]

    art = os.path.join(OUT, "artifacts")
    os.makedirs(art, exist_ok=True)

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
        out = os.path.join(art, f"{ds}-{head}.f32")
        if not os.path.exists(out):
            with open(src, "rb") as f:
                import numpy as np
                np.load(src).astype(np.float32).tofile(out)
        doc_vectors[head] = out
    canonical_out = os.path.join(art, f"{ds}-CANONICAL.f32")
    if not os.path.exists(canonical_out):
        import numpy as np
        np.load(head_src("CANONICAL")).astype(np.float32).tofile(canonical_out)
    qvec_out = os.path.join(art, f"{ds}-QUERIES.f32")
    if not os.path.exists(qvec_out):
        import numpy as np
        np.load(head_src("QUERIES")).astype(np.float32).tofile(qvec_out)
    nq_all = len(query_ids)

    mode_full = mode
    if mode_full == "B1":
        coll_heads = ["CANONICAL"]; attend = ["CANONICAL"]
        grid_doc_vectors = {"CANONICAL": canonical_out}
    else:
        coll_heads = list(doc_vectors.keys()); attend = list(doc_vectors.keys())
        grid_doc_vectors = dict(doc_vectors)
        grid_doc_vectors["CANONICAL"] = canonical_out

    return {
        "run_id": run_id, "mode": mode_full, "k": K, "seed": SEED,
        "warmup": 20, "n_queries": nq_all, "n_docs": n_docs,
        "dim": dim, "subsample": keep,
        "collection_heads": coll_heads, "attend_heads": attend,
        "qrels": {str(r): {str(d): g for d, g in v.items()} for r, v in qrels.items()},
        "query_vectors": qvec_out, "doc_vectors": grid_doc_vectors,
        "candidate_budget": cand, "min_candidates_per_head": 20,
        "ef_search": ef, "ef_construction": 400, "m": 16,
        "modelcard": (os.path.join(CS, "c2", "b3", "modelcards",
                      "b3-lodo-nf-v2-s20260925-h32-lr0.01.json")
                      if mode_full == "B3" else ""),
        "configuration_id": f"{run_id}__{level_key}",
    }

def run_cell(run_id, ds, mode, level_key, ef, cand, reps=2):
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    cfg_path = os.path.join(OUT, f"{run_id}__{level_key}.cfg.json")
    out_path = os.path.join(OUT, f"{run_id}__{level_key}.rep1.json")
    cfg = build_cfg(run_id, ds, mode, level_key, ef, cand)
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)
    # B0 reference pass
    b0 = dict(cfg)
    b0["mode"] = "B0"
    b0["collection_heads"] = []
    b0["attend_heads"] = []
    b0["doc_vectors"] = {"CANONICAL": cfg["doc_vectors"].get("CANONICAL")}
    b0c = os.path.join(OUT, f"{run_id}__B0.cfg.json")
    b0o = os.path.join(OUT, f"{run_id}__B0.json")
    with open(b0c, "w", encoding="utf8") as f:
        json.dump(b0, f, indent=2)
    subprocess.run([C2PILOT, "--config", b0c, "--out", b0o], check=True,
                   capture_output=True, text=True, timeout=1800)
    b0arr = json.load(open(b0o, encoding="utf8"))
    keep = sorted(set(b0arr["per_query"][0].keys()) if False else
                  [p["query_row"] for p in b0arr["per_query"]])
    b0_rec = sum(p["recall10_qrels"] for p in b0arr["per_query"]) / len(b0arr["per_query"])
    for rep in range(1, reps + 1):
        o = os.path.join(OUT, f"{run_id}__{level_key}.rep{rep}.json")
        subprocess.run([C2PILOT, "--config", cfg_path, "--out", o], check=True,
                       capture_output=True, text=True, timeout=1800)
    arr = json.load(open(os.path.join(OUT, f"{run_id}__{level_key}.rep1.json"), encoding="utf8"))
    pq = arr["per_query"]
    rec_mean = sum(p["recall10_qrels"] for p in pq) / len(pq)
    lat = sorted(p.get("latency_us", 0) for p in pq)
    p50 = lat[len(lat) // 2] if lat else None
    return {"recall10_qrels": rec_mean, "b0_recall10_qrels": b0_rec,
            "p50_us": p50, "n_queries": len(pq)}

def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {"name": "C4-BINVERIFY-001",
               "purpose": "behavioral equivalence new deterministic binary vs C4.2",
               "c2pilot_sha256_new": BIN_NEW,
               "c2pilot_sha256_c4_2": "9024974731d8d5c0b182348eae2601037768819ebae5eff459c82c6931f543ce",
               "tolerance_rule": "abs(new - recorded) <= 0.01 + 1e-9 (C4 matching rule)",
               "cells": []}
    all_ok = True
    for run_id, ds, mode, level_key, rec in C4_2_SELECTED:
        ef = int(level_key.replace("EF", "").split("-")[0]) if level_key != "EFdef" else 64
        cand = int(level_key.split("-C")[1]) if "-C" in level_key else 500
        r = run_cell(run_id, ds, mode, level_key, ef, cand)
        ok = abs(r["recall10_qrels"] - rec) <= 0.01 + 1e-9
        all_ok = all_ok and ok
        summary["cells"].append({
            "run_id": run_id, "mode": mode, "dataset": ds, "level": level_key,
            "recorded_c4_2_recall10_qrels": rec, "new_recall10_qrels": r["recall10_qrels"],
            "new_b0_recall10_qrels": r["b0_recall10_qrels"],
            "p50_us": r["p50_us"], "within_tolerance": ok,
            "tolerance": 0.01})
        print(f"{run_id} {level_key}: recorded={rec:.4f} new={r['recall10_qrels']:.4f} "
              f"B0={r['b0_recall10_qrels']:.4f} ok={ok}")
    summary["verdict"] = "PASS" if all_ok else "MISMATCH"
    with open(os.path.join(OUT, "C4-BINVERIFY-001.json"), "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "status.txt"), "w", encoding="utf8") as f:
        f.write(summary["verdict"] + "\nnew binary sha256: " + BIN_NEW)
    reg = (f"- run_id: C4-BINVERIFY-001\n  status: {summary['verdict']}\n"
           f"  purpose: 'behavioral equivalence new deterministic binary '\n"
           f"  c2pilot_sha256: {BIN_NEW}\n"
           f"  result: " + ", ".join(
               f"{c['run_id']} {c['level']} rec={c['new_recall10_qrels']:.4f}"
               f"(recorded {c['recorded_c4_2_recall10_qrels']:.4f})"
               for c in summary["cells"]) + "\n")
    with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a", encoding="utf8") as f:
        f.write(reg)

if __name__ == "__main__":
    main()