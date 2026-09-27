"""C3 preflight: formalize on-disk NFCorpus verification as a registered run.

Recomputes sha256 of corpus/queries/qrels + the 5 embedding .npy on THIS host and
compares against C2-DATA-NFCORPUS-003 / C2-EMBED-NFCORPUS-003 manifests.
Self-contained Windows-safe (mirrors C2 RUN conventions without adb_common).
"""
import hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone

REPO = r"H:\Attention-DB\AttentionDB"
RAW = os.path.join(REPO, "research", "comparative-study", "raw")
RUN_ID = "C3DATA-NFC-VERIFY-001"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    run_dir = os.path.join(RAW, RUN_ID)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=False)
    base = os.path.join(RAW, "datasets", "beir", "nfcorpus")
    embed_dir = os.path.join(RAW, "C2-EMBED-NFCORPUS-003", "artifacts")

    checks = {}
    # corpus/queries/qrels vs data manifest
    dm = None
    import glob
    for p in glob.glob(os.path.join(RAW, "C2-DATA-NFCORPUS-*", "artifacts", "DS-NFCORPUS-manifest.yaml")):
        dm = p
    checks["_data_manifest"] = dm
    if dm:
        import yaml
        m = yaml.safe_load(open(dm))["files"]
        for name in ("corpus.jsonl",):
            p = os.path.join(base, name)
            checks[f"corpus.jsonl"] = {"expected": m[name]["sha256"], "observed": sha256_file(p),
                                        "match": m[name]["sha256"] == sha256_file(p)}
        for name in ("queries.jsonl", "qrels.dev.tsv", "qrels.test.tsv", "qrels.train.tsv", "hf-qrels.dev.tsv", "hf-qrels.test.tsv", "hf-qrels.train.tsv"):
            if name in m:
                p = os.path.join(base, name)
                checks[name] = {"expected": m[name]["sha256"], "observed": sha256_file(p),
                                "match": m[name]["sha256"] == sha256_file(p)}
    # embeddings vs embed manifest export_hashes
    em = json.load(open(os.path.join(RAW, "C2-EMBED-NFCORPUS-003", "metrics.json")))
    for view, h in em["manifest"]["export_hashes"].items():
        p = os.path.join(embed_dir, f"DS-NFCORPUS-{view.replace('.npy', '')}.npy")
        checks[f"embed:{view}"] = {"expected": h, "observed": sha256_file(p), "match": h == sha256_file(p)}

    all_ok = all(v.get("match") for k, v in checks.items() if isinstance(v, dict) and "match" in v)
    report = {"run_id": RUN_ID, "all_match": all_ok, "checks": checks,
              "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                           capture_output=True, text=True).stdout.strip()}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
        f.write(json.dumps(report, indent=2))
    with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf8") as f:
        f.write("PASS\n" if all_ok else "FAILED\n")
    print(json.dumps(report, indent=2))
    print("STATUS:", "PASS" if all_ok else "FAILED")

if __name__ == "__main__":
    main()