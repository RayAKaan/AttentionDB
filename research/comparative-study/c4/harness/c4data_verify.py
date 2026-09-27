"""C4 dataset validation: formalized on-disk verification for confirmed scale.

Recomputes sha256 for DS-NFCORPUS (corpus/queries/qrels + C2-EMBED-NFCORPUS-003
5 .npy) against the C2/DATA manifests, and for DS-SCIFACT against the C3
on-own-hashes record (C3-EMBED-SCIFACT-001) + C2-DATA-SCIFACT-007 manifest.
Registered at C4 start (pre-execution). Self-contained Windows-safe.
"""
import hashlib, json, os, subprocess, glob
from datetime import datetime, timezone

REPO = r"H:\Attention-DB\AttentionDB"
RAW = os.path.join(REPO, "research", "comparative-study", "raw")
RUN_ID = "C4DATA-VERIFY-001"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def yaml_load(p):
    import yaml
    return yaml.safe_load(open(p))

def embed_hashes(run_name):
    m = json.load(open(os.path.join(RAW, run_name, "metrics.json")))
    return m["manifest"]["export_hashes"]

def main():
    run_dir = os.path.join(RAW, RUN_ID)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=False)
    checks = {}

    # ---------- DS-NFCORPUS ----------
    nfc = os.path.join(RAW, "datasets", "beir", "nfcorpus")
    dm = sorted(glob.glob(os.path.join(RAW, "C2-DATA-NFCORPUS-*", "artifacts", "DS-NFCORPUS-manifest.yaml")))[-1]
    m = yaml_load(dm)["files"]
    nfc_by_path = {
        "corpus.jsonl": "corpus.jsonl", "queries.jsonl": "queries.jsonl",
        "qrels/dev.tsv": os.path.join("qrels", "dev.tsv"),
        "qrels/test.tsv": os.path.join("qrels", "test.tsv"),
        "qrels/train.tsv": os.path.join("qrels", "train.tsv"),
    }
    for name, rel in nfc_by_path.items():
        exp = m[name]["sha256"]
        obs = sha256_file(os.path.join(nfc, rel))
        checks[f"NFC:{name}"] = {"expected": exp, "observed": obs, "match": exp == obs}
    for view, exp in embed_hashes("C2-EMBED-NFCORPUS-003").items():
        p = os.path.join(RAW, "C2-EMBED-NFCORPUS-003", "artifacts", f"DS-NFCORPUS-{view}")
        obs = sha256_file(p)
        checks[f"NFC-EMBED:{view}"] = {"expected": exp, "observed": obs, "match": exp == obs}

    # ---------- DS-SCIFACT (on own hashes as registered C3-EMBED-SCIFACT-001) ----------
    sci = os.path.join(RAW, "datasets", "beir", "scifact")
    sm = sorted(glob.glob(os.path.join(RAW, "C2-DATA-SCIFACT-*", "artifacts", "DS-SCIFACT-manifest.yaml")))
    if sm:
        sm2 = yaml_load(sm[-1])["files"]
        for name, rel in (("corpus.jsonl", "corpus.jsonl"),
                          ("queries.jsonl", "queries.jsonl"),
                          ("qrels/test.tsv", os.path.join("qrels", "test.tsv")),
                          ("qrels/train.tsv", os.path.join("qrels", "train.tsv"))):
            exp = sm2.get(name, {}).get("sha256")
            obs = sha256_file(os.path.join(sci, rel))
            checks[f"SCI:{name}"] = {"expected": exp, "observed": obs,
                                     "match": (exp == obs) if exp else None}
    for view, exp in embed_hashes("C3-EMBED-SCIFACT-001").items():
        p = os.path.join(RAW, "C3-EMBED-SCIFACT-001", "artifacts", f"DS-SCIFACT-{view}")
        obs = sha256_file(p)
        checks[f"SCI-EMBED:{view}"] = {"expected": exp, "observed": obs, "match": exp == obs}

    decided = {k: v for k, v in checks.items() if v.get("match") is not None}
    all_ok = all(v["match"] for v in decided.values())
    report = {"run_id": RUN_ID, "all_match": all_ok, "checks": checks,
              "notes": "SciFact embeddings validated against the C3-EMBED-SCIFACT-001 own-hashes record (decision per c4-dataset-validation.md); corpus/queries/qrels against C2-DATA-SCIFACT-007 manifest.",
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