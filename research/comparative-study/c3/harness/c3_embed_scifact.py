"""C3 re-materialization: SciFact pinned MiniLM embeddings (byte-verify vs C2).

Reproduces C2-EMBED-SCIFACT-003 with the manifest-pinned toolchain so the
export can be byte-compared against the recorded export_hashes. Self-contained
(Windows-safe; the C2 harness adb_common.py is Linux-only).

Run:  python c3_embed_scifact.py
"""
import ctypes, hashlib, json, os, platform, shutil, subprocess, sys, threading, time
from datetime import datetime, timezone

REPO = r"H:\Attention-DB\AttentionDB"
RAW = os.path.join(REPO, "research", "comparative-study", "raw")
C2ROOT = os.path.join(REPO, "research", "comparative-study", "c2")
RUN_ID = "C3-EMBED-SCIFACT-001"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIM = 384
BATCH = 32

EXPECTED_MODEL_FILES_SHA256 = {
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "config_sentence_transformers.json": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
}

# C2 export hashes this run must reproduce (source of truth).
C2_METRICS_PATH = os.path.join(RAW, "C2-EMBED-SCIFACT-003", "metrics.json")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unavailable"

class MemInfo(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

def mem_kb():
    m = MemInfo()
    m.dwLength = ctypes.sizeof(MemInfo)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys // 1024, m.ullAvailPhys // 1024

def disk_free_mb(path=REPO):
    return shutil.disk_usage(path).free // (1024 * 1024)

def assert_phase3e_isolation():
    bad = [k for k in ("PH3E_CRASH_AT", "PH3E_CRASH_HIT", "PH3E_CRASH_MODEL", "PH3E_CRASH_MARKER")
           if k in os.environ]
    if bad:
        raise RuntimeError(f"PH3E crashgate env set, hard abort per guardrails: {bad}")

def capture_env():
    total_kb, avail_kb = mem_kb()
    return {
        "timestamp_utc": now_utc(),
        "os": platform.system() + " " + platform.release(),
        "kernel": platform.version(),
        "cpu_model": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "mem_total_kb": total_kb,
        "mem_available_kb": avail_kb,
        "disk_free_mb": disk_free_mb(),
        "filesystem": shutil.disk_usage(REPO).__class__.__name__,
        "python": platform.python_version(),
        "git_commit": git_commit(),
    }

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def key_sentences(text):
    t = " ".join(text.split())
    parts = [s for s in t.replace("! ", ". ").replace("? ", ". ").split(". ") if s]
    if len(parts) <= 2:
        return t
    return parts[0] + ". " + parts[-1] + "."

def main():
    assert_phase3e_isolation()
    run_dir = os.path.join(RAW, RUN_ID)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=False)  # immutable: must not already exist
    log = open(os.path.join(run_dir, "stdout.log"), "w", encoding="utf8")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)

    env = capture_env()
    with open(os.path.join(run_dir, "environment.yaml"), "w", encoding="utf8") as f:
        f.write(json.dumps(env, indent=2).replace("{", "").replace("}", "") + "\n")
    print(json.dumps(env, indent=2))

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf8") as f:
        f.write(json.dumps({"dataset_id": "DS-SCIFACT", "model_id": MODEL_ID,
                            "revision": REVISION, "seed": None, "batch": BATCH}, indent=2) + "\n")

    c2_export_hashes = None
    c2_manifest = None
    if os.path.exists(C2_METRICS_PATH):
        c2 = json.load(open(C2_METRICS_PATH, encoding="utf8"))
        c2_export_hashes = c2["manifest"]["export_hashes"]
        c2_manifest = c2["manifest"]
        print("C2 reference export_hashes loaded from", C2_METRICS_PATH)
    else:
        print("WARNING: C2 reference metrics missing; cannot byte-verify.")

    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch, transformers, sentence_transformers as st_mod
    print(f"versions: torch={torch.__version__} st={st_mod.__version__} transformers={transformers.__version__}")

    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu")

    from huggingface_hub import snapshot_download
    snap = snapshot_download(MODEL_ID, revision=REVISION)
    file_hashes = {fp: sha256_file(os.path.join(snap, fp))
                   for fp in EXPECTED_MODEL_FILES_SHA256 if os.path.exists(os.path.join(snap, fp))}
    files_ok = all(file_hashes.get(fp) == h for fp, h in EXPECTED_MODEL_FILES_SHA256.items())
    print("model files byte-verified against manifest:", "PASS" if files_ok else "FAIL")
    print("model file hashes:", json.dumps(file_hashes))

    base = os.path.join(RAW, "datasets", "beir", "scifact")
    corpus = [json.loads(l) for l in open(os.path.join(base, "corpus.jsonl"), encoding="utf8")]
    queries = [json.loads(l) for l in open(os.path.join(base, "queries.jsonl"), encoding="utf8")]
    title = [d.get("title", "") for d in corpus]
    text = [d.get("text", "") for d in corpus]
    canonical = [a + " " + b for a, b in zip(title, text)]

    def encode(texts):
        t0 = time.time()
        emb = model.encode(texts, batch_size=BATCH, convert_to_numpy=True,
                           show_progress_bar=False).astype(np.float32)
        print(f"encoded {len(texts)} texts in {time.time()-t0:.1f}s")
        return emb

    views = {
        "HEAD-TITLE": encode(title),
        "HEAD-BODY": encode(text),
        "HEAD-CITE": encode([key_sentences(t) for t in text]),
        "CANONICAL": encode(canonical),
        "QUERIES": encode([q["text"] for q in queries]),
    }

    re_canon = model.encode(canonical, batch_size=BATCH, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)
    det_ok = bool(np.array_equal(re_canon, views["CANONICAL"]))
    one = model.encode([canonical[0]], convert_to_numpy=True).astype(np.float32)
    batch1_differs = not np.array_equal(one[0], views["CANONICAL"][0])

    src_hash = sha256_file(os.path.join(base, "corpus.jsonl"))
    metrics = {k: {"dim_ok": bool(v.shape == (len(corpus) if k != "QUERIES" else len(queries), DIM)),
                   "finite_no_nan_inf": bool(np.isfinite(v).all()),
                   "dtype": str(v.dtype), "n": int(v.shape[0])}
               for k, v in views.items()}

    export_hashes = {}
    for name, emb in views.items():
        p = os.path.join(artifacts_dir, f"DS-SCIFACT-{name}.npy")
        np.save(p, emb)
        export_hashes[f"{name}.npy"] = sha256_file(p)

    byte_equal = {}
    if c2_export_hashes:
        byte_equal = {name: (export_hashes.get(name) == c2_export_hashes.get(name))
                      for name in c2_export_hashes}
    print("my export_hashes:", json.dumps(export_hashes, indent=2))
    print("byte_equal_to_c2:", json.dumps(byte_equal))

    manifest = {
        "dataset_id": "DS-SCIFACT",
        "source_dataset_hash_sha256": src_hash,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files_sha256": file_hashes,
        "sentence_transformers_version": st_mod.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "inference": "sentence-transformers encode, float32, batch 32, sort_by_length off, no normalization (cosine metric applied downstream)",
        "embedding_dimension": DIM, "metric": "cosine", "dtype": "float32",
        "records": {k: int(v.shape[0]) for k, v in views.items()},
        "head_view_definitions": {
            "HEAD-TITLE": "embed(title)", "HEAD-BODY": "embed(text)",
            "HEAD-CITE": "embed(first + final sentence of text; query-independent, fixed before any run)",
            "CANONICAL": "embed(title + ' ' + text); the single vector every external single-vector system receives",
            "QUERIES": "embed(query text)"},
        "determinism_full_view_reencode_byte_equal": det_ok,
        "batch1_bitwise_differs": batch1_differs,
        "export_hashes": export_hashes,
        "byte_equal_to_C2_reference": byte_equal,
        "repro_run_id": RUN_ID,
        "env": env,
    }
    with open(os.path.join(run_dir, "manifest.yaml"), "w", encoding="utf8") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf8") as f:
        f.write(json.dumps({**metrics, "manifest": manifest}, indent=2))
    with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf8") as f:
        f.write("PASS\n")

    all_metrics_ok = all(m["dim_ok"] and m["finite_no_nan_inf"] for m in metrics.values())
    status = "PASS" if (files_ok and det_ok and all_metrics_ok) else "FAILED"
    byte_match = all(byte_equal.values()) if byte_equal else None
    print(f"RESULT status={status} det={det_ok} files_ok={files_ok} "
          f"byte_match_c2={byte_match if byte_match is not None else 'N/A'}")
    print("DONE")
    log.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())