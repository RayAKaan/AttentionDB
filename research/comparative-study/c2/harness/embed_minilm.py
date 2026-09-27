"""C2 pinned embedding pipeline: all-MiniLM-L6-v2 (protocol §9/§10/§21).

HEAD-TITLE  = embed(title)
HEAD-BODY   = embed(text)
HEAD-CITE   = embed(first + final sentence of text)  [fixed, query-independent]
CANONICAL   = embed(title + " " + text)
Query views = embed(query text)
Exports float32 .npy + manifest with model file hashes. Deterministic re-encode
check on a sample. Run: python3 embed_minilm.py DS-SCIFACT [DS-NFCORPUS]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from adb_common import Run, sha256_file, RAW

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # HF api /models sha, resolved 2026-09-24
DIM = 384
BATCH = 32

def key_sentences(text: str) -> str:
    """Preregistered deterministic rule (fixed before any run): first + final sentence."""
    t = " ".join(text.split())
    parts = [s for s in t.replace("! ", ". ").replace("? ", ". ").split(". ") if s]
    if len(parts) <= 2:
        return t
    return parts[0] + ". " + parts[-1] + "."

def model_manifest_bits(model) -> dict:
    from huggingface_hub import snapshot_download
    snap = snapshot_download(MODEL_ID, revision=REVISION)
    files = {}
    for fp in ["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json",
               "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
               "special_tokens_map.json", "1_Pooling/config.json", "model.safetensors"]:
        p = os.path.join(snap, fp)
        if os.path.exists(p):
            files[fp] = sha256_file(p)
    return {"model_id": MODEL_ID, "revision": REVISION, "files_sha256": files,
            "sentence_transformers_version": __import__("sentence_transformers").__version__,
            "torch_version": __import__("torch").__version__,
            "transformers_version": __import__("transformers").__version__,
            "inference": "sentence-transformers encode, float32, batch 32, sort_by_length off, no normalization (cosine metric applied downstream)"}

def encode(model, texts, run):
    t0 = time.time()
    emb = model.encode(texts, batch_size=BATCH, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
    dt = time.time() - t0
    run.print(f"encoded {len(texts)} texts in {dt:.1f}s")
    return emb, dt

def checks(emb, run, n):
    ok_dim = emb.shape == (n, DIM)
    ok_nan = bool(np.isfinite(emb).all())
    return {"dim_ok": ok_dim, "finite_no_nan_inf": ok_nan, "dtype": str(emb.dtype), "n": int(n)}

def materialize(ds_id):
    inner = {"DS-SCIFACT": "scifact", "DS-NFCORPUS": "nfcorpus"}[ds_id]
    base = f"{RAW}/datasets/beir/{inner}"
    run = Run(f"C2-EMBED-{ds_id.replace('DS-', '').replace('-', '')}-003",
              f"pinned MiniLM embeddings for {ds_id} (protocol §9/§10/§21)",
              {"dataset_id": ds_id, "model_id": MODEL_ID, "revision": REVISION,
               "seed": None, "batch": BATCH}, use_sampler=True)
    from sentence_transformers import SentenceTransformer
    if REVISION == "PENDING":
        raise SystemExit("resolve revision first")
    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu")
    corpus = [json.loads(l) for l in open(f"{base}/corpus.jsonl", encoding="utf8")]
    queries = [json.loads(l) for l in open(f"{base}/queries.jsonl", encoding="utf8")]
    ids = [d["_id"] for d in corpus]
    title = [d.get("title", "") for d in corpus]
    text = [d.get("text", "") for d in corpus]
    views = {}
    times = {}
    views["HEAD-TITLE"], times["HEAD-TITLE"] = encode(model, title, run)
    views["HEAD-BODY"], times["HEAD-BODY"] = encode(model, text, run)
    views["HEAD-CITE"], times["HEAD-CITE"] = encode(model, [key_sentences(t) for t in text], run)
    views["CANONICAL"], times["CANONICAL"] = encode(model, [a + " " + b for a, b in zip(title, text)], run)
    views["QUERIES"], times["QUERIES"] = encode(model, [q["text"] for q in queries], run)
    # determinism: re-encode the FULL canonical view with identical batching and
    # byte-compare. (Single-text re-encodes are NOT byte-comparable: transformer
    # padding depends on batch composition — recorded as an export-pinning note.)
    re_canon = model.encode([a + " " + b for a, b in zip(title, text)], batch_size=BATCH,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    det_ok = bool(np.array_equal(re_canon, views["CANONICAL"]))
    one = model.encode([title[0] + " " + text[0]], convert_to_numpy=True).astype(np.float32)
    batch1_bitwise_differs = not np.array_equal(one[0], views["CANONICAL"][0])
    ds_manifest_path = os.path.join(RAW, f"C2-DATA-{ds_id.replace('DS-', '').replace('-', '')}-001",
                                    "artifacts", f"{ds_id}-manifest.yaml")
    src_hash = None
    try:
        import yaml
        src_hash = yaml.safe_load(open(ds_manifest_path))["files"]["corpus.jsonl"]["sha256"]
    except Exception:
        src_hash = sha256_file(f"{base}/corpus.jsonl")
    emb_manifest = {
        "dataset_id": ds_id, "source_dataset_hash_sha256": src_hash,
        **model_manifest_bits(model),
        "embedding_dimension": DIM, "metric": "cosine", "dtype": "float32",
        "records": {k: int(v.shape[0]) for k, v in views.items()},
        "encode_seconds": {k: round(v, 2) for k, v in times.items()},
        "head_view_definitions": {
            "HEAD-TITLE": "embed(title)", "HEAD-BODY": "embed(text)",
            "HEAD-CITE": "embed(first + final sentence of text; query-independent, fixed before any run)",
            "CANONICAL": "embed(title + ' ' + text); the single vector every external single-vector system receives",
            "QUERIES": "embed(query text)"},
        "determinism_full_view_reencode_byte_equal": det_ok,
        "batching_sensitivity_note": "single-text re-encode bitwise differs from batch-32 export (padding effects); exports pin batch=32 + order, regeneration uses the same script",
        "batch1_bitwise_differs": batch1_bitwise_differs,
        "export_hashes": {},
    }
    metrics = {}
    for name, emb in views.items():
        p = os.path.join(run.dir, "artifacts", f"{ds_id}-{name}.npy")
        np.save(p.replace(".npy", ""), emb) if False else np.save(p, emb)
        emb_manifest["export_hashes"][f"{name}.npy"] = sha256_file(p)
        metrics[name] = checks(emb, run, emb.shape[0])
    run.print(json.dumps(emb_manifest["export_hashes"], indent=2))
    run.finish("PASS" if (det_ok and all(m["dim_ok"] and m["finite_no_nan_inf"] for m in metrics.values()))
               else "FAILED", {**metrics, "manifest": emb_manifest})

if __name__ == "__main__":
    for ds in sys.argv[1:]:
        materialize(ds)
