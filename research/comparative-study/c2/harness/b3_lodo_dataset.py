"""Build the LODO NFCorpus-derived B3 training dataset (C1 LODO rule).

Features: canonical NFCorpus test-query embeddings (384-d, pinned MiniLM).
Targets: B0 exact per-head recall@10 on NFCorpus head views (title/body/cite)
vs qrels positives; nDCG/MRR from graded gains. SciFact is never read here.
Disclosed per C1: NFCorpus TEST queries serve as TRAINING inputs (the held-out
headline dataset for this arm is SciFact).
Output: c2/b3/datasets/gating-lodo-nfcorpus.json (GatingDataset schema).
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from adb_common import Run, RAW

def load_views(ds_id, tag):
    run_dir = glob_dir = None
    import glob
    cands = sorted(glob.glob(f"{RAW}/C2-EMBED-{tag}-00*/artifacts/{ds_id}-*.npy"))
    views = {}
    for p in cands:
        name = os.path.basename(p).replace(f"{ds_id}-", "").replace(".npy", "")
        views[name] = np.load(p)
    return views

def exact_topk(X, Q, k):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)
    S = Qn @ Xn.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    sc = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-sc, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(sc, order, axis=1)

def main():
    run = Run("C2-B3-DATA-LODO-002", "LODO NFCorpus-derived B3 training dataset construction",
              {"component": "b3-target-generation", "seed": 20260925, "top_k": 10,
               "disclosure": "NFCorpus test queries used as TRAINING inputs (C1 LODO rule); SciFact never read"})
    base = f"{RAW}/datasets/beir/nfcorpus"
    corpus = [json.loads(l) for l in open(f"{base}/corpus.jsonl", encoding="utf8")]
    queries = [json.loads(l) for l in open(f"{base}/queries.jsonl", encoding="utf8")]
    qrels = {}
    for split in ("train", "dev", "test"):
        for r in csv.DictReader(open(f"{base}/qrels/{split}.tsv"), delimiter="\t"):
            qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    views = load_views("DS-NFCORPUS", "NFCORPUS")
    assert {"HEAD-TITLE", "HEAD-BODY", "HEAD-CITE", "QUERIES"} <= set(views), views.keys()
    did2i = {d["_id"]: i for i, d in enumerate(corpus)}
    qid2i = {q["_id"]: i for i, q in enumerate(queries)}
    qids = [q for q in qid2i if q in qrels and len(qrels[q]) >= 1]
    run.print(f"judged queries: {len(qids)}")
    K = 10
    heads = ["HEAD-TITLE", "HEAD-BODY", "HEAD-CITE"]
    tops = {}
    for h in heads:
        t, s = exact_topk(views[h].astype(np.float32), views["QUERIES"].astype(np.float32), K)
        tops[h] = (t, s)
    rng = np.random.default_rng(20260925)
    perm = rng.permutation(len(qids))
    n_val = max(1, len(qids) // 5)
    val_set = {qids[i] for i in perm[:n_val]}
    out_queries = []
    for qi, qid in enumerate(qids):
        rel = qrels[qid]
        gts = [did for did, g in sorted(rel.items(), key=lambda kv: -kv[1]) if did in did2i]
        heads_out = []
        for h in heads:
            t, s = tops[h]
            cands = [corpus[i]["_id"] for i in t[qid2i[qid]]]
            scores = s[qid2i[qid]].tolist()
            hits = [1.0 if c in rel else 0.0 for c in cands]
            recall = sum(hits) / min(K, len(gts))
            dcg = sum((rel[c] / np.log2(r + 2)) for r, c in enumerate(cands) if c in rel)
            ideal = sum((g / np.log2(r + 2)) for r, g in enumerate(sorted(rel.values(), reverse=True)[:K]))
            ndcg = dcg / ideal if ideal > 0 else 0.0
            mrr = next((1.0 / (r + 1) for r, c in enumerate(cands) if c in rel), 0.0)
            heads_out.append({
                "candidates": [did2i[c] for c in cands],
                "raw_scores": scores, "norm_scores": scores, "exact_scores": scores,
                "recall_at_k": round(recall, 6), "ndcg_at_k": round(ndcg, 6), "mrr": round(mrr, 6)})
        out_queries.append({
            "query_id": qi, "query": views["QUERIES"][qid2i[qid]].tolist(),
            "query_group": None,
            "split": "Val" if qid in val_set else "Train",
            "ground_truth": [did2i[d] for d in gts[:K]],
            "heads": heads_out})
    ds = {"format": "attentiondb-gating-dataset", "version": 1, "num_heads": 3,
          "input_dim": 384, "top_k": K,
          "corpus_desc": "lodo-train-nfcorpus-384d-minilm-views-recall10",
          "seed": 20260925, "queries": out_queries}
    out = f"/home/user/AttentionDB/research/comparative-study/c2/b3/datasets/gating-lodo-nfcorpus.json"
    json.dump(ds, open(out, "w"))
    counts = {}
    for q in out_queries:
        counts[q["split"]] = counts.get(q["split"], 0) + 1
    run.print(json.dumps(counts))
    run.finish("PASS", {"judged_queries": len(qids), "splits": counts, "dim": 384, "out": out})

if __name__ == "__main__":
    main()
