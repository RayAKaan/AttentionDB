"""C2 BEIR dataset materialization + manifests (protocol §8/§9/§10/§24).

Materializes DS-SCIFACT + DS-NFCORPUS (primary), and runs the SciDocs /
FiQA structure gates. Verifies counts, schema, qrels->doc closure, graded
labels; writes sha256 manifests. No embeddings here (embed_minilm.py).
"""
import csv, io, json, os, sys, zipfile
sys.path.insert(0, os.path.dirname(__file__))
from adb_common import Run, sha256_file, RAW

DL = f"{RAW}/datasets/downloads"
OUT = f"{RAW}/datasets/beir"

SPECS = {
    "DS-SCIFACT":   {"zip": "scifact.zip", "inner": "scifact", "corpus": 5183, "queries": 300,
                     "hf_qrels": "scifact",
                     "upstream": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
                     "origin": "BEIR (HF BeIR/scifact); original SciFact https://openreview.net/forum?id=wF5DuRiKfMq",
                     "license_recorded": "BEIR card: cc-by-sa-4.0; underlying SciFact CC BY-NC 4.0 (research use; C1 verify-at-download flag)"},
    "DS-NFCORPUS":  {"zip": "nfcorpus.zip", "inner": "nfcorpus", "corpus": 3633, "queries": 323,
                     "hf_qrels": "nfcorpus",
                     "upstream": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
                     "origin": "BEIR (HF BeIR/nfcorpus); original NFCorpus UIUC",
                     "license_recorded": "cc-by-sa-4.0 per HF card (C1 record); verify-at-download flag"},
    "DS-SCIDOCS":   {"zip": "scidocs.zip", "inner": "scidocs", "corpus": 25657, "queries": 1000,
                     "upstream": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
                     "origin": "BEIR (HF BeIR/scidocs)",
                     "license_recorded": "cc-by-sa-4.0 per HF card (C1 record)"},
    "DS-FIQA-2018": {"zip": "fiqa.zip", "inner": "fiqa", "corpus": 57638, "queries": 648,
                     "upstream": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
                     "origin": "BEIR (HF BeIR/fiqa)",
                     "license_recorded": "FiQA 2018; per HF card (C1 record)"},
}

def load_hf_qrels(ds_id, spec):
    """Official HF BeIR/<ds>-qrels (positives-only TSV; the source C1 cited).
    Falls back to the zip's own qrels when the HF set has no file for a split."""
    import shutil
    hf_dir = os.path.join(RAW, "datasets", "hf-qrels", spec.get("hf_qrels", ""))
    base = os.path.join(OUT, spec["inner"])
    qrels, provenance = {}, {}
    import glob as _glob
    for p in sorted(_glob.glob(f"{base}/qrels/*.tsv")):
        split = os.path.basename(p)[:-4]
        rows = list(csv.DictReader(open(p, encoding="utf8"), delimiter="\t"))
        qrels[split] = rows
        provenance[split] = "tu-darmstadt-zip"
    for p in sorted(_glob.glob(f"{hf_dir}/*.tsv")):
        split = os.path.basename(p)[:-4]
        rows = list(csv.DictReader(open(p, encoding="utf8"), delimiter="\t"))
        qrels[split] = rows
        provenance[split] = "hf-BeIR-qrels (official; positives-only)"
    return qrels, provenance

def load_zip(ds_id, spec):
    inner = spec["inner"]
    base = os.path.join(OUT, inner)
    if not os.path.exists(base):
        with zipfile.ZipFile(os.path.join(DL, spec["zip"])) as z:
            z.extractall(OUT)
    corpus = [json.loads(l) for l in open(f"{base}/corpus.jsonl", encoding="utf8")]
    queries = [json.loads(l) for l in open(f"{base}/queries.jsonl", encoding="utf8")]
    qrels = {}
    import glob as _glob
    for p in sorted(_glob.glob(f"{base}/qrels/*.tsv")):
        split = os.path.basename(p)[:-4]
        rows = list(csv.DictReader(open(p, encoding="utf8"), delimiter="\t"))
        qrels[split] = rows
    return base, corpus, queries, qrels

def structure_gate(ds_id, spec, run):
    base, corpus, queries, zip_qrels = load_zip(ds_id, spec)
    qrels, qrels_provenance = load_hf_qrels(ds_id, spec)
    fields = sorted({k for d in corpus[:2000] for k in d.keys()})
    graded = sorted({int(r["score"]) for rows in qrels.values() for r in rows})
    qids = {q["_id"] for q in queries}
    dids = {d["_id"] for d in corpus}
    test_qrel_qids = {r["query-id"] for r in qrels["test"]}
    dangling_q = test_qrel_qids - qids
    dangling_d = {r["corpus-id"] for rows in qrels.values() for r in rows} - dids
    counts_ok = (len(corpus) == spec["corpus"] and len(test_qrel_qids) == spec["queries"])
    return {
        "dataset_id": ds_id, "corpus_docs": len(corpus), "queries": len(queries),
        "queries_file_note": "queries.jsonl holds ALL split queries; C1 'queries' count refers to TEST qrels queries",
        "qrels_rows": {k: len(v) for k, v in qrels.items()},
        "qrels_provenance": qrels_provenance,
        "corpus_fields": fields, "graded_levels": graded,
        "graded_levels_note": ("BEIR qrels are positives-only; explicit 0 rows do not exist in the release. "
                               "SciFact ships score=1 only (binary); NFCorpus ships 1 and 2. "
                               "C1 recorded 'graded 0/1/2' for SciFact — recorded as C1-vs-data discrepancy (vocabulary vs shipped levels); "
                               "acceptance rule (defensible labels) unaffected."),
        "qrel_test_queries": len(test_qrel_qids),
        "dangling_qrel_query_ids": len(dangling_q), "dangling_qrel_doc_ids": len(dangling_d),
        "counts_match_c1_record": counts_ok,
    }

def manifest_for(ds_id, spec, run, gate):
    base = os.path.join(OUT, spec["inner"])
    files = {}
    import glob as _glob
    qrel_files = sorted(os.path.relpath(p, base) for p in _glob.glob(f"{base}/qrels/*.tsv"))
    for fn in ["corpus.jsonl", "queries.jsonl"] + qrel_files:
        p = os.path.join(base, fn)
        files[fn] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    hf_dir = os.path.join(RAW, "datasets", "hf-qrels", spec.get("hf_qrels", ""))
    for p in sorted(_glob.glob(f"{hf_dir}/*.tsv")):
        fn = "hf-qrels/" + os.path.basename(p)
        files[fn] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p),
                     "source": f"https://huggingface.co/datasets/BeIR/{spec.get('hf_qrels')}-qrels"}
    manifest = {
        "dataset_id": ds_id,
        "upstream_url": spec["upstream"],
        "origin": spec["origin"],
        "download_timestamp_utc": run.manifest["started_utc"],
        "license_metadata": spec["license_recorded"],
        "zip_sha256": sha256_file(os.path.join(DL, spec["zip"])),
        "files": files,
        "preprocessing_version": "c2-beir-v1 (raw pass-through; no dedup, no text mutation)",
        "preprocessing_script": "research/comparative-study/c2/harness/beir_materialize.py",
        "preprocessing_script_git_commit": run.manifest.get("git_commit"),
        "corpus_size": gate["corpus_docs"], "query_count": gate["queries"],
        "qrels_count": gate["qrels_rows"],
        "integrity": gate,
    }
    return manifest

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "primary"
    os.makedirs(OUT, exist_ok=True)
    if which == "primary":
        seq = {"DS-SCIFACT": "007", "DS-NFCORPUS": "003"}  # SCIFACT-001 = INVALID-STARTUP (missing qrels/dev.tsv assumption), preserved
        for ds_id in ("DS-SCIFACT", "DS-NFCORPUS"):
            spec = SPECS[ds_id]
            run = Run(f"C2-DATA-{ds_id.replace('DS-', '').replace('-', '')}-{seq[ds_id]}",
                      f"materialize + verify + manifest {ds_id} (protocol §8/§9)",
                      {"dataset_id": ds_id, "component": "dataset-materialization"})
            try:
                gate = structure_gate(ds_id, spec, run)
                mf = manifest_for(ds_id, spec, run, gate)
                import yaml
                with open(os.path.join(run.dir, "artifacts", f"{ds_id}-manifest.yaml"), "w") as f:
                    yaml.safe_dump(mf, f, sort_keys=False)
                run.artifact(f"{ds_id}-manifest.yaml", os.path.join(run.dir, "artifacts", f"{ds_id}-manifest.yaml"))
                # admissibility per charter: labels defensible (positive labels
                # exist, levels within the dataset's declared scale), closure clean,
                # counts match. BEIR qrels are positives-only by design (recorded).
                ok = (gate["counts_match_c1_record"] and gate["dangling_qrel_query_ids"] == 0
                      and gate["dangling_qrel_doc_ids"] == 0
                      and set(gate["graded_levels"]) <= {0, 1, 2} and 1 in gate["graded_levels"])
                run.print(json.dumps(gate, indent=2))
                run.finish("PASS" if ok else "FAILED", gate)
            except Exception as e:
                run.finish("FAILED", {"error": repr(e)}, reason=repr(e))
                raise
    elif which == "gates":
        # §24: SciDocs facet-structure gate + FiQA Track-B single-vector gate
        run = Run("C2-GATE-SCIDOCS-FIQA-003",
                  "conditional dataset structure gates (protocol §24)",
                  {"datasets": ["DS-SCIDOCS", "DS-FIQA-2018"], "component": "conditional-gate"})
        out = {}
        try:
            sd = structure_gate("DS-SCIDOCS", SPECS["DS-SCIDOCS"], run)
            # C1 intended use requires a reproducible FACET structure; the BEIR
            # release carries only title/text. Record what actually exists.
            facets_present = [f for f in sd["corpus_fields"] if f not in ("_id", "title", "text")]
            sd["facet_fields_beyond_title_text"] = facets_present
            sd["facet_structure_reproducible"] = bool(facets_present)
            out["DS-SCIDOCS"] = sd
            fq = structure_gate("DS-FIQA-2018", SPECS["DS-FIQA-2018"], run)
            fq["single_vector_role_fields_ok"] = set(fq["corpus_fields"]) >= {"_id", "text"}
            out["DS-FIQA-2018"] = fq
            import yaml
            with open(os.path.join(run.dir, "artifacts", "conditional-gates.yaml"), "w") as f:
                yaml.safe_dump(out, f, sort_keys=False)
            run.print(json.dumps(out, indent=2))
            status = "PASS" if (out["DS-FIQA-2018"]["single_vector_role_fields_ok"]) else "FAILED"
            run.finish(status, out)
        except Exception as e:
            run.finish("FAILED", out or {"error": repr(e)}, reason=repr(e))

if __name__ == "__main__":
    main()
