# raw/datasets/downloads/ — volatile artifact cache (EMPTY after 2026-09-24 reset)

Everything here was lost to the workspace snapshot budget cap in the 2026-09-24 session
reset (~2.4 GB: scifact/nfcorpus/fiqa/scidocs zips, glove-25/50 hdf5, COCO annotations +
val2017, esci-train-{0,1}.parquet, bin/{qdrant,elasticsearch,weaviate} tarballs).

Re-procurement rules (see ../../../../c2/environment-reset-2026-09-24.md):

1. Re-download from the sources recorded in `raw/datasets/README.md` and the harness
   scripts (`c2/harness/gates_conditional.py`, `embed_minilm.py`, `ann_materialize.py`).
2. sha256 EVERY artifact on arrival and record it in `c2/dataset-manifests/` at that time
   (this supersedes the deferred "sha256 all downloads" task — it now happens per
   re-download).
3. Keep end-of-turn budget in mind: large archives should be fetched into `/var/tmp`
   (outside snapshot scope) or re-fetched per session; only what a run needs is staged
   here. The recorded byte sizes of the lost artifacts are preserved in session records
   and must be matched on re-download.
4. `raw/datasets/beir/` (processed SciFact/NFCorpus corpora) and `raw/datasets/hf-qrels/`
   SURVIVED and are the working copies for the primary datasets.
