# C2 Dataset Downloads — provenance and hashes

All files under this directory were downloaded 2026-09-24 (UTC) for study
comparative-study-001 C2. Every artifact's sha256 is recorded in the
corresponding run manifest (`raw/C2-DATA-*` / `raw/C2-SMOKE-COCO-001`).
License notes live in the dataset manifests (verify-at-download flags from C1
honored at download time).

| file | source | note |
|---|---|---|
| scifact.zip | public.ukp.informatik.tu-darmstadt.de (BEIR) | qrels positives-only (score=1) |
| nfcorpus.zip | same | qrels incl. dev; levels 1/2 |
| scidocs.zip | same | structure gate only (C2-GATE-SCIDOCS-FIQA-003) |
| fiqa.zip | same | Track-B structure gate only |
| hf-qrels/<ds>/*.tsv | huggingface.co/datasets/BeIR/<ds>-qrels | official BEIR qrels (C1's cited source) |
| glove-25-angular.hdf5 | huggingface.co/datasets/hhy3/ann-datasets (mirror) | 127,359,688 B |
| glove-50-angular.hdf5 | ann-benchmarks.com (canonical) | official pre-split HDF5 |
| ann_trainval2017.zip | images.cocodataset.org | COCO 2017 annotations (CC BY 4.0) |
| val2017.zip | images.cocodataset.org | 5,000 images (CC BY 4.0) |
| esci-train-{0,1}.parquet | huggingface.co/datasets/tasksource/esci | mirror of amazon-science/esci-data (CC BY-4.0 reported) |
| bin/qdrant.tar.gz | github qdrant v1.12.4 release | smoke binary |
| bin/elasticsearch.tar.gz | artifacts.elastic.co 8.15.2 | smoke tarball |
| bin/weaviate.tar.gz | github weaviate v1.39.6 release | smoke binary |

## Retention caveat (recorded, not hidden)

This workspace snapshot excludes multi-hundred-MB binaries from long-term
persistence; large files here are live for the C2 session. Nothing is lost
scientifically: every manifest records source URL + sha256 + byte size, so
any later phase can re-materialize byte-identical data. Small artifacts
(embedding exports, oracle outputs, manifests, logs) persist with the repo.

## Key data findings (full discussion in c2-validation-report.md)

1. BEIR qrels are positives-only: SciFact ships score=1 only (1,258 rows
   total — matching C1's recorded row counts); NFCorpus ships 1 and 2.
   C1's "graded 0/1/2" described the label vocabulary; explicit 0 rows do
   not exist in the release. Acceptance rule (defensible labels) unaffected.
2. TU-Darmstadt zips and HF BeIR qrels agree on positives; HF files used as
   the qrels source of record (dual provenance hashed in manifests).
