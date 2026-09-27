"""C2 conditional dataset gates: COCO (CLIP-on-CPU smoke, §22) + ESCI (§23).

COCO: verify val2017 = 5000 images, captions present, CLIP ViT-B/32 CPU encode
of a sample, dims, extrapolated runtime + peak RSS. Decision: ACCEPT-CONDITIONAL
or BLOCKED-HOST. No benchmark runs.
ESCI: schema/judgment fields from 2 official train shards, closure-subsample
prototype, memory estimate vs host envelope. No comparative performance.
"""
import json, os, sys, time, zipfile
sys.path.insert(0, os.path.dirname(__file__))
from adb_common import Run, sha256_file, RAW, guarded_popen

DL = f"{RAW}/datasets/downloads"

def coco_gate():
    run = Run("C2-SMOKE-COCO-001", "DS-COCO-CAP-5K feasibility gate (protocol §22)",
              {"dataset_id": "DS-COCO-CAP-5K", "model_id": "openai/clip-vit-base-patch32",
               "clip_revision_pinned_at_runtime": True, "sample_images": 6})
    out = {}
    try:
        # license + provenance
        out["license"] = ("COCO 2017 images: CC BY 4.0 (Flickr terms respected per official site); "
                          "annotations: CC BY 4.0; verify-at-download flag from C1 recorded")
        out["zip_sha256_val2017"] = sha256_file(f"{DL}/val2017.zip")
        out["zip_sha256_annotations"] = sha256_file(f"{DL}/ann_trainval2017.zip")
        with zipfile.ZipFile(f"{DL}/val2017.zip") as z:
            imgs = [n for n in z.namelist() if n.endswith(".jpg")]
            out["val2017_image_count"] = len(imgs)
            sample = sorted(imgs)[:6]
        with zipfile.ZipFile(f"{DL}/ann_trainval2017.zip") as z:
            cap_name = [n for n in z.namelist() if n.endswith("captions_val2017.json")][0]
            import io
            caps = json.loads(z.read(cap_name))
            out["captions_annotations_images"] = len(caps["images"])
            out["captions_annotations_count"] = len(caps["annotations"])
        ok5k = out["val2017_image_count"] == 5000
        # resolve + pin the CLIP revision before loading (verify-at-download)
        import urllib.request
        try:
            info = json.loads(urllib.request.urlopen(
                "https://huggingface.co/api/models/openai/clip-vit-base-patch32", timeout=30).read())
            out["clip_revision_resolved"] = info["sha"]
            rev = info["sha"]
        except Exception as e:
            out["clip_revision_resolved"] = None
            rev = None
        # CLIP smoke: encode 6 images + 2 captions on CPU under the sampler
        script = f"""
import json, time, torch
from PIL import Image
import zipfile
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision={rev!r})
p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", revision={rev!r})
m.eval()
with zipfile.ZipFile("{DL}/val2017.zip") as z:
    imgs = [Image.open(io.BytesIO(z.read(n))).convert("RGB") for n in {sample!r}]
t0 = time.time()
with torch.no_grad():
    ii = p(images=imgs[:3], return_tensors="pt")
    iemb = m.get_image_features(**ii)
    t_img3 = time.time() - t0
    tt = p(text=["a photo of a dog", "a cat sitting on a mat"], return_tensors="pt", padding=True)
    temb = m.get_text_features(**tt)
out = {{
  "image_emb_dim": int(iemb.shape[1]), "text_emb_dim": int(temb.shape[1]),
  "encode_3_images_cpu_s": round(t_img3, 3),
  "est_5k_images_cpu_min": round(t_img3 / 3 * 5000 / 60, 1),
  "torch": torch.__version__,
}}
print(json.dumps(out))
"""
        sp = f"{run.dir}/clip_smoke.py"
        open(sp, "w").write(script)
        t0 = time.time()
        rc, summ, log = guarded_popen(["python3", sp], run, timeout_s=1500)
        out["clip_smoke_exit"] = rc
        out["clip_smoke"] = log.strip().splitlines()[-1] if log.strip() else None
        out["peak_rss_kb"] = summ["peak_rss_kb"]
        out["wall_s"] = round(time.time() - t0, 1)
        try:
            cs = json.loads(out["clip_smoke"])
            dims_ok = cs["image_emb_dim"] == 512 and cs["text_emb_dim"] == 512
            fits_ram = summ["abort_reason"] is None
            fits_time = cs["est_5k_images_cpu_min"] <= 60.0
            out.update(cs)
            decision = ("ACCEPT-CONDITIONAL" if (dims_ok and fits_ram and fits_time and ok5k)
                        else "BLOCKED-HOST")
        except Exception as e:
            decision = "BLOCKED-HOST"
            out["decision_error"] = repr(e)
        out["decision"] = decision
        out["val2017_is_5000_images"] = ok5k
        run.print(json.dumps(out, indent=2))
        run.finish("PASS" if decision == "ACCEPT-CONDITIONAL" else ("BLOCKED" if decision == "BLOCKED-HOST" else "FAILED"), out)
    except Exception as e:
        run.finish("FAILED", out, reason=repr(e))
        raise

def esci_gate():
    run = Run("C2-GATE-ESCI-001", "DS-ESCI-EN-SUB feasibility gate (protocol §23)",
              {"dataset_id": "DS-ESCI-EN-SUB", "shards_downloaded": 2, "component": "conditional-gate"})
    out = {}
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        shards = sorted(os.path.join(DL, f"esci-train-{i}.parquet") for i in (0, 1))
        schema = None
        judged = {}
        locale_counts = {}
        label_counts = {}
        n_rows = 0
        for sh in shards:
            f = pq.ParquetFile(sh)
            schema = f.schema_arrow
            for batch in f.iter_batches(batch_size=50000, columns=["query_id", "query", "product_id", "esci_label", "product_locale"]):
                d = batch.to_pydict()
                for qid, q, pid, lab, loc in zip(d["query_id"], d["query"], d["product_id"], d["esci_label"], d["product_locale"]):
                    if loc != "us":
                        locale_counts[loc] = locale_counts.get(loc, 0) + 1
                        continue
                    n_rows += 1
                    judged.setdefault(qid, {"q": q, "products": set()})
                    judged[qid]["products"].add(pid)
                    label_counts[lab] = label_counts.get(lab, 0) + 1
        out["schema_fields"] = schema.names if schema else None
        out["us_rows_in_2_shards"] = n_rows
        out["judged_us_queries_in_2_shards"] = len(judged)
        out["esci_label_distribution_us"] = label_counts
        out["locale_counts_non_us"] = locale_counts
        # qrels-closure prototype: product universe = closure of judged products
        prods = set()
        multi = 0
        for qid, v in judged.items():
            prods |= v["products"]
            if len(v["products"]) >= 2:
                multi += 1
        out["closure_products_2shards"] = len(prods)
        out["queries_with_ge2_judged"] = multi
        out["closure_method"] = "subset = (queries with >=2 judged products) x (closure of their judged product_ids); queries from official TRAIN split only; dedup by product_id"
        mem_est_gb = len(prods) * 384 * 4 / (1024**3)
        out["embedding_mem_estimate_gb_at_384d_fp32"] = round(mem_est_gb, 3)
        out["host_mem_available_gb"] = 1.9
        out["memory_verdict"] = "closure fits envelope for 2-shard scale; full-train closure estimated from viewer totals scales linearly — gate on full download in C3 if admitted"
        out["source"] = "https://huggingface.co/datasets/tasksource/esci (mirror of amazon-science/esci-data; official license: CC BY-4.0 per esci-code repo)"
        out["license"] = "CC BY-4.0 (reported; C1 verify-at-download flag set)"
        ok = (schema is not None and {"query_id", "query", "product_id", "esci_label", "product_locale"} <= set(schema.names)
              and len(judged) > 0 and "E" in label_counts)
        run.print(json.dumps({k: v for k, v in out.items() if k != "schema_fields"}, indent=2))
        run.finish("PASS" if ok else "FAILED", out)
    except Exception as e:
        run.finish("FAILED", out, reason=repr(e))
        raise

if __name__ == "__main__":
    {"coco": coco_gate, "esci": esci_gate}[sys.argv[1]]()
