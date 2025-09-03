````markdown
# FAISS Index Builder

Build **inner-product (IP)** FAISS indexes from precomputed embedding JSONs.  
Backwards-compatible with legacy `*_text_embeddings.json` and the new `*_caption/joint_embeddings.json`.

---

## Inputs (from `--in-dir`)
Any of the following (files are optional, detected if present):

- `<model>_<dataset>_image_embeddings.json`
- `<model>_<dataset>_text_embeddings.json`  _(legacy, caption-equivalent)_
- `<model>_<dataset>_caption_embeddings.json`
- `<model>_<dataset>_joint_embeddings.json`

Each file is a JSON list with:
```json
{"image_name": "...", "caption": "... (optional)", "embedding": [ ... ] }
````

> **Note:** Index uses IP search; ensure embeddings are **L2-normalized** at export time.

---

## Outputs (to `--out-dir` or `config.yaml → vector_store.faiss.index_dir`)

* `<model>_<dataset>_faiss_img.index`   + `<model>_<dataset>_img_names.json`
* `<model>_<dataset>_faiss_txt.index`   + `<model>_<dataset>_txt_meta.json`
* `<model>_<dataset>_faiss_joint.index` + `<model>_<dataset>_joint_meta.json`

Files are produced only for available inputs.

---

## Quick Start

### Run

```bash
# Uses paths from config.yaml by default
python build_faiss_from_json.py --model uniir --dataset flickr
```

Common flags:

* `--model`: model tag in filenames (e.g., `uniir`, `flava`, `minilm`)
* `--dataset`: `flickr` or `coco` (defaults to `selected_config.dataset` in `config.yaml`)
* `--in-dir`: where the embedding JSONs live (default: `<project_root>/data/embeddings`)
* `--out-dir`: override output directory (default: `config.yaml → vector_store.faiss.index_dir`)

**Examples**

```bash
# UNI-IR, COCO, default in/out dirs from config.yaml
python build_faiss_from_json.py --model uniir --dataset coco

# FLAVA, Flickr, custom input/output dirs
python build_faiss_from_json.py --model flava --dataset flickr \
  --in-dir /mnt/storage/embeddings \
  --out-dir /mnt/storage/faiss_indices
```

---

