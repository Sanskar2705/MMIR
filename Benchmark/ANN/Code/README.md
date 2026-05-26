# ANN `Code/` — scripts and run order

All scripts assume:

```bash
export PYTHONPATH=/mnt/storage/RSystemsBenchmarking/gitProject
cd /mnt/storage/RSystemsBenchmarking/gitProject
ENV_PYTHON=/mnt/storage/RSystemsBenchmarking/envs/39myenv/bin/python
```

Paths, encoders, and index parameter grids are defined in **`configs.py`**. Shared I/O and FAISS helpers are in **`common.py`**.

---

## Run order (recommended)

Run these steps **in sequence**. Later steps depend on outputs from earlier ones.

| Step | Script | Requires | Produces |
|------|--------|----------|----------|
| **1** | `build_indexes.py` | Precomputed embeddings in `Benchmark/data/embeddings/` | `../Index/{HNSW,IVFPQ,OIVFPQ}/` |
| **2** | `run_knn.py` | Same embeddings + query JSON | `../KNN/*.json` |
| **3** | `run_retrieval.py` | Indexes from step 1 | `../HNSW/`, `../IVFPQ/`, `../OIVFPQ/*.json` |
| **4** | `compute_overlap_csv.py` | KNN + ANN JSON from 2–3 | `../Result/overlap_summary_all.csv` |
| **5** | `blip2_rerank_batch.py` | Step 4 CSV + ANN JSON | `../Reranker/{HNSW,IVFPQ,OIVFPQ}/` |
| **6** | `evaluate_reranker_recall.py` | Reranker JSON + query GT | `../Result/reranker/reranker_recall_summary.csv` |

Steps **1–4** are automated by `../run_all_batches.sh`. Steps **5–6** use the BLIP2 stack (`blip2_loader.py`); typical Python env: `py3_9_neuralReranker1_0` (see `../run_blip2_rerank.sh`).

### One-liner examples

```bash
# 1 — Build all index types (CPU)
$ENV_PYTHON Benchmark/code/ANN/Code/build_indexes.py \
  --encoder all --dataset all --index-type all --faiss-threads 8

# 2 — Exact KNN reference (GPU query encoding)
$ENV_PYTHON Benchmark/code/ANN/Code/run_knn.py \
  --encoder openclip_image --dataset flickr --device cuda

# 3 — ANN retrieval for one index family
$ENV_PYTHON Benchmark/code/ANN/Code/run_retrieval.py \
  --encoder openclip_image --dataset flickr --index-type hnsw --device cuda

# 4 — Overlap vs KNN
$ENV_PYTHON Benchmark/code/ANN/Code/compute_overlap_csv.py \
  --encoder all --index-type all

# 5 — BLIP2 rerank (after overlap CSV exists)
BLIP2_PYTHON=/path/to/py3_9_neuralReranker1_0/bin/python
$BLIP2_PYTHON Benchmark/code/ANN/Code/blip2_rerank_batch.py \
  --csv Benchmark/code/ANN/Result/overlap_summary_all.csv --index-type all

# 6 — Recall on reranked results
$ENV_PYTHON Benchmark/code/ANN/Code/evaluate_reranker_recall.py
```

Use `--encoder`, `--dataset`, and `--index-type` with `all` or a single value (`clip_image`, `coco`, `hnsw`, etc.). Add `--rebuild` on `build_indexes.py` to overwrite existing indexes.

---

## File reference

### `configs.py`

Central configuration: encoder/dataset names, embedding filenames, query/ingest paths, image roots, `INDEX_ROOT`, `TOP_K=100`, and functions `hnsw_configs()`, `ivfpq_configs()`, `oivfpq_configs()`, `result_filename()`.

**Use:** imported by all other modules; edit here to change parameter grids or paths.

---

### `common.py`

Shared utilities:

- Load/normalize embeddings JSON
- Index path helpers (`index_paths`, `results_dir`)
- Save/load FAISS index bundles (index + names + meta)
- Query loading, caption lookups, hit formatting for JSON output
- `set_hnsw_ef_search`, `set_ivf_nprobe`

**Use:** library only — not run directly.

---

### `embedders.py`

GPU batch encoders for query captions: CLIP, OpenCLIP, UniIR (`get_batch_encoder(encoder, device)`).

**Use:** imported by `run_knn.py` and `run_retrieval.py`.

---

### `build_indexes.py`

**Use case:** Build FAISS indexes from corpus embeddings for HNSW, IVFPQ, and O-IVFPQ (all config variants in `configs.py`).

**How to run:**

```bash
$ENV_PYTHON Benchmark/code/ANN/Code/build_indexes.py \
  [--encoder clip_image|openclip_image|uniir_joint|all] \
  [--dataset coco|flickr|all] \
  [--index-type hnsw|ivfpq|oivfpq|all] \
  [--faiss-threads 8] [--rebuild]
```

**Output:** `../Index/{TYPE}/{encoder}_{dataset}/{encoder}_{dataset}_{cfg_tag}.index` (+ names/meta JSON).

---

### `run_knn.py`

**Use case:** Exact **flat inner-product** search (no ANN approximation). Produces the reference top-100 lists used in overlap evaluation.

**How to run:**

```bash
$ENV_PYTHON Benchmark/code/ANN/Code/run_knn.py \
  [--encoder ...] [--dataset ...] \
  [--top-k 100] [--encode-batch-size 128] [--device cuda] \
  [--skip-existing]
```

**Output:** `../KNN/results_{encoder}_{dataset}_knn_flat__0.json`

**When:** Step **2**, after embeddings exist; before or in parallel with step 3 (overlap needs both KNN and ANN files).

---

### `run_retrieval.py`

**Use case:** Encode test queries and run batched FAISS search on each built index config (HNSW, IVFPQ, or O-IVFPQ).

**How to run:**

```bash
$ENV_PYTHON Benchmark/code/ANN/Code/run_retrieval.py \
  [--encoder ...] [--dataset ...] [--index-type hnsw|ivfpq|oivfpq|all] \
  [--top-k 100] [--encode-batch-size 128] [--device cuda] \
  [--faiss-threads 16] [--skip-existing]
```

**Output:** `../{HNSW|IVFPQ|OIVFPQ}/results_{encoder}_{dataset}_{index_type}_{cfg_tag}__0.json`

Each JSON entry: `query`, `list_of_top_k` (image paths, scores, ranks, captions).

**When:** Step **3**, after `build_indexes.py`.

---

### `compute_overlap_csv.py`

**Use case:** Compare each ANN result file to the matching KNN reference: mean top-100 overlap accuracy and Jaccard. Writes per-encoder CSVs and a combined summary used as the **rerank job list**.

**How to run:**

```bash
$ENV_PYTHON Benchmark/code/ANN/Code/compute_overlap_csv.py \
  [--encoder all] [--index-type hnsw|ivfpq|oivfpq|all] \
  [--top-n 100] [--output-dir ../Result]
```

**Output:**

- `../Result/{encoder}_{index_type}_overlap.csv`
- `../Result/overlap_summary_all.csv` (column `ann_method_file` → input for reranking)

**When:** Step **4**, after steps 2 and 3.

---

### `blip2_loader.py`

**Use case:** Load BLIP2-ITM model and processors in-process (minimal LAVIS import). Used only by the reranker.

**Use:** library — set `LAVIS_ROOT` if needed. Not run directly.

---

### `blip2_rerank_batch.py`

**Use case:** Rerank each ANN top-100 list with BLIP2 image–text matching scores (batched GPU, no HTTP service).

**How to run:**

```bash
$BLIP2_PYTHON Benchmark/code/ANN/Code/blip2_rerank_batch.py \
  --csv Benchmark/code/ANN/Result/overlap_summary_all.csv \
  --ann-root Benchmark/code/ANN \
  [--index-type hnsw|ivfpq|oivfpq|all] \
  [--top-k 100] [--device cuda] [--skip-existing]
```

**Input:** ANN JSON under `../HNSW/`, `../IVFPQ/`, `../OIVFPQ/` (paths from CSV).

**Output:** `../Reranker/{HNSW|IVFPQ|OIVFPQ}/*_rerank-blip2.json` with `list_of_top_k_rerank`.

**When:** Step **5**, after overlap CSV exists.

---

### `evaluate_reranker_recall.py`

**Use case:** Score reranked results against query ground truth (Karpathy COCO / Flickr30k test queries): recall@1, @5, @10.

**How to run:**

```bash
$ENV_PYTHON Benchmark/code/ANN/Code/evaluate_reranker_recall.py \
  [--reranker-root ../Reranker] \
  [--output-csv ../Result/reranker/reranker_recall_summary.csv] \
  [--format wide|detail]
```

**When:** Step **6**, after reranker JSONs exist.

---

### `__init__.py`

Package marker (empty). Scripts are normally invoked as modules from the project root with `PYTHONPATH` set.

---

## Result JSON naming

```text
results_{encoder}_{dataset}_{index_type}_{config_tag}__0.json
```

Examples:

- `results_openclip_image_flickr_hnsw_m32_efc200_efs128__0.json`
- `results_clip_image_coco_ivfpq_nlist128_m48_nbits8_nprobe16__0.json`

Reranked: same stem + `_rerank-blip2.json` under `../Reranker/`.

---

## Dependencies

| Step | Typical env | Notes |
|------|-------------|--------|
| 1–4 | `39myenv` | `faiss`, `torch`, `clip` / `open_clip` |
| 5 | `py3_9_neuralReranker1_0` | LAVIS BLIP2-ITM, `PIL`, `tqdm` |
| 6 | `39myenv` | stdlib + json only |

Model paths for CLIP/OpenCLIP/UniIR come from `Benchmark/config/config.yaml` via `embedders.py`.

---

## Parent folder

Directory layout, shell wrappers, and git notes: **[../README.md](../README.md)**
