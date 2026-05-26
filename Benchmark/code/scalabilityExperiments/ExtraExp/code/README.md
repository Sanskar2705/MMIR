# Indexing code (CLIP / LAION ~7M)

Stream-build FAISS indexes from CLIP10M embedding shards, run text-query retrieval, write top-100 JSON per config.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `build_indexes32.py` | HNSW → `index/hnsw/clip_hnsw.index` |
| 2 | `build_indexes.py` | IVFPQ → `index/ivfpq/quantization/*.index` |
| 3 | `run_retrieval_eval.py` | JSON under `output_testing1_clip/{hnsw,ivfpq}/` |

Or run all steps: **`bash run_all.sh`**

## Modules

- **`common.py`** — embedding shard I/O, fixed 1000-query extraction from LAION tars.
- **`build_indexes32.py`** — HNSW index (despite the name: HNSW builder, not a separate “32” format).
- **`build_indexes.py`** — IVF-PQ index (train + stream add).
- **`run_retrieval_eval.py`** — CLIP text encode + FAISS search; sweeps HNSW `efSearch` and IVFPQ `nprobe`.

## Env vars

`MAX_VECTORS`, `HNSW_M`, `EF_CONSTRUCTION`, `IVFPQ_NLIST`, `IVFPQ_NPROBE`, `IVFPQ_TRAIN_SIZE` — see script headers.


