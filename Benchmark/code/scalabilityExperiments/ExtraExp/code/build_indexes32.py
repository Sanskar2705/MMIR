#!/usr/bin/env python3
"""
Build FAISS HNSW inner-product index from CLIP embeddings (STREAMING, RAM-safe).

- No training needed
- Adds vectors in streaming fashion
- Saves:
    clip_image_hnsw.index
    image_names.json
"""

import os
import json
import faiss
import numpy as np
from tqdm import tqdm

from common import assert_normalized, iter_embedding_files, load_shard

# ================= PATHS =================

EMB_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "CLIP10M/clip/embeddings"
)

BASE_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "testing/laionRetriveal/extra_exp/code/indexing_code/index/hnsw"
)

OUT_DIR = os.path.join(BASE_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(OUT_DIR, "clip_hnsw.index")
NAMES_PATH = os.path.join(OUT_DIR, "image_names.json")

MAX_VECTORS = int(os.getenv("MAX_VECTORS", "7000000"))

# ================= HNSW PARAMS =================

HNSW_M = int(os.getenv("HNSW_M", "64"))
EF_CONSTRUCTION = int(os.getenv("EF_CONSTRUCTION", "200"))

# ================= MAIN =================

def main():
    files = iter_embedding_files(EMB_DIR)

    print(f"[INFO] Found {len(files)} embedding files")
    print(f"[INFO] MAX_VECTORS={MAX_VECTORS:,}")

    index = None
    image_names = []
    total_added = 0

    for fname in tqdm(files, desc="Building HNSW index"):
        if total_added >= MAX_VECTORS:
            break

        shard_path = os.path.join(EMB_DIR, fname)
        names, embs = load_shard(shard_path)

        if embs.size == 0:
            continue

        if index is None:
            dim = embs.shape[1]

            index = faiss.IndexHNSWFlat(
                dim,
                HNSW_M,
                faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = EF_CONSTRUCTION

            print(
                f"[INFO] HNSW initialized "
                f"(dim={dim}, M={HNSW_M}, efC={EF_CONSTRUCTION})"
            )

        remaining = MAX_VECTORS - total_added
        if embs.shape[0] > remaining:
            embs = embs[:remaining]
            names = names[:remaining]

        assert embs.shape[1] == dim
        assert_normalized(embs)

        index.add(embs)
        image_names.extend(names)
        total_added += len(names)

        del embs, names

    print(f"[INFO] Total indexed images: {total_added:,}")

    assert index.ntotal == len(image_names)

    # ================= SAVE =================

    faiss.write_index(index, INDEX_PATH)

    with open(NAMES_PATH, "w") as f:
        json.dump(image_names, f)

    print("[DONE] HNSW index built")
    print("Index:", INDEX_PATH)
    print("Names:", NAMES_PATH)
    print(f"Index type: HNSW (M={HNSW_M}, efC={EF_CONSTRUCTION})")


if __name__ == "__main__":
    main()