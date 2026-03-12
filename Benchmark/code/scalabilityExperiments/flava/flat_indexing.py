#!/usr/bin/env python3
"""
Build FAISS index from FLAVA image embeddings (STREAMING, RAM-safe)
"""

import os
import json
import faiss
import numpy as np
from tqdm import tqdm

# ================= PATHS =================

EMB_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "Flava10M/flava/embeddings"
)

BASE_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "testing/testing1/flava"
)

OUT_DIR = os.path.join(BASE_DIR, "index")
INDEX_PATH = os.path.join(OUT_DIR, "flava_image.index")
NAMES_PATH = os.path.join(OUT_DIR, "image_names.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ================= MAIN =================

def main():
    files = sorted(
        f for f in os.listdir(EMB_DIR)
        if f.startswith("embeddings_part_") and f.endswith(".json")
    )

    print(f"[INFO] Found {len(files)} embedding files")

    index = None
    image_names = []

    total_added = 0

    for fname in tqdm(files, desc="Indexing embeddings"):
        path = os.path.join(EMB_DIR, fname)

        with open(path, "r") as f:
            data = json.load(f)

        # extract embeddings + names for THIS FILE ONLY
        names = [row["image_name"] for row in data]
        embs = np.asarray(
            [row["embedding"] for row in data],
            dtype="float32"
        )

        # initialize index lazily
        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            print(f"[INFO] FAISS index initialized (dim={dim})")

        # safety check: normalized
        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Embeddings not normalized in {fname}")

        # add to index
        index.add(embs)
        image_names.extend(names)
        total_added += len(names)

        # free memory explicitly
        del embs
        del names
        del data

        if total_added % 1_000_000 == 0:
            print(f"[INFO] Indexed {total_added:,} images")

    print(f"[INFO] Total indexed images: {total_added:,}")

    # save index + mapping
    print("[INFO] Saving FAISS index...")
    faiss.write_index(index, INDEX_PATH)

    print("[INFO] Saving image name mapping...")
    with open(NAMES_PATH, "w") as f:
        json.dump(image_names, f)

    # final sanity check
    assert index.ntotal == len(image_names)

    print("=" * 60)
    print("[DONE] FAISS index built successfully (streaming)")
    print(f"Index: {INDEX_PATH}")
    print(f"Names: {NAMES_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
