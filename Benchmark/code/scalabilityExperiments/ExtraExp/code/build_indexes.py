#!/usr/bin/env python3
"""
Build FAISS IVF-PQ inner-product index from CLIP embeddings (STREAMING, RAM-safe).

- Trains IVF-PQ on a sampled subset
- Adds all vectors in streaming way
- Saves:
    clip_image_ivfpq*.index
    image_names*.json
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
    "testing/laionRetriveal/extra_exp/code/indexing_code/index/ivfpq"
)

OUT_DIR = os.path.join(BASE_DIR, "quantization")
os.makedirs(OUT_DIR, exist_ok=True)

# Build only a bounded subset to reduce index size.
MAX_VECTORS = int(os.getenv("MAX_VECTORS", "7000000"))

# ================= IVF-PQ PARAMS =================
# Good defaults for ~10M vectors, dim ~768
NLIST = int(os.getenv("IVFPQ_NLIST", "4096"))
M = 128
NBITS = 8
NPROBE = int(os.getenv("IVFPQ_NPROBE", "16"))
TRAIN_SIZE = int(os.getenv("IVFPQ_TRAIN_SIZE", "200000"))
INDEX_SUFFIX = os.getenv("IVFPQ_INDEX_SUFFIX", f"_nprobe_{NPROBE}_nlist_{NLIST}")
INDEX_PATH = os.path.join(
    OUT_DIR,
    f"clip_ivfpq_nlist{NLIST}_m{M}_nbits{NBITS}.index"
)

NAMES_PATH = os.path.join(
    OUT_DIR,
    f"image_names_nlist{NLIST}_m{M}.json"
)
def _embedding_files():
    return iter_embedding_files(EMB_DIR)


def sample_training_vectors(files, dim):
    """
    Collect up to TRAIN_SIZE vectors across shards without loading everything.
    """
    train_vecs = []
    collected = 0

    target_train = min(int(TRAIN_SIZE), int(MAX_VECTORS))
    for fname in tqdm(files, desc="Sampling train vectors"):
        shard_path = os.path.join(EMB_DIR, fname)
        _, embs = load_shard(shard_path)

        assert embs.shape[1] == dim
        assert_normalized(embs)

        remaining = target_train - collected
        if remaining <= 0:
            break

        take = min(remaining, embs.shape[0])
        train_vecs.append(embs[:take])
        collected += take

        del embs

    if collected == 0:
        raise RuntimeError("Could not collect training vectors.")

    train_x = np.vstack(train_vecs)
    return train_x

# ================= MAIN =================

def main():
    files = _embedding_files()
    print(f"[INFO] Found {len(files)} embedding files")
    print(f"[INFO] MAX_VECTORS={MAX_VECTORS:,}")

    # ---- Step 1: read first shard to get dim ----
    first_path = os.path.join(EMB_DIR, files[0])
    _, first_embs = load_shard(first_path)

    dim = first_embs.shape[1]
    assert_normalized(first_embs)
    del first_embs

    print(f"[INFO] dim={dim}")

    # ---- Step 2: build IVF-PQ index skeleton ----
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        int(NLIST),
        int(M),
        int(NBITS),
        faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = int(NPROBE)

    # ---- Step 3: train IVF-PQ ----
    print(f"[INFO] Training IVF-PQ with TRAIN_SIZE={TRAIN_SIZE:,} vectors...")
    train_x = sample_training_vectors(files, dim)
    print(f"[INFO] Training matrix shape: {train_x.shape}")

    index.train(train_x)
    del train_x

    print("[INFO] Training done")

    # ---- Step 4: add all vectors streaming ----
    image_names = []
    total_added = 0

    for fname in tqdm(files, desc="Adding vectors"):
        if total_added >= MAX_VECTORS:
            break

        shard_path = os.path.join(EMB_DIR, fname)
        names, embs = load_shard(shard_path)

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

    # ---- Step 5: save index + metadata ----
    faiss.write_index(index, INDEX_PATH)
    with open(NAMES_PATH, "w") as f:
        json.dump(image_names, f)

    print("[DONE] IVF-PQ index built")
    print("Index:", INDEX_PATH)
    print("Names:", NAMES_PATH)
    print(f"Index type: IVF-PQ (nlist={NLIST}, m={M}, nbits={NBITS}, nprobe={NPROBE})")

if __name__ == "__main__":
    main()
