#!/usr/bin/env python3
"""
CLIP Text → Image Retrieval Evaluation (LAION)
"""

import os
import json
import tarfile
import faiss
import numpy as np
import torch
from tqdm import tqdm
import clip

# ================= PATHS =================

BASE_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "testing/testing1/clip"
)

# switch index here: flat or hnsw
INDEX_PATH = os.path.join(BASE_DIR, "index/clip_image_flat.index")
# INDEX_PATH = os.path.join(BASE_DIR, "index/clip_image_hnsw.index")

NAMES_PATH = os.path.join(BASE_DIR, "index/image_names.json")

TAR_PATH = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "laion/tar/Worker0/tars/shard_00000.tar"
)

DEVICE = "cuda"
BATCH_SIZE = 256
TOPK = 1000
DEBUG_QUERIES = 5

MODEL_NAME = "ViT-L/14@336px"

# ================= LOAD INDEX =================

print("[INFO] Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

with open(NAMES_PATH, "r") as f:
    image_names = json.load(f)

assert index.ntotal == len(image_names)
print(f"[INFO] Index size: {index.ntotal:,}")

# ================= LOAD CAPTIONS =================

def load_captions_from_tar(tar_path, worker="worker_0"):
    captions, gt_images = [], []

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".json")]

        for m in tqdm(members, desc="Reading captions"):
            f = tar.extractfile(m)
            if f is None:
                continue

            data = json.load(f)
            caption = data.get("caption", "").strip()
            if not caption:
                continue

            shard, fname = m.name.split("/")
            img = fname.replace(".json", ".jpg")
            gt = f"{worker}/{shard}/{img}"

            captions.append(caption)
            gt_images.append(gt)

    return captions, gt_images

captions, gt_images = load_captions_from_tar(TAR_PATH)
print(f"[INFO] Queries: {len(captions)}")

# ================= LOAD CLIP =================

print("[INFO] Loading CLIP model...")
model, _ = clip.load(MODEL_NAME, device=DEVICE)
model.eval()

# ================= TEXT ENCODING =================

@torch.no_grad()
def encode_captions(captions):
    embs = []
    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Encoding captions"):
        batch = captions[i:i+BATCH_SIZE]
        tokens = clip.tokenize(batch, truncate=True).to(DEVICE)

        e = model.encode_text(tokens)
        e = e.float()
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu().numpy())

    return np.vstack(embs).astype("float32")

text_embs = encode_captions(captions)

# ================= FAISS SEARCH ===============
