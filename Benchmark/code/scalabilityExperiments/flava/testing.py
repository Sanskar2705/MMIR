#!/usr/bin/env python3
"""
FLAVA Text → Image Retrieval (FULL TEST SCRIPT)

Includes:
- GT existence checks
- Correct tokenizer truncation
- FAISS retrieval
- Recall@K
- Debug prints
"""

import os
import json
import tarfile
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaModel

# ========================= PATHS =========================

BASE_DIR = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/testing1/flava"

INDEX_PATH = os.path.join(BASE_DIR, "index/flava_image.index")
NAMES_PATH = os.path.join(BASE_DIR, "index/image_names.json")

TAR_PATH = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "laion/tar/Worker0/tars/shard_00000.tar"
)

DEVICE = "cuda"
BATCH_SIZE = 128
TOPK = 1000
DEBUG_QUERIES = 5   # print debug for first N queries

# ========================= LOAD INDEX =========================

print("[INFO] Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("[INFO] Loading image-name mapping...")
with open(NAMES_PATH, "r") as f:
    image_names = json.load(f)

assert index.ntotal == len(image_names)
print(f"[INFO] Index size: {index.ntotal:,} images")

# ========================= LOAD CAPTIONS =========================

def load_captions_from_tar(tar_path, worker="worker_0"):
    captions = []
    gt_images = []

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".json")]

        for m in tqdm(members, desc="Reading captions"):
            f = tar.extractfile(m)
            if f is None:
                continue

            data = json.load(f)
            caption = data["caption"]

            shard, fname = m.name.split("/")
            img = fname.replace(".json", ".jpg")
            gt_path = f"{worker}/{shard}/{img}"

            captions.append(caption)
            gt_images.append(gt_path)

    return captions, gt_images

print("[INFO] Loading captions + GT...")
captions, gt_images = load_captions_from_tar(TAR_PATH)
print(f"[INFO] Queries: {len(captions)}")

# ========================= GT CONSISTENCY CHECK =========================

missing = sum(1 for gt in gt_images if gt not in image_names)
print(f"[CHECK] GT images missing from index: {missing}/{len(gt_images)}")

if missing > 0:
    print("[WARNING] Some GT images are missing from the index!")

# ========================= LOAD FLAVA =========================

print("[INFO] Loading FLAVA model...")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model = FlavaModel.from_pretrained("facebook/flava-full").to(DEVICE)
model.eval()

# ========================= TEXT ENCODING =========================

@torch.no_grad()
def encode_captions(captions):
    all_embs = []

    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Encoding captions"):
        batch = captions[i:i+BATCH_SIZE]

        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        out = model.get_text_features(**inputs)

        # CLS token pooling (IMPORTANT)
        emb = out[:, 0]
        emb = emb / emb.norm(dim=-1, keepdim=True)

        all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs).astype("float32")

print("[INFO] Encoding captions...")
text_embs = encode_captions(captions)

# ========================= FAISS RETRIEVAL =========================

print("[INFO] Running FAISS search...")
_, indices = index.search(text_embs, TOPK)

# ========================= RECALL METRICS =========================

def recall_at_k(indices, gt, names, ks=(1,10,100,1000)):
    out = {k: 0 for k in ks}

    for i, gt_img in enumerate(gt):
        retrieved = [names[idx] for idx in indices[i]]
        for k in ks:
            if gt_img in retrieved[:k]:
                out[k] += 1

    total = len(gt)
    return {k: v / total for k, v in out.items()}

print("[INFO] Computing recall...")
recalls = recall_at_k(indices, gt_images, image_names)

print("\n" + "=" * 60)
for k in sorted(recalls):
    print(f"Recall@{k}: {recalls[k]:.4f}")
print("=" * 60)

# ========================= DEBUG OUTPUT =========================

print("\n[DEBUG] Sample retrieval results")
for i in range(min(DEBUG_QUERIES, len(captions))):
    print("\nQuery:", captions[i])
    print("GT:", gt_images[i])
    print("Top-10 retrieved:")
    for r in indices[i][:10]:
        print("   ", image_names[r])

print("\n[DONE] Retrieval test complete.")
