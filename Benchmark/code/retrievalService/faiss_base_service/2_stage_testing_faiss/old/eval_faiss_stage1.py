#!/usr/bin/env python3
"""
Evaluate Stage-1 retrieval (FAISS only, no re-ranking).

Examples
--------
# Flickr30k, caption→image (use the 5th caption like your sample), Flat IP:
python eval_faiss_stage1.py \
  --dataset flickr \
  --model clip \
  --target caption \
  --engine faiss \
  --ann /mnt/storage/RSystemsBenchmarking/data/datasets/vision/flickr30k/annotations/test.json

# COCO Karpathy test, caption→image, Flat IP:
python eval_faiss_stage1.py \
  --dataset coco \
  --model clip \
  --target caption \
  --engine faiss \
  --ann /mnt/storage/RSystemsBenchmarking/data/datasets/coco/annotations/coco_karpathy_test.json
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import faiss
import torch
from tqdm import tqdm

# project imports
sys.path.append('/mnt/storage/RSystemsBenchmarking/gitProject')
from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import get_embedder

os.environ["HF_HUB_DISABLE_XET"] = "1"


# ---------------- helpers ---------------- #

def _dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _engine_tag(engine: str) -> str:
    e = (engine or "").lower()
    if e == "faiss":
        return "faiss"
    if e in ("faiss-hnsw", "hnsw"):
        return "faiss-hnsw"
    raise ValueError("engine must be 'faiss' or 'faiss-hnsw'")

def _normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x / (np.linalg.norm(x) + 1e-12)
        return x.astype("float32", copy=False)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x.astype("float32", copy=False)

def _kind_meta_files(index_dir: Path, model: str, dataset: str, engine_tag: str, target: str) -> Tuple[Path, Path, str]:
    """
    target ∈ {"image","caption","text","joint-image-text"}; "text" -> "caption"
    Returns (index_path, meta_path, normalized_target)
    """
    t = target.lower()
    if t == "text":
        t = "caption"

    if t == "image":
        kind = "img"
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
        meta = index_dir / f"{model}_{dataset}_img_names.json"  # list[str]
    elif t == "caption":
        kind = "txt"
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
        meta = index_dir / f"{model}_{dataset}_txt_meta.json"   # list[{"image_name","caption"}]
    elif t == "joint-image-text":
        kind = "joint-image-text"
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
        meta = index_dir / f"{model}_{dataset}_joint-image-text_meta.json"
    else:
        raise ValueError("target must be image | caption | text | joint-image-text")

    if not idx.exists() or not meta.exists():
        raise FileNotFoundError(f"Missing index/meta at {index_dir}: {idx.name} / {meta.name}")
    return idx, meta, t

def _load_meta(meta_path: Path, target_norm: str) -> List[Dict]:
    data = json.loads(meta_path.read_text())
    if target_norm == "image":
        return [{"image_name": n, "caption": ""} for n in data]  # unify shape
    for r in data:
        r.setdefault("image_name", r.get("image_name", ""))
        r.setdefault("caption", r.get("caption", ""))
    return data

def _canon(name: str) -> str:
    return os.path.basename(str(name)).lower()

# ---------------- annotation readers ---------------- #

def _iter_flickr30k(ann_path: Path, caption_idx: int = 4) -> Iterable[Tuple[str, str]]:
    """
    Yields (gt_image_name, query_caption)
    Expects entries like:
      {"image": "flickr30k-images/100012.jpg", "caption": ["..", "...", ...]}
    """
    data = json.loads(ann_path.read_text())
    for entry in data:
        img = entry.get("image", "")
        caps = entry.get("caption", [])
        if not caps:
            continue
        idx = min(max(caption_idx, 0), len(caps)-1)
        yield _canon(img), str(caps[idx]).strip()

def _iter_coco_karpathy(ann_path: Path) -> Iterable[Tuple[str, str]]:
    """
    Yields (gt_image_name, query_caption) for COCO Karpathy split format:
      {"images":[{"filename":"COCO_val2014_000000391895.jpg",
                  "sentences":[{"raw":"a man ..."}, ...]}, ...]}
    """
    data = json.loads(ann_path.read_text())
    images = data["images"] if "images" in data else data  # some dumps flatten it
    for img in images:
        fname = img.get("filename", img.get("image", ""))
        for s in img.get("sentences", []):
            raw = s.get("raw") or s.get("caption") or ""
            if raw:
                yield _canon(fname), str(raw).strip()

def _iter_annotations(dataset: str, ann_path: Path, caption_idx: int = 4) -> Iterable[Tuple[str, str]]:
    ds = dataset.lower()
    if ds.startswith("flickr"):
        return _iter_flickr30k(ann_path, caption_idx=caption_idx)
    if ds.startswith("coco"):
        return _iter_coco_karpathy(ann_path)
    # default: try generic list of {image, caption[list]|caption[str]}
    def _generic():
        data = json.loads(ann_path.read_text())
        for entry in data:
            img = _canon(entry.get("image", entry.get("filename","")))
            cap = entry.get("caption", "")
            if isinstance(cap, list) and cap:
                cap = cap[min(caption_idx, len(cap)-1)]
            if img and cap:
                yield img, str(cap).strip()
    return _generic()

# ---------------- main eval ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="flickr | coco | ...")
    ap.add_argument("--model",   required=True, help="clip | flava | minilm | uniir | ...")
    ap.add_argument("--target",  default="caption", choices=["image","caption","text","joint-image-text"],
                    help="Which FAISS index to query for Stage-1.")
    ap.add_argument("--engine",  default="faiss", choices=["faiss","faiss-hnsw"],
                    help="Stage-1 FAISS index type (default: faiss = Flat IP).")
    ap.add_argument("--ann",     required=True, type=Path, help="Annotation JSON path")
    ap.add_argument("--topk",    type=int, default=10, help="Top-K rows to retrieve from FAISS (Stage-1).")
    ap.add_argument("--max_samples", type=int, default=None, help="Limit number of queries for speed.")
    ap.add_argument("--caption_idx", type=int, default=4, help="For Flickr30k: which caption index to use.")
    ap.add_argument("--collapse_unique_images", action="store_true",
                    help="Collapse Stage-1 rows to unique images (order-preserving) before computing recall.")
    ap.add_argument("--ef", type=int, default=None, help="Optional efSearch for HNSW (ignored for Flat IP).")
    args = ap.parse_args()

    cfg = load_config()
    dev = _dev()
    embed = get_embedder(args.model, cfg, dev)

    index_dir = Path(cfg["vector_store"]["faiss"]["index_dir"])
    eng_tag = _engine_tag(args.engine)
    idx_path, meta_path, tnorm = _kind_meta_files(index_dir, args.model, args.dataset, eng_tag, args.target)
    meta = _load_meta(meta_path, tnorm)

    index = faiss.read_index(str(idx_path))
    # Optional HNSW tuning
    if eng_tag == "faiss-hnsw" and args.ef is not None:
        try:
            index.hnsw.efSearch = int(args.ef)
        except Exception:
            pass

    # Build array that maps row->image_name (for caption/joint indices with many rows per image)
    row_to_image = [ _canon(row["image_name"]) for row in meta ]

    # ---- Evaluation loop ----
    r1 = r5 = r10 = 0
    enc_times: List[float] = []
    srch_times: List[float] = []
    total = 0

    ann_iter = _iter_annotations(args.dataset, args.ann, caption_idx=args.caption_idx)
    if args.max_samples is not None:
        # materialize limited slice for progress display
        ann_iter = list(ann_iter)[:args.max_samples]

    for gt_img, query in tqdm(ann_iter, desc="Evaluating", unit="q"):
        try:
            t0 = get_time()
            qv = embed(query)              # np.ndarray (or torch -> np inside embedder)
            enc_times.append(get_time() - t0)
            qv = _normalize(qv)

            t1 = get_time()
            D, I = index.search(qv[None, :].astype("float32"), args.topk)
            srch_times.append(get_time() - t1)

            # Convert Stage-1 rows to image list
            row_ids = [int(i) for i in I[0] if int(i) >= 0]
            if args.collapse_unique_images:
                seen = set()
                images = []
                for rid in row_ids:
                    im = row_to_image[rid]
                    if im not in seen:
                        images.append(im)
                        seen.add(im)
                retrieved_images = images
            else:
                # no collapse: one row = one vote (can include dups)
                retrieved_images = [row_to_image[rid] for rid in row_ids]

            total += 1
            # Compare using canonical/basename (your annotations often include subdirs)
            gt = _canon(gt_img)
            # compute recall on the first N retrieved images
            # If not collapsed, duplicates are fine: membership in slice still works.
            if gt in retrieved_images[:1]:
                r1 += 1
            if gt in retrieved_images[:5]:
                r5 += 1
            if gt in retrieved_images[:10]:
                r10 += 1

        except Exception as e:
            # Skip bad samples but continue
            # (print once in a while if you want)
            continue

    if total == 0:
        print("No samples evaluated. Check your --ann and dataset format.")
        return

    print(f"\nEvaluated on {total} samples")
    print(f"R@1:  {r1/total:.4f}")
    print(f"R@5:  {r5/total:.4f}")
    print(f"R@10: {r10/total:.4f}")
    print(f"Avg encode time:  {np.mean(enc_times):.4f}s")
    print(f"Avg search time:  {np.mean(srch_times):.4f}s")
    print(f"Engine: {eng_tag}  | Target index: {tnorm}  | TopK rows: {args.topk}  | Collapse unique images: {args.collapse_unique_images}")
    

if __name__ == "__main__":
    main()
