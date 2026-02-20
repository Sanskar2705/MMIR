#!/usr/bin/env python3
"""
Interactive Semantic Search over FAISS
Usage:
  python faiss_generic_semantic_search.py --model flava --dataset coco --target caption --engine faiss-hnsw --k 10
Targets: image | caption | joint-image-text
Engines: faiss | faiss-hnsw
"""
import sys, os, argparse, json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch

sys.path.append('/mnt/storage/RSystemsBenchmarking/gitProject')
from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import get_embedder


os.environ["HF_HUB_DISABLE_XET"] = "1"

def _d(): return "cuda" if torch.cuda.is_available() else "cpu"

def _engine_tag(e: str) -> str:
    e = e.lower()
    if e == "faiss": return "faiss"
    if e in ("faiss-hnsw","hnsw"): return "faiss-hnsw"
    raise ValueError("engine must be 'faiss' or 'faiss-hnsw'")

def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype("float32")

def _paths(index_dir: Path, model: str, dataset: str, target: str, engine_tag: str):
    t = target.lower()
    if t == "text": t = "caption"
    if t == "image":
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_img.index"
        meta = index_dir / f"{model}_{dataset}_img_names.json"
    elif t == "caption":
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_txt.index"
        meta = index_dir / f"{model}_{dataset}_txt_meta.json"
    elif t == "joint-image-text":
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_joint-image-text.index"
        meta = index_dir / f"{model}_{dataset}_joint-image-text_meta.json"
    else:
        raise ValueError("bad target")
    return idx, meta

def _load_meta(meta_path: Path, target: str):
    data = json.loads(meta_path.read_text())
    if target == "image":
        return [{"image_name": n, "caption": ""} for n in data]
    for r in data: r.setdefault("caption","")
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", default="caption", choices=["image","caption","text","joint-image-text"])
    ap.add_argument("--engine", default="faiss", choices=["faiss","faiss-hnsw"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=None, help="efSearch override for HNSW")
    args = ap.parse_args()

    cfg = load_config()
    dev = _d()
    embed = get_embedder(args.model, cfg, dev)

    index_dir = Path(cfg["vector_store"]["faiss"]["index_dir"])
    engine_tag = _engine_tag(args.engine)
    idx_path, meta_path = _paths(index_dir, args.model, args.dataset, args.target, engine_tag)

    index = faiss.read_index(str(idx_path))
    if engine_tag == "faiss-hnsw" and args.ef is not None:
        try: index.hnsw.efSearch = int(args.ef)
        except Exception: pass
    meta = _load_meta(meta_path, "image" if args.target=="image" else "caption")

    print(f"\nInteractive Semantic Search (FAISS) – {args.model}_{args.dataset}_{args.target}_{engine_tag}")
    print("Type 'exit' to quit.")
    while True:
        try:
            q = input("\nEnter your search query: ").strip()
            if q.lower() in ("exit","quit"): break
            if not q: continue

            t0 = get_time()
            vec = embed(q)
            enc = get_time() - t0
            vec = _normalize(vec)

            t1 = get_time()
            D, I = index.search(vec[None,:], args.k)
            rt = get_time() - t1

            print(f"Encoding: {enc:.4f}s | Retrieval: {rt:.4f}s")
            for rank, (i, s) in enumerate(zip(I[0], D[0]), start=1):
                row = meta[i] if i >= 0 else {"image_name":"<none>","caption":""}
                cap = row.get("caption","")
                cap = cap if len(cap)<120 else cap[:117]+"…"
                print(f"{rank:2d}. {row['image_name']}  [score={float(s):.6f}]  {cap}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
