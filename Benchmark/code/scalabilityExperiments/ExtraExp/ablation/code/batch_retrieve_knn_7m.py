#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from tqdm import tqdm


class ClipTextEncoder:
    def __init__(self, model_name: str, device: str):
        import torch
        import clip

        self.torch = torch
        self.clip = clip
        self.device = device if (device and torch.cuda.is_available()) else "cpu"
        self.model, _ = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.no_grad():
            toks = self.clip.tokenize(list(texts), truncate=True).to(self.device)
            with self.torch.cuda.amp.autocast(
                enabled=self.device.startswith("cuda"), dtype=self.torch.float16
            ):
                emb = self.model.encode_text(toks)
            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.detach().cpu().numpy().astype("float32")


def batched(items: Sequence[str], batch_size: int) -> Iterable[Tuple[int, Sequence[str]]]:
    for i in range(0, len(items), batch_size):
        yield i, items[i : i + batch_size]


def load_queries(queries_file: str, max_queries: int) -> List[str]:
    with open(queries_file, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("queries", [])
    if not isinstance(data, list):
        raise ValueError(f"Unsupported query file format: {queries_file}")
    queries = [str(x).strip() for x in data if str(x).strip()][:max_queries]
    if len(queries) < max_queries:
        raise RuntimeError(f"Only {len(queries)} queries found, expected {max_queries}")
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch retrieval for 7M clip flat (KNN) index.")
    ap.add_argument(
        "--index_path",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/testing1/clip/index/clip_image_flat.index",
    )
    ap.add_argument(
        "--names_path",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/testing1/clip/index/image_names.json",
    )
    ap.add_argument(
        "--queries_file",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/ablation/results/fixed_queries_1000.json",
    )
    ap.add_argument(
        "--results_dir",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/ablation/results",
    )
    ap.add_argument("--max_queries", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--encode_batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip_model_name", default="ViT-L/14@336px")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    queries = load_queries(args.queries_file, int(args.max_queries))
    print(f"[INFO] Loaded {len(queries)} fixed queries from {args.queries_file}", flush=True)

    with open(args.names_path, "r") as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(f"Names file is not a list: {args.names_path}")

    print(f"[INFO] Reading index: {args.index_path}", flush=True)
    index = faiss.read_index(args.index_path)
    print(f"[INFO] Index ntotal={index.ntotal}", flush=True)

    encoder = ClipTextEncoder(model_name=args.clip_model_name, device=args.device)

    rows: List[dict] = []
    for start, q_batch in tqdm(
        batched(queries, int(args.encode_batch_size)),
        total=(len(queries) + int(args.encode_batch_size) - 1) // int(args.encode_batch_size),
        desc="KNN retrieval",
    ):
        embs = encoder.encode_batch(q_batch)
        scores, idxs = index.search(embs.astype("float32"), int(args.top_k))
        for row_i, query in enumerate(q_batch):
            top = []
            for fid, score in zip(idxs[row_i].tolist(), scores[row_i].tolist()):
                if fid < 0:
                    continue
                top.append({"image_path": names[fid], "score": float(score)})
            rows.append(
                {
                    "query_id": start + row_i + 1,
                    "query": query,
                    "result_of_top_100": top,
                }
            )

    out_name = os.path.basename(args.index_path).replace(".index", ".json")
    out_path = os.path.join(args.results_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[DONE] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
