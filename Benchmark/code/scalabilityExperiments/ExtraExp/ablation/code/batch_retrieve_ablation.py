#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tarfile
from typing import Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from tqdm import tqdm


def collect_fixed_queries(tar_dir: str, max_queries: int) -> List[str]:
    queries: List[str] = []
    seen = set()
    tar_files = sorted(
        os.path.join(tar_dir, name) for name in os.listdir(tar_dir) if name.endswith(".tar")
    )
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {tar_dir}")

    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tf:
            members = sorted(m for m in tf.getnames() if m.endswith(".json"))
            for member in members:
                if len(queries) >= max_queries:
                    break
                fobj = tf.extractfile(member)
                if not fobj:
                    continue
                try:
                    payload = json.loads(fobj.read().decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                caption = str(payload.get("caption", "")).strip()
                if not caption or caption in seen:
                    continue
                seen.add(caption)
                queries.append(caption)
        if len(queries) >= max_queries:
            break

    if len(queries) < max_queries:
        raise RuntimeError(f"Only {len(queries)} queries found, expected {max_queries}")
    return queries


def load_or_build_queries(queries_file: str, tar_dir: str, max_queries: int) -> List[str]:
    if os.path.exists(queries_file):
        with open(queries_file, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("queries", [])
        if not isinstance(data, list):
            raise ValueError(f"Unsupported query file format: {queries_file}")
        queries = [str(x).strip() for x in data if str(x).strip()][:max_queries]
        if len(queries) < max_queries:
            raise RuntimeError(
                f"Existing queries_file has only {len(queries)} queries, expected {max_queries}"
            )
        return queries

    queries = collect_fixed_queries(tar_dir=tar_dir, max_queries=max_queries)
    os.makedirs(os.path.dirname(queries_file) or ".", exist_ok=True)
    with open(queries_file, "w") as f:
        json.dump(queries, f, indent=2)
    return queries


def get_matching_names_path(index_path: str) -> str:
    index_name = os.path.basename(index_path)
    if index_name.startswith("clip_ivfpq_"):
        suffix = index_name[len("clip_ivfpq_") : -len(".index")]
    elif index_name.startswith("clip_hnsw_"):
        suffix = index_name[len("clip_hnsw_") : -len(".index")]
    else:
        raise ValueError(f"Unsupported index naming format: {index_name}")
    names_name = f"image_names_{suffix}.json"
    return os.path.join(os.path.dirname(index_path), names_name)


def list_indexes(index_dirs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for d in index_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Index dir not found: {d}")
        for name in sorted(os.listdir(d)):
            if name.endswith(".index"):
                out.append(os.path.join(d, name))
    if not out:
        raise RuntimeError("No .index files found in configured index dirs.")
    return out


class ClipTextEncoder:
    def __init__(self, model_name: str, device: str):
        import torch
        import clip

        self.torch = torch
        self.device = device if (device and torch.cuda.is_available()) else "cpu"
        self.clip = clip
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


def _set_ivf_nprobe(index: faiss.Index, nprobe: int) -> None:
    """Set nprobe on flat IVF indices and on OPQ/PreTransform-wrapped IVF-PQ indexes."""
    nprobe = max(1, int(nprobe))
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
        return
    try:
        ivf = faiss.extract_index_ivf(index)
    except Exception:
        return
    if hasattr(ivf, "nprobe"):
        ivf.nprobe = nprobe


def run_index_retrieval(
    index_path: str,
    names: Sequence[str],
    queries: Sequence[str],
    encoder: ClipTextEncoder,
    top_k: int,
    encode_batch_size: int,
    *,
    nprobe: Optional[int] = None,
    ef_search: Optional[int] = None,
) -> List[dict]:
    index = faiss.read_index(index_path)

    n_val = nprobe if nprobe is not None else int(os.getenv("IVFPQ_NPROBE", "16"))
    n_val = max(1, int(n_val))
    is_ivf = hasattr(index, "nprobe")
    if not is_ivf:
        try:
            faiss.extract_index_ivf(index)
            is_ivf = True
        except Exception:
            is_ivf = False
    if is_ivf:
        _set_ivf_nprobe(index, n_val)
    if hasattr(index, "hnsw"):
        ef_val = ef_search if ef_search is not None else int(os.getenv("HNSW_EFSEARCH", "128"))
        index.hnsw.efSearch = max(1, int(ef_val))

    output: List[dict] = []
    for start, q_batch in tqdm(
        batched(queries, encode_batch_size),
        total=(len(queries) + encode_batch_size - 1) // encode_batch_size,
        desc=f"Retrieving {os.path.basename(index_path)}",
    ):
        embs = encoder.encode_batch(q_batch)
        scores, idxs = index.search(embs.astype("float32"), int(top_k))

        for row_i, query in enumerate(q_batch):
            results = []
            for fid, score in zip(idxs[row_i].tolist(), scores[row_i].tolist()):
                if fid < 0:
                    continue
                results.append(
                    {
                        "image_path": names[fid],
                        "score": float(score),
                    }
                )
            output.append(
                {
                    "query_id": start + row_i + 1,
                    "query": query,
                    "result_of_top_100": results,
                }
            )
    return output


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch retrieval for all ablation HNSW/IVFPQ indexes (no API service)."
    )
    ap.add_argument(
        "--ivfpq_dir",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/ablation/index/ivfpq",
    )
    ap.add_argument(
        "--hnsw_dir",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/ablation/index/hnsw",
    )
    ap.add_argument(
        "--results_dir",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/ablation/results",
    )
    ap.add_argument(
        "--queries_file",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/ablation/results/fixed_queries_1000.json",
    )
    ap.add_argument(
        "--tar_dir",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/laion/worker1_20/tar1/worker_0",
    )
    ap.add_argument("--max_queries", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--encode_batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip_model_name", default="ViT-L/14@336px")
    ap.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="IVFPQ nprobe (default: IVFPQ_NPROBE env or 16). Ignored for HNSW indexes.",
    )
    ap.add_argument(
        "--efsearch",
        type=int,
        default=None,
        help="HNSW efSearch (default: HNSW_EFSEARCH env or 128). Ignored if index has no hnsw.",
    )
    ap.add_argument(
        "--output",
        default="",
        help="Write JSON to this path instead of results_dir/<index_basename>.json",
    )
    ap.add_argument(
        "--index_path",
        default="",
        help="Run retrieval for this single .index file only (skips directory scan)",
    )
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    queries = load_or_build_queries(
        queries_file=args.queries_file, tar_dir=args.tar_dir, max_queries=int(args.max_queries)
    )
    print(f"[INFO] Fixed queries: {len(queries)} from {args.queries_file}")

    index_path_arg = (args.index_path or "").strip()
    if index_path_arg:
        if not os.path.isfile(index_path_arg):
            raise FileNotFoundError(index_path_arg)
        index_paths = [index_path_arg]
        print(f"[INFO] Single index: {index_path_arg}")
    else:
        index_paths = list_indexes([args.ivfpq_dir, args.hnsw_dir])
        print(f"[INFO] Found {len(index_paths)} indexes.")

    encoder = ClipTextEncoder(model_name=args.clip_model_name, device=args.device)

    for index_path in index_paths:
        names_path = get_matching_names_path(index_path)
        with open(names_path, "r") as f:
            names = json.load(f)
        if not isinstance(names, list):
            raise ValueError(f"Names file is not list: {names_path}")

        print(f"[START] {os.path.basename(index_path)}")
        rows = run_index_retrieval(
            index_path=index_path,
            names=names,
            queries=queries,
            encoder=encoder,
            top_k=int(args.top_k),
            encode_batch_size=int(args.encode_batch_size),
            nprobe=args.nprobe,
            ef_search=args.efsearch,
        )

        out_name = os.path.basename(index_path).replace(".index", ".json")
        out_path = (args.output or "").strip()
        if not out_path:
            out_path = os.path.join(args.results_dir, out_name)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
