#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from tqdm import tqdm

from common import build_fixed_queries, chunked


class ClipTextEncoder:
    def __init__(self, device: str):
        import torch
        try:
            import clip
        except ImportError:
            print("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
            raise

        self.device = device if device and torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.model.eval()

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        import torch
        import clip

        # 🔥 FIX: truncate long text
        def truncate_text(t, max_words=50):
            words = (t or "").strip().split()
            return " ".join(words[:max_words])

        clean = [truncate_text(t) for t in texts]

        with torch.no_grad():
            toks = clip.tokenize(clean, truncate=True).to(self.device)
            use_cuda = self.device.startswith("cuda")
            with torch.amp.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
                emb = self.model.encode_text(toks)

            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.detach().cpu().numpy().astype("float32")

def load_index_and_names(index_path: str, names_path: str):
    index = faiss.read_index(index_path)
    with open(names_path, "r") as f:
        names = json.load(f)
    if index.ntotal != len(names):
        raise ValueError(f"Index ntotal={index.ntotal} does not match names={len(names)}")
    return index, names


def find_first_file(candidates: List[str], fallback_glob: str = None) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    if fallback_glob:
        matches = sorted(Path("/").glob(fallback_glob.lstrip("/")))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(
        "Could not find required file. Checked candidates:\n"
        + "\n".join([f" - {c}" for c in candidates])
        + (f"\nGlob: {fallback_glob}" if fallback_glob else "")
    )


def resolve_query_tar_root(user_path: str) -> str:
    candidates = [
        user_path,
        "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/laion/worker1_20/tar1/worker_0",
        "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/laion400m",
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p

    raise FileNotFoundError(
        "No valid query tar directory found.\n"
        "Checked:\n" + "\n".join([f" - {x}" for x in candidates])
    )


def prepare_knn_gpu_if_needed(index, gpu_id: int):
    if isinstance(index, faiss.IndexFlat):
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, gpu_id, index), "gpu"
    return index, "cpu"


def set_runtime_search_param(index, index_type: str, runtime_value: int):
    if index_type == "hnsw":
        index.hnsw.efSearch = int(runtime_value)
    elif index_type == "ivfpq":
        index.nprobe = int(runtime_value)


def evaluate_single_config(
    encoder: ClipTextEncoder,
    queries: List[str],
    index,
    names: List[str],
    output_file: str,
    config: Dict[str, object],
    top_k: int,
    batch_size: int,
):
    results = []
    for batch in tqdm(list(chunked(queries, batch_size)), desc=f"Retrieval {os.path.basename(output_file)}"):
        qemb = encoder.encode_batch(batch)
        scores, idxs = index.search(qemb, top_k)
        for i, query in enumerate(batch):
            topk = []
            for fid, score in zip(idxs[i].tolist(), scores[i].tolist()):
                if fid < 0:
                    continue
                topk.append({"image_path": names[fid], "score": float(score)})
            results.append({"query": query, "topk": topk})

    payload = {
        "configuration": config,
        "num_queries": len(queries),
        "top_k": top_k,
        "results": results,
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[DONE] Saved => {output_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clip_root",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/testing1/clip",
        help="Root folder that contains index/ and quantization/ for CLIP indices",
    )
    ap.add_argument(
        "--query_tar_root",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/laion/worker1_20/tar1/worker_0",
    )
    ap.add_argument(
        "--output_root",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/output_testing1_clip",
    )
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    encoder = ClipTextEncoder(device=args.device)

    query_tar_root = resolve_query_tar_root(args.query_tar_root)
    print(f"[INFO] Query tar root: {query_tar_root}")

    query_file = os.path.join(args.output_root, "queries.json")
    queries = build_fixed_queries(
        tar_dir=query_tar_root,
        out_path=query_file,
        max_queries=1000,
    )

    hnsw_values = [1024, 2048]
    ivfpq_values = [32, 64, 128, 256, 512]

    index_dir = os.path.join(args.clip_root, "index")
    quant_dir = os.path.join(args.clip_root, "quantization")

    hnsw_index_path = find_first_file(
        [os.path.join(index_dir, "clip_image_hnsw.index")]
    )
    hnsw_names_path = find_first_file(
        [
            os.path.join(index_dir, "image_names.json"),
            os.path.join(quant_dir, "image_names.json"),
        ]
    )

    ivfpq_index_path = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/code/indexing_code/index/ivfpq/quantization/clip_image_ivfpq_nprobe_16_nlist_4096.index"
 
    ivfpq_names_path = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/code/indexing_code/index/ivfpq/quantization/image_names_nprobe_16_nlist_4096.json"

    knn_index_path = find_first_file(
        [os.path.join(index_dir, "clip_image_flat.index")]
    )
    knn_names_path = find_first_file(
        [
            os.path.join(index_dir, "image_names.json"),
            os.path.join(quant_dir, "image_names.json"),
        ]
    )

    print(f"[INFO] HNSW index: {hnsw_index_path}")
    print(f"[INFO] IVFPQ index: {ivfpq_index_path}")
    print(f"[INFO] KNN index: {knn_index_path}")

    hnsw_index, hnsw_names = load_index_and_names(hnsw_index_path, hnsw_names_path)
    for ef_search in hnsw_values:
        set_runtime_search_param(hnsw_index, "hnsw", ef_search)
        out_file = os.path.join(
            args.output_root,
            "hnsw",
            f"hnsw_m32_efc200_efs{ef_search}_top{args.top_k}.json",
        )
        evaluate_single_config(
            encoder=encoder,
            queries=queries,
            index=hnsw_index,
            names=hnsw_names,
            output_file=out_file,
            config={
                "index_type": "hnsw",
                "M": 32,
                "efConstruction": 200,
                "efSearch": ef_search,
                "search_device": "cpu",
            },
            top_k=args.top_k,
            batch_size=args.batch_size,
        )

    ivfpq_index, ivfpq_names = load_index_and_names(ivfpq_index_path, ivfpq_names_path)
    for nprobe in ivfpq_values:
        set_runtime_search_param(ivfpq_index, "ivfpq", nprobe)
        out_file = os.path.join(
            args.output_root,
            "ivfpq",
            f"ivfpq_nlist4096_m32_nbits8_nprobe{nprobe}_top{args.top_k}.json",
        )
        evaluate_single_config(
            encoder=encoder,
            queries=queries,
            index=ivfpq_index,
            names=ivfpq_names,
            output_file=out_file,
            config={
                "index_type": "ivfpq",
                "nlist": 4096,
                "m": 32,
                "nbits": 8,
                "nprobe": nprobe,
                "search_device": "cpu",
            },
            top_k=args.top_k,
            batch_size=args.batch_size,
        )

    # knn_index, knn_names = load_index_and_names(knn_index_path, knn_names_path)
    # knn_index, knn_device = prepare_knn_gpu_if_needed(knn_index, args.gpu_id)
    # out_file = os.path.join(args.output_root, "knn", f"knn_flat_ip_top{args.top_k}.json")
    # evaluate_single_config(
    #     encoder=encoder,
    #     queries=queries,
    #     index=knn_index,
    #     names=knn_names,
    #     output_file=out_file,
    #     config={
    #         "index_type": "knn",
    #         "flat": "IndexFlatIP",
    #         "search_device": knn_device,
    #         "gpu_id": args.gpu_id if knn_device == "gpu" else None,
    #     },
    #     top_k=args.top_k,
    #     batch_size=args.batch_size,
    # )


if __name__ == "__main__":
    main()