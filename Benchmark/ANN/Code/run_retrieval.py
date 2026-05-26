#!/usr/bin/env python3
"""Batched ANN retrieval: top-100 per query, minimal JSON output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.code.ANN.Code.common import (
    build_caption_lookup,
    format_hit,
    index_paths,
    load_index_bundle,
    load_ingest_caption_lookup,
    load_queries,
    merge_caption_lookups,
    results_dir,
    set_hnsw_ef_search,
    set_ivf_nprobe,
)
from Benchmark.code.ANN.Code.configs import (
    BASE_IMAGE_DIRS,
    DATASETS,
    ENCODERS,
    INGEST_FILES,
    QUERY_FILES,
    TOP_K,
    DatasetName,
    EncoderName,
    IndexType,
    hnsw_configs,
    ivfpq_configs,
    oivfpq_configs,
    result_filename,
)
from Benchmark.code.ANN.Code.embedders import get_batch_encoder


def search_batch(
    index: faiss.Index,
    names: List[str],
    caption_lookup: Dict[str, str],
    query_matrix: np.ndarray,
    top_k: int,
    index_type: IndexType,
    search_param: int,
    dataset: DatasetName,
    encoder: EncoderName,
) -> List[List[Dict[str, Any]]]:
    if index_type == "hnsw":
        set_hnsw_ef_search(index, search_param)
    else:
        set_ivf_nprobe(index, search_param)

    base_dir = BASE_IMAGE_DIRS[dataset]
    target = "joint-image-text" if encoder == "uniir_joint" else "image"
    all_hits: List[List[Dict[str, Any]]] = []

    search_bs = 256
    for start in range(0, query_matrix.shape[0], search_bs):
        qbatch = query_matrix[start : start + search_bs]
        scores, idxs = index.search(qbatch, top_k)
        for row_idx, row_scores in zip(idxs, scores):
            hits = []
            rank = 1
            for fid, sc in zip(row_idx.tolist(), row_scores.tolist()):
                if fid < 0:
                    continue
                hits.append(
                    format_hit(names[fid], sc, rank, base_dir, caption_lookup, target)
                )
                rank += 1
            all_hits.append(hits)
    return all_hits


def write_results(
    queries: List[Dict[str, str]],
    all_hits: List[List[Dict[str, Any]]],
    output_path: str,
) -> None:
    output = [
        {"query": q["caption"], "list_of_top_k": hits}
        for q, hits in zip(queries, all_hits)
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVE] {output_path} ({len(output)} queries)")


def caption_lookup_for(encoder: EncoderName, dataset: DatasetName, meta_rows: List[Dict]) -> Dict[str, str]:
    return merge_caption_lookups(
        build_caption_lookup(meta_rows),
        load_ingest_caption_lookup(INGEST_FILES[dataset]),
    )


def iter_jobs(
    encoder: EncoderName,
    dataset: DatasetName,
    index_type: IndexType,
) -> List[Tuple[str, str, int]]:
    """(cfg_tag, search_param_value) per config."""
    if index_type == "hnsw":
        return [(c.tag, c.ef_search) for c in hnsw_configs(dataset)]
    if index_type == "ivfpq":
        return [(c.tag, c.nprobe) for c in ivfpq_configs(dataset)]
    if index_type == "oivfpq":
        return [(c.tag, c.nprobe) for c in oivfpq_configs(dataset)]
    return []


def run_encoder_dataset(
    encoder: EncoderName,
    dataset: DatasetName,
    index_types: List[IndexType],
    top_k: int,
    encode_batch_size: int,
    device: str,
    skip_existing: bool,
) -> None:
    query_path = QUERY_FILES[dataset]
    queries = load_queries(query_path)
    captions = [q["caption"] for q in queries]

    print(f"[ENCODE] {encoder}/{dataset} ({len(captions)} queries)")
    batch_enc = get_batch_encoder(encoder, device)
    query_matrix = batch_enc.encode_all(captions, batch_size=encode_batch_size)
    del batch_enc
    if device.startswith("cuda"):
        torch = __import__("torch")
        torch.cuda.empty_cache()

    for idx_type in index_types:
        out_root = results_dir(idx_type.upper())
        jobs = iter_jobs(encoder, dataset, idx_type)
        for cfg_tag, search_param in jobs:
            out_path = os.path.join(
                out_root,
                result_filename(encoder, dataset, idx_type, cfg_tag),
            )
            if skip_existing and os.path.isfile(out_path):
                print(f"[SKIP] {out_path}")
                continue

            idx_path, names_path, meta_path = index_paths(idx_type, encoder, dataset, cfg_tag)
            if not os.path.isfile(idx_path):
                print(f"[WARN] missing index: {idx_path}")
                continue

            print(f"[SEARCH] {encoder}/{dataset}/{idx_type}/{cfg_tag}")
            index, names, meta_rows = load_index_bundle(idx_path, names_path, meta_path)
            cap_lookup = caption_lookup_for(encoder, dataset, meta_rows)
            hits = search_batch(
                index, names, cap_lookup, query_matrix, top_k,
                idx_type, search_param, dataset, encoder,
            )
            write_results(queries, hits, out_path)
            del index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=ENCODERS + ["all"], default="all")
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    ap.add_argument("--index-type", choices=["hnsw", "ivfpq", "oivfpq", "all"], default="all")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--encode-batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--faiss-threads", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    faiss.omp_set_num_threads(int(args.faiss_threads))

    encoders = ENCODERS if args.encoder == "all" else [args.encoder]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    index_types: List[IndexType] = (
        ["hnsw", "ivfpq", "oivfpq"] if args.index_type == "all" else [args.index_type]
    )

    for encoder in encoders:
        for dataset in datasets:
            run_encoder_dataset(
                encoder, dataset, index_types,
                args.top_k, args.encode_batch_size, args.device, args.skip_existing,
            )


if __name__ == "__main__":
    main()
