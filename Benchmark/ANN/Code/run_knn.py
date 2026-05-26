#!/usr/bin/env python3
"""Flat (exact) KNN retrieval — top-100 per query, batched GPU encode + FAISS search."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import faiss
import numpy as np

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.code.ANN.Code.common import (
    build_caption_lookup,
    embedding_path,
    format_hit,
    load_embeddings_json,
    load_ingest_caption_lookup,
    load_queries,
    merge_caption_lookups,
    results_dir,
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
    result_filename,
)
from Benchmark.code.ANN.Code.embedders import get_batch_encoder


def search_flat_batch(
    index: faiss.Index,
    names: List[str],
    caption_lookup: Dict[str, str],
    query_matrix: np.ndarray,
    top_k: int,
    dataset: DatasetName,
    encoder: EncoderName,
) -> List[List[Dict[str, Any]]]:
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
    print(f"[SAVE] {output_path} ({len(output)} queries, top-{len(all_hits[0]) if all_hits else 0})")


def run_knn(
    encoder: EncoderName,
    dataset: DatasetName,
    top_k: int,
    encode_batch_size: int,
    device: str,
    skip_existing: bool,
    save_index: bool,
) -> None:
    out_path = os.path.join(
        results_dir("KNN"),
        result_filename(encoder, dataset, "knn", "flat"),
    )
    if skip_existing and os.path.isfile(out_path):
        print(f"[SKIP] {out_path}")
        return

    emb_path = embedding_path(encoder, dataset)
    names, meta_rows, embeddings = load_embeddings_json(emb_path)
    n, dim = embeddings.shape
    print(f"[KNN] {encoder}/{dataset}: corpus N={n}, dim={dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    if save_index:
        idx_dir = os.path.join(results_dir("KNN"), f"{encoder}_{dataset}")
        os.makedirs(idx_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(idx_dir, f"{encoder}_{dataset}_flat.index"))
        with open(os.path.join(idx_dir, f"{encoder}_{dataset}_names.json"), "w") as f:
            json.dump(names, f)

    queries = load_queries(QUERY_FILES[dataset])
    captions = [q["caption"] for q in queries]
    print(f"[ENCODE] {len(captions)} queries")
    batch_enc = get_batch_encoder(encoder, device)
    query_matrix = batch_enc.encode_all(captions, batch_size=encode_batch_size)
    del batch_enc
    if device.startswith("cuda"):
        import torch
        torch.cuda.empty_cache()

    cap_lookup = merge_caption_lookups(
        build_caption_lookup(meta_rows),
        load_ingest_caption_lookup(INGEST_FILES[dataset]),
    )

    print(f"[SEARCH] flat IP top-{top_k}")
    hits = search_flat_batch(index, names, cap_lookup, query_matrix, top_k, dataset, encoder)
    write_results(queries, hits, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flat KNN retrieval (exact search)")
    ap.add_argument("--encoder", choices=ENCODERS + ["all"], default="all")
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--encode-batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--faiss-threads", type=int, default=16)
    ap.add_argument("--save-index", action="store_true", help="Also save flat .index under KNN/")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    faiss.omp_set_num_threads(int(args.faiss_threads))

    encoders = ENCODERS if args.encoder == "all" else [args.encoder]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for encoder in encoders:
        for dataset in datasets:
            run_knn(
                encoder, dataset,
                args.top_k, args.encode_batch_size, args.device,
                args.skip_existing, args.save_index,
            )


if __name__ == "__main__":
    main()
