#!/usr/bin/env python3
"""Build HNSW, IVFPQ, and O-IVFPQ FAISS indexes for COCO/Flickr."""

from __future__ import annotations

import argparse
import sys

import faiss
import numpy as np

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.code.ANN.Code.common import (
    embedding_path,
    index_paths,
    load_embeddings_json,
    save_index_bundle,
)
from Benchmark.code.ANN.Code.configs import (
    DATASETS,
    ENCODERS,
    DatasetName,
    EncoderName,
    IndexType,
    hnsw_configs,
    ivfpq_configs,
    oivfpq_configs,
)


def build_hnsw(
    embeddings: np.ndarray,
    names: list,
    meta_rows: list,
    m: int,
    ef_construction: int,
    index_path: str,
    names_path: str,
    meta_path: str,
) -> None:
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, int(m), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)
    index.add(embeddings)
    save_index_bundle(index, names, meta_rows, index_path, names_path, meta_path)


def build_ivfpq(
    embeddings: np.ndarray,
    names: list,
    meta_rows: list,
    nlist: int,
    m: int,
    nbits: int,
    index_path: str,
    names_path: str,
    meta_path: str,
) -> None:
    n, dim = embeddings.shape
    if dim % int(m) != 0:
        raise ValueError(f"dim={dim} not divisible by m={m}")
    nlist = min(int(nlist), n)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, nlist, int(m), int(nbits), faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings)
    index.add(embeddings)
    save_index_bundle(index, names, meta_rows, index_path, names_path, meta_path)


def build_oivfpq(
    embeddings: np.ndarray,
    names: list,
    meta_rows: list,
    factory_spec: str,
    index_path: str,
    names_path: str,
    meta_path: str,
) -> None:
    n, dim = embeddings.shape
    index = faiss.index_factory(dim, factory_spec, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)
    save_index_bundle(index, names, meta_rows, index_path, names_path, meta_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=ENCODERS + ["all"], default="all")
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    ap.add_argument("--index-type", choices=["hnsw", "ivfpq", "oivfpq", "all"], default="all")
    ap.add_argument("--faiss-threads", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    faiss.omp_set_num_threads(int(args.faiss_threads))

    encoders = ENCODERS if args.encoder == "all" else [args.encoder]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    index_types = (
        ["hnsw", "ivfpq", "oivfpq"]
        if args.index_type == "all"
        else [args.index_type]
    )

    for encoder in encoders:
        for dataset in datasets:
            emb_path = embedding_path(encoder, dataset)
            names, meta_rows, embeddings = load_embeddings_json(emb_path)
            print(f"[LOAD] {encoder}/{dataset}: N={len(names)}, dim={embeddings.shape[1]}")

            if "hnsw" in index_types:
                for cfg in hnsw_configs(dataset):
                    idx_path, names_path, meta_path = index_paths("hnsw", encoder, dataset, cfg.tag)
                    if not args.rebuild and __import__("os").path.isfile(idx_path):
                        print(f"[SKIP][HNSW] {cfg.tag}")
                        continue
                    print(f"[BUILD][HNSW] {encoder}/{dataset} {cfg.tag}")
                    build_hnsw(
                        embeddings, names, meta_rows,
                        cfg.m, cfg.ef_construction,
                        idx_path, names_path, meta_path,
                    )

            if "ivfpq" in index_types:
                for cfg in ivfpq_configs(dataset):
                    idx_path, names_path, meta_path = index_paths("ivfpq", encoder, dataset, cfg.tag)
                    if not args.rebuild and __import__("os").path.isfile(idx_path):
                        print(f"[SKIP][IVFPQ] {cfg.tag}")
                        continue
                    print(f"[BUILD][IVFPQ] {encoder}/{dataset} {cfg.tag}")
                    build_ivfpq(
                        embeddings, names, meta_rows,
                        cfg.nlist, cfg.m, cfg.nbits,
                        idx_path, names_path, meta_path,
                    )

            if "oivfpq" in index_types:
                for cfg in oivfpq_configs(dataset):
                    idx_path, names_path, meta_path = index_paths("oivfpq", encoder, dataset, cfg.tag)
                    if not args.rebuild and __import__("os").path.isfile(idx_path):
                        print(f"[SKIP][OIVFPQ] {cfg.tag}")
                        continue
                    print(f"[BUILD][OIVFPQ] {encoder}/{dataset} {cfg.tag} ({cfg.factory_spec})")
                    build_oivfpq(
                        embeddings, names, meta_rows,
                        cfg.factory_spec,
                        idx_path, names_path, meta_path,
                    )

    print("[DONE] Index build complete.")


if __name__ == "__main__":
    main()
