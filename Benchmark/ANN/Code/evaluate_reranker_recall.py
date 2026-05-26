#!/usr/bin/env python3
"""
Evaluate BLIP2-reranked ANN results (HNSW / IVFPQ / OIVFPQ) against query ground truth.

Ground truth: caption -> image pairs in QUERY_FILES (COCO Karpathy test, Flickr30k test).
Metrics: recall@1, recall@5, recall@10 on reranked top-k lists.

Writes a single CSV to code/ANN/Result/reranker/reranker_recall_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.code.ANN.Code.configs import (
    ANN_ROOT,
    DATASETS,
    ENCODERS,
    QUERY_FILES,
    DatasetName,
    EncoderName,
    IndexType,
    hnsw_configs,
    ivfpq_configs,
    oivfpq_configs,
)
RERANKER_ROOT = os.path.join(ANN_ROOT, "Reranker")
RESULT_DIR = os.path.join(ANN_ROOT, "Result", "reranker")
OUTPUT_CSV = os.path.join(RESULT_DIR, "reranker_recall_summary.csv")

INDEX_TYPES: List[IndexType] = ["hnsw", "ivfpq", "oivfpq"]
INDEX_FOLDERS = {"hnsw": "HNSW", "ivfpq": "IVFPQ", "oivfpq": "OIVFPQ"}
RECALL_KS = (1, 5, 10)
RERANK_SUFFIX = "_rerank-blip2.json"


def normalize_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("\\", "/").strip()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def get_gt_image(annotation: Dict[str, Any]) -> Optional[str]:
    return normalize_path(
        annotation.get("image") or annotation.get("image_path") or annotation.get("image_name")
    )


def get_retrieved_paths(result_row: Dict[str, Any]) -> List[str]:
    items = result_row.get("list_of_top_k_rerank")
    if not isinstance(items, list):
        items = result_row.get("list_of_top_k") or []
    paths: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        p = normalize_path(
            item.get("image_path") or item.get("image") or item.get("image_name")
        )
        if p:
            paths.append(p)
    return paths


def compute_recall(
    annotations: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    ks: Sequence[int] = RECALL_KS,
) -> Tuple[Dict[int, float], int]:
    if len(annotations) != len(results):
        raise ValueError(
            f"Query count mismatch: ground_truth={len(annotations)} results={len(results)}"
        )

    hits = {k: 0 for k in ks}
    total = 0
    for ann, res in zip(annotations, results):
        gt_image = get_gt_image(ann)
        if not gt_image:
            continue
        retrieved = get_retrieved_paths(res)
        total += 1
        for k in ks:
            if gt_image in retrieved[:k]:
                hits[k] += 1

    recalls = {k: (hits[k] / total if total else 0.0) for k in ks}
    return recalls, total


def parse_rerank_filename(fname: str) -> Optional[Tuple[EncoderName, DatasetName, IndexType, str]]:
    """
    results_clip_image_coco_hnsw_m16_efc100_efs64__0_rerank-blip2.json
    -> (clip_image, coco, hnsw, m16_efc100_efs64)
    """
    if not fname.endswith(RERANK_SUFFIX):
        return None
    body = fname[: -len(RERANK_SUFFIX)]
    if not body.startswith("results_"):
        return None
    body = body[len("results_") :]
    if body.endswith("__0"):
        body = body[:-3]

    for encoder in ENCODERS:
        prefix = f"{encoder}_"
        if not body.startswith(prefix):
            continue
        rest = body[len(prefix) :]
        for dataset in DATASETS:
            ds_prefix = f"{dataset}_"
            if not rest.startswith(ds_prefix):
                continue
            rest2 = rest[len(ds_prefix) :]
            for index_type in INDEX_TYPES:
                idx_prefix = f"{index_type}_"
                if rest2.startswith(idx_prefix):
                    cfg_tag = rest2[len(idx_prefix) :]
                    return encoder, dataset, index_type, cfg_tag
    return None


def config_rank(dataset: DatasetName, index_type: IndexType, cfg_tag: str) -> Optional[int]:
    if index_type == "hnsw":
        tags = [c.tag for c in hnsw_configs(dataset)]
    elif index_type == "ivfpq":
        tags = [c.tag for c in ivfpq_configs(dataset)]
    else:
        tags = [c.tag for c in oivfpq_configs(dataset)]
    try:
        return tags.index(cfg_tag)
    except ValueError:
        return None


def discover_rerank_files(reranker_root: str) -> List[str]:
    paths: List[str] = []
    for index_type, folder in INDEX_FOLDERS.items():
        sub = os.path.join(reranker_root, folder)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            if fname.endswith(RERANK_SUFFIX):
                paths.append(os.path.join(sub, fname))
    return paths


def evaluate_all(reranker_root: str) -> List[Dict[str, Any]]:
    detail_rows: List[Dict[str, Any]] = []
    gt_cache: Dict[DatasetName, List[Dict[str, Any]]] = {}

    for path in discover_rerank_files(reranker_root):
        fname = os.path.basename(path)
        parsed = parse_rerank_filename(fname)
        if parsed is None:
            print(f"[WARN] Skip unparseable filename: {fname}")
            continue
        encoder, dataset, index_type, cfg_tag = parsed

        if dataset not in gt_cache:
            gt_cache[dataset] = load_json(QUERY_FILES[dataset])
        annotations = gt_cache[dataset]
        results = load_json(path)

        try:
            recalls, total = compute_recall(annotations, results)
        except ValueError as exc:
            print(f"[WARN] {fname}: {exc}")
            continue

        rank = config_rank(dataset, index_type, cfg_tag)
        row = {
            "encoder": encoder,
            "dataset": dataset,
            "index_type": index_type,
            "config": cfg_tag,
            "config_rank": rank if rank is not None else "",
            "rerank_method_file": fname,
            "num_queries": total,
            "recall@1": round(recalls[1], 4),
            "recall@5": round(recalls[5], 4),
            "recall@10": round(recalls[10], 4),
        }
        detail_rows.append(row)
        print(
            f"{encoder}/{dataset}/{index_type}/{cfg_tag}: "
            f"R@1={recalls[1]:.4f} R@5={recalls[5]:.4f} R@10={recalls[10]:.4f}"
        )
    return detail_rows


def _fill_wide_bucket(bucket: Dict[str, Any], row: Dict[str, Any]) -> None:
    prefix = row["index_type"]
    bucket[f"{prefix}_config"] = row["config"]
    bucket[f"{prefix}_recall@1"] = row["recall@1"]
    bucket[f"{prefix}_recall@5"] = row["recall@5"]
    bucket[f"{prefix}_recall@10"] = row["recall@10"]
    bucket[f"{prefix}_rerank_file"] = row["rerank_method_file"]


def build_wide_rows(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Wide CSV: hnsw/ivfpq/oivfpq recall@1/5/10 per row.

    - clip_image: one row per (dataset, config_rank) — tier 0/1/2 from configs.py
    - openclip_image: one row per dataset (single rerank run per index type)
    """
    buckets: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for row in detail_rows:
        rank = row.get("config_rank")
        if row["encoder"] == "openclip_image":
            key = (row["encoder"], row["dataset"])
            if key not in buckets:
                buckets[key] = {
                    "encoder": row["encoder"],
                    "dataset": row["dataset"],
                    "config_rank": "single",
                }
        else:
            if rank == "" or rank is None:
                continue
            key = (row["encoder"], row["dataset"], int(rank))
            if key not in buckets:
                buckets[key] = {
                    "encoder": row["encoder"],
                    "dataset": row["dataset"],
                    "config_rank": int(rank),
                }
        _fill_wide_bucket(buckets[key], row)

    return [buckets[k] for k in sorted(buckets.keys())]


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[SKIP] No rows for {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate reranker recall@1/5/10 vs ground truth.")
    ap.add_argument("--reranker-root", default=RERANKER_ROOT)
    ap.add_argument("--output-csv", default=OUTPUT_CSV)
    ap.add_argument(
        "--format",
        choices=["detail", "wide"],
        default="wide",
        help="wide (default): one row per encoder/dataset/config_rank with hnsw/ivfpq/oivfpq recall columns",
    )
    args = ap.parse_args()

    detail_rows = evaluate_all(args.reranker_root)
    if not detail_rows:
        raise SystemExit("No reranker result files found.")

    out_rows = detail_rows if args.format == "detail" else build_wide_rows(detail_rows)
    write_csv(args.output_csv, out_rows)

    print(f"\n[DONE] Evaluated {len(detail_rows)} reranker result files -> {args.output_csv}")


if __name__ == "__main__":
    main()
