#!/usr/bin/env python3
"""
Top-100 overlap: KNN (flat) vs ANN (HNSW / IVFPQ / OIVFPQ).

Writes per-encoder, per-index-type CSVs under code/ANN/Result/:
  clip_image_hnsw_overlap.csv
  clip_image_ivfpq_overlap.csv
  clip_image_oivfpq_overlap.csv
  openclip_image_*.csv
  uniir_joint_*.csv
  overlap_summary_all.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Set, Tuple

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.code.ANN.Code.configs import (
    ANN_ROOT,
    DATASETS,
    ENCODERS,
    EncoderName,
    IndexType,
    result_filename,
)
from Benchmark.code.ANN.Code.common import results_dir


RESULT_DIR = os.path.join(ANN_ROOT, "Result")
TOP_N_DEFAULT = 100

INDEX_TYPES: List[IndexType] = ["hnsw", "ivfpq", "oivfpq"]


def image_paths_from_results(rows: Sequence[Dict[str, Any]]) -> List[List[str]]:
    out: List[List[str]] = []
    for row in rows:
        items = row.get("list_of_top_k") or []
        paths = [str(it.get("image_path", "")) for it in items if it.get("image_path")]
        out.append(paths)
    return out


def overlap_stats(
    knn_paths: Sequence[str], ann_paths: Sequence[str], n: int
) -> Tuple[float, int, float]:
    """Returns (overlap_accuracy, overlap_count, jaccard)."""
    knn_n = list(knn_paths[:n])
    ann_n = list(ann_paths[:min(n, len(ann_paths))])
    if not knn_n or not ann_n:
        return 0.0, 0, 0.0
    set_k = set(knn_n)
    set_a = set(ann_n)
    common = set_k & set_a
    overlap_count = len(common)
    accuracy = overlap_count / float(n)
    union = len(set_k | set_a)
    jaccard = overlap_count / union if union else 0.0
    return accuracy, overlap_count, jaccard


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def extract_config_tag(method_file: str, encoder: str, dataset: str, index_type: str) -> str:
    """e.g. results_clip_image_coco_hnsw_m32_efc200_efs128__0.json -> m32_efc200_efs128"""
    prefix = f"results_{encoder}_{dataset}_{index_type}_"
    name = os.path.basename(method_file)
    if not name.startswith(prefix):
        return name
    tag = name[len(prefix) :]
    if tag.endswith(".json"):
        tag = tag[:-5]
    if tag.endswith("__0"):
        tag = tag[:-3]
    return tag


def compute_rows_for_encoder_index(
    encoder: EncoderName,
    index_type: IndexType,
    top_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ann_root = results_dir(index_type.upper())

    for dataset in DATASETS:
        knn_file = os.path.join(
            results_dir("KNN"),
            result_filename(encoder, dataset, "knn", "flat"),
        )
        if not os.path.isfile(knn_file):
            print(f"[WARN] Missing KNN: {knn_file}")
            continue

        knn_rows = load_json(knn_file)
        knn_lists = image_paths_from_results(knn_rows)
        prefix = f"results_{encoder}_{dataset}_{index_type}_"

        if not os.path.isdir(ann_root):
            continue

        for fname in sorted(os.listdir(ann_root)):
            if not fname.startswith(prefix) or not fname.endswith(".json"):
                continue

            ann_path = os.path.join(ann_root, fname)
            ann_rows = load_json(ann_path)
            if len(ann_rows) != len(knn_rows):
                print(f"[WARN] Query mismatch {fname}: knn={len(knn_rows)} ann={len(ann_rows)}")
                continue

            ann_lists = image_paths_from_results(ann_rows)
            accs: List[float] = []
            counts: List[int] = []
            jaccards: List[float] = []

            for kp, ap in zip(knn_lists, ann_lists):
                acc, cnt, jac = overlap_stats(kp, ap, top_n)
                accs.append(acc)
                counts.append(cnt)
                jaccards.append(jac)

            nq = len(accs)
            cfg_tag = extract_config_tag(fname, encoder, dataset, index_type)
            rows.append(
                {
                    "encoder": encoder,
                    "dataset": dataset,
                    "index_type": index_type,
                    "config": cfg_tag,
                    "ann_method_file": fname,
                    "knn_reference": os.path.basename(knn_file),
                    "top_n": top_n,
                    "num_queries": nq,
                    "mean_overlap_accuracy": sum(accs) / nq if nq else 0.0,
                    "mean_overlap_count": sum(counts) / nq if nq else 0.0,
                    "mean_jaccard_topn": sum(jaccards) / nq if nq else 0.0,
                    "min_overlap_accuracy": min(accs) if accs else 0.0,
                    "max_overlap_accuracy": max(accs) if accs else 0.0,
                }
            )
            print(
                f"  {encoder}/{dataset}/{index_type}/{cfg_tag}: "
                f"acc={rows[-1]['mean_overlap_accuracy']:.4f} "
                f"count={rows[-1]['mean_overlap_count']:.1f}"
            )
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[SKIP] No rows for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=ENCODERS + ["all"], default="all")
    ap.add_argument("--index-type", choices=INDEX_TYPES + ["all"], default="all")
    ap.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--output-dir", default=RESULT_DIR)
    args = ap.parse_args()

    encoders = ENCODERS if args.encoder == "all" else [args.encoder]
    index_types = INDEX_TYPES if args.index_type == "all" else [args.index_type]
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for encoder in encoders:
        for index_type in index_types:
            print(f"\n=== {encoder} | KNN vs {index_type.upper()} ===")
            rows = compute_rows_for_encoder_index(encoder, index_type, args.top_n)
            all_rows.extend(rows)
            out_name = f"{encoder}_{index_type}_overlap.csv"
            write_csv(os.path.join(args.output_dir, out_name), rows)

    write_csv(os.path.join(args.output_dir, "overlap_summary_all.csv"), all_rows)
    print(f"\n[DONE] CSVs written to {args.output_dir}")


if __name__ == "__main__":
    main()
