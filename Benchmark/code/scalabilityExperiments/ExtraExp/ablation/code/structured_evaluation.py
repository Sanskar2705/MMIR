#!/usr/bin/env python3
"""
Structured top-n overlap evaluation vs flat KNN reference.

Supports both:
  - legacy flat files directly under `results/`
  - structured files under:
      results/hnsw/efsearch_*/clip_hnsw_*.json
      results/ivfpq/nprobe_*/clip_ivfpq_*.json

Outputs CSVs with explicit columns to distinguish:
  - index type (`hnsw`, `ivfpq`, `flat`)
  - search parameter source folder (`efsearch_256`, `nprobe_64`, ...)
  - base filename and a stable run_id
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class MethodRun:
    path: str
    method_file: str
    index_type: str
    search_setting: str
    search_param_name: str
    search_param_value: Optional[int]
    run_id: str


def _image_paths(row: Mapping[str, Any]) -> List[str]:
    items = row.get("result_of_top_100") or []
    out: List[str] = []
    for it in items:
        p = it.get("image_path")
        if p is not None:
            out.append(str(p))
    return out


def load_retrieval_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}")
    rows: List[Dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def align_by_query_id(
    knn_rows: Sequence[Mapping[str, Any]], other_rows: Sequence[Mapping[str, Any]]
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    by_id: Dict[int, Mapping[str, Any]] = {}
    for r in other_rows:
        by_id[int(r["query_id"])] = r

    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    missing: List[int] = []
    for r in knn_rows:
        qid = int(r["query_id"])
        o = by_id.get(qid)
        if o is None:
            missing.append(qid)
        else:
            pairs.append((r, o))
    if missing:
        raise ValueError(f"Missing query_ids in candidate file: {len(missing)} (e.g. {missing[:5]})")
    return pairs


def top_n_slice(paths: Sequence[str], n: int) -> List[str]:
    return list(paths[:n])


def rank_map(ordered: Sequence[str]) -> Dict[str, int]:
    return {p: i + 1 for i, p in enumerate(ordered)}


def query_overlap_detail(
    knn_paths: Sequence[str], method_paths: Sequence[str], n: int
) -> Tuple[float, int, int, List[Tuple[str, int, int]]]:
    """
    Returns (accuracy, overlap_count, k_method, list of (image_path, rank_knn, rank_method)).
    Accuracy denominator is fixed at `n`.
    """
    if n < 1:
        return 0.0, 0, 0, []
    knn_take = min(n, len(knn_paths))
    m_take = min(n, len(method_paths))
    if knn_take < 1 or m_take < 1:
        return 0.0, 0, m_take, []

    knn_n = top_n_slice(knn_paths, knn_take)
    m_n = top_n_slice(method_paths, m_take)
    rk = rank_map(knn_n)
    rm = rank_map(m_n)
    common: Set[str] = set(knn_n) & set(m_n)

    overlap_count = len(common)
    accuracy = overlap_count / float(n)

    detail: List[Tuple[str, int, int]] = []
    for img in sorted(common, key=lambda x: rk[x]):
        detail.append((img, rk[img], rm[img]))
    return accuracy, overlap_count, m_take, detail


def default_results_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))


def default_evaluation_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))


def _split_param(setting_name: str) -> Tuple[str, Optional[int]]:
    if "_" not in setting_name:
        return "setting", None
    key, value = setting_name.split("_", 1)
    try:
        return key, int(value)
    except ValueError:
        return key, None


def _make_run(results_dir: str, path: str, index_type: str, search_setting: str) -> MethodRun:
    method_file = os.path.basename(path)
    rel = os.path.relpath(path, results_dir)
    param_name, param_value = _split_param(search_setting)
    run_id = f"{index_type}/{search_setting}/{method_file}"
    return MethodRun(
        path=path,
        method_file=method_file,
        index_type=index_type,
        search_setting=search_setting,
        search_param_name=param_name,
        search_param_value=param_value,
        run_id=run_id if rel != method_file else f"{index_type}/root/{method_file}",
    )


def discover_method_runs(results_dir: str, knn_basename: str) -> List[MethodRun]:
    runs: List[MethodRun] = []
    seen: Set[str] = set()
    skip = {knn_basename, "fixed_queries_1000.json"}

    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if os.path.isfile(path) and name.endswith(".json"):
            if name in skip:
                continue
            if not (name.startswith("clip_hnsw_") or name.startswith("clip_ivfpq_")):
                continue
            index_type = "hnsw" if name.startswith("clip_hnsw_") else "ivfpq"
            run = _make_run(results_dir, path, index_type=index_type, search_setting="root")
            if run.run_id not in seen:
                runs.append(run)
                seen.add(run.run_id)

    for index_type in ("hnsw", "ivfpq"):
        base = os.path.join(results_dir, index_type)
        if not os.path.isdir(base):
            continue
        for setting in sorted(os.listdir(base)):
            setting_dir = os.path.join(base, setting)
            if not os.path.isdir(setting_dir):
                continue
            for method_file in sorted(os.listdir(setting_dir)):
                if not method_file.endswith(".json"):
                    continue
                if method_file in skip:
                    continue
                if index_type == "hnsw" and not method_file.startswith("clip_hnsw_"):
                    continue
                if index_type == "ivfpq" and not method_file.startswith("clip_ivfpq_"):
                    continue
                path = os.path.join(setting_dir, method_file)
                run = _make_run(results_dir, path, index_type=index_type, search_setting=setting)
                if run.run_id not in seen:
                    runs.append(run)
                    seen.add(run.run_id)
    return runs


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Structured top-n overlap accuracy vs KNN")
    p.add_argument("--results-dir", default=default_results_dir(), help="Results directory")
    p.add_argument("--knn-file", default="clip_image_flat.json", help="KNN baseline JSON basename")
    p.add_argument("--top-n", type=int, default=100, help="Prefix size N for overlap")
    p.add_argument("--evaluation-dir", default=default_evaluation_dir(), help="CSV output directory")
    p.add_argument(
        "--accuracy-csv",
        default="top100_overlap_accuracy_structured.csv",
        help="Per-run summary CSV basename",
    )
    p.add_argument(
        "--ranks-csv",
        default="top100_overlap_ranks_structured.csv",
        help="Per-query overlap + rank CSV basename",
    )
    p.add_argument(
        "--index-types",
        default="",
        help="Optional comma-separated filter for index_type (e.g. 'ivfpq' or 'hnsw,ivfpq').",
    )
    p.add_argument("--no-save", action="store_true", help="Print summary only; skip CSV writing")
    args = p.parse_args(list(argv) if argv is not None else None)

    n = args.top_n
    if n < 1:
        raise ValueError("--top-n must be >= 1")

    results_dir = os.path.abspath(args.results_dir)
    knn_path = os.path.join(results_dir, args.knn_file)
    if not os.path.isfile(knn_path):
        raise FileNotFoundError(knn_path)

    knn_rows = load_retrieval_json(knn_path)
    min_knn = min(len(_image_paths(r)) for r in knn_rows)
    if min_knn < n:
        raise ValueError(
            f"Every KNN list must have at least {n} results (shortest has {min_knn})"
        )

    runs = discover_method_runs(results_dir, args.knn_file)
    if not runs:
        raise FileNotFoundError(
            f"No method JSON files discovered under {results_dir} (hnsw/ivfpq, root or nested)"
        )

    type_filter_raw = (args.index_types or "").strip()
    if type_filter_raw:
        allowed = {t.strip().lower() for t in type_filter_raw.split(",") if t.strip()}
        runs = [r for r in runs if r.index_type.lower() in allowed]
        if not runs:
            raise FileNotFoundError(
                f"No runs left after --index-types filter {allowed!r} under {results_dir}"
            )

    eval_dir = os.path.abspath(args.evaluation_dir)
    accuracy_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []

    for run in runs:
        cand_rows = load_retrieval_json(run.path)
        pairs = align_by_query_id(knn_rows, cand_rows)

        acc_sum = 0.0
        overlap_sum = 0
        m_take_sum = 0

        for knn_row, approx_row in pairs:
            qid = int(knn_row["query_id"])
            knn_p = _image_paths(knn_row)
            approx_p = _image_paths(approx_row)
            accuracy, overlap_count, m_take, detail = query_overlap_detail(knn_p, approx_p, n)

            acc_sum += accuracy
            overlap_sum += overlap_count
            m_take_sum += m_take

            for img, rk, rm in detail:
                rank_rows.append(
                    {
                        "query_id": qid,
                        "run_id": run.run_id,
                        "index_type": run.index_type,
                        "search_setting": run.search_setting,
                        "search_param_name": run.search_param_name,
                        "search_param_value": run.search_param_value,
                        "method_file": run.method_file,
                        "image_path": img,
                        "rank_knn": rk,
                        "rank_method": rm,
                    }
                )

        num_q = len(pairs)
        mean_acc = acc_sum / num_q
        mean_overlap = overlap_sum / num_q
        mean_prefix = m_take_sum / num_q
        accuracy_rows.append(
            {
                "run_id": run.run_id,
                "index_type": run.index_type,
                "search_setting": run.search_setting,
                "search_param_name": run.search_param_name,
                "search_param_value": run.search_param_value,
                "method_file": run.method_file,
                "top_n": n,
                "num_queries": num_q,
                "mean_accuracy": mean_acc,
                "mean_overlap_count": mean_overlap,
                "mean_method_prefix_len": mean_prefix,
            }
        )
        print(
            f"{run.run_id}  mean_accuracy={mean_acc:.6f}  "
            f"mean|overlap|={mean_overlap:.4f}  mean_method_prefix_len={mean_prefix:.2f}"
        )

    if not args.no_save:
        os.makedirs(eval_dir, exist_ok=True)

        acc_path = os.path.join(eval_dir, args.accuracy_csv)
        with open(acc_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "index_type",
                    "search_setting",
                    "search_param_name",
                    "search_param_value",
                    "method_file",
                    "top_n",
                    "num_queries",
                    "mean_accuracy",
                    "mean_overlap_count",
                    "mean_method_prefix_len",
                ],
            )
            w.writeheader()
            w.writerows(accuracy_rows)
        print(f"\nWrote {acc_path}")

        ranks_path = os.path.join(eval_dir, args.ranks_csv)
        with open(ranks_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "query_id",
                    "run_id",
                    "index_type",
                    "search_setting",
                    "search_param_name",
                    "search_param_value",
                    "method_file",
                    "image_path",
                    "rank_knn",
                    "rank_method",
                ],
            )
            w.writeheader()
            w.writerows(rank_rows)
        print(f"Wrote {ranks_path}  ({len(rank_rows)} rows)")


if __name__ == "__main__":
    main()
