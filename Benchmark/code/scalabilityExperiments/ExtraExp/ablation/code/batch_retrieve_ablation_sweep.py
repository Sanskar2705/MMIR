#!/usr/bin/env python3
"""
Sweep IVFPQ nprobe and HNSW efSearch, writing results under a nested layout:

  results_dir/hnsw/efsearch_<ef>/clip_hnsw_<...>.json
  results_dir/ivfpq/nprobe_<np>/clip_ivfpq_<...>.json

Skips by default (already completed in the flat run with env defaults):
  - HNSW: efSearch == 128
  - IVFPQ: nprobe == 16

Uses ``run_index_retrieval`` from ``batch_retrieve_ablation.py`` (same encoder and JSON shape).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from batch_retrieve_ablation import (  # noqa: E402
    ClipTextEncoder,
    get_matching_names_path,
    list_indexes,
    load_or_build_queries,
    run_index_retrieval,
)


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    return out


def _under_root(path: str, root: str) -> bool:
    ap = os.path.abspath(path)
    ar = os.path.abspath(root)
    return ap == ar or ap.startswith(ar + os.sep)


def split_index_paths(
    index_paths: Sequence[str], hnsw_dir: str, ivfpq_dir: str
) -> Tuple[List[str], List[str]]:
    hnsw: List[str] = []
    ivfpq: List[str] = []
    for p in index_paths:
        if _under_root(p, hnsw_dir) or os.path.basename(p).startswith("clip_hnsw_"):
            hnsw.append(p)
        elif _under_root(p, ivfpq_dir) or os.path.basename(p).startswith("clip_ivfpq_"):
            ivfpq.append(p)
        else:
            raise ValueError(f"Index path not under hnsw/ivfpq dirs: {p}")
    return sorted(hnsw), sorted(ivfpq)


def _extract_ivfpq_nlist(index_path: str) -> int | None:
    base = os.path.basename(index_path)
    m = re.search(r"nlist(\d+)", base)
    if m:
        return int(m.group(1))
    ivf_nums = [int(x) for x in re.findall(r"_ivf(\d+)", base, flags=re.IGNORECASE)]
    if ivf_nums:
        return max(ivf_nums)
    ivf_nums2 = [int(x) for x in re.findall(r"ivf(\d+)", base, flags=re.IGNORECASE)]
    if ivf_nums2:
        return max(ivf_nums2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep nprobe (IVFPQ) and efSearch (HNSW) into hnsw/efsearch_* and ivfpq/nprobe_*."
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
        "--hnsw-efsearch",
        default="256,512,1024,2048",
        help="Comma-separated efSearch values for every HNSW index.",
    )
    ap.add_argument(
        "--ivfpq-nprobe",
        default="512,1024,2048,4096,8192,16384,32768",
        help="Comma-separated nprobe values for every IVFPQ index.",
    )
    ap.add_argument(
        "--run-hnsw",
        action="store_true",
        help="Include HNSW retrieval jobs in this sweep.",
    )
    ap.add_argument(
        "--run-ivfpq",
        action="store_true",
        help="Include IVFPQ retrieval jobs in this sweep.",
    )
    ap.add_argument(
        "--skip-hnsw-efsearch",
        default="128",
        help="Do not run HNSW jobs with this efSearch (comma-separated). Default skips completed 128.",
    )
    ap.add_argument(
        "--skip-ivfpq-nprobe",
        default="16",
        help="Do not run IVFPQ jobs with this nprobe (comma-separated). Default skips completed 16.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JSON outputs (default: skip if file exists).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned paths only; do not load model or run retrieval.",
    )
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    hnsw_dir = os.path.abspath(args.hnsw_dir)
    ivfpq_dir = os.path.abspath(args.ivfpq_dir)
    os.makedirs(os.path.join(results_dir, "hnsw"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "ivfpq"), exist_ok=True)

    hnsw_ef_list = _parse_int_list(args.hnsw_efsearch)
    ivfpq_np_list = _parse_int_list(args.ivfpq_nprobe)
    skip_ef = set(_parse_int_list(args.skip_hnsw_efsearch))
    skip_np = set(_parse_int_list(args.skip_ivfpq_nprobe))

    index_paths = list_indexes([args.ivfpq_dir, args.hnsw_dir])
    hnsw_indexes, ivfpq_indexes = split_index_paths(index_paths, hnsw_dir, ivfpq_dir)

    jobs: List[Tuple[str, str, dict]] = []
    # (label, out_path, kwargs for run_index_retrieval extras as dict with index_path later)

    run_hnsw = bool(args.run_hnsw)
    run_ivfpq = bool(args.run_ivfpq)
    if not run_hnsw and not run_ivfpq:
        # Backward-compatible behavior: run both if no selector is provided.
        run_hnsw = True
        run_ivfpq = True

    if run_hnsw:
        for index_path in hnsw_indexes:
            base = os.path.basename(index_path).replace(".index", ".json")
            for ef in hnsw_ef_list:
                if ef in skip_ef:
                    continue
                sub = os.path.join(results_dir, "hnsw", f"efsearch_{ef}", base)
                jobs.append(("hnsw", sub, {"index_path": index_path, "ef_search": ef, "nprobe": None}))

    if run_ivfpq:
        for index_path in ivfpq_indexes:
            base = os.path.basename(index_path).replace(".index", ".json")
            nlist = _extract_ivfpq_nlist(index_path)
            for np_ in ivfpq_np_list:
                if np_ in skip_np:
                    continue
                effective_np = min(np_, nlist) if nlist is not None else np_
                sub = os.path.join(results_dir, "ivfpq", f"nprobe_{np_}", base)
                jobs.append(
                    (
                        "ivfpq",
                        sub,
                        {"index_path": index_path, "nprobe": effective_np, "ef_search": None},
                    )
                )

    print(f"[INFO] Planned jobs: {len(jobs)}")
    for kind, out_path, meta in jobs:
        print(f"  {kind}  ef={meta.get('ef_search')} nprobe={meta.get('nprobe')} -> {out_path}")

    if args.dry_run:
        return

    queries = load_or_build_queries(
        queries_file=args.queries_file, tar_dir=args.tar_dir, max_queries=int(args.max_queries)
    )
    encoder = ClipTextEncoder(model_name=args.clip_model_name, device=args.device)

    for kind, out_path, meta in jobs:
        index_path = meta["index_path"]
        if not args.force and os.path.isfile(out_path):
            print(f"[SKIP] exists {out_path}")
            continue

        names_path = get_matching_names_path(index_path)
        with open(names_path, "r") as f:
            names = json.load(f)
        if not isinstance(names, list):
            raise ValueError(f"Names file is not list: {names_path}")

        print(f"[START] {kind} {os.path.basename(index_path)} -> {out_path}")
        rows = run_index_retrieval(
            index_path=index_path,
            names=names,
            queries=queries,
            encoder=encoder,
            top_k=int(args.top_k),
            encode_batch_size=int(args.encode_batch_size),
            nprobe=meta.get("nprobe"),
            ef_search=meta.get("ef_search"),
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
