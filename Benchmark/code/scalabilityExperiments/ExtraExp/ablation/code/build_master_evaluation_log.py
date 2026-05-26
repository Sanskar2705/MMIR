#!/usr/bin/env python3
"""
Build master ablation log: all runs under results/ (root, hnsw/, ivfpq/) with
top-N overlap vs flat KNN and parsed index/search configuration.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Must match defaults in code/batch_retrieve_ablation.py:run_index_retrieval when
# CLI does not pass --nprobe / --efsearch (i.e. JSON files directly under results/).
DEFAULT_IVFPQ_NPROBE = int(os.getenv("MASTER_LOG_DEFAULT_IVFPQ_NPROBE", "16"))
DEFAULT_HNSW_EFSEARCH = int(os.getenv("MASTER_LOG_DEFAULT_HNSW_EFSEARCH", "128"))


def parse_hnsw_index(method_file: str) -> str:
    m = re.match(r"clip_hnsw_m(\d+)_efc(\d+)\.json", method_file)
    if not m:
        return ""
    return f"M={m.group(1)}, efConstruction={m.group(2)}"


def parse_ivfpq_standard(method_file: str) -> str:
    m = re.match(r"clip_ivfpq_nlist(\d+)_m(\d+)_nbits(\d+)\.json", method_file)
    if not m:
        return ""
    return f"nlist={m.group(1)}, m={m.group(2)}, nbits={m.group(3)}"


def parse_ivfpq_factory(method_file: str) -> str:
    m = re.match(r"clip_ivfpq_fac_opq(\d+)_ivf(\d+)_pq(\d+)\.json", method_file)
    if not m:
        return ""
    return f"index_factory OPQ{m.group(1)},IVF{m.group(2)},PQ{m.group(3)}"


def index_build_notes(index_type: str, method_file: str) -> str:
    if index_type == "hnsw":
        return parse_hnsw_index(method_file) or "unknown HNSW"
    if "fac_opq" in method_file:
        return parse_ivfpq_factory(method_file) or "unknown factory IVFPQ"
    return parse_ivfpq_standard(method_file) or "unknown IVFPQ"


def retrieval_notes(index_type: str, row: Dict[str, Any]) -> str:
    setting = row.get("search_setting") or ""
    if setting == "root":
        if index_type == "hnsw":
            return "default efSearch (HNSW_EFSEARCH env or 128)"
        return "default nprobe (IVFPQ_NPROBE env or 16)"
    param = row.get("search_param_name") or ""
    val = row.get("search_param_value")
    if val is not None and val != "":
        return f"{param}={val}"
    return setting


def ivfpq_tier(method_file: str) -> str:
    if "fac_opq" in method_file:
        return "factory_opq"
    if re.search(r"nlist(?:16384|32768)_m96", method_file):
        return "standard_high_nlist"
    return "standard_grid"


def ivfpq_nlist_from_method_file(method_file: str) -> Optional[int]:
    """IVF nlist encoded in result filename (classic IVF-PQ or factory name)."""
    m = re.match(r"clip_ivfpq_nlist(\d+)_m\d+_nbits\d+\.json", method_file)
    if m:
        return int(m.group(1))
    m = re.match(r"clip_ivfpq_fac_opq\d+_ivf(\d+)_pq\d+\.json", method_file)
    if m:
        return int(m.group(1))
    return None


def effective_ivfpq_nprobe(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (effective_nprobe_str, note).

    - root runs: batch_retrieve_ablation default (IVFPQ_NPROBE env or 16). JSON does
      not store nprobe; not reproducible as a swept folder because this study starts
      at nprobe_32 — roots are strictly below nprobe_32 in overlap for all standard
      indexes, matching default 16.
    - nprobe_* folders: folder name is the *requested* probe count; Faiss uses
      min(requested, nlist) (see batch_retrieve_ablation_sweep.py).
    """
    setting = row.get("search_setting") or ""
    mf = row.get("method_file") or ""
    nlist = ivfpq_nlist_from_method_file(mf)

    if setting == "root":
        return (
            str(DEFAULT_IVFPQ_NPROBE),
            f"default from batch_retrieve_ablation.py: IVFPQ_NPROBE env or {DEFAULT_IVFPQ_NPROBE} "
            "(not in JSON; inferred from code defaults + overlap < nprobe_32 for standard indexes)",
        )

    raw = row.get("search_param_value")
    if raw is None or str(raw).strip() == "":
        return ("", "missing search_param_value")
    try:
        requested = int(raw)
    except ValueError:
        return ("", f"non-integer search_param_value={raw!r}")

    if nlist is None:
        return (str(requested), "nlist not parsed from filename; using requested nprobe")

    eff = min(requested, nlist)
    if eff != requested:
        return (str(eff), f"effective=min(requested={requested}, nlist={nlist}) per sweep script")
    return (str(eff), "effective equals requested (requested <= nlist)")


def effective_hnsw_efsearch(row: Dict[str, Any]) -> Tuple[str, str]:
    """Root HNSW: HNSW_EFSEARCH env or 128 (batch_retrieve_ablation.py)."""
    setting = row.get("search_setting") or ""
    if setting == "root":
        return (
            str(DEFAULT_HNSW_EFSEARCH),
            f"default from batch_retrieve_ablation.py: HNSW_EFSEARCH env or {DEFAULT_HNSW_EFSEARCH} "
            "(not in JSON)",
        )
    raw = row.get("search_param_value")
    if raw is None or str(raw).strip() == "":
        return ("", "missing search_param_value")
    return (str(int(raw)), "from results/hnsw/efsearch_* folder")


def load_accuracy_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def enrich_row(row: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    index_type = row["index_type"]
    method_file = row["method_file"]
    run_id = row["run_id"]
    if row.get("search_setting") == "root":
        result_path = os.path.join(results_dir, method_file)
    else:
        result_path = os.path.join(results_dir, *run_id.split("/"))
    out = dict(row)
    out["result_json"] = result_path
    out["index_build"] = index_build_notes(index_type, method_file)
    out["ivfpq_tier"] = ivfpq_tier(method_file) if index_type == "ivfpq" else ""

    if index_type == "ivfpq":
        nprobe_s, nprobe_note = effective_ivfpq_nprobe(row)
        out["effective_ivfpq_nprobe"] = nprobe_s
        out["effective_search_note"] = nprobe_note
        out["effective_hnsw_efsearch"] = ""
        if row.get("search_setting") == "root":
            out["retrieval"] = (
                f"nprobe={nprobe_s} (IVFPQ_NPROBE env or {DEFAULT_IVFPQ_NPROBE}; see effective_search_note)"
            )
        else:
            if nprobe_s:
                out["retrieval"] = f"nprobe={nprobe_s} ({nprobe_note})"
            else:
                out["retrieval"] = retrieval_notes(index_type, row)
    else:
        ef_s, ef_note = effective_hnsw_efsearch(row)
        out["effective_hnsw_efsearch"] = ef_s
        out["effective_ivfpq_nprobe"] = ""
        out["effective_search_note"] = ef_note
        if row.get("search_setting") == "root":
            out["retrieval"] = (
                f"efSearch={ef_s} (HNSW_EFSEARCH env or {DEFAULT_HNSW_EFSEARCH}; see effective_search_note)"
            )
        else:
            if ef_s:
                out["retrieval"] = f"efsearch={ef_s} ({ef_note})"
            else:
                out["retrieval"] = retrieval_notes(index_type, row)

    out["mean_accuracy_pct"] = f"{100.0 * float(row['mean_accuracy']):.2f}%"
    return out


def best_row(rows: List[Dict[str, Any]], pred) -> Optional[Dict[str, Any]]:
    filtered = [r for r in rows if pred(r)]
    if not filtered:
        return None
    return max(filtered, key=lambda r: float(r["mean_accuracy"]))


def write_master_log(
    rows: List[Dict[str, Any]],
    out_path: str,
    *,
    results_dir: str,
    knn_file: str,
    top_n: int,
    generated_at: str,
) -> None:
    hnsw_rows = [r for r in rows if r["index_type"] == "hnsw"]
    ivfpq_rows = [r for r in rows if r["index_type"] == "ivfpq"]

    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("LAION ABLATION — MASTER EVALUATION LOG")
    lines.append("=" * 88)
    lines.append(f"Generated (UTC):     {generated_at}")
    lines.append(f"Results root:        {results_dir}")
    lines.append(f"KNN reference:       {os.path.join(results_dir, knn_file)}")
    lines.append(f"Queries:             1000 (fixed_queries_1000.json)")
    lines.append(f"Metric top_n:        {top_n}")
    lines.append(
        "Overlap accuracy:    |intersection(KNN top-N, method top-N)| / N  (mean over queries)"
    )
    lines.append(f"Total runs logged:   {len(rows)}")
    lines.append("")
    lines.append(
        "Root JSONs (under results/): IVFPQ nprobe defaults to "
        f"{DEFAULT_IVFPQ_NPROBE} and HNSW efSearch to {DEFAULT_HNSW_EFSEARCH} in "
        "code/batch_retrieve_ablation.py when env vars are unset. Result JSON files "
        "do not record these values — see effective_ivfpq_nprobe / effective_hnsw_efsearch in master_ablation_overlap.csv."
    )
    lines.append("")

    lines.append("-" * 88)
    lines.append("EXECUTIVE SUMMARY (best mean overlap accuracy per family)")
    lines.append("-" * 88)

    summaries: List[Tuple[str, Optional[Dict[str, Any]]]] = [
        ("HNSW — best overall (any efSearch)", best_row(hnsw_rows, lambda r: True)),
        (
            "HNSW — best root/default search only",
            best_row(hnsw_rows, lambda r: r["search_setting"] == "root"),
        ),
        (
            "IVFPQ standard grid — best overall",
            best_row(
                ivfpq_rows,
                lambda r: r["ivfpq_tier"] == "standard_grid" and r["search_setting"] != "root",
            ),
        ),
        (
            "IVFPQ standard grid — best root/default",
            best_row(
                ivfpq_rows,
                lambda r: r["ivfpq_tier"] == "standard_grid" and r["search_setting"] == "root",
            ),
        ),
        (
            "IVFPQ high-nlist (16384/32768, m96) — best overall",
            best_row(ivfpq_rows, lambda r: r["ivfpq_tier"] == "standard_high_nlist"),
        ),
        (
            "IVFPQ factory OPQ — best overall",
            best_row(ivfpq_rows, lambda r: r["ivfpq_tier"] == "factory_opq"),
        ),
    ]

    for title, row in summaries:
        lines.append(f"\n{title}")
        if row is None:
            lines.append("  (no runs)")
            continue
        lines.append(f"  mean_accuracy:     {row['mean_accuracy']} ({row['mean_accuracy_pct']})")
        lines.append(f"  mean|overlap|:      {row['mean_overlap_count']}")
        lines.append(f"  run_id:            {row['run_id']}")
        lines.append(f"  index build:       {row['index_build']}")
        lines.append(f"  retrieval:         {row['retrieval']}")
        lines.append(f"  result JSON:       {row['result_json']}")

    lines.append("")
    lines.append("-" * 88)
    lines.append("INDEX BUILD CONFIGURATIONS PRESENT IN THIS STUDY")
    lines.append("-" * 88)
    lines.append("")
    lines.append("HNSW (IndexHNSWFlat, build_ablation_indexes.py + hnsw_best extras):")
    hnsw_indexes = sorted({r["method_file"] for r in hnsw_rows})
    for mf in hnsw_indexes:
        lines.append(f"  - {mf}  →  {parse_hnsw_index(mf)}")
    lines.append("")
    lines.append("IVFPQ standard grid (build_ablation_indexes.py):")
    for mf in sorted({r["method_file"] for r in ivfpq_rows if r["ivfpq_tier"] == "standard_grid"}):
        lines.append(f"  - {mf}  →  {parse_ivfpq_standard(mf)}")
    lines.append("")
    lines.append("IVFPQ high-nlist:")
    for mf in sorted(
        {r["method_file"] for r in ivfpq_rows if r["ivfpq_tier"] == "standard_high_nlist"}
    ):
        lines.append(f"  - {mf}  →  {parse_ivfpq_standard(mf)}")
    lines.append("")
    lines.append("IVFPQ factory (build_high_recall_indexes.py):")
    for mf in sorted({r["method_file"] for r in ivfpq_rows if r["ivfpq_tier"] == "factory_opq"}):
        lines.append(f"  - {mf}  →  {parse_ivfpq_factory(mf)}")

    lines.append("")
    lines.append("-" * 88)
    lines.append("RETRIEVAL SWEEPS (search parameters on disk)")
    lines.append("-" * 88)
    hnsw_settings = sorted(
        {r["search_setting"] for r in hnsw_rows},
        key=lambda s: (s != "root", s),
    )
    ivfpq_settings = sorted(
        {r["search_setting"] for r in ivfpq_rows},
        key=lambda s: (s != "root", s),
    )
    lines.append(f"HNSW efSearch folders: {', '.join(hnsw_settings)}")
    lines.append(f"IVFPQ nprobe folders:  {', '.join(ivfpq_settings)}")
    lines.append("Note: effective nprobe = min(requested nprobe, index nlist) in sweep script.")

    lines.append("")
    lines.append("-" * 88)
    lines.append("ALL RUNS (sorted by index_type, search_setting, mean_accuracy desc)")
    lines.append("-" * 88)

    def sort_key(r: Dict[str, Any]) -> Tuple:
        ef = r.get("search_param_value") or 0
        np_ = r.get("search_param_value") or 0
        sp = int(ef) if r["index_type"] == "hnsw" and r.get("search_param_value") else int(np_) if r.get("search_param_value") else 0
        return (
            r["index_type"],
            r["search_setting"],
            -float(r["mean_accuracy"]),
            r["method_file"],
        )

    for r in sorted(rows, key=sort_key):
        lines.append("")
        lines.append(
            f"[{r['index_type'].upper()}] {r['run_id']}  "
            f"accuracy={r['mean_accuracy']} ({r['mean_accuracy_pct']})  "
            f"overlap={r['mean_overlap_count']}"
        )
        lines.append(f"  index:     {r['index_build']}")
        lines.append(f"  search:    {r['retrieval']}")
        lines.append(f"  json:      {r['result_json']}")

    lines.append("")
    lines.append("=" * 88)
    lines.append("END OF MASTER LOG")
    lines.append("=" * 88)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_enriched_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    fieldnames = [
        "run_id",
        "index_type",
        "search_setting",
        "search_param_name",
        "search_param_value",
        "effective_ivfpq_nprobe",
        "effective_hnsw_efsearch",
        "effective_search_note",
        "method_file",
        "ivfpq_tier",
        "index_build",
        "retrieval",
        "top_n",
        "num_queries",
        "mean_accuracy",
        "mean_accuracy_pct",
        "mean_overlap_count",
        "mean_method_prefix_len",
        "result_json",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    p = argparse.ArgumentParser(description="Build master ablation evaluation log")
    p.add_argument(
        "--accuracy-csv",
        default=os.path.join(base, "evaluation", "final", "master_overlap_accuracy.csv"),
    )
    p.add_argument(
        "--results-dir",
        default=os.path.join(base, "results"),
    )
    p.add_argument("--knn-file", default="clip_image_flat.json")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument(
        "--out-dir",
        default=os.path.join(base, "evaluation", "final"),
    )
    args = p.parse_args()

    if not os.path.isfile(args.accuracy_csv):
        raise FileNotFoundError(
            f"Missing {args.accuracy_csv}; run structured_evaluation.py first."
        )

    raw = load_accuracy_csv(args.accuracy_csv)
    results_dir = os.path.abspath(args.results_dir)
    rows = [enrich_row(r, results_dir) for r in raw]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log_path = os.path.join(out_dir, "master_ablation_overlap.log")
    csv_path = os.path.join(out_dir, "master_ablation_overlap.csv")

    write_master_log(
        rows,
        log_path,
        results_dir=results_dir,
        knn_file=args.knn_file,
        top_n=args.top_n,
        generated_at=generated_at,
    )
    write_enriched_csv(rows, csv_path)

    print(f"Wrote {log_path}  ({len(rows)} runs)")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
