#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

"""
Direct batched BLIP2-ITM reranking for ANN JSONs (no HTTP service).

Input:  HNSW/IVFPQ/OIVFPQ/*.json  (list_of_top_k, 100 candidates)
Output: Reranker/{HNSW,IVFPQ,OIVFPQ}/*_rerank-blip2.json
        fields: query, list_of_top_k_rerank

Jobs from: Result/overlap_summary_all.csv (ann_method_file column)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.code.ANN.Code.configs import ANN_ROOT

def _load_blip2():
    from Benchmark.code.ANN.Code.blip2_loader import load_blip2_itm
    return load_blip2_itm


def _index_type_to_folder(index_type: str) -> str:
    m = index_type.lower().strip()
    if m == "hnsw":
        return "HNSW"
    if m == "ivfpq":
        return "IVFPQ"
    if m in ("oivfpq", "o_ivfpq"):
        return "OIVFPQ"
    return m.upper()


def load_jobs(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def balance_coco_flickr_jobs(jobs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Interleave Flickr (1k queries) and COCO (5k) for steadier progress across workers."""
    coco = [j for j in jobs if j.get("dataset") == "coco"]
    flickr = [j for j in jobs if j.get("dataset") == "flickr"]
    out: List[Dict[str, str]] = []
    i = j = 0
    while i < len(coco) or j < len(flickr):
        if j < len(flickr):
            out.append(flickr[j])
            j += 1
        if i < len(coco):
            out.append(coco[i])
            i += 1
    return out


def input_path(ann_root: str, index_type: str, fname: str, results_subdir: str = "") -> str:
    folder = _index_type_to_folder(index_type)
    if results_subdir:
        return os.path.join(ann_root, folder, results_subdir, fname)
    return os.path.join(ann_root, folder, fname)


def output_path(
    ann_root: str,
    index_type: str,
    fname: str,
    output_suffix: str = "",
    reranker_output_root: str = "",
) -> str:
    folder = _index_type_to_folder(index_type)
    stem = fname[:-5] if fname.endswith(".json") else fname
    suf = output_suffix or ""
    root = reranker_output_root or os.path.join(ann_root, "Reranker")
    return os.path.join(root, folder, f"{stem}_rerank-blip2{suf}.json")


def standard_output_path(
    ann_root: str,
    index_type: str,
    fname: str,
    reranker_output_root: str = "",
) -> str:
    return output_path(
        ann_root, index_type, fname, output_suffix="", reranker_output_root=reranker_output_root
    )


def open_image(path: str) -> Optional[Image.Image]:
    try:
        if path and os.path.isfile(path):
            return Image.open(path).convert("RGB")
    except Exception:
        pass
    return None


class Blip2ITMReranker:
    """GPU-batched ITM scoring; one model load per process."""

    def __init__(
        self,
        device: str,
        image_batch_size: int = 32,
        use_fp16: bool = True,
        load_workers: int = 8,
    ):
        self.device = torch.device(device)
        self.image_batch_size = max(1, int(image_batch_size))
        self.load_workers = max(1, int(load_workers))
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        print(
            f"[BLIP2] Loading ITM on {self.device} "
            f"(image_batch={self.image_batch_size}, load_workers={self.load_workers})"
        )
        self.model, self.vis_processors, self.text_processors = _load_blip2()(str(self.device))
        self.model.eval()
        if self.use_fp16:
            self.model = self.model.half()

    def _load_candidate_image(
        self, cand: Dict[str, Any]
    ) -> Tuple[Optional[Image.Image], str]:
        p = cand.get("image_abs_path") or cand.get("image_path", "")
        return open_image(str(p)), str(p)

    @torch.inference_mode()
    def _score_same_query_images(
        self, query: str, images: List[Optional[Image.Image]]
    ) -> List[float]:
        """BLIP2 ITM expects one caption per image batch; all images share `query`."""
        scores = [0.0] * len(images)
        bs = self.image_batch_size

        for start in range(0, len(images), bs):
            chunk = images[start : start + bs]
            proc_imgs: List[Any] = []
            idx_map: List[int] = []

            for local_i, im in enumerate(chunk):
                if im is None:
                    continue
                proc_imgs.append(self.vis_processors["eval"](im))
                idx_map.append(start + local_i)

            if not proc_imgs:
                continue

            img_t = torch.stack(proc_imgs).to(self.device)
            if self.use_fp16:
                img_t = img_t.half()

            samples = {"image": img_t, "text_input": [query] * len(proc_imgs)}
            with torch.amp.autocast(
                "cuda", enabled=self.use_fp16, dtype=torch.float16
            ):
                logits = self.model(samples, match_head="itm").float()
                probs = F.softmax(logits, dim=1)[:, 1]

            for j, global_i in enumerate(idx_map):
                scores[global_i] = float(probs[j].item())

        return scores

    @torch.inference_mode()
    def score_aligned_pairs(
        self, queries: List[str], images: List[Optional[Image.Image]]
    ) -> List[float]:
        """Score (query_i, image_i) pairs; group by query for tokenizer compatibility."""
        n = len(images)
        scores = [0.0] * n
        if n == 0:
            return scores

        by_query: Dict[str, List[Tuple[int, Image.Image]]] = defaultdict(list)
        for i, (q, im) in enumerate(zip(queries, images)):
            if im is not None:
                by_query[q].append((i, im))

        for q, pairs in by_query.items():
            idxs = [p[0] for p in pairs]
            imgs = [p[1] for p in pairs]
            sub = self._score_same_query_images(q, imgs)
            for j, global_i in enumerate(idxs):
                scores[global_i] = sub[j]

        return scores

    @torch.inference_mode()
    def score_batch(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        scores = [0.0] * len(candidates)
        bs = self.image_batch_size

        for start in range(0, len(candidates), bs):
            chunk = candidates[start : start + bs]
            images: List[Optional[Image.Image]] = []
            idx_map: List[int] = []

            with ThreadPoolExecutor(max_workers=self.load_workers) as ex:
                loaded = list(ex.map(self._load_candidate_image, chunk))
            for local_i, (img, _) in enumerate(loaded):
                if img is not None:
                    images.append(img)
                    idx_map.append(start + local_i)

            if not images:
                continue

            queries = [query] * len(images)
            pair_scores = self.score_aligned_pairs(queries, images)
            for j, global_i in enumerate(idx_map):
                scores[global_i] = pair_scores[j]

        return scores

    def rerank_entries_batch(
        self, entries: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank multiple queries; parallel image load, one GPU caption per query."""
        results: List[Dict[str, Any]] = []
        for entry in entries:
            q = entry.get("query", "")
            items = [dict(c) for c in (entry.get("list_of_top_k") or [])[:top_k]]
            if not items:
                results.append({"query": q, "list_of_top_k_rerank": []})
                continue
            paths = [c.get("image_abs_path") or c.get("image_path", "") for c in items]
            with ThreadPoolExecutor(max_workers=self.load_workers) as ex:
                imgs = list(ex.map(lambda p: open_image(str(p)), paths))
            chunk_scores = self._score_same_query_images(q, imgs)
            for it, s in zip(items, chunk_scores):
                it["reranking_score"] = s
            out = sorted(
                items, key=lambda x: x.get("reranking_score", 0.0), reverse=True
            )
            for r, it in enumerate(out, start=1):
                it["rank"] = r
            results.append({"query": q, "list_of_top_k_rerank": out})
        return results

    def rerank_one_query(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        row = self.rerank_entries_batch(
            [{"query": query, "list_of_top_k": candidates}], top_k
        )
        return row[0]["list_of_top_k_rerank"] if row else []


def partial_path(out: str) -> str:
    return out + ".partial.json"


def load_partial(out: str) -> Tuple[List[Dict[str, Any]], int]:
    p = partial_path(out)
    if not os.path.isfile(p):
        return [], 0
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("output", []), int(data.get("completed", 0))


def save_partial(out: str, output: List[Dict[str, Any]], completed: int) -> None:
    p = partial_path(out)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"completed": completed, "output": output}, f)


def process_file(
    inp: str,
    out: str,
    reranker: Blip2ITMReranker,
    top_k: int,
    skip_existing: bool,
    promote_standard_path: str = "",
    reranker_output_root: str = "",
    query_batch_size: int = 1,
    checkpoint_every: int = 0,
    resume_partial: bool = True,
) -> None:
    if skip_existing and os.path.isfile(out):
        print(f"[SKIP] {out}")
        return
    if (
        skip_existing
        and promote_standard_path
        and os.path.isfile(promote_standard_path)
    ):
        print(f"[SKIP] standard exists {promote_standard_path}")
        return
    if not os.path.isfile(inp):
        print(f"[WARN] missing {inp}")
        return

    with open(inp, "r", encoding="utf-8") as f:
        rows = json.load(f)

    output: List[Dict[str, Any]] = []
    start_idx = 0
    if resume_partial:
        output, start_idx = load_partial(out)
        if start_idx:
            print(f"[RESUME] {Path(inp).name} from query {start_idx}/{len(rows)}")

    qbs = max(1, int(query_batch_size))
    ckpt = max(0, int(checkpoint_every))

    pbar = tqdm(
        total=len(rows),
        initial=start_idx,
        desc=Path(inp).name,
        leave=True,
    )
    i = start_idx
    while i < len(rows):
        batch = rows[i : i + qbs]
        if qbs > 1:
            chunk_out = reranker.rerank_entries_batch(batch, top_k)
        else:
            chunk_out = []
            for entry in batch:
                reranked = reranker.rerank_one_query(
                    entry.get("query", ""),
                    entry.get("list_of_top_k") or [],
                    top_k,
                )
                chunk_out.append(
                    {
                        "query": entry.get("query", ""),
                        "list_of_top_k_rerank": reranked,
                    }
                )
        output.extend(chunk_out)
        i += len(batch)
        pbar.update(len(batch))
        if ckpt and i % ckpt < qbs:
            save_partial(out, output, i)

    pbar.close()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVE] {out} ({len(output)} queries)")
    partial = partial_path(out)
    if os.path.isfile(partial):
        os.remove(partial)

    if promote_standard_path and not os.path.isfile(promote_standard_path):
        import shutil

        shutil.copy2(out, promote_standard_path)
        print(f"[PROMOTE] {promote_standard_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct BLIP2 batch rerank for ANN results")
    ap.add_argument("--csv", default=os.path.join(ANN_ROOT, "Result", "overlap_summary_all.csv"))
    ap.add_argument(
        "--jobs-json",
        default="",
        help="Optional JSON list of job rows (overrides --csv / --index-type filtering)",
    )
    ap.add_argument("--ann-root", default=ANN_ROOT)
    ap.add_argument("--index-type", choices=["hnsw", "ivfpq", "oivfpq", "all"], default="all")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--image-batch-size", type=int, default=64)
    ap.add_argument(
        "--query-batch-size",
        type=int,
        default=8,
        help="Rerank this many queries per GPU mega-batch (flattened image pairs)",
    )
    ap.add_argument(
        "--load-workers",
        type=int,
        default=8,
        help="Parallel threads for loading images from disk",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save .partial.json every N queries (0=off)",
    )
    ap.add_argument(
        "--no-resume-partial",
        dest="resume_partial",
        action="store_false",
        help="Do not resume from .partial.json",
    )
    ap.set_defaults(resume_partial=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--job-start", type=int, default=0)
    ap.add_argument("--job-end", type=int, default=-1)
    ap.add_argument(
        "--balance-coco-flickr",
        action="store_true",
        help="Interleave Flickr and COCO jobs (similar wall-clock per worker)",
    )
    ap.add_argument(
        "--reverse-order",
        action="store_true",
        help="Process jobs last-to-first (meet forward workers in the middle)",
    )
    ap.add_argument(
        "--output-suffix",
        default="",
        help="Extra suffix before .json, e.g. '_1' -> *_rerank-blip2_1.json",
    )
    ap.add_argument(
        "--promote-to-standard",
        action="store_true",
        help="After save, copy to *_rerank-blip2.json if that file is missing",
    )
    ap.add_argument(
        "--pending-only",
        action="store_true",
        help="Skip jobs whose standard *_rerank-blip2.json output already exists",
    )
    ap.add_argument(
        "--results-subdir",
        default="",
        help="ANN results subfolder, e.g. LowerConfig -> HNSW/LowerConfig/*.json",
    )
    ap.add_argument(
        "--reranker-output-root",
        default="",
        help="Override rerank output root (default: ann-root/Reranker)",
    )
    args = ap.parse_args()

    if args.jobs_json:
        with open(args.jobs_json, "r", encoding="utf-8") as f:
            jobs = json.load(f)
    else:
        jobs = load_jobs(args.csv)
        if args.index_type != "all":
            jobs = [j for j in jobs if j.get("index_type", "").lower() == args.index_type]
    if args.balance_coco_flickr:
        jobs = balance_coco_flickr_jobs(jobs)

    if args.pending_only:
        kept: List[Dict[str, str]] = []
        for job in jobs:
            std = standard_output_path(
                args.ann_root,
                job["index_type"],
                job["ann_method_file"],
                reranker_output_root=args.reranker_output_root,
            )
            if not os.path.isfile(std):
                kept.append(job)
        jobs = kept

    if args.reverse_order:
        jobs = list(reversed(jobs))

    end = len(jobs) if args.job_end < 0 else args.job_end
    jobs = jobs[args.job_start : end]

    if not jobs:
        print("[WARN] no jobs")
        return

    reranker = Blip2ITMReranker(
        args.device,
        args.image_batch_size,
        use_fp16=not args.no_fp16,
        load_workers=args.load_workers,
    )

    for i, job in enumerate(jobs):
        itype = job["index_type"]
        fname = job["ann_method_file"]
        inp = input_path(args.ann_root, itype, fname, args.results_subdir)
        out = output_path(
            args.ann_root,
            itype,
            fname,
            args.output_suffix,
            reranker_output_root=args.reranker_output_root,
        )
        std = standard_output_path(
            args.ann_root, itype, fname, reranker_output_root=args.reranker_output_root
        )
        promote = std if args.promote_to_standard else ""
        print(f"\n[{i+1}/{len(jobs)}] {itype} {fname}")
        process_file(
            inp,
            out,
            reranker,
            args.top_k,
            args.skip_existing,
            promote,
            reranker_output_root=args.reranker_output_root,
            query_batch_size=args.query_batch_size,
            checkpoint_every=args.checkpoint_every,
            resume_partial=args.resume_partial,
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
