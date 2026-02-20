#!/usr/bin/env python3
#faiss_two_stage_staged_client.py
# -*- coding: utf-8 -*-

import json, argparse, os, sys
from pathlib import Path
from tqdm import tqdm
import requests

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

def main():
    ap = argparse.ArgumentParser("Run FAISS 2-stage in two calls: stage1_only → stage2_from_file (brute-force)")
    ap.add_argument("--dataset", "-d", required=True, choices=["coco","flickr"])
    ap.add_argument("--stage1_model", required=True)
    ap.add_argument("--stage1_core_type", required=True, choices=["text","image","joint-image-text"])
    ap.add_argument("--stage2_model", required=True)
    ap.add_argument("--stage2_core_type", required=True, choices=["text","image","joint-image-text"])
    ap.add_argument("--stage1_k", type=int, default=100)
    ap.add_argument("--stage2_k", type=int, default=10)
    ap.add_argument("--server", default="http://localhost:5059", help="Base URL of faiss_two_stage_service")
    ap.add_argument("--out", required=True, help="Output JSON file for per-query results")
    ap.add_argument("--save_dir", default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/retrievalService/faiss_base_service/2_stage_testing_faiss/stage1_files", help="Where to save Stage-1 JSONs")
    ap.add_argument("--max", type=int, default=None, help="Limit #queries (debug)")
    args = ap.parse_args()

    cfg = load_config()
    ann = cfg["paths"]["dataset"][args.dataset]["query_annotations_path"]
    data = json.loads(Path(ann).read_text())
    if args.max: data = data[:args.max]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for idx, item in enumerate(tqdm(data, desc="Two-stage (staged)")):
        q = item["caption"].strip()
        stage1_file = save_dir / f"stage1_{args.dataset}_{idx:06d}.json"

        # --- Stage 1
        s1_body = {
            "query": q,
            "dataset": args.dataset,
            "stage1_model": args.stage1_model,
            "stage1_core_type": args.stage1_core_type,
            "stage1_k": args.stage1_k,
            "save_to": str(stage1_file)
        }
        s1 = requests.post(f"{args.server}/stage1_only", json=s1_body)
        s1.raise_for_status()

        # --- Stage 2 (from the saved file)
        s2_body = {
            "query": q,
            "dataset": args.dataset,
            "stage2_model": args.stage2_model,
            "stage2_core_type": args.stage2_core_type,
            "stage2_k": args.stage2_k,
            "stage1_file": str(stage1_file)
        }
        s2 = requests.post(f"{args.server}/stage2_from_file", json=s2_body)
        s2.raise_for_status()
        res2 = s2.json()

        out.append({
            "query": q,
            "list_of_top_k": res2.get("list_of_top_k", []),
            "encoding_time": res2.get("encoding_time", {}),
            "retrieval_time": res2.get("retrieval_time", {})
        })

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
