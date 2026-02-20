import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
import sys
import re

# add your project path if needed
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config


# ---------- safe JSON loader ----------
def load_json(path):
    try:
        with open(path, "r") as f:
            data = f.read()
        # quick fix: remove trailing commas
        data = re.sub(r",\s*([}\]])", r"\1", data)
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"\n⚠️ JSON Decode Error in: {path}")
        print(f"    {e}")
        return None
    except Exception as e:
        print(f"\n⚠️ Error reading {path}: {e}")
        return None


# ---------- evaluation ----------
def evaluate(annotation_file, results_file):
    """Return (R@1, R@5, R@10, total)."""
    annotations = load_json(annotation_file)
    results = load_json(results_file)

    if annotations is None or results is None:
        return None

    total = r1 = r5 = r10 = 0
    for ann, res in zip(annotations, results):
        gt_image = ann["image"]
        retrieved = [r["image_path"] for r in res["list_of_top_k"]]

        total += 1
        if gt_image in retrieved[:1]:
            r1 += 1
        if gt_image in retrieved[:5]:
            r5 += 1
        if gt_image in retrieved[:10]:
            r10 += 1

    return r1 / total, r5 / total, r10 / total, total


# ---------- filename parser ----------
def parse_info(filename):
    """
    Parse filenames like:
    results_rrf_clip_text_coco_faiss_uniir_image_coco_faiss__0.json
    """
    fname = os.path.basename(filename)
    parts = fname.split("_")

    # dataset is at position 4 (coco/flickr/etc.)
    dataset = parts[4]

    # stage1 = "clip_text" or "flava_text"
    stage1 = f"{parts[2]}_{parts[3]}"

    # stage2 = "uniir_image" or similar
    stage2 = f"{parts[6]}_{parts[7]}"

    return dataset, stage1, stage2


# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all RRF result files")
    parser.add_argument("--results_dir", "-r",
                        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/results/rrf/rrf_cross_model",
                        help="Directory containing results files")
    parser.add_argument("--output_csv", "-o", default="evaluation_results_rrf_reranker.csv",
                        help="Where to save the CSV")
    args = parser.parse_args()

    config = load_config()
    rows = []

    for file in os.listdir(args.results_dir):
        if file.startswith("results_rrf") and file.endswith(".json"):
            results_file = os.path.join(args.results_dir, file)
            dataset, stage1, stage2 = parse_info(results_file)

            if dataset not in config["paths"]["dataset"]:
                print(f"Skipping {results_file} — dataset '{dataset}' not in config")
                continue
            annotation_file = config["paths"]["dataset"][dataset]["query_annotations_path"]

            print(f"Evaluating: {results_file} (dataset={dataset}, stage1={stage1}, stage2={stage2})")
            result = evaluate(annotation_file, results_file)
            if result is None:
                print(f"⚠️ Skipping invalid JSON file: {results_file}")
                continue
            r1, r5, r10, total = result

            rows.append({
                "dataset": dataset,
                "stage1": stage1,
                "stage2": stage2,
                "R@1": round(r1, 4),
                "R@5": round(r5, 4),
                "R@10": round(r10, 4),
                "total_samples": total,
                "file": file
            })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")
    print(df)
