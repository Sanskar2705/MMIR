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

        # optional quick fix: remove trailing commas before } or ]
        data = re.sub(r",\s*([}\]])", r"\1", data)

        return json.loads(data)

    except json.JSONDecodeError as e:
        print(f"\n JSON Decode Error in: {path}")
        print(f"    {e}")
        return None
    except Exception as e:
        print(f"\n Error reading {path}: {e}")
        return None


# ---------- evaluation ----------
def evaluate(annotation_file, results_file):
    """Return (R@1, R@5, R@10, total)."""
    annotations = load_json(annotation_file)
    results = load_json(results_file)

    if annotations is None or results is None:
        return None  # skip

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
    Extract dataset, stage1, stage2 from filename.
    Example filename:
    result_twoStage_coco_clip_image_flava_image_solr_0.json
    """
    fname = os.path.basename(filename)
    parts = fname.split("_")
    dataset = parts[2]  # coco or flickr etc.
    stage1 = f"{parts[3]}_{parts[4]}"
    stage2 = f"{parts[5]}_{parts[6]}"
    return dataset, stage1, stage2


# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all result_twoStage files automatically detecting dataset")
    parser.add_argument("--results_dir", "-r",
                        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/results",
                        help="Directory containing results files")
    parser.add_argument("--output_csv", "-o", default="evaluation_results.csv",
                        help="Where to save the CSV")
    args = parser.parse_args()

    # Load config once
    config = load_config()

    rows = []

    for file in os.listdir(args.results_dir):
        if file.startswith("result_twoStage") and file.endswith(".json"):
            results_file = os.path.join(args.results_dir, file)
            dataset, stage1, stage2 = parse_info(results_file)

            # get annotation file for the dataset
            if dataset not in config["paths"]["dataset"]:
                print(f"Skipping {results_file} — dataset '{dataset}' not in config")
                continue
            annotation_file = config["paths"]["dataset"][dataset]["query_annotations_path"]

            print(f"Evaluating: {results_file} (dataset={dataset})")
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
                "total_samples": total
            })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")
    print(df)
