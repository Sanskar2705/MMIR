import os
import json
from tqdm import tqdm
import argparse
import pandas as pd
import sys

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate(annotation_file, results_file):
    annotations = load_json(annotation_file)
    results = load_json(results_file)

    # Basic validation
    if not isinstance(results, list) or len(results) == 0:
        print(f"⚠️ Skipping {results_file} (not a list or empty)")
        return None

    if "list_of_top_k" not in results[0]:
        print(f"⚠️ Skipping {results_file} (missing list_of_top_k format)")
        return None

    total = r1 = r5 = r10 = 0
    for ann, res in tqdm(zip(annotations, results), total=min(len(annotations), len(results))):
        gt_image = ann.get("image") or ann.get("image_name") or ann.get("image_path")
        if gt_image is None:
            continue

        retrieved = [r.get("image_path") for r in res.get("list_of_top_k", []) if "image_path" in r]

        total += 1
        if gt_image in retrieved[:1]:
            r1 += 1
        if gt_image in retrieved[:5]:
            r5 += 1
        if gt_image in retrieved[:10]:
            r10 += 1

    if total == 0:
        return None

    return {
        "R@1": r1 / total,
        "R@5": r5 / total,
        "R@10": r10 / total,
        "total": total
    }


def infer_dataset_from_filename(filename):
    """
    Tries to extract dataset from name like:
    results_uniir_image_flickr_faiss_0.json  -> flickr
    results_clip_text_coco_faiss_0.json     -> coco
    """
    parts = filename.lower().split("_")
    for d in ["coco", "flickr"]:
        if d in parts:
            return d
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all result JSON files in a directory")
    parser.add_argument(
        "--results_dir", "-r",
        default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/results/New_exp/Variable_k",
        help="Directory containing result JSON files"
    )
    parser.add_argument("--csv_out", "-o", default="evaluation_all.csv", help="Output CSV filename")
    args = parser.parse_args()

    config = load_config()

    results_dir = args.results_dir
    all_files = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])

    rows = []

    for fname in all_files:
        if fname == "energy_log.json":
            continue

        full_path = os.path.join(results_dir, fname)

        dataset_name = infer_dataset_from_filename(fname)
        if dataset_name is None:
            print(f"⚠️ Skipping {fname} (cannot infer dataset)")
            continue

        annotation_file = config["paths"]["dataset"][dataset_name]["query_annotations_path"]

        print(f"\n✅ Evaluating: {fname} (dataset={dataset_name})")
        metrics = evaluate(annotation_file, full_path)

        if metrics is None:
            print(f"⚠️ Could not evaluate {fname}")
            continue

        print(f"R@1={metrics['R@1']:.4f}, R@5={metrics['R@5']:.4f}, R@10={metrics['R@10']:.4f}  (N={metrics['total']})")

        rows.append({
            "file": fname,
            "dataset": dataset_name,
            "R@1": round(metrics["R@1"], 4),
            "R@5": round(metrics["R@5"], 4),
            "R@10": round(metrics["R@10"], 4),
            "total": metrics["total"]
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, args.csv_out)
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved summary CSV to: {csv_path}")
    print(df)
