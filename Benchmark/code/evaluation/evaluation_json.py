import json
from tqdm import tqdm

import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

import argparse

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def evaluate(annotation_file, results_file):
    annotations = load_json(annotation_file)
    results     = load_json(results_file)
    total = r1 = r5 = r10 = 0
    for ann, res in tqdm(zip(annotations, results), total=len(annotations)):
        gt_image  = ann["image"]
        retrieved = [r["image_path"] for r in res["list_of_top_k"]]

        total += 1
        if gt_image in retrieved[:1]:  r1 += 1
        if gt_image in retrieved[:5]:  r5 += 1
        if gt_image in retrieved[:10]: r10 += 1

    print(f"\nEvaluated on {total} samples")
    print(f"R@1:  {r1 / total:.4f}")
    print(f"R@5:  {r5 / total:.4f}")
    print(f"R@10: {r10 / total:.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This code will compute correctness metrics from the given results file")

    # Load config
    config = load_config()
    selected_dataset = config["selected_config"]["dataset"]

    # Optional argument
    parser.add_argument("--result", "-r", help="results json file path")
    parser.add_argument("--dataset", "-d", help="Dataset ", default=selected_dataset)
    
    args = parser.parse_args()
    results_file = args.result
    dataset_name = args.dataset

    annotation_file = config["paths"]["dataset"][dataset_name]["query_annotations_path"]

    evaluate(annotation_file, results_file)
