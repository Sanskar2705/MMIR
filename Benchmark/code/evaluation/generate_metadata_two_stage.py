import json
import requests
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import pynvml
import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time

# GPU energy measurement
handle = None
def get_gpu_energy():
    global handle
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

# Disable proxies for local requests
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

MAX_SAMPLES = None  # Set None for full dataset

def two_stage_query(service_url, annotation_file, output_file, dataset, 
                    stage1_model, stage1_core_type, stage2_model, stage2_core_type,
                    stage1_k=100, stage2_k=10, reranking_url=None):

    with open(annotation_file, "r") as f:
        data = json.load(f)

    output = []

    for entry in tqdm(data[:MAX_SAMPLES], desc="Processing queries"):
        query_caption = entry["caption"].strip()
        per_query_log = {"query": query_caption}

        try:
            start_time_total = get_time()
            start_energy_total = get_gpu_energy()

            # Two-stage API call
            params = {
                "query": query_caption,
                "dataset": dataset,
                "stage1_model": stage1_model,
                "stage1_core_type": stage1_core_type,
                "stage2_model": stage2_model,
                "stage2_core_type": stage2_core_type,
                "stage1_k": stage1_k,
                "stage2_k": stage2_k
            }

            resp = requests.get(service_url, params=params)
            resp.raise_for_status()
            results = resp.json()

            end_time_total = get_time()
            end_energy_total = get_gpu_energy()

            # Extract stage-wise times and encodings
            stage1_time = results.get("retrieval_time", {}).get("stage1", 0)
            stage2_time = results.get("retrieval_time", {}).get("stage2", 0)
            stage1_encoding_time = results.get("encoding_time", {}).get("stage1", 0)
            stage2_encoding_time = results.get("encoding_time", {}).get("stage2", 0)
            query_time = results.get("query_time", stage1_time + stage2_time)

            # Default values before reranking
            per_query_log.update({
                "running_time_without_reranking": stage1_time,
                "running_time": end_time_total - start_time_total,
                "energy_util": end_energy_total - start_energy_total,
                "reranking_time": 0,
                "reranking_energy": 0,
                "list_of_top_k": results.get("list_of_top_k", []),
                "query_time_service": query_time,
                "stage1_encoding_time": stage1_encoding_time,
                "stage2_encoding_time": stage2_encoding_time,
                "stage1_time_service": stage1_time,
                "stage2_time_service": stage2_time
            })

            # Apply reranking if URL is provided
            if reranking_url is not None:
                top_k_list_fetched = results.get("list_of_top_k", [])
                rerank_start_time = get_time()
                rerank_start_energy = get_gpu_energy()
                
                reranked_resp = requests.get(
                    reranking_url,
                    params={"query": query_caption, "topk_list": json.dumps(top_k_list_fetched), "K": stage2_k}
                )
                reranked_resp.raise_for_status()
                reranked_results = reranked_resp.json()
                
                rerank_end_time = get_time()
                rerank_end_energy = get_gpu_energy()
                
                # Update logs with reranking
                per_query_log.update({
                    "running_time": (get_time() - start_time_total),
                    "energy_util": rerank_end_energy - start_energy_total,
                    "reranking_time": rerank_end_time - rerank_start_time,
                    "reranking_energy": rerank_end_energy - rerank_start_energy,
                    "list_of_top_k": reranked_results.get("list_of_top_k", [])
                })

                for key in reranked_results:
                    if key != "list_of_top_k":
                        per_query_log[f"{key}_reranking_service"] = reranked_results[key]

        except Exception as e:
            print(f"Error for query '{query_caption}': {e}")
            per_query_log.update({
                "running_time_without_reranking": 0,
                "running_time": 0,
                "energy_util": 0,
                "reranking_time": 0,
                "reranking_energy": 0,
                "list_of_top_k": [],
                "query_time_service": 0,
                "stage1_encoding_time": 0,
                "stage2_encoding_time": 0,
                "stage1_time_service": 0,
                "stage2_time_service": 0
            })

        output.append(per_query_log)

    with open(output_file, "w") as f_out:
        json.dump(output, f_out, indent=2)
    print(f"\nSaved results to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Two-stage retrieval metadata generator")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name (coco/flickr)")
    parser.add_argument("--stage1_model", required=True, help="Stage 1 model")
    parser.add_argument("--stage1_core_type", required=True, help="Stage 1 core type (text/image)")
    parser.add_argument("--stage2_model", required=True, help="Stage 2 model")
    parser.add_argument("--stage2_core_type", required=True, help="Stage 2 core type (text/image)")
    parser.add_argument("--stage1_k", type=int, default=50, help="Top K for stage 1")
    parser.add_argument("--stage2_k", type=int, default=10, help="Top K for stage 2")
    parser.add_argument("--db", default="solr", help="Backend DB (solr/faiss)")
    parser.add_argument("--reranking_model", default=None, help="Optional reranking model endpoint")
    parser.add_argument("--suffix", "-s", help="Suffix to result file", default="_0")  

    args = parser.parse_args()

    config = load_config()
    annotation_file = config["paths"]["dataset"][args.dataset]["query_annotations_path"]
    result_folder = Path(config["paths"]["project_root"]) / "results"
    result_folder.mkdir(exist_ok=True)

    output_file = result_folder / (
        f"result_twoStage_{args.dataset}_{args.stage1_model}_{args.stage1_core_type}_"
        f"{args.stage2_model}_{args.stage2_core_type}_{args.db}{args.suffix}.json"
    )

    # Initialize GPU measurement
    pynvml.nvmlInit()
    global handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    batch_start_energy = get_gpu_energy()


    # service_url = config["endpoints"]["two_stage_retrieval"]
    if args.db.lower() == "solr":
        service_url = config["endpoints"]["two_stage_retrieval_solr"]
    elif args.db.lower() == "faiss":
        service_url = config["endpoints"]["two_stage_retrieval_faiss"]
    else:
        raise ValueError(f"Unknown DB backend {args.db}. Use solr or faiss.")



    reranking_url = None
    if args.reranking_model is not None:
        reranking_url = config["endpoints"]["neural_reranking"] + args.reranking_model

    two_stage_query(
        service_url=service_url,
        annotation_file=annotation_file,
        output_file=output_file,
        dataset=args.dataset,
        stage1_model=args.stage1_model,
        stage1_core_type=args.stage1_core_type,
        stage2_model=args.stage2_model,
        stage2_core_type=args.stage2_core_type,
        stage1_k=args.stage1_k,
        stage2_k=args.stage2_k,
        reranking_url=reranking_url
    )
    batch_end_energy = get_gpu_energy()
    batch_energy_consumption = batch_end_energy - batch_start_energy
    print(f"BATCH ENERGY COMNSUMPTION : {batch_energy_consumption}")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
