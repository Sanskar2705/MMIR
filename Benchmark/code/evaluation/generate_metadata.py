import json
import requests
from tqdm import tqdm
import os

import sys
from pathlib import Path
import argparse


import pynvml

# For GPU Energy measurements
handle = None
def get_gpu_energy():
    global handle
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    return start_energy



# Allow: from Benchmark.config.config_utils import load_config
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

# Import time utils
from Benchmark.code.evaluation.time_util import get_time



# Disable proxies for local requests
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
proxies = {
            'http': '',
            'https': ''
        }



MAX_SAMPLES = None  # Set to None to use full dataset

def query_and_save(service_url, annotation_file, output_file,k=10, methods=None,reranking_url=None, rerank_candidates=100):
    
    print(locals())
    
    with open(annotation_file, "r") as f:
        data = json.load(f)

    output = []

    for entry in tqdm(data[:MAX_SAMPLES]):
        query_caption = entry["caption"].strip()


        try:
            per_query_log = {}
            per_query_log["query"] = query_caption

            # start_time = get_time()
            # start_energy = get_gpu_energy()

            
            # if methods: 
            #     # Fusion query - pass list of methods
            #     response = requests.get(service_url, params={
            #         "q": query_caption,
            #         "methods": methods
            #     })
                 
            #     duration = get_time() - start_time
            #     energy_util = get_gpu_energy() - start_energy
            #     response.raise_for_status()
            #     results = response.json()
            # else:
            #     # Single model query
            #     candidate_k = 30 if reranking_url is not None else k
            #     response = requests.get(service_url, params={"q": query_caption, "k": candidate_k})
            
            #this code works for 30
            # start_time = get_time()
            # start_energy = get_gpu_energy()

            # candidate_k = 30 if reranking_url is not None else k

            # if methods:
            #     # Fusion query - pass list of methods
            #     response = requests.get(service_url, params={
            #         "q": query_caption,
            #         "methods": methods,
            #         "k": candidate_k
            #     })

            #     duration = get_time() - start_time
            #     energy_util = get_gpu_energy() - start_energy
            #     response.raise_for_status()
            #     results = response.json()
            # else:
            #     # Single model query
            #     response = requests.get(service_url, params={"q": query_caption, "k": candidate_k})

            #     duration = get_time() - start_time
            #     energy_util = get_gpu_energy() - start_energy
            #     response.raise_for_status()
            #     results = response.json()
            start_time = get_time()
            start_energy = get_gpu_energy()

            # final_k = k            # final output size (top-10)
            # candidate_k = 30       # fetch 30 from each endpoint for fusion

            # if methods:
            #     # RRF fusion call: fetch 30 from each model, return top-10 fused
            #     response = requests.get(
            #         service_url,
            #         params={
            #             "q": query_caption,
            #             "methods": methods,
            #             "k": final_k,                 #  final top-10
            #             "candidate_k": candidate_k    #  per-endpoint top-30
            #         },
            #         proxies=proxies,
            #         timeout=120
            #     )
            # else:
            #     # Single model: return top-k as usual
            #     response = requests.get(
            #         service_url,
            #         params={"q": query_caption, "k": final_k},
            #         proxies=proxies,
            #         timeout=120
            #     )
            final_k = k
            rerank_candidate_k = rerank_candidates if reranking_url is not None else final_k

            if methods:
                # RRF fusion case: ask fusion endpoint to RETURN 100 fused candidates
                response = requests.get(
                    service_url,
                    params={
                        "q": query_caption,
                        "methods": methods,
                        "k": rerank_candidate_k,        # RETURN 100 fused
                        "candidate_k": rerank_candidate_k  # each model fetch size (can keep 30 if you want)
                    },
                    proxies=proxies,
                    timeout=120
                )
            else:
                # Single model case: ask retrieval endpoint to return 100 candidates
                response = requests.get(
                    service_url,
                    params={"q": query_caption, "k": rerank_candidate_k},
                    proxies=proxies,
                    timeout=120
                )


            duration = get_time() - start_time
            energy_util = get_gpu_energy() - start_energy

            response.raise_for_status()
            results = response.json()


                # response = requests.get(service_url, params={"q": query_caption,"k":k})
                # duration = get_time() - start_time
                # energy_util = get_gpu_energy() - start_energy
                # response.raise_for_status()
                # results = response.json()
            
            
            per_query_log["running_time_without_reranking"] = duration
            per_query_log["running_time"] = duration
            per_query_log["energy_util"] = energy_util
            per_query_log["reranking_time"] = 0
            per_query_log["reranking_energy"] = 0

            for key in results:
                if (key != "list_of_top_k") :
                    per_query_log[key+"_service"] = results[key]
                elif (key == "list_of_top_k" and reranking_url is not None):
                    per_query_log["list_of_top_k"+"_not_reranked"] = results[key]
                else:
                    per_query_log["list_of_top_k"] = results[key]

                    

            if reranking_url is not None:
                
                per_query_log["running_time_without_reranking"] = duration
                top_k_list_fetched = results.get("list_of_top_k", [])
                reranking_start_energy = get_gpu_energy() 
                reranking_start_time = get_time()
                reranked_response = requests.get(reranking_url, params={"query": query_caption,'topk_list':json.dumps(top_k_list_fetched),"K": rerank_candidate_k})
                reranking_end_time = get_time()
                reranking_end_energy = get_gpu_energy()
                duration = get_time() - start_time
                reranked_response.raise_for_status()
                reranked_results = reranked_response.json()

                per_query_log["running_time"] = duration
                per_query_log["energy_util"] = ( reranking_end_energy - start_energy)
                per_query_log["reranking_time"] = reranking_end_time - reranking_start_time
                per_query_log["reranking_energy"] = reranking_end_energy - reranking_start_energy
                per_query_log["list_of_top_k"] = reranked_results["list_of_top_k"]
                
                for key in reranked_results:
                    if "list_of_top_k" != key:
                        per_query_log[key+"_reranking_service"] = reranked_results[key]

        except Exception as e:
            print(f"Error for query: {query_caption} -> {e}")
            results = []
            duration = None

 
        output.append(per_query_log)

    with open(output_file, "w") as f_out:
        json.dump(output, f_out, indent=2)

    print(f"\nSaved results to {output_file}")


def main():

    parser = argparse.ArgumentParser(description="This code will generate clip embeddings")

    # Load config
    config = load_config()
    debug = config["debug_logs"]

    # Optional argument
    parser.add_argument("--suffix", "-s", help="suffix to result file", default="_0")
    parser.add_argument("--dataset", "-d", help="Enter dataset to be evaluated")
    parser.add_argument("--rerankingmodel","-r",help="Enter type of reranking",default=None)
    # parser.add_argument("--k","-k",help="top k results will be fetched",default=config["k"])
    parser.add_argument("--k","-k", type=int, help="top k results will be fetched", default=int(config["k"]))
    parser.add_argument("--rerank_candidates", type=int, default=100)


    args = parser.parse_args()
    op_suffix = args.suffix   
    reranking_model = args.rerankingmodel
    k = args.k 
  
    ENDPOINTS = config["endpoints"]

    # Initialize pynvml
    pynvml.nvmlInit()
    global handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
    
    batch_start_energy = get_gpu_energy()


    mode = input("Select mode ('single' or 'rrf' ): ").strip().lower()

    result_folder = Path(config["paths"]["project_root"]) / "results" 
    result_folder.mkdir(exist_ok=True)

    if mode == "single":
        print("\nAvailable Single Endpoints:")
        for i, (name, url) in enumerate(ENDPOINTS.items(), start=1):
            print(f"{i}. {name}")

        selection = input("\nEnter number or endpoint name: ").strip()

        if selection.isdigit():
            selection = list(ENDPOINTS.keys())[int(selection) - 1]

        if selection not in ENDPOINTS:
            print("Invalid endpoint. Exiting.")
            return

        url = ENDPOINTS[selection]
        model = (url.split("/")[-1]).split("_")[0]
        modality = (url.split("/")[-1]).split("_")[1]
        dataset = (url.split("/")[-1]).split("_")[2]
        db = (url.split("/")[-1]).split("_")[3]


        # parts = (url.split("/")[-1]).split("_")

        # if parts[0] == "two" and parts[1] == "stage":
        #     # Handle two_stage_*_dataset_db
        #     model = parts[2]          # clip
        #     dataset = parts[3]        # coco/flickr
        #     db = parts[4]             # solr/faiss
        #     modality = "two_stage"
        # else:
        #     model = parts[0]          # clip/flava/etc.
        #     modality = parts[1]       # image/text
        #     dataset = parts[2]        # coco/flickr
        #     db = parts[3]             # solr/faiss



        annotation_file = config["paths"]["dataset"][dataset]["query_annotations_path"]

        
        output_file = result_folder / f"results_{selection.lower()}_{op_suffix}.json"

        methods = None
        print(f"\nRunning query for: {selection}")

    elif mode == "rrf":

        dataset = args.dataset
        annotation_file = config["paths"]["dataset"][dataset]["query_annotations_path"]

 
        FUSION_ENDPOINT = ENDPOINTS["rrf"]
        valid_endpoints = []
        print("\nEnter method names for RRF fusion (comma-separated, must match endpoint keys).")
        print("Available options:")
        for name in ENDPOINTS:
            if not (name=='rrf' or name=="neural_reranking"):
                # print("Potential Name",name)
                endpoint_dataset = name.split("_")[2]
                if endpoint_dataset == dataset:
                    print(f" - {name}")
                    valid_endpoints.append(name)

        method_input = input("\nMethods (comma-separated): ").strip()
        methods = [m.strip() for m in method_input.split(",") if m.strip()]

        if not methods or any(m not in valid_endpoints for m in methods):
            print("One or more method names are invalid. Exiting.")
            return
        
        output_file = result_folder / f"results_rrf_{'_'.join(m.lower() for m in methods)}_{op_suffix}.json"
        url = FUSION_ENDPOINT
        print(f"\nRunning RRF fusion with: {', '.join(methods)}")
    else:
        print("Invalid mode. Choose either 'single' or 'rrf'.")

    if reranking_model is not None:
        neural_reranking_url = config["endpoints"]["neural_reranking"]+reranking_model

        if debug:
            print(f"Neural Reranking Selected with Model {reranking_model}")
    else:
        neural_reranking_url = None
    

     
    # query_and_save(url, annotation_file, output_file,k, methods,reranking_url=neural_reranking_url)
    query_and_save(
        url,
        annotation_file,
        output_file,
        k,
        methods,
        reranking_url=neural_reranking_url,
        rerank_candidates=args.rerank_candidates,
    )


    batch_end_energy = get_gpu_energy()
    batch_energy_consumption = batch_end_energy - batch_start_energy
    print(f"BATCH ENERGY COMNSUMPTION : {batch_energy_consumption}")

    pynvml.nvmlShutdown()  # Clean up


if __name__ == "__main__":
    main()
 