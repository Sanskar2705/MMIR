import itertools
import subprocess
import sys
import json
import re
from pathlib import Path

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

config = load_config()
SCRIPT_PATH = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/evaluation/generate_metadata.py"
ENERGY_LOG_PATH = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/results/energy_log.json"

# Explicit list of RRF endpoint pairs
# Example: ("bm25text_solr", "clipimage_solr")

# faiss

RRF_PAIRS = [  
    ("minilm_text_flickr_faiss", "uniir_joint-image-text_flickr_faiss"),
    ("minilm_text_flickr_faiss", "clip_image_flickr_faiss"),
    ("minilm_text_flickr_faiss", "flava_image_flickr_faiss"),
    ("minilm_text_flickr_faiss", "uniir_image_flickr_faiss"),
    
    ("clip_text_flickr_faiss", "uniir_joint-image-text_flickr_faiss"),
    ("clip_text_flickr_faiss", "flava_image_flickr_faiss"),
    ("clip_text_flickr_faiss", "uniir_image_flickr_faiss"),

    ("flava_text_flickr_faiss", "uniir_joint-image-text_flickr_faiss"),
    ("flava_text_flickr_faiss", "uniir_image_flickr_faiss"),

    ("clip_image_flickr_faiss", "uniir_joint-image-text_flickr_faiss"),
    ("clip_image_flickr_faiss", "flava_text_flickr_faiss"),
    ("clip_image_flickr_faiss", "uniir_text_flickr_faiss"),


    ("flava_image_flickr_faiss", "uniir_joint-image-text_flickr_faiss"),
    ("flava_image_flickr_faiss", "uniir_text_flickr_faiss"),


    ("minilm_text_coco_faiss", "uniir_joint-image-text_coco_faiss"),
    ("minilm_text_coco_faiss", "clip_image_coco_faiss"),
    ("minilm_text_coco_faiss", "flava_image_coco_faiss"),
    ("minilm_text_coco_faiss", "uniir_image_coco_faiss"),

    ("clip_image_coco_faiss", "uniir_joint-image-text_coco_faiss"),
    ("clip_image_coco_faiss", "flava_text_coco_faiss"),
    ("clip_image_coco_faiss", "uniir_text_coco_faiss"),

    ("clip_text_coco_faiss", "uniir_joint-image-text_coco_faiss"),
    ("clip_text_coco_faiss", "flava_image_coco_faiss"),
    ("clip_text_coco_faiss", "uniir_image_coco_faiss"),

    ("flava_image_coco_faiss", "uniir_joint-image-text_coco_faiss"),
    ("flava_image_coco_faiss", "uniir_text_coco_faiss"),

    ("flava_text_coco_faiss", "uniir_joint-image-text_coco_faiss"),
    ("flava_text_coco_faiss", "uniir_image_coco_faiss"),
]

# RRF_PAIRS = [  
#     ("uniir_image_flickr_faiss", "uniir_joint-image-text_flickr_faiss"),
#     ("uniir_text_flickr_faiss", "uniir_joint-image-text_flickr_faiss"),
# ]


# RRF_PAIRS = [  
#     ("minilm_text_flickr_solr", "uniir_joint-image-text_flickr_solr"),
#     ("uniir_text_flickr_solr", "uniir_joint-image-text_flickr_solr"),
#     ("flava_image_flickr_solr", "uniir_joint-image-text_flickr_solr"),
#     ("uniir_image_flickr_solr", "uniir_joint-image-text_flickr_solr"),
#     ("flava_text_flickr_solr", "uniir_joint-image-text_flickr_solr"),

#     ("flava_image_coco_solr", "uniir_joint-image-text_coco_solr"),
#     ("uniir_image_coco_solr", "uniir_joint-image-text_coco_solr"),
#     ("uniir_text_coco_solr", "uniir_joint-image-text_coco_solr"),
#     ("flava_text_coco_solr", "uniir_joint-image-text_coco_solr"),
#     ("minilm_text_coco_solr", "uniir_joint-image-text_coco_solr"),
# ]



# RRF_PAIRS = [
#     ("flava_text_coco_solr", "uniir_image_coco_solr"),
#     ("uniir_text_coco_solr", "uniir_image_coco_solr"),
#     ("minilm_text_coco_solr", "uniir_image_coco_solr"),
#     ("flava_image_coco_solr", "uniir_image_coco_solr"),
#     ("clip_text_coco_solr", "uniir_image_coco_solr"),
      
#     ("flava_image_coco_solr", "minilm_text_coco_solr"),
#     ("uniir_text_coco_solr", "flava_image_coco_fais"),

    
#     ("uniir_joint-image-text_coco_solr", "uniir_text_coco_solr"),
#     ("uniir_joint-image-text_coco_solr", "flava_image_coco_solr"),
#     ("uniir_joint-image-text_coco_solr", "clip_image_coco_solr"),
#     ("uniir_joint-image-text_coco_solr", "flava_text_coco_solr"),
#     ("uniir_joint-image-text_coco_solr", "uniir_text_coco_solr"),
    
    

#     ("flava_text_flickr_solr", "uniir_image_flickr_solr"),
#     ("uniir_text_flickr_solr", "uniir_image_flickr_solr"),
#     ("minilm_text_flickr_solr", "uniir_image_flickr_solr"),
#     ("flava_image_flickr_solr", "uniir_image_flickr_solr"),
#     ("clip_text_flickr_solr", "uniir_image_flickr_solr"),

#     ("flava_image_flickr_solr", "minilm_text_flickr_solr"),
#     ("uniir_text_flickr_solr", "flava_image_flickr_solr"),

    
#     ("uniir_joint-image-text_flickr_solr", "minilm_text_flickr_solr"),
#     ("uniir_joint-image-text_flickr_solr", "flava_image_coco_solr"),
#     ("uniir_joint-image-text_flickr_solr", "clip_image_coco_solr"),
#     ("uniir_joint-image-text_flickr_solr", "uniir_text_coco_solr"),

#     ("bm25_text_coco_solr", "minilm_text_coco_solr"),
#     ("bm25_text_flickr_solr", "minilm_text_flickr_solr"),
    
# ]

# RRF_PAIRS = [
#     ("clip_image_coco_solr", "minilm_text_coco_solr"),
#     ("clip_image_flickr_solr", "minilm_text_flickr_solr"),
#     ("clip_image_coco_solr", "bm25_text_coco_solr"),
#     ("clip_image_flickr_solr", "bm25_text_flickr_solr"),

#     ("clip_image_coco_solr", "minilm_text_coco_solr"),
#     ("clip_image_flickr_solr", "minilm_text_flickr_solr"),
#     ("clip_image_coco_solr", "bm25_text_coco_solr"),
#     ("clip_image_flickr_solr", "bm25_text_flickr_solr"),
# ]

# Load existing energy log
energy_log = []
if Path(ENERGY_LOG_PATH).exists():
    try:
        with open(ENERGY_LOG_PATH, "r") as f:
            content = f.read().strip()
            if content:
                energy_log = json.loads(content)
    except Exception as e:
        print(f"Warning: Could not load existing energy log. Starting fresh. Error: {e}")


def run_metadata(dataset="flickr", iteration=0):
    ENDPOINTS = config["endpoints"]

    print(f"\n=== Running for dataset: {dataset.upper()} (iter={iteration}) ===")

    for ep1, ep2 in RRF_PAIRS:
        # Skip pairs if not present in config
        if ep1 not in ENDPOINTS or ep2 not in ENDPOINTS:
            print(f"Skipping pair ({ep1}, {ep2}) - not found in config")
            continue

        methods = f"{ep1},{ep2}"
        print(f"\n>>> Running RRF fusion: {methods}")

        proc = subprocess.Popen(
            # ["python3", SCRIPT_PATH, "--dataset", dataset, "--suffix", f"_rerank-blip2_{j}", "--rerankingmodel" , "blip2"],
            ["python3", SCRIPT_PATH, "--dataset", dataset, "--suffix", f"_{iteration}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        inputs = f"rrf\n{methods}\n"
        stdout, stderr = proc.communicate(inputs)

        if proc.returncode != 0:
            print(f"Error: Process for {methods} exited with code {proc.returncode}")
            print(stderr)
        else:
            print(f"Completed successfully for {methods}")

            # Extract energy consumption
            match = re.search(r"BATCH ENERGY COMNSUMPTION\s*:\s*(\d+)", stdout)
            if match:
                energy_value = int(match.group(1))
                output_filename = f"results_rrf_{ep1}_{ep2}_{iteration}.json"
                energy_log.append({
                    "batch_energy": energy_value,
                    "iteration": iteration,
                    "dataset": dataset,
                    "pair": [ep1, ep2],
                    "output_file": output_filename
                })


if __name__ == "__main__":
    for j in range(1):
        
        run_metadata("flickr", iteration=j)
        run_metadata("coco", iteration=j)
        

    # Save updated energy log
    with open(ENERGY_LOG_PATH, "w") as f:
        json.dump(energy_log, f, indent=2)
