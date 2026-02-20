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

# endpoints = list(config["endpoints"].keys())
# single_endpoints = [ep for ep in endpoints if ep.lower() != "rrf"]
# single_endpoints = single_endpoints[0:18]
# [bm25text,clipimage] [minimltext , clip image]

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


def run_metadata(dataset="coco", iteration=0):
    ENDPOINTS = config["endpoints"]

    # Filter endpoints by dataset and backend type
    solr_endpoints = [name for name in ENDPOINTS if dataset in name and "solr" in name]
    faiss_endpoints = [name for name in ENDPOINTS if dataset in name and "faiss" in name]

    print(f"\n=== Running for dataset: {dataset.upper()} (iter={iteration}) ===")

    def run_pairs(endpoint_list, backend_type):
        for combo in itertools.combinations(endpoint_list, 2):
            methods = ",".join(combo)
            print(f"\n>>> Running RRF fusion: {methods}")

            proc = subprocess.Popen(
                ["python3", SCRIPT_PATH, "--dataset", dataset, "--suffix", f"_rerank-blip2_{j}", "--rerankingmodel" , "blip2"],
                # ["python3", SCRIPT_PATH, "--dataset", dataset, "--suffix", f"_{iteration}"],
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
                    output_filename = f"results_rrf_{'_'.join(m.lower() for m in combo)}_{iteration}.json"
                    energy_log.append({
                        "batch_energy": energy_value,
                        "iteration": iteration,
                        "output_file": output_filename
                    })

    if solr_endpoints:
        print("\n--- Solr Endpoints ---")
        run_pairs(solr_endpoints, "solr")

    if faiss_endpoints:
        print("\n--- Faiss Endpoints ---")
        run_pairs(faiss_endpoints, "faiss")


if __name__ == "__main__":
    # Example: iterate same as file 1
    for j in range(80, 100):
        run_metadata("coco", iteration=j)
        run_metadata("flickr", iteration=j)

    # Save updated energy log
    with open(ENERGY_LOG_PATH, "w") as f:
        json.dump(energy_log, f, indent=2)

