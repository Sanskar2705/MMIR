import subprocess
import sys
import json
import re
from pathlib import Path

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

config = load_config()
endpoints = list(config["endpoints"].keys())
single_endpoints = [ep for ep in endpoints if ep.lower() != "rrf"]
single_endpoints = single_endpoints[0:13]

SCRIPT_PATH = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/evaluation/generate_metadata.py"
ENERGY_LOG_PATH = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/results/energy_log.json"

# Load existing energy log or initialize empty list
energy_log = []
if Path(ENERGY_LOG_PATH).exists():
    try:
        with open(ENERGY_LOG_PATH, "r") as f:
            content = f.read().strip()
            if content:
                energy_log = json.loads(content)
    except Exception as e:
        print(f"Warning: Could not load existing energy log. Starting fresh. Error: {e}")


for j in range(80,100):
# for j in range(0,5):
    for i, endpoint in enumerate(single_endpoints, 1):
        print(f"\n===== Running for Endpoint {i}: {endpoint} =====")
        proc = subprocess.Popen(
            # ["python3", SCRIPT_PATH, "--suffix", f"_rerank_blip2_{j}", "--rerankingmodel" , "blip2"],
            ["python3", SCRIPT_PATH, "--suffix", f"_{j}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        inputs = f"single\n{endpoint}\n"
        stdout, stderr = proc.communicate(inputs)

        if proc.returncode != 0:
            print(f"Error: Process for {endpoint} exited with code {proc.returncode}")
            print(stderr)
        else:
            print(f"Completed successfully for {endpoint}")
            # Extract energy consumption
            match = re.search(r"BATCH ENERGY COMNSUMPTION\s*:\s*(\d+)", stdout)
            if match:
                energy_value = int(match.group(1))
                output_filename = f"results_{endpoint.lower()}_{j}.json"
                energy_log.append({
                    "batch_energy": energy_value,
                    "iteration": j,
                    "output_file": output_filename
                })

# Save updated energy log
with open(ENERGY_LOG_PATH, "w") as f:
    json.dump(energy_log, f, indent=2)
