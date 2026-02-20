import subprocess
import sys
import json
import re
from pathlib import Path
import os
env = os.environ.copy()

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

config = load_config()
endpoints = list(config["endpoints"].keys())
single_endpoints = [ep for ep in endpoints if ep.lower() != "rrf"]
# single_endpoints = single_endpoints[0:18]
# single_endpoints = single_endpoints[18:34]
# single_endpoints = single_endpoints[18:34]
# single_endpoints = single_endpoints[4:5]
# single_endpoints = single_endpoints[-4:]
# single_endpoints = [ single_endpoints[20]]+ [single_endpoints[22]] + [single_endpoints[24]] + [single_endpoints[26]] + [single_endpoints[27]] + [single_endpoints[28]] +[single_endpoints[30]]   
single_endpoints = [ single_endpoints[19]]+ [single_endpoints[21]] + [single_endpoints[23]] + [single_endpoints[25]] + [single_endpoints[29]] + [single_endpoints[31]] +[single_endpoints[32]] + [single_endpoints[33]]    


for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    env.pop(k, None)

env["no_proxy"] = "localhost,127.0.0.1"
env["NO_PROXY"] = "localhost,127.0.0.1"

print(single_endpoints)

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


# for j in range(0,40):
for j in range(0,1):
# for j in range(40,80):
    for i, endpoint in enumerate(single_endpoints, 1):
        print(f"\n===== Running for Endpoint {i}: {endpoint} =====")
        proc = subprocess.Popen(
            ["python3", SCRIPT_PATH, "--suffix", f"_rerank-blip2_{j}", "--rerankingmodel" , "blip2", "--rerank_candidates", "100"], 
            # ["python3", SCRIPT_PATH, "--suffix", f"_{j}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
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
