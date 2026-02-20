#!/usr/bin/env python3
"""
Batch runner for generate_metadata_two_stage.py using the exact suffix pattern
and subprocess.Popen-style invocation you used in the single-endpoint script.

- Uses RUN_LIST of (dataset, stage1_model, stage1_core_type, stage2_model, stage2_core_type)
- Two modes per configuration:
    * SIMPLE run -> uses suffix "_{j}"
    * RERANK run  -> uses suffix "_rerank-<reranker>_{j}" and passes --reranking_model <name>
- Uses subprocess.Popen(..., stdin=..., stdout=..., stderr=..., text=True) like your original.
- Collects BATCH ENERGY COMNSUMPTION from stdout (if printed) and logs entries to ENERGY_LOG_PATH.

Adjust RUN_LIST, ITER_START/ITER_END, DO_SIMPLE_RUN and DO_RERANK_RUN as needed.
"""

import subprocess
import sys
import json
import re
from pathlib import Path

# ensure repo imports work for load_config
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

# ----------------- CONFIG -----------------
config = load_config()

SCRIPT_PATH = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/evaluation/generate_metadata_two_stage.py"
ENERGY_LOG_PATH = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/results/energy_log.json"

RUN_LIST = [

    # ("coco", "clip",  "text", "uniir", "joint-image-text"),
    # ("coco", "flava", "text", "uniir", "joint-image-text"),
    # ("coco", "clip",  "image","uniir", "joint-image-text"),
    # ("coco", "flava", "image","uniir", "joint-image-text"),
    # ("coco", "uniir", "image", "uniir", "joint-image-text"),
    # ("coco", "uniir", "joint-image-text", "clip",  "text"),
    # ("coco", "uniir", "joint-image-text", "minilm","text"),
    # ("coco", "uniir", "joint-image-text", "flava", "text"),
    # ("coco", "uniir", "joint-image-text", "uniir", "text"),
    # ("coco", "uniir", "joint-image-text", "bm25",  "text"),
    # ("coco", "uniir", "joint-image-text", "clip",  "image"),
    # ("coco", "uniir", "joint-image-text", "flava", "image"),
    ("coco", "uniir", "text", "uniir", "joint-image-text"),
    # ("coco", "uniir", "Image", "uniir", "joint-image-text")



    # ("coco", "uniir", "joint-image-text", "flava", "image"),
    # ("coco", "uniir", "joint-image-text", "flava", "text"),

    # ("flickr", "uniir", "joint-image-text", "flava", "text"),
    # ("flickr", "uniir", "joint-image-text", "flava", "image"),
]


# RUN_LIST = [
#     # ("coco", "bm25", "text", "uniir", "joint-image-text"),
#     # ("coco", "clip", "image", "uniir", "joint-image-text"),
#     # ("coco", "clip", "text", "uniir", "joint-image-text"),
#     # ("coco", "uniir", "joint-image-text", "bm25", "text"),
#     # ("coco", "uniir", "joint-image-text", "clip", "image"),

#     # ("coco", "uniir", "joint-image-text", "clip", "text"),
#     # ("coco", "uniir", "joint-image-text", "flava", "image"),
#     # ("coco", "uniir", "joint-image-text", "flava", "text"),
#     # ("coco", "uniir", "joint-image-text", "minilm", "text"),
#     ("coco", "uniir", "joint-image-text", "uniir", "image"),
#     ("coco", "uniir", "joint-image-text", "uniir", "text"),

#     ("flickr", "bm25", "text", "uniir", "joint-image-text"),
#     ("flickr", "clip", "image", "uniir", "joint-image-text"),
#     ("flickr", "clip", "text", "uniir", "joint-image-text"),
#     ("flickr", "uniir", "joint-image-text", "bm25", "text"),
#     ("flickr", "uniir", "joint-image-text", "clip", "image"),

#     ("flickr", "uniir", "joint-image-text", "clip", "text"),
#     ("flickr", "uniir", "joint-image-text", "flava", "image"),
#     ("flickr", "uniir", "joint-image-text", "flava", "text"),
#     ("flickr", "uniir", "joint-image-text", "minilm", "text"),
#     ("flickr", "uniir", "joint-image-text", "uniir", "image"),
#     ("flickr", "uniir", "joint-image-text", "uniir", "text"),
# ]

# RUN_LIST = [ 

#     ("flickr", "flava", "image", "uniir", "joint-image-text"),
#     ("flickr", "flava", "text", "uniir", "joint-image-text"),
#     ("flickr", "minilm", "text", "uniir", "joint-image-text"),
#     ("flickr", "uniir", "image", "uniir", "joint-image-text"),
#     ("flickr", "uniir", "text", "uniir", "joint-image-text"),
   
#     ("coco", "flava", "image", "uniir", "joint-image-text"),
#     ("coco", "uniir", "image", "uniir", "joint-image-text"),
#     ("coco", "uniir", "text", "uniir", "joint-image-text"),
#     ("coco", "flava", "text", "uniir", "joint-image-text"),
#     ("coco", "minilm", "text", "uniir", "joint-image-text"),
# ]

# RUN_LIST = [ 
#     #   ("flickr", "uniir", "joint-image-text", "uniir", "image"),
#     #   ("flickr", "uniir", "joint-image-text", "uniir", "text"),
#     #   ("flickr", "uniir", "joint-image-text", "flava", "image"),
#     #   ("flickr", "uniir", "joint-image-text", "flava", "text"),
#     #   ("flickr", "uniir", "joint-image-text", "clip", "image"),
#     #   ("flickr", "uniir", "joint-image-text", "clip", "text"),
#     #   ("flickr", "uniir", "joint-image-text", "minilm", "text"),
#     #   ("flickr", "uniir", "joint-image-text", "bm25", "text"),

#       ("flickr", "uniir", "image", "uniir", "joint-image-text"),
#       ("flickr", "uniir", "text", "uniir", "joint-image-text"),
#       ("flickr", "flava", "image", "uniir", "joint-image-text"),
#     #   ("flickr", "flava", "text", "uniir", "joint-image-text"),
#       ("flickr", "clip", "image", "uniir", "joint-image-text"),
#     #   ("flickr", "clip", "text", "uniir", "joint-image-text"),
#       ("flickr", "minilm", "text", "uniir", "joint-image-text"),
#     #   ("flickr", "bm25", "text", "uniir", "joint-image-text"),

#     #   ("coco", "uniir", "joint-image-text", "uniir", "image"),
#     #   ("coco", "uniir", "joint-image-text", "uniir", "text"),
#     #   ("coco", "uniir", "joint-image-text", "flava", "image"),
#     #   ("coco", "uniir", "joint-image-text", "flava", "text"),
#     #   ("coco", "uniir", "joint-image-text", "clip", "image"),
#     #   ("coco", "uniir", "joint-image-text", "clip", "text"),
#     #   ("coco", "uniir", "joint-image-text", "minilm", "text"),
#     #   ("coco", "uniir", "joint-image-text", "bm25", "text"),

#       ("coco", "uniir", "image", "uniir", "joint-image-text"),
#       ("coco", "uniir", "text", "uniir", "joint-image-text"),
#       ("coco", "flava", "image", "uniir", "joint-image-text"),
#       ("coco", "flava", "text", "uniir", "joint-image-text"),
#       ("coco", "clip", "image", "uniir", "joint-image-text"),
#     #   ("coco", "clip", "text", "uniir", "joint-image-text"),
#     #   ("coco", "minilm", "text", "uniir", "joint-image-text"),
#     #   ("coco", "bm25", "text", "uniir", "joint-image-text"),
# ]


# (dataset, stage1_model, stage1_core_type, stage2_model, stage2_core_type)
#coco stage1 bm25 with all is pending, 
# RUN_LIST = [
#     ("flickr", "minilm", "text", "uniir", "text"),
#     ("flickr", "minilm", "text", "uniir", "image"),
#     ("flickr", "minilm", "text", "clip", "text"),
#     ("flickr", "minilm", "text", "clip", "image"),
#     ("flickr", "minilm", "text", "flava", "text"),
#     ("flickr", "minilm", "text", "flava", "image"),
#     # ("flickr", "minilm", "text", "bm25", "text"),

#     # ("flickr", "bm25", "text", "uniir", "text"),
#     # ("flickr", "bm25", "text", "uniir", "image"),
#     # ("flickr", "bm25", "text", "clip", "text"),
#     # ("flickr", "bm25", "text", "clip", "image"),
#     # ("flickr", "bm25", "text", "flava", "text"),
#     # ("flickr", "bm25", "text", "flava", "image"),
#     # ("flickr", "bm25", "text", "minilm", "text"),
    
#     ("flickr", "clip", "text", "uniir", "text"),
#     ("flickr", "clip", "text", "uniir", "image"),
#     ("flickr", "clip", "text", "minilm", "text"),
#     ("flickr", "clip", "text", "clip", "image"),
#     ("flickr", "clip", "text", "flava", "text"),
#     ("flickr", "clip", "text", "flava", "image"),
#     # ("flickr", "clip", "text", "bm25", "text"),
    
#     ("flickr", "clip", "image", "uniir", "text"),
#     ("flickr", "clip", "image", "uniir", "image"),
#     ("flickr", "clip", "image", "clip", "text"),
#     ("flickr", "clip", "image", "minilm", "text"),
#     ("flickr", "clip", "image", "flava", "text"),
#     ("flickr", "clip", "image", "flava", "image"),
#     # ("flickr", "clip", "image", "bm25", "text"),
    
#     ("flickr", "flava", "text", "uniir", "text"),
#     ("flickr", "flava", "text", "uniir", "image"),
#     ("flickr", "flava", "text", "clip", "text"),
#     ("flickr", "flava", "text", "clip", "image"),
#     ("flickr", "flava", "text", "minilm", "text"),
#     ("flickr", "flava", "text", "flava", "image"),
#     # ("flickr", "flava", "text", "bm25", "text"),
    
#     ("flickr", "flava", "image", "uniir", "text"),
#     ("flickr", "flava", "image", "uniir", "image"),
#     ("flickr", "flava", "image", "clip", "text"),
#     ("flickr", "flava", "image", "clip", "image"),
#     ("flickr", "flava", "image", "flava", "text"),
#     ("flickr", "flava", "image", "minilm", "text"),
#     # ("flickr", "flava", "image", "bm25", "text"),

#     ("flickr", "uniir", "text", "minilm", "text"),
#     ("flickr", "uniir", "text", "uniir", "image"),
#     ("flickr", "uniir", "text", "clip", "text"),
#     ("flickr", "uniir", "text", "clip", "image"),
#     ("flickr", "uniir", "text", "flava", "text"),
#     ("flickr", "uniir", "text", "flava", "image"),
#     # ("flickr", "uniir", "text", "bm25", "text"),
    
#     ("flickr", "uniir", "image", "uniir", "text"),
#     ("flickr", "uniir", "image", "minilm", "text"),
#     ("flickr", "uniir", "image", "clip", "text"),
#     ("flickr", "uniir", "image", "clip", "image"),
#     ("flickr", "uniir", "image", "flava", "text"),
#     ("flickr", "uniir", "image", "flava", "image"),
#     # ("flickr", "uniir", "image", "bm25", "text"),

#     ("coco", "minilm", "text", "uniir", "text"),
#     ("coco", "minilm", "text", "uniir", "image"),
#     ("coco", "minilm", "text", "clip", "text"),
#     ("coco", "minilm", "text", "clip", "image"),
#     ("coco", "minilm", "text", "flava", "text"),
#     ("coco", "minilm", "text", "flava", "image"),
#     # ("coco", "minilm", "text", "bm25", "text"),
    
#     ("coco", "clip", "text", "uniir", "text"),
#     ("coco", "clip", "text", "uniir", "image"),
#     ("coco", "clip", "text", "minilm", "text"),
#     ("coco", "clip", "text", "clip", "image"),
#     ("coco", "clip", "text", "flava", "text"),
#     ("coco", "clip", "text", "flava", "image"),
#     # ("coco", "clip", "text", "bm25", "text"),
    
#     ("coco", "clip", "image", "uniir", "text"),
#     ("coco", "clip", "image", "uniir", "image"),
#     ("coco", "clip", "image", "clip", "text"),
#     ("coco", "clip", "image", "minilm", "text"),
#     ("coco", "clip", "image", "flava", "text"),
#     ("coco", "clip", "image", "flava", "image"),
#     # ("coco", "clip", "image", "bm25", "text"),  
    
#     ("coco", "flava", "text", "uniir", "text"),
#     ("coco", "flava", "text", "uniir", "image"),
#     ("coco", "flava", "text", "clip", "text"),
#     ("coco", "flava", "text", "clip", "image"),
#     ("coco", "flava", "text", "minilm", "text"),
#     ("coco", "flava", "text", "flava", "image"),
#     # ("coco", "flava", "text", "bm25", "text"),
    
#     ("coco", "flava", "image", "uniir", "text"),
#     ("coco", "flava", "image", "uniir", "image"),
#     ("coco", "flava", "image", "clip", "text"),
#     ("coco", "flava", "image", "clip", "image"),
#     ("coco", "flava", "image", "flava", "text"),
#     ("coco", "flava", "image", "minilm", "text"),
#     # ("coco", "flava", "image", "bm25", "text"),

#     ("coco", "uniir", "text", "minilm", "text"),
#     ("coco", "uniir", "text", "uniir", "image"),
#     ("coco", "uniir", "text", "clip", "text"),
#     ("coco", "uniir", "text", "clip", "image"),
#     ("coco", "uniir", "text", "flava", "text"),
#     ("coco", "uniir", "text", "flava", "image"),
#     # ("coco", "uniir", "text", "bm25", "text"),
    
#     ("coco", "uniir", "image", "uniir", "text"),
#     ("coco", "uniir", "image", "minilm", "text"),
#     ("coco", "uniir", "image", "clip", "text"),
#     ("coco", "uniir", "image", "clip", "image"),
#     ("coco", "uniir", "image", "flava", "text"),
#     ("coco", "uniir", "image", "flava", "image"),
#     # ("coco", "uniir", "image", "bm25", "text")
    
# ]


# DB backend to pass to the script (change if needed)
DB = "faiss"

# Number of iterations: runs j in [ITER_START, ITER_END)
ITER_START = 0
ITER_END   = 1  # exclusive; set to e.g. 5 to repeat 5 times

# Which run types to execute
DO_SIMPLE_RUN = True
DO_RERANK_RUN  = False
RERANK_MODEL   = "blip2"  # used only if DO_RERANK_RUN True

# Optional stage K values (passed if you want)
STAGE1_K = 50
STAGE2_K = 10

# ------------------------------------------

# Load existing energy log (if any)
energy_log = []
energy_log_path = Path(ENERGY_LOG_PATH)
if energy_log_path.exists():
    try:
        with energy_log_path.open("r") as f:
            c = f.read().strip()
            if c:
                energy_log = json.loads(c)
    except Exception as e:
        print(f"Warning: could not load existing energy log: {e}")

# Helper to run one command via Popen and capture stdout/stderr
def _run_popen(cmd, stdin_text=""):
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate(stdin_text)
    return proc.returncode, stdout, stderr

# Main loop
if __name__ == "__main__":
    for j in range(ITER_START, ITER_END):
        print(f"\n===== ITERATION {j} =====")
        for i, (dataset, s1m, s1c, s2m, s2c) in enumerate(RUN_LIST, start=1):
            print(f"\n--- Config {i}: dataset={dataset}, S1={s1m}/{s1c}, S2={s2m}/{s2c} ---")

            # SIMPLE run: suffix format exactly like your previous code -> "_{j}"
            if DO_SIMPLE_RUN:
                suffix_simple = f"_{j}"
                cmd_simple = [
                    "python3", SCRIPT_PATH,
                    "--dataset", dataset,
                    "--stage1_model", s1m,
                    "--stage1_core_type", s1c,
                    "--stage2_model", s2m,
                    "--stage2_core_type", s2c,
                    "--db", DB,
                    "--stage1_k", str(STAGE1_K),
                    "--stage2_k", str(STAGE2_K),
                    "--suffix", suffix_simple
                ]

                # call via Popen (generate_metadata_two_stage.py expects no interactive stdin,
                # so we send empty input to keep the pattern identical to your other script)
                rc, out, err = _run_popen(cmd_simple, stdin_text="")

                if rc != 0:
                    print(f"[ERROR] SIMPLE run failed for {dataset} {s1m}->{s2m} (rc={rc})")
                    print(err)
                else:
                    print(f"[OK] SIMPLE run completed for {dataset} {s1m}->{s2m}")
                    # extract energy if printed by the inner script
                    m = re.search(r"BATCH ENERGY COMNSUMPTION\s*:\s*(\d+)", out, re.IGNORECASE | re.MULTILINE)
                    energy_val = int(m.group(1)) if m else 0  # default 0 if not found
                    out_fname = f"result_twoStage_{s1m}_{s1c}_{s2m}_{s2c}_{DB}{suffix_simple}.json"
                    out_path = Path(config["paths"]["project_root"]) / "results" / "two_stage_joint_faiss" /out_fname
                    energy_log.append({
                        "batch_energy": energy_val,
                        "iteration": j,
                        "output_file": str(out_path)
                    }) 


            # RERANK run: suffix "_rerank-<model>_{j}" and pass --reranking_model <model>
            if DO_RERANK_RUN:
                suffix_rerank = f"_rerank-{RERANK_MODEL}_{j}"
                cmd_rerank = [
                    "python3", SCRIPT_PATH,
                    "--dataset", dataset,
                    "--stage1_model", s1m,
                    "--stage1_core_type", s1c,
                    "--stage2_model", s2m,
                    "--stage2_core_type", s2c,
                    "--db", DB,
                    "--stage1_k", str(STAGE1_K),
                    "--stage2_k", str(STAGE2_K),
                    "--suffix", suffix_rerank,
                    "--reranking_model", RERANK_MODEL
                ]

                rc, out, err = _run_popen(cmd_rerank, stdin_text="")

                if rc != 0:
                    print(f"[ERROR] RERANK run failed for {dataset} {s1m}->{s2m} (rc={rc})")
                    print(err)
                else:
                    print(f"[OK] RERANK run completed for {dataset} {s1m}->{s2m}")
                    m = re.search(r"BATCH ENERGY COMNSUMPTION\s*:\s*(\d+)", out)
                    energy_val = int(m.group(1)) if m else None
                    out_fname = f"result_twoStage_{s1m}_{s1c}_{s2m}_{s2c}_{DB}{suffix_rerank}.json"
                    out_path = Path(config["paths"]["project_root"]) / "results" / out_fname
                    energy_log.append({
                        "batch_energy": energy_val,
                        "iteration": j,
                        "output_file": str(out_path)
                    })


    # Save energy log
    try:
        energy_log_path.parent.mkdir(parents=True, exist_ok=True)
        with energy_log_path.open("w") as f:
            json.dump(energy_log, f, indent=2)
        print(f"\nSaved energy log to {ENERGY_LOG_PATH}")
    except Exception as e:
        print(f"Warning: could not save energy log: {e}")
