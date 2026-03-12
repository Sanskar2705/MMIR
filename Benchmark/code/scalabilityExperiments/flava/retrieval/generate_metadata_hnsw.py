#!/usr/bin/env python3
import os
import json
import tarfile
import argparse
import requests
from tqdm import tqdm

import pynvml

import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.code.evaluation.time_util import get_time


# -------- disable proxies for local requests --------
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
PROXIES = {"http": "", "https": ""}


# -------- GPU energy --------
handle = None

def get_gpu_energy():
    # returns millijoules (mJ) on supported GPUs
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


def iter_captions_from_tar(tar_path: str, n: int):
    """Yield up to n captions from one tar."""
    with tarfile.open(tar_path, "r") as tf:
        members = sorted([m for m in tf.getnames() if m.endswith(".json")])
        count = 0
        for m in members:
            if count >= n:
                break
            f = tf.extractfile(m)
            if not f:
                continue
            try:
                d = json.loads(f.read().decode("utf-8", errors="ignore"))
                cap = (d.get("caption") or "").strip()
                if cap:
                    yield cap
                    count += 1
            except Exception:
                continue


def iter_captions_from_tar_dir(tar_dir: str, per_tar: int, max_total: int):
    """Yield captions across many tars until max_total reached."""
    tars = sorted([os.path.join(tar_dir, x) for x in os.listdir(tar_dir) if x.endswith(".tar")])
    seen = 0
    for tp in tars:
        for cap in iter_captions_from_tar(tp, per_tar):
            yield cap
            seen += 1
            if seen >= max_total:
                return


# def query_service(url: str, query: str, k: int):
#     r = requests.get(url, params={"q": query, "k": k}, proxies=PROXIES, timeout=120)
#     r.raise_for_status()
#     return r.json()
def query_service(url: str, query: str, k: int, ef: int):
    r = requests.get(url, params={"q": query, "k": k, "ef": ef}, proxies=PROXIES, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar_dir", default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/laion/tar/Worker0/tars")
    ap.add_argument("--service_url", default="http://127.0.0.1:8001/flava_faiss_search")
    ap.add_argument("--out", default="laion_flava_faiss_metadata_10k.json")
    ap.add_argument("--per_tar", type=int, default=20)
    ap.add_argument("--max_queries", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=128) 
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # ---- NVML init ----
    pynvml.nvmlInit()
    global handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    batch_start_energy = get_gpu_energy()
    batch_start_time = get_time()

    captions = list(iter_captions_from_tar_dir(args.tar_dir, args.per_tar, args.max_queries))
    print(f"[INFO] Loaded {len(captions)} captions from tar files")

    output = []

    for cap in tqdm(captions):
        per_query_log = {"query": cap}

        try:
            start_time = get_time()
            start_energy = get_gpu_energy()

            # results = query_service(args.service_url, cap, args.k)
            results = query_service(args.service_url, cap, args.k, args.ef)

            end_time = get_time()
            end_energy = get_gpu_energy()

            duration = end_time - start_time
            energy_util = end_energy - start_energy

            per_query_log["running_time_without_reranking"] = float(duration)
            per_query_log["running_time"] = float(duration)
            per_query_log["energy_util"] = float(energy_util)

            per_query_log["reranking_time"] = 0
            per_query_log["reranking_energy"] = 0

            for key in results:
                if key != "list_of_top_k":
                    per_query_log[key + "_service"] = results[key]
                else:
                    per_query_log["list_of_top_k"] = results[key]

        except Exception as e:
            print(f"[ERROR] query failed: {e}")
            per_query_log["running_time_without_reranking"] = None
            per_query_log["running_time"] = None
            per_query_log["energy_util"] = None
            per_query_log["reranking_time"] = 0
            per_query_log["reranking_energy"] = 0
            per_query_log["list_of_top_k"] = []

        output.append(per_query_log)

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    batch_end_energy = get_gpu_energy()
    batch_end_time = get_time()

    print(f"[DONE] saved => {args.out}")
    print(f"[BATCH] total_time = {batch_end_time - batch_start_time:.4f} sec")
    print(f"[BATCH] total_energy = {batch_end_energy - batch_start_energy:.4f} mJ")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
