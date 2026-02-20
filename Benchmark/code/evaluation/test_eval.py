#!/usr/bin/env python3
import json, requests, os
from   tqdm import tqdm

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
proxies = {"http": "", "https": ""}

# ───────── endpoints to test ─────────
ENDPOINTS = {
    "flava‑flickr‑image"   : "http://localhost:5051/search/flava/flickr/image",
    "flava‑flickr‑caption" : "http://localhost:5051/search/flava/flickr/caption",
    "uniir‑flickr‑image"   : "http://localhost:5051/search/uniir/flickr/image",
    "uniir‑flickr‑caption" : "http://localhost:5051/search/uniir/flickr/caption",
}
# ─────────────────────────────────────

ANNOTATION_FILE = (
    "/mnt/storage/RSystemsBenchmarking/data/datasets/vision/flickr30k/"
    "annotations/test.json"
)
TOP_K       = 10
MAX_SAMPLES = None          # None = evaluate all entries


def evaluate(service_url: str) -> None:
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    r1 = r5 = r10 = total = 0

    for entry in tqdm(data[:MAX_SAMPLES], leave=False):
        image_gt      = entry["image"]
        query_caption = entry["caption"][4].strip()

        try:
            resp = requests.get(service_url,
                                params={"q": query_caption, "k": TOP_K},
                                proxies=proxies,
                                timeout=30)
            resp.raise_for_status()
            results = resp.json()["list_of_top_k"]
        except Exception as e:
            print(f"[warn] query failed → {e}")
            continue

        retrieved = [r["image_path"] for r in results]

        total += 1
        if image_gt in retrieved[:1]:   r1  += 1
        if image_gt in retrieved[:5]:   r5  += 1
        if image_gt in retrieved[:10]:  r10 += 1

    print(f"\nEndpoint: {service_url}")
    print(f"Samples:  {total}")
    if total == 0:
        print("No valid responses.\n" + "-"*28)
        return
    print(f"R@1   {r1/total:7.4f}")
    print(f"R@5   {r5/total:7.4f}")
    print(f"R@10  {r10/total:7.4f}")
    print("-" * 28)


if __name__ == "__main__":
    for name, url in ENDPOINTS.items():
        print(f"\n=== Evaluating {name} ===")
        evaluate(url)
