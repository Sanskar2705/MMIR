import json
import requests
from tqdm import tqdm
import os

# Disable proxies for local requests
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# CONFIG
ENDPOINTS = {
    "ClipImage_mscoco": "http://localhost:5004/ClipImage_mscoco",
    "ClipCaption_mscoco": "http://localhost:5004/ClipCaption_mscoco",
    "ClipImage_flickr": "http://localhost:5004/ClipImage_flickr",
    "ClipCaption_flickr": "http://localhost:5004/ClipCaption_flickr",
    "BM25Caption_mscoco": "http://localhost:5004/BM25Caption_mscoco",
    "BM25Caption_flickr": "http://localhost:5004/BM25Caption_flickr",
    "MiniLmCaption_mscoco": "http://localhost:5004/MiniLmCaption_mscoco",
    "MiniLmCaption_flickr": "http://localhost:5004/MiniLmCaption_flickr",
    "ClipCaptionImage_mscoco": "http://localhost:5004/ClipCaptionImage_mscoco",
    
}

FUSION_ENDPOINT = "http://localhost:5004/rrf_fusion"

ANNOTATION_FILES = {
    "coco": "/mnt/storage/sanskar/samy/Benchmark/Benchmark/code/Split_query/coco_karpathy_test_query.json",
    "flickr": "/mnt/storage/sanskar/samy/Benchmark/Benchmark/code/Split_query/flickr_test_query.json",
}

TOP_K = 10
MAX_SAMPLES = None  # Set to None to use all entries


def evaluate(service_url, annotation_file, methods=None):
    with open(annotation_file, "r") as f:
        data = json.load(f)

    r1 = r5 = r10 = total = 0
    for entry in tqdm(data[:MAX_SAMPLES]):
        image_gt = entry["image"]
        query_caption = entry["caption"]
        try:
            if methods:
                # Fusion Case
                response = requests.get(service_url, params={"q": query_caption, "methods": methods})
                response.raise_for_status()
                results = response.json()["list_of_top_k"]
            else:
                # Single model retrieval
                response = requests.get(service_url, params={"q": query_caption})
                response.raise_for_status()
                results = response.json()["list_of_top_k"]
        except Exception as e:
            print(f"Error with query: {query_caption} -> {e}")
            continue

        retrieved_images = [r["image_path"] for r in results]

        total += 1
        if image_gt in retrieved_images[:1]:
            r1 += 1
        if image_gt in retrieved_images[:5]:
            r5 += 1
        if image_gt in retrieved_images[:10]:
            r10 += 1

    print(f"\nEvaluated on {total} samples")
    print(f"R@1:  {r1 / total:.4f}")
    print(f"R@5:  {r5 / total:.4f}")
    print(f"R@10: {r10 / total:.4f}")


def main():
    mode = input("Select mode ('single' or 'rrf'): ").strip().lower()

    if mode == "single":
        print("\nAvailable Single Endpoints:")
        for i, name in enumerate(ENDPOINTS.keys(), start=1):
            print(f"{i}. {name}")

        selection = input("\nEnter number or endpoint name: ").strip()

        if selection.isdigit():
            selection = list(ENDPOINTS.keys())[int(selection) - 1]

        if selection not in ENDPOINTS:
            print("Invalid endpoint. Exiting.")
            return

        url = ENDPOINTS[selection]
        dataset_type = "flickr" if "flickr" in selection.lower() else "coco"
        annotation_file = ANNOTATION_FILES[dataset_type]

        print(f"\nEvaluating: {selection}")
        evaluate(url, annotation_file)

    elif mode == "rrf":
        print("\nEnter method names for RRF fusion (comma-separated, must match endpoint keys):")
        print("Available options:")
        for name in ENDPOINTS:
            print(f" - {name}")

        method_input = input("\nMethods (comma-separated): ").strip()
        methods = [m.strip() for m in method_input.split(",") if m.strip()]

        if not methods or any(m not in ENDPOINTS for m in methods):
            print("One or more method names are invalid. Exiting.")
            return

        dataset_type = "flickr" if any("flickr" in m.lower() for m in methods) else "coco"
        annotation_file = ANNOTATION_FILES[dataset_type]

        print(f"\nEvaluating RRF fusion with: {', '.join(methods)}")
        evaluate(FUSION_ENDPOINT, annotation_file, methods=methods)

    else:
        print("Invalid mode. Choose either 'single' or 'rrf'.")

if __name__ == "__main__":
    main()
