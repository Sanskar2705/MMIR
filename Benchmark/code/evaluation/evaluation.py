import json
import requests
from tqdm import tqdm
import os
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

proxies = {
            'http': '',
            'https': ''
        }


# CONFIG
# SERVICE_URL_IMAGE = "http://localhost:5005/ClipImage_mscoco"
# SERVICE_URL_IMAGE = "http://localhost:8083/rerank_with_blip2_ziv"
# SERVICE_URL_IMAGE = "http://localhost:8080/flava_text2img_flickr30k"
# SERVICE_URL_IMAGE = "http://localhost:8101/coco-flava-text"
# SERVICE_URL_IMAGE = "http://localhost:8100/coco-flava-image"
# SERVICE_URL_IMAGE = "http://localhost:8102/flickr-flava-image"
# SERVICE_URL_IMAGE = "http://localhost:8103/flickr-flava-text"

# CLIP_IMG_FLICKR
# SERVICE_URL_IMAGE = "http://localhost:8083/rerank_mscoco_clip_only"
# SERVICE_URL_IMAGE = "http://localhost:5005/ClipImage_flickr"
# SERVICE_URL_IMAGE = "http://localhost:5052/uniir_joint_coco_faiss"
#SERVICE_URL_IMAGE = "http://localhost:8084/neuralrerank_blip2_withendpoint"
SERVICE_URL_IMAGE = "http://localhost:5053/uniir_joint-image-text_flickr_faiss" 
# test_endpoint = "http://localhost:5050/clip_image_coco_solr"
# test_endpoint = "http://localhost:5052/uniir_caption_flickr_faiss"
test_endpoint = "http://localhost:5053/uniir_joint-image-text_flickr_faiss"

# SERVICE_URL_CAPTION = "http://localhost:5005/ClipCaption_mscoco"
#ANNOTATION_FILE = "/mnt/storage/RSystemsBenchmarking/data/datasets/coco/annotations/coco_karpathy_test.json"
ANNOTATION_FILE = "/mnt/storage/RSystemsBenchmarking/data/datasets/vision/flickr30k/annotations/test.json"

TOP_K = 10


MAX_SAMPLES = None  # Set None to use all entries

def evaluate(service_url):
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    r1, r5, r10 = 0, 0, 0
    total = 0

    for entry in tqdm(data[:MAX_SAMPLES]):
        image_gt = entry["image"]
        query_caption = entry["caption"][4].strip()

        try:
            response = requests.get(service_url, params={"q": query_caption,"methods":[test_endpoint]},proxies=proxies)
            response.raise_for_status()
            print(response.json())
            results = response.json()["list_of_top_k"]
        except Exception as e:
            print(f"Error with query: {query_caption} -> {e}")
            continue

        # Get predicted image names
        retrieved_images = [r["image_path"] for r in results]

        total += 1
        if image_gt in retrieved_images[:1]:
            r1 += 1
        if image_gt in retrieved_images[:5]:
            r5 += 1
        if image_gt in retrieved_images[:10]:
            r10 += 1

    print(f"Evaluated on {total} samples")
    print(f"R@1:  {r1 / total:.4f}")
    print(f"R@5:  {r5 / total:.4f}")
    print(f"R@10: {r10 / total:.4f}")

print(" Evaluating CLIP Image Endpoint")
evaluate(SERVICE_URL_IMAGE)

# print("\n Evaluating CLIP Caption Endpoint")
# evaluate(SERVICE_URL_CAPTION)
