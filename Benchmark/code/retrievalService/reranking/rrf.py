# # rrf_fusion.py
# import requests
# from typing import List, Dict

# def run_rrf_fusion(query: str, url1: str, url2: str, k: int = 10, rrf_k: int = 60) -> List[Dict]:
#     def fetch_results(url):
#         response = requests.get(url, params={"q": query})
#         response.raise_for_status()
#         data = response.json()
#         return data.get("list_of_top_k", [])  # <- FIXED HERE

#     results1 = fetch_results(url1)
#     results2 = fetch_results(url2)

#     score_dict = {}

#     for rank, item in enumerate(results1[:k]):
#         img = item["image_path"]  # <- USE image_path as key
#         score_dict[img] = score_dict.get(img, 0) + 1 / (rrf_k + rank+1)

#     for rank, item in enumerate(results2[:k]):
#         img = item["image_path"]
#         score_dict[img] = score_dict.get(img, 0) + 1 / (rrf_k + rank+1)

#     # Collect metadata (caption, abs path)
#     metadata_map = {item["image_path"]: item for item in results1 + results2}
#     fused = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

#     fused_result = []
#     for img, score in fused[:k]:
#         item = metadata_map.get(img, {"image_path": img})
#         item["rrf_score"] = round(score, 4)
#         fused_result.append(item)

#     return fused_result

# rrf_fusion.py

import requests
from typing import List, Dict 

proxies = {
            'http': '',
            'https': ''
        }

def run_rrf_fusion(query: str, urls: List[str], k: int = 10, rrf_k: int = 60) -> List[Dict]:
    def fetch_results(url):
        response = requests.get(url, params={"q": query},proxies=proxies)
        response.raise_for_status()
        data = response.json()
        return data.get("list_of_top_k", []) 

    score_dict = {}
    metadata_map = {}

    for url in urls:
        results = fetch_results(url)
        for rank, item in enumerate(results[:k]):
            img = item["image_path"]
            score_dict[img] = score_dict.get(img, 0) + 1 / (rrf_k + rank + 1)
            metadata_map[img] = item  

    fused = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    fused_result = []
    for img, score in fused[:k]:
        item = metadata_map.get(img, {"image_path": img})
        item["rrf_score"] = round(score, 4)
        fused_result.append(item)

    return fused_result

