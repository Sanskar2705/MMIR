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

# import requests 
# from typing import List, Dict 
# from Benchmark.code.evaluation.time_util import get_time

# proxies = {
#             'http': '',
#             'https': ''
#         }

# def run_rrf_fusion(query: str, urls: List[str], k: int = 10, rrf_k: int = 60) -> List[Dict]:
#     def fetch_results(url):
#         response = requests.get(url, params={"q": query},proxies=proxies)
#         response.raise_for_status()
#         data = response.json()
#         return data.get("list_of_top_k", []) 

#     score_dict = {}
#     metadata_map = {}
#     rrf_time = get_time()
#     for url in urls:
#         start_time = get_time()
#         for rank, item in enumerate(results[:k]):
#             img = item["image_path"]
#             score_dict[img] = score_dict.get(img, 0) + 1 / (rrf_k + rank + 1)
#             metadata_map[img] = item  

#     fused = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
#     rrf_time = get_time() - rrf_time

#     fused_result = []
#     for img, score in fused[:k]:
#         item = metadata_map.get(img, {"image_path": img})
#         item["rrf_score"] = round(score, 4)
#         fused_result.append(item)

#     return fused_result, rrf_time

# #code work
# import requests
# from typing import List, Dict
# from Benchmark.code.evaluation.time_util import get_time

# def run_rrf_fusion(query: str, urls: List[str], k: int = 10, rrf_k: int = 60) -> List[Dict]:
#     def fetch_results(url):
#         response = requests.get(url, params={"q": query})
#         response.raise_for_status()
#         data = response.json()
#         return data.get("list_of_top_k", [])  # standard format

#     score_dict = {}
#     metadata_map = {}
#     for url in urls:
#         results = fetch_results(url)
#         rrf_time = get_time()
#         for rank, item in enumerate(results[:k]):
#             img = item["image_path"]
#             score_dict[img] = score_dict.get(img, 0) + 1 / (rrf_k + rank + 1)
#             metadata_map[img] = item  

#     fused = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
#     rrf_time = get_time() - rrf_time

#     fused_result = []
#     for img, score in fused[:k]:
#         item = metadata_map.get(img, {"image_path": img})
#         item["rrf_score"] = round(score, 4)
#         fused_result.append(item)

#     return fused_result, rrf_time


import httpx
import asyncio
from typing import List, Dict, Tuple
from Benchmark.code.evaluation.time_util import get_time
# proxies = {
#             'http': '',
#             'https': ''
#         }
async def run_rrf_fusion(
    query: str,
    urls: List[str],
    k: int = 10,                 # final output size
    candidate_k: int = 30,       # per-model fetch size
    rrf_k: int = 60
) -> Tuple[List[Dict], float]:

    async def fetch_results(client, url):
        response = await client.get(url, params={"q": query, "k": candidate_k})
        response.raise_for_status()
        data = response.json()
        return data.get("list_of_top_k", [])

    score_dict = {}
    metadata_map = {}

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [fetch_results(client, url) for url in urls]
        all_results = await asyncio.gather(*tasks)

    rrf_start = get_time()

    for results in all_results:
        for rank, item in enumerate(results[:candidate_k]):
            img = item["image_path"]
            score_dict[img] = score_dict.get(img, 0) + 1 / (rrf_k + rank + 1)
            metadata_map[img] = item

    rrf_time = get_time() - rrf_start

    fused = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    fused_result = []
    for img, score in fused[:k]:   #  final output limited to top-k
        item = metadata_map.get(img, {"image_path": img})
        item["rrf_score"] = float(score)  # keep full precision if you want
        fused_result.append(item)

    return fused_result, rrf_time

# async def run_rrf_fusion(query: str, urls: List[str], k: int = 10, rrf_k: int = 60 ) -> Tuple[List[Dict], float]:

#     async def fetch_results(client, url):
#         response = await client.get(url, params={"q": query})
#         response.raise_for_status()
#         data = response.json()
#         return data.get("list_of_top_k", [])

#     score_dict = {}
#     metadata_map = {}

#     async with httpx.AsyncClient(timeout=30) as client:
#         tasks = [fetch_results(client, url) for url in urls]
#         all_results = await asyncio.gather(*tasks)

#     rrf_start = get_time()

#     for results in all_results:
#         for rank, item in enumerate(results[:k]):
#             img = item["image_path"]
#             score_dict[img] = score_dict.get(img, 0) + 1 / (rrf_k + rank + 1)
#             metadata_map[img] = item

#     rrf_time = get_time() - rrf_start

#     fused = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

#     fused_result = []
#     for img, score in fused[:k]:
#         item = metadata_map.get(img, {"image_path": img})
#         item["rrf_score"] = round(score, 4)
#         fused_result.append(item)

#     return fused_result, rrf_time
