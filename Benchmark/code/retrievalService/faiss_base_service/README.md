The service currently supports the following models
1) FLAVA
2) UNI-IR

# Setup
To install the dependenices please setup a python environment using requirements.txt file given above.

## Start the server

```bash
uvicorn faiss_retrieval_server:app --host 0.0.0.0 --port 5052 --reload
```

**Endpoints**

* `GET /available` → lists discovered indexes
* `GET /{model}_{target}_{dataset}_faiss?q=...&k=10`

  * `model ∈ {uniir, flava}`
  * `target ∈ {image, caption, joint}`
  * `dataset ∈ {coco, flickr}`
* `POST /refresh` → reloads `config.yaml`, rescans indices, clears embedder cache

**Example**

```bash
curl "http://localhost:5052/uniir_joint_coco_faiss?q=a+red+bus&k=10"
curl -X POST "http://localhost:5052/refresh"
```

