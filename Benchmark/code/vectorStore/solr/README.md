

## 1. Ingesting Data into Solr

Solr 9.8.1 (with vector support) is used to store embeddings. Ingestion scripts are located at:

```
Benchmark/code/vectorStore/solr/ingestDataIntoSolr
```

### Scripts:

* `bm25_to_solr.py` – For classic BM25 indexing
* `image_modality_to_solr.py` – For indexing image embeddings
* `associated_text_modality_to_solr.py` – For indexing text embeddings

###  How to Use

```
python image_modality_to_solr.py --modality text --dataset flickr --model clip
```

* `--modality` or `-m`: Either `text` or `image`
* `--dataset` or `-d`: `flickr` or `coco`
* `--model` or `-M`: `clip` or `minilm`

## 2. Retrieval from Solr

Retrieval functions are located at:

```
Benchmark/code/vectorStore/solr/Retriveal
```

These are **not standalone scripts**. They are **imported and used inside the Solr retrieval service (query_solr.py)**.


## 4. FastAPI-Based Retrieval Service

Solr is served via FastAPI at:

```
Benchmark/code/retrievalService/solr_base_service/query_solr.py
```

###  How to Start the Service

```
uvicorn query_solr:app --port 5050
```

* Base URL: `http://localhost:5050`
* Swagger UI: `http://localhost:5050/docs`

