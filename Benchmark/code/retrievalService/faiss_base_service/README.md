# FAISS Retrieval Service

## Aim
This directory contains the base service for FAISS-based retrieval. It provides a FastAPI server to handle retrieval requests using pre-built FAISS indices.

## Usage

### `faiss_retrieval_server_updated.py`
Starts the FAISS retrieval server.
```bash
uvicorn faiss_retrieval_server_updated:app --host 0.0.0.0 --port 5053
```
(Note: This file is typically run using `uvicorn` as it defines a FastAPI app.)

### `embed_utils.py`
A utility module containing the model loaders and embedding functions. It is not meant to be run directly but is imported by the server scripts.
