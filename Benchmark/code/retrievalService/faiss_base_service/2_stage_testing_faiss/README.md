# FAISS Two-Stage Testing

## Aim
This directory contains scripts for testing and orchestrating a two-stage retrieval pipeline using FAISS. It includes an orchestrator for interactive testing, a server for the two-stage pipeline, and a client for staged execution.

## Usage

### `faiss_two_stage_orchestrator.py`
Runs an interactive CLI for testing the two-stage retrieval pipeline.
```bash
python faiss_two_stage_orchestrator.py
```

### `faiss_two_stage_server.py`
Starts the two-stage retrieval server.
```bash
uvicorn faiss_two_stage_server:app --host 0.0.0.0 --port 5059
```

### `faiss_two_stage_staged_client.py`
A client script to run the two-stage retrieval in two separate calls (Stage 1 only -> Stage 2 from file).
```bash
python faiss_two_stage_staged_client.py --dataset [coco|flickr] --stage1_model [str] --stage1_core_type [text|image|joint-image-text] --stage2_model [str] --stage2_core_type [text|image|joint-image-text] --out [file] --save_dir [dir]
```
