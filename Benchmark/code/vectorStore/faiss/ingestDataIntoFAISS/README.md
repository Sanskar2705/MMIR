# FAISS Index Building

## Aim
This directory contains scripts to ingest embeddings into FAISS indices. It supports building indices for images, text/captions, and joint embeddings.

## Usage

### `build_faiss_updated.py`
Builds FAISS indices from JSON embeddings.
```bash
python build_faiss_updated.py --dataset [coco|flickr] --model [str] --in-dir [dir] --out-dir [dir]
```

<!-- ### `new_build_faiss.py`
A newer script to build FAISS indices, supporting cases where only image or text embeddings are present.
```bash
python new_build_faiss.py --dataset [coco|flickr] --model [str] --in-dir [dir] --out-dir [dir]
``` -->
