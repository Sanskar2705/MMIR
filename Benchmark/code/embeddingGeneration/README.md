# Embedding Generation

## Aim
This directory contains scripts for generating embeddings from various multi-modal models. These embeddings are used for downstream retrieval tasks.

## Usage

### `clip_model.py`
Generates CLIP embeddings for images or text.
```bash
python clip_model.py --modality [text|image] --dataset [coco|flickr]
```

### `flava_model.py`
Generates FLAVA embeddings for images, text, or both.
```bash
python flava_model.py --dataset [coco|flickr] --modality [image|text|both] --batch-img [int] --batch-text [int]
```

### `miniLM_model.py`
Generates MiniLM embeddings (text only).
```bash
python miniLM_model.py --modality text --dataset [coco|flickr]
```

### `test_preflmr.py`
Runs PreFLMR indexing/embedding generation.
```bash
python test_preflmr.py --dataset [coco|flickr] --checkpoint [path] --image-processor [path] --index-root [path] --experiment [name] --index-name [name] --nbits [int] --doc-maxlen [int] --use-gpu
```

### `uniir_model.py`
Generates UniIR embeddings with support for different variants (CLIP-SF, BLIP-FF).
```bash
python uniir_model.py --dataset [coco|flickr] --variant [clip_sf|blip_ff] --modality [all|image|text|joint] --batch-img [int] --batch-text [int] --fp16 --w3 [float] --w4 [float] --suffix [str]
```

### `uniir_model_new.py`
Newer version of UniIR embedding generation.
```bash
python uniir_model_new.py --dataset [coco|flickr] --modality [all|image|text|joint] --batch-img [int] --batch-text [int] --fp16 --w3 [float] --w4 [float]
```
