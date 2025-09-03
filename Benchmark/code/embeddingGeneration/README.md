## 1. Embedding Generation

Embedding scripts are located in:

```

Benchmark/code/embeddingGeneration

````

These scripts generate vector representations (embeddings) of your data using a specified model and modality.

###  How to Use

```
python clip_model.py --modality image --dataset flickr
```

* `--modality` or `-m`: Choose between `text` or `image`.
* `--dataset` or `-d`: Choose between `flickr` or `COCO`.


## 2. FLAVA (Image + Text)

Generates **both image and text** embeddings in one run.

```bash
# Flickr
python code/embeddingGeneration/flava_embeddings.py \
  --dataset flickr --modality both --batch-img 32 --batch-text 128

# COCO
python code/embeddingGeneration/flava_embeddings.py \
  --dataset coco --modality both --batch-img 32 --batch-text 128
```

**Arguments:**

* `--modality`: `image`, `text`, or `both`
* `--batch-img`: Batch size for image embeddings
* `--batch-text`: Batch size for text embeddings

## 3. UNIIR CLIP-SF (Image / Caption / Joint)

Generates:

1. Image-only embeddings
2. Caption-only embeddings
3. Joint embeddings (score-level fusion of image + caption)

```bash
python code/embeddingGeneration/uniir_clip_sf_exporter.py \
  --dataset flickr \
  --arch ViT-L-14-quickgelu \
  --ckpt /path/to/clip_sf_large.pth \
  --batch-img 32 --batch-text 128 --w3 1.0 --w4 1.0
```

**Arguments:**

* `--dataset`: `flickr` or `coco`
* `--arch`: CLIP backbone architecture
* `--ckpt`: Path to UniIR checkpoint
* `--batch-img`: Batch size for image embeddings
* `--batch-text`: Batch size for caption/joint embeddings
* `--w3`, `--w4`: Fusion weights for image & text (candidate side)
* `--fp16`: Use half precision (optional)

