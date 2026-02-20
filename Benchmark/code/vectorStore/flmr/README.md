# FLMR Index Building

## Aim
This directory contains scripts to build ColBERT-format indices for PreFLMR. It creates indices for caption, joint, and image retrieval.

## Usage

### `build_preflmr_colbert_indices_all3.py`
Builds official ColBERT-format indices for PreFLMR.
```bash
python build_preflmr_colbert_indices_all3.py --dataset [coco|flickr] --index-name-prefix [str] --nbits [int] --doc-maxlen [int] --indexing-batch-size [int] --nranks [int] --overwrite --use-gpu
```
