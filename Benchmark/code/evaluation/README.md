# Evaluation Scripts

This folder contains scripts for evaluating the performance of various retrieval models (Single, RRF, Two-Stage) and measuring their energy consumption.

## 📂 Directory Structure

- **`Results/`**: Directory where evaluation results (JSON files) are typically saved.
- **`energy.py`**: Utility to measure GPU energy consumption.
- **`time_util.py`**: Utility to measure execution time (supports CUDA synchronization).

## 📜 Key Scripts & Usage

### 1. Metadata Generation (Running Queries)

These scripts query the retrieval endpoints and save the results (top-k images, timing, energy) to JSON files.

#### `generate_metadata.py`
**Purpose**: Runs queries against a single endpoint or an RRF fusion of endpoints. Measures time and energy.
**Usage**:
```bash
python3 generate_metadata.py --dataset <dataset> --suffix <suffix> [--rerankingmodel <model>] [--k <k>]
```
- Interactive mode: Run without arguments to interactively select "single" or "rrf" mode.
- Arguments:
    - `--dataset`: `coco` or `flickr`
    - `--suffix`: Suffix for the output filename (e.g., `_0`).
    - `--rerankingmodel`: Optional neural reranking model (e.g., `blip2`).
    - `--k`: Number of results to fetch (default from config).

#### `generate_metadata_all.py`
**Purpose**: Batch runs `generate_metadata.py` for all single endpoints defined in the config.
**Usage**:
```bash
python3 generate_metadata_all.py
```
- Iterates through endpoints and runs evaluation.
- Updates `energy_log.json` with batch energy consumption.

#### `generate_metadata_two_stage.py`
**Purpose**: Runs queries for a two-stage retrieval pipeline (e.g., Stage 1: CLIP, Stage 2: UniIR).
**Usage**:
```bash
python3 generate_metadata_two_stage.py --dataset <dataset> \
    --stage1_model <model> --stage1_core_type <type> \
    --stage2_model <model> --stage2_core_type <type> \
    --db <solr|faiss> [--reranking_model <model>]
```
- Example:
  ```bash
  python3 generate_metadata_two_stage.py --dataset coco --stage1_model clip --stage1_core_type text --stage2_model uniir --stage2_core_type joint-image-text --db faiss
  ```

#### `two_stage_all.py`
**Purpose**: Batch runner for `generate_metadata_two_stage.py`.
**Usage**:
```bash
python3 two_stage_all.py
```
- Configured via `RUN_LIST` variable inside the script.
- Supports "SIMPLE" runs and "RERANK" runs.

### 2. RRF Evaluation (Fusion)

#### `rrf_evaluation.py`
**Purpose**: Generates RRF results for combinations of endpoints (Solr/Faiss).
**Usage**:
```bash
python3 rrf_evaluation.py
```
- Automatically generates pairs of endpoints for RRF fusion.

#### `rrf_evaluation2.py`
**Purpose**: Similar to `rrf_evaluation.py` but uses an **explicit list** of RRF pairs defined in `RRF_PAIRS`.
**Usage**:
```bash
python3 rrf_evaluation2.py
```

### 3. Metric Calculation (R@K)

These scripts calculate Recall@1, Recall@5, and Recall@10 from the generated JSON result files.

#### `evaluation_json.py`
**Purpose**: Calculates metrics for a **single** result JSON file.
**Usage**:
```bash
python3 evaluation_json.py --result <path_to_json> --dataset <dataset>
```

#### `evaluation_json_all.py`
**Purpose**: Batch evaluates **all** `result_twoStage*.json` files in a directory and saves a CSV report.
**Usage**:
```bash
python3 evaluation_json_all.py --results_dir <dir> --output_csv <file.csv>
```

#### `evaluation_json_rrf_all.py`
**Purpose**: Batch evaluates **all** `results_rrf*.json` files in a directory and saves a CSV report.
**Usage**:
```bash
python3 evaluation_json_rrf_all.py --results_dir <dir> --output_csv <file.csv>
```

### 4. Other Evaluation Scripts

- **`evaluation.py`**: Simple script to evaluate a single service URL (hardcoded in script) against ground truth.
- **`evalation_updated.py`**: Updated version of evaluation logic, supports single and RRF modes via hardcoded endpoints.
- **`test_eval.py`**: Quick test script for specific hardcoded endpoints (Flava/UniIR).
- **`test_evalu.py`**: Evaluation script for "preflmr" or similar variants, supports `variant` parameter.

## 🛠 Dependencies

- `requests`
- `tqdm`
- `pandas`
- `pynvml` (for energy measurement)
- `torch` (for time synchronization)
- Project config (`Benchmark.config.config_utils`)

## 📝 Notes

- Ensure the backend services (Solr/Faiss/Flask) are running before executing metadata generation scripts.
- Scripts often assume a specific directory structure for config and data (e.g., `/mnt/storage/...`). Check `sys.path.append` and config paths if running in a new environment.
