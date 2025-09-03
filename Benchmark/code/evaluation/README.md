
## 1.  Metadata Generation & Logging

Benchmark metadata is generated via:

```
Benchmark/code/evaluation/generate_metadata.py
```

This script logs:

* Query results
* Energy consumption
* Query time
* Encoding time
* Retrieval time

## 2.  Evaluation Metrics (Recall\@K)

To compute standard retrieval metrics such as Recall\@1, Recall\@5, and Recall\@10, use:

```
Benchmark/code/evaluation/evaluation_json.py
```
