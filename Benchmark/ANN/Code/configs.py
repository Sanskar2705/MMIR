"""ANN parameter grids tuned for small corpora (COCO ~5k, Flickr ~1k)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

EncoderName = Literal["clip_image", "openclip_image", "uniir_joint"]
DatasetName = Literal["coco", "flickr"]
IndexType = Literal["hnsw", "ivfpq", "oivfpq", "knn"]


@dataclass(frozen=True)
class HNSWConfig:
    m: int
    ef_construction: int
    ef_search: int

    @property
    def tag(self) -> str:
        return f"m{self.m}_efc{self.ef_construction}_efs{self.ef_search}"


@dataclass(frozen=True)
class IVFPQConfig:
    nlist: int
    m: int
    nbits: int
    nprobe: int

    @property
    def tag(self) -> str:
        return f"nlist{self.nlist}_m{self.m}_nbits{self.nbits}_nprobe{self.nprobe}"


@dataclass(frozen=True)
class OIVFPQConfig:
    factory_spec: str
    nprobe: int

    @property
    def tag(self) -> str:
        cleaned = (
            self.factory_spec.replace(",", "_")
            .replace("OPQ", "opq")
            .replace("IVF", "ivf")
            .replace("PQ", "pq")
        )
        return f"{cleaned}_nprobe{self.nprobe}"


ENCODERS: List[EncoderName] = ["clip_image", "openclip_image", "uniir_joint"]
DATASETS: List[DatasetName] = ["coco", "flickr"]

EMBEDDING_FILES: Dict[EncoderName, Dict[DatasetName, str]] = {
    "clip_image": {
        "coco": "clip_coco_image_embeddings.json",
        "flickr": "clip_flickr_image_embeddings.json",
    },
    "openclip_image": {
        "coco": "openclip_coco_image_embeddings.json",
        "flickr": "openclip_flickr_image_embeddings.json",
    },
    "uniir_joint": {
        "coco": "uniir_coco_joint-image-text_embeddings.json",
        "flickr": "uniir_flickr_joint-image-text_embeddings.json",
    },
}

QUERY_FILES: Dict[DatasetName, str] = {
    "coco": "/mnt/storage/RSystemsBenchmarking/data/datasets/coco/processed_mapping/coco_karpathy_test_query.json",
    "flickr": "/mnt/storage/RSystemsBenchmarking/data/datasets/vision/flickr30k/processed_mapping/flickr_test_query.json",
}

INGEST_FILES: Dict[DatasetName, str] = {
    "coco": "/mnt/storage/RSystemsBenchmarking/data/datasets/coco/processed_mapping/coco_karpathy_test_ingest.json",
    "flickr": "/mnt/storage/RSystemsBenchmarking/data/datasets/vision/flickr30k/processed_mapping/flickr_test_ingest.json",
}

BASE_IMAGE_DIRS: Dict[DatasetName, str] = {
    "coco": "/mnt/storage/RSystemsBenchmarking/data/datasets/coco/images/val2014/",
    "flickr": "/mnt/storage/RSystemsBenchmarking/data/datasets/vision/flickr30k/images/flickr30k-images/flickr30k-images/",
}

EMBEDDING_ROOT = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/embeddings"
ANN_ROOT = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/ANN"
INDEX_ROOT = f"{ANN_ROOT}/Index"
RESULT_SUFFIX = "__0"

TOP_K = 100


def hnsw_configs(dataset: DatasetName) -> List[HNSWConfig]:
    if dataset == "coco":
        return [
            HNSWConfig(16, 100, 64),
            HNSWConfig(32, 200, 128),
            HNSWConfig(48, 200, 256),
        ]
    return [
        HNSWConfig(16, 100, 32),
        HNSWConfig(32, 200, 64),
        HNSWConfig(48, 200, 128),
    ]


def ivfpq_configs(dataset: DatasetName) -> List[IVFPQConfig]:
    if dataset == "coco":
        return [
            IVFPQConfig(64, 48, 8, 8),
            IVFPQConfig(128, 48, 8, 16),
            IVFPQConfig(256, 48, 8, 32),
        ]
    return [
        IVFPQConfig(32, 48, 8, 4),
        IVFPQConfig(64, 48, 8, 8),
        IVFPQConfig(128, 48, 8, 16),
    ]


def oivfpq_configs(dataset: DatasetName) -> List[OIVFPQConfig]:
    if dataset == "coco":
        return [
            OIVFPQConfig("OPQ64,IVF64,PQ48", 8),
            OIVFPQConfig("OPQ64,IVF128,PQ48", 16),
            OIVFPQConfig("OPQ64,IVF256,PQ48", 32),
        ]
    return [
        OIVFPQConfig("OPQ64,IVF32,PQ48", 4),
        OIVFPQConfig("OPQ64,IVF64,PQ48", 8),
        OIVFPQConfig("OPQ64,IVF128,PQ48", 16),
    ]


def result_filename(encoder: EncoderName, dataset: DatasetName, index_type: IndexType, cfg_tag: str) -> str:
    return f"results_{encoder}_{dataset}_{index_type}_{cfg_tag}{RESULT_SUFFIX}.json"
