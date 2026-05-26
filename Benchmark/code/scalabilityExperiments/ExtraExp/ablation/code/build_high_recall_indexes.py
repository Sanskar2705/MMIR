#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np
from tqdm import tqdm


def iter_embedding_files(emb_dir: str) -> List[str]:
    files = sorted(
        f for f in os.listdir(emb_dir) if f.startswith("embeddings_part_") and f.endswith(".json")
    )
    if not files:
        raise FileNotFoundError(f"No embedding shards found in: {emb_dir}")
    return files


def load_shard(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = [row["image_name"] for row in data]
    embs = np.asarray([row["embedding"] for row in data], dtype="float32")
    return names, embs


def assert_normalized(embs: np.ndarray, atol: float = 1e-3) -> None:
    norms = np.linalg.norm(embs, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError("Embeddings are not L2-normalized.")


def get_dim(emb_dir: str, files: Sequence[str]) -> int:
    _, first_embs = load_shard(os.path.join(emb_dir, files[0]))
    assert_normalized(first_embs)
    return int(first_embs.shape[1])


def collect_image_names(emb_dir: str, files: Sequence[str], max_vectors: int) -> List[str]:
    image_names: List[str] = []
    total = 0
    for fname in tqdm(files, desc="Collecting image names"):
        if total >= max_vectors:
            break
        names, _ = load_shard(os.path.join(emb_dir, fname))
        remaining = max_vectors - total
        if len(names) > remaining:
            names = names[:remaining]
        image_names.extend(names)
        total += len(names)
    return image_names


def sample_training_vectors(
    emb_dir: str, files: Sequence[str], dim: int, target_train: int, max_vectors: int
) -> np.ndarray:
    train_vecs: List[np.ndarray] = []
    collected = 0
    target_train = min(target_train, max_vectors)
    for fname in tqdm(files, desc="Sampling train vectors"):
        if collected >= target_train:
            break
        _, embs = load_shard(os.path.join(emb_dir, fname))
        if embs.shape[1] != dim:
            raise ValueError(f"Unexpected dim in {fname}: {embs.shape[1]} != {dim}")
        assert_normalized(embs)
        remaining = target_train - collected
        take = min(remaining, embs.shape[0])
        train_vecs.append(embs[:take])
        collected += take
    if collected == 0:
        raise RuntimeError("Could not collect training vectors")
    return np.vstack(train_vecs)


def build_hnsw(
    emb_dir: str,
    files: Sequence[str],
    dim: int,
    image_names: Sequence[str],
    out_dir: str,
    max_vectors: int,
    hnsw_configs: Sequence[Tuple[int, int]],
) -> None:
    if not hnsw_configs:
        print("[INFO] No HNSW configs requested; skipping HNSW build.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for hnsw_m, efc in hnsw_configs:
        cfg_tag = f"m{hnsw_m}_efc{efc}"
        index_path = os.path.join(out_dir, f"clip_hnsw_{cfg_tag}.index")
        names_path = os.path.join(out_dir, f"image_names_{cfg_tag}.json")
        if os.path.isfile(index_path) and os.path.isfile(names_path):
            print(f"[SKIP][HNSW] exists {cfg_tag}")
            continue

        print(f"[START][HNSW] {cfg_tag}")
        index = faiss.IndexHNSWFlat(dim, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(efc)

        total = 0
        for fname in tqdm(files, desc=f"Adding to HNSW {cfg_tag}"):
            if total >= max_vectors:
                break
            _, embs = load_shard(os.path.join(emb_dir, fname))
            remaining = max_vectors - total
            if embs.shape[0] > remaining:
                embs = embs[:remaining]
            if embs.shape[1] != dim:
                raise ValueError(f"Unexpected dim in {fname}: {embs.shape[1]} != {dim}")
            assert_normalized(embs)
            index.add(embs)
            total += embs.shape[0]

        if total != len(image_names):
            raise RuntimeError(f"HNSW count mismatch: {total} != {len(image_names)}")

        faiss.write_index(index, index_path)
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(list(image_names), f)
        print(f"[DONE][HNSW] {cfg_tag} -> {index_path}")


def parse_ivfpq_factory_specs(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def max_ivf_nlist_from_factory_spec(spec: str) -> int:
    found = [int(m.group(1)) for m in re.finditer(r"IVF(\d+)", spec, flags=re.IGNORECASE)]
    if not found:
        raise ValueError(
            f"Factory spec {spec!r} has no IVF<nlist> clause "
            "(example: OPQ192,IVF16384,PQ192)"
        )
    return max(found)


def prepare_ivfpq_training_vectors(
    emb_dir: str,
    files: Sequence[str],
    dim: int,
    max_vectors: int,
    train_size: int,
    max_nlist: int,
) -> Tuple[np.ndarray, int]:
    """
    Sample training vectors for IVF/PQ training. Uses FAISS's common rule of thumb:
    use at least ~39 * nlist training points to avoid the clustering warning.
    """
    required_train_min = 39 * int(max_nlist) if int(max_nlist) > 0 else 0
    effective_train_size = int(train_size)
    if required_train_min and effective_train_size < required_train_min:
        print(
            "[WARN] --ivfpq-train-size is below FAISS IVF recommendation "
            f"(requested={effective_train_size:,}, recommended_min={required_train_min:,} "
            f"for nlist={max_nlist:,}). Auto-bumping to {required_train_min:,}."
        )
        effective_train_size = required_train_min
    if effective_train_size > int(max_vectors):
        print(
            "[WARN] IVFPQ train size exceeds --max-vectors. "
            f"Clamping to max_vectors={int(max_vectors):,}."
        )
        effective_train_size = int(max_vectors)
    train_x = sample_training_vectors(
        emb_dir, files, dim, target_train=effective_train_size, max_vectors=max_vectors
    )
    return train_x, effective_train_size


def build_ivfpq(
    emb_dir: str,
    files: Sequence[str],
    dim: int,
    image_names: Sequence[str],
    out_dir: str,
    max_vectors: int,
    ivfpq_configs: Sequence[Tuple[int, int, int]],
    train_x: np.ndarray,
) -> None:
    if not ivfpq_configs:
        print("[INFO] No IVFPQ configs requested; skipping IVFPQ build.")
        return
    os.makedirs(out_dir, exist_ok=True)

    for nlist, pq_m, nbits in ivfpq_configs:
        cfg_tag = f"nlist{nlist}_m{pq_m}_nbits{nbits}"
        index_path = os.path.join(out_dir, f"clip_ivfpq_{cfg_tag}.index")
        names_path = os.path.join(out_dir, f"image_names_{cfg_tag}.json")
        if os.path.isfile(index_path) and os.path.isfile(names_path):
            print(f"[SKIP][IVFPQ] exists {cfg_tag}")
            continue

        print(f"[START][IVFPQ] {cfg_tag}")
        if not (1 <= int(nbits) <= 8):
            raise ValueError(
                f"Invalid IVFPQ config {cfg_tag}: nbits={nbits} "
                "(FAISS IndexIVFPQ requires 1 <= nbits <= 8)"
            )
        if dim % int(pq_m) != 0:
            raise ValueError(f"Invalid IVFPQ config {cfg_tag}: dim={dim} is not divisible by m={pq_m}")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            int(nlist),
            int(pq_m),
            int(nbits),
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(train_x)

        total = 0
        for fname in tqdm(files, desc=f"Adding to IVFPQ {cfg_tag}"):
            if total >= max_vectors:
                break
            _, embs = load_shard(os.path.join(emb_dir, fname))
            remaining = max_vectors - total
            if embs.shape[0] > remaining:
                embs = embs[:remaining]
            if embs.shape[1] != dim:
                raise ValueError(f"Unexpected dim in {fname}: {embs.shape[1]} != {dim}")
            assert_normalized(embs)
            index.add(embs)
            total += embs.shape[0]

        if total != len(image_names):
            raise RuntimeError(f"IVFPQ count mismatch: {total} != {len(image_names)}")

        faiss.write_index(index, index_path)
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(list(image_names), f)
        print(f"[DONE][IVFPQ] {cfg_tag} -> {index_path}")


def _factory_spec_to_tag(spec: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", spec.strip()).strip("_").lower()
    return cleaned or "factory"


def build_ivfpq_index_factory(
    emb_dir: str,
    files: Sequence[str],
    dim: int,
    image_names: Sequence[str],
    out_dir: str,
    max_vectors: int,
    factory_specs: Sequence[str],
    train_x: np.ndarray,
) -> None:
    if not factory_specs:
        print("[INFO] No IVFPQ index_factory specs; skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for spec in factory_specs:
        tag = _factory_spec_to_tag(spec)
        cfg_tag = f"fac_{tag}"
        index_path = os.path.join(out_dir, f"clip_ivfpq_{cfg_tag}.index")
        names_path = os.path.join(out_dir, f"image_names_{cfg_tag}.json")
        if os.path.isfile(index_path) and os.path.isfile(names_path):
            print(f"[SKIP][IVFPQ factory] exists {spec}")
            continue
        print(f"[START][IVFPQ factory] {spec}")
        index = faiss.index_factory(int(dim), spec, faiss.METRIC_INNER_PRODUCT)
        index.train(train_x)

        total = 0
        for fname in tqdm(files, desc=f"Adding IVFPQ factory {tag}"):
            if total >= max_vectors:
                break
            _, embs = load_shard(os.path.join(emb_dir, fname))
            remaining = max_vectors - total
            if embs.shape[0] > remaining:
                embs = embs[:remaining]
            if embs.shape[1] != dim:
                raise ValueError(f"Unexpected dim in {fname}: {embs.shape[1]} != {dim}")
            assert_normalized(embs)
            index.add(embs)
            total += embs.shape[0]

        if total != len(image_names):
            raise RuntimeError(f"IVFPQ factory count mismatch: {total} != {len(image_names)}")

        faiss.write_index(index, index_path)
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(list(image_names), f)
        print(f"[DONE][IVFPQ factory] {spec} -> {index_path}")


def parse_hnsw_configs(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    raw = (raw or "").strip()
    if not raw:
        return out
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        m_s, efc_s = token.split(":")
        out.append((int(m_s), int(efc_s)))
    return out


def parse_ivfpq_configs(raw: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    raw = (raw or "").strip()
    if not raw:
        return out
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        nlist_s, m_s, nbits_s = token.split(":")
        out.append((int(nlist_s), int(m_s), int(nbits_s)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build high-recall HNSW and IVFPQ indexes.")
    ap.add_argument(
        "--emb-dir",
        default=(
            "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/CLIP10M/clip/embeddings"
        ),
    )
    ap.add_argument(
        "--hnsw-out-dir",
        default=(
            "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/"
            "laionRetriveal/extra_exp/ablation/index/hnsw_best"
        ),
    )
    ap.add_argument(
        "--ivfpq-out-dir",
        default=(
            "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/"
            "laionRetriveal/extra_exp/ablation/index/ivfpq_best"
        ),
    )
    ap.add_argument("--max-vectors", type=int, default=7_000_000)
    ap.add_argument("--ivfpq-train-size", type=int, default=1_400_000)
    ap.add_argument(
        "--hnsw-configs",
        default="64:800,80:1200",
        help="Comma-separated m:efConstruction configs",
    )
    ap.add_argument(
        "--ivfpq-configs",
        default="16384:96:8,16384:192:8,32768:192:8",
        help="Comma-separated nlist:m:nbits configs (nbits must be 1..8 for IndexIVFPQ)",
    )
    ap.add_argument(
        "--ivfpq-factory-specs",
        default="",
        help=(
            "Optional Faiss index_factory specs for stronger IVFPQ (e.g. OPQ+IVF+PQ). "
            "Separate multiple specs with '|'. Example: "
            "'OPQ192,IVF16384,PQ192|OPQ192,IVF32768,PQ192'"
        ),
    )
    args = ap.parse_args()

    files = iter_embedding_files(args.emb_dir)
    print(f"[INFO] embedding files={len(files)}")
    dim = get_dim(args.emb_dir, files)
    print(f"[INFO] dim={dim}")
    image_names = collect_image_names(args.emb_dir, files, max_vectors=int(args.max_vectors))
    print(f"[INFO] vectors={len(image_names):,}")

    hnsw_configs = parse_hnsw_configs(args.hnsw_configs)
    ivfpq_configs = parse_ivfpq_configs(args.ivfpq_configs)
    factory_specs = parse_ivfpq_factory_specs(args.ivfpq_factory_specs)
    print(f"[INFO] HNSW configs={hnsw_configs}")
    print(f"[INFO] IVFPQ configs={ivfpq_configs}")
    print(f"[INFO] IVFPQ index_factory specs={factory_specs}")

    max_nlist = 0
    for nlist, _, _ in ivfpq_configs:
        max_nlist = max(max_nlist, int(nlist))
    for spec in factory_specs:
        max_nlist = max(max_nlist, max_ivf_nlist_from_factory_spec(spec))

    train_x: Optional[np.ndarray] = None
    if ivfpq_configs or factory_specs:
        if max_nlist <= 0:
            raise ValueError("Could not determine max IVF nlist for IVFPQ training.")
        train_x, eff_train = prepare_ivfpq_training_vectors(
            emb_dir=args.emb_dir,
            files=files,
            dim=dim,
            max_vectors=int(args.max_vectors),
            train_size=int(args.ivfpq_train_size),
            max_nlist=max_nlist,
        )
        rec_min = 39 * max_nlist
        print(
            f"[INFO] IVFPQ training sample: shape={train_x.shape}, "
            f"effective_train_size={eff_train:,}, max_nlist={max_nlist:,}, "
            f"FAISS_recommended_min_train≈{rec_min:,}"
        )

    build_hnsw(
        emb_dir=args.emb_dir,
        files=files,
        dim=dim,
        image_names=image_names,
        out_dir=args.hnsw_out_dir,
        max_vectors=int(args.max_vectors),
        hnsw_configs=hnsw_configs,
    )
    if train_x is not None:
        build_ivfpq(
            emb_dir=args.emb_dir,
            files=files,
            dim=dim,
            image_names=image_names,
            out_dir=args.ivfpq_out_dir,
            max_vectors=int(args.max_vectors),
            ivfpq_configs=ivfpq_configs,
            train_x=train_x,
        )
        build_ivfpq_index_factory(
            emb_dir=args.emb_dir,
            files=files,
            dim=dim,
            image_names=image_names,
            out_dir=args.ivfpq_out_dir,
            max_vectors=int(args.max_vectors),
            factory_specs=factory_specs,
            train_x=train_x,
        )
    print("[DONE] High-recall index build complete.")


if __name__ == "__main__":
    main()
