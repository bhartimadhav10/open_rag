"""Recall@20 verification: does INT8 + HNSW actually return the same top-20
docs as FP32 + exact flat search?

Ground truth = FP32 embeddings + flat FAISS (exact inner product).
Compares three production-adjacent configs against it, reporting recall@20
per query and a mean.

Run from the project root:  python scripts/legacy/recall_check.py
"""
from __future__ import annotations
import time
from pathlib import Path
import faiss
import numpy as np

from openrag.embed_onnx import OnnxEmbedder
from openrag.bench import DEFAULT_DOCS_URL, DEFAULT_QUERIES, fetch_docs


CACHE_DIR = Path("data/recall-cache")


def build_flat(vecs):
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx


def build_hnsw(vecs, M=32, efc=200, efs=64):
    idx = faiss.IndexHNSWFlat(vecs.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = efc
    idx.hnsw.efSearch = efs
    idx.add(vecs)
    return idx


def topk(idx, qvec, k=20):
    _, i = idx.search(qvec, k)
    return set(i[0].tolist())


def get_or_embed(embedder, texts, cache_path: Path, label: str):
    if cache_path.exists():
        print(f"[recall] loading cached {label} vectors from {cache_path}  (skip embed)")
        return np.load(cache_path)
    print(f"[recall] embedding {len(texts)} docs with {label}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    vecs = embedder.encode_docs(texts, batch_size=256)
    print(f"[recall] {label} embed took {time.perf_counter()-t0:.1f}s  →  saving to {cache_path}")
    np.save(cache_path, vecs)
    return vecs


def main():
    print("[recall] loading corpus...")
    docs = fetch_docs(DEFAULT_DOCS_URL)
    texts = [d["text"] for d in docs]
    print(f"[recall] {len(docs)} docs, {len(DEFAULT_QUERIES)} queries")

    e_fp32 = OnnxEmbedder(device="cpu", precision="fp32")
    vecs_fp32 = get_or_embed(e_fp32, texts, CACHE_DIR / f"vecs_fp32_{len(docs)}.npy", "FP32")

    e_int8 = OnnxEmbedder(device="cpu", precision="int8")
    vecs_int8 = get_or_embed(e_int8, texts, CACHE_DIR / f"vecs_int8_{len(docs)}.npy", "INT8")

    gt_idx = build_flat(vecs_fp32)           # ground truth
    hnsw_fp32_idx = build_hnsw(vecs_fp32)    # HNSW cost alone
    flat_int8_idx = build_flat(vecs_int8)    # INT8 cost alone
    hnsw_int8_idx = build_hnsw(vecs_int8)    # combined (what production uses)

    print(f"\n{'query':<40} {'fp32+hnsw':>10} {'int8+flat':>10} {'int8+hnsw':>10}")
    print("-" * 72)
    totals = {"a": 0.0, "b": 0.0, "c": 0.0}
    for q in DEFAULT_QUERIES:
        qv_fp32 = e_fp32.encode_query(q)
        qv_int8 = e_int8.encode_query(q)
        gt = topk(gt_idx, qv_fp32)
        a = len(topk(hnsw_fp32_idx, qv_fp32) & gt) / 20
        b = len(topk(flat_int8_idx,  qv_int8) & gt) / 20
        c = len(topk(hnsw_int8_idx,  qv_int8) & gt) / 20
        totals["a"] += a; totals["b"] += b; totals["c"] += c
        qd = q if len(q) <= 40 else q[:37] + "..."
        print(f"{qd:<40} {a*100:>9.1f}% {b*100:>9.1f}% {c*100:>9.1f}%")

    n = len(DEFAULT_QUERIES)
    print("-" * 72)
    print(f"{'MEAN recall@20':<40} {totals['a']/n*100:>9.1f}% {totals['b']/n*100:>9.1f}% {totals['c']/n*100:>9.1f}%")
    print("\nInterpretation:")
    print("  fp32+hnsw: cost of HNSW alone — healthy is 97–99%")
    print("  int8+flat: cost of INT8 alone — healthy is 98–100%")
    print("  int8+hnsw: combined (what your bench actually used) — healthy is 96–99%")
    print("  If int8+hnsw < 95%, your 3.4 ms claim has a hidden quality cost.")


if __name__ == "__main__":
    main()
