"""BEIR-style retrieval quality evaluation.

Downloads a small BEIR dataset (default: SciFact, ~5k docs / ~300 queries),
embeds it with the same OnnxEmbedder used in production (INT8 + HNSW by
default), then computes recall@k, MRR, and nDCG@10 against the official
human relevance judgments.

Why this exists: the Moss bench_100k_docs corpus is synthetic (20 templates
varied ~5,000x). recall@20 by doc_id is broken there. BEIR has real human-
labeled qrels, so quality numbers actually mean something.

Usage (run from the project root):
  python scripts/eval_beir.py                          # SciFact, INT8 + HNSW
  python scripts/eval_beir.py --dataset nfcorpus
  python scripts/eval_beir.py --precision fp32 --index-type flat
  python scripts/eval_beir.py --dataset scifact --precision int8 --index-type hnsw
"""
from __future__ import annotations
import argparse
import io
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import faiss
import httpx
import numpy as np

from openrag.embed_onnx import OnnxEmbedder


BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATA_DIR = Path("data/beir")
RESULTS_DIR = Path("data/quality")


def download_beir(dataset: str) -> Path:
    target = DATA_DIR / dataset
    if target.exists() and (target / "corpus.jsonl").exists():
        print(f"[beir] using cached {target}")
        return target
    url = BEIR_URL.format(name=dataset)
    print(f"[beir] downloading {url} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes(chunk_size=1 << 20):
            buf.write(chunk)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(DATA_DIR)
    print(f"[beir] extracted to {target}")
    return target


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """BEIR qrels are TSV: query-id<TAB>corpus-id<TAB>score (with header)."""
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score <= 0:
                continue
            qrels.setdefault(qid, {})[did] = score
    return qrels


def build_index(vecs: np.ndarray, kind: str) -> faiss.Index:
    if kind == "flat":
        idx = faiss.IndexFlatIP(vecs.shape[1])
    elif kind == "hnsw":
        idx = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 64
    else:
        raise ValueError(f"unknown index type: {kind!r}")
    idx.add(vecs)
    return idx


def recall_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Dict[str, int]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        rel = relevant.get(d, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact",
                    help="BEIR dataset name (e.g. scifact, nfcorpus, arguana, scidocs)")
    ap.add_argument("--precision", default="int8", choices=["fp32", "int8"])
    ap.add_argument("--index-type", default="hnsw", choices=["flat", "hnsw"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--split", default="test", help="qrels split (test or dev)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = download_beir(args.dataset)
    corpus = load_jsonl(root / "corpus.jsonl")
    queries_all = load_jsonl(root / "queries.jsonl")
    qrels_path = root / "qrels" / f"{args.split}.tsv"
    if not qrels_path.exists():
        # fallback to train if test/dev not present
        alt = root / "qrels" / "train.tsv"
        if alt.exists():
            print(f"[beir] {qrels_path.name} missing — falling back to train.tsv")
            qrels_path = alt
        else:
            raise FileNotFoundError(f"no qrels found at {root/'qrels'}")
    qrels = load_qrels(qrels_path)

    queries = [q for q in queries_all if q["_id"] in qrels]
    if not queries:
        raise RuntimeError("no queries matched the qrels — check split or dataset")

    print(f"[beir] {args.dataset}: {len(corpus):,} docs, {len(queries)} queries (split={qrels_path.stem})")

    doc_ids = [d["_id"] for d in corpus]
    doc_texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in corpus]

    embedder = OnnxEmbedder(device=args.device, precision=args.precision)
    print(f"[beir] embedding {len(doc_texts):,} docs (precision={args.precision})...")
    t0 = time.perf_counter()
    doc_vecs = embedder.encode_docs(doc_texts, batch_size=args.batch)
    print(f"[beir] doc embed: {time.perf_counter()-t0:.1f}s")

    print(f"[beir] embedding {len(queries)} queries...")
    q_vecs = np.vstack([embedder.encode_query(q["text"]) for q in queries])

    idx = build_index(doc_vecs, args.index_type)

    print(f"[beir] searching top-{args.top_k} ({args.index_type})...")
    t0 = time.perf_counter()
    _, I = idx.search(q_vecs, args.top_k)
    print(f"[beir] search: {time.perf_counter()-t0:.2f}s for {len(queries)} queries")

    recalls_5, recalls_10, recalls_20 = [], [], []
    mrrs, ndcgs_10 = [], []
    for q, retrieved_idxs in zip(queries, I):
        retrieved_ids = [doc_ids[i] for i in retrieved_idxs]
        rel = qrels[q["_id"]]
        recalls_5.append(recall_at_k(retrieved_ids, rel, 5))
        recalls_10.append(recall_at_k(retrieved_ids, rel, 10))
        recalls_20.append(recall_at_k(retrieved_ids, rel, 20))
        mrrs.append(reciprocal_rank(retrieved_ids, rel))
        ndcgs_10.append(ndcg_at_k(retrieved_ids, rel, 10))

    def m(xs): return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "dataset": args.dataset,
        "split": qrels_path.stem,
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "config": {
            "precision": args.precision,
            "index_type": args.index_type,
            "top_k": args.top_k,
            "embed_model": "BAAI/bge-small-en-v1.5",
        },
        "metrics": {
            "recall@5": round(m(recalls_5), 4),
            "recall@10": round(m(recalls_10), 4),
            "recall@20": round(m(recalls_20), 4),
            "mrr": round(m(mrrs), 4),
            "ndcg@10": round(m(ndcgs_10), 4),
        },
    }

    print("\n=== BEIR quality summary ===")
    print(f"  dataset:    {summary['dataset']} ({summary['n_docs']:,} docs, {summary['n_queries']} queries)")
    print(f"  config:     precision={args.precision}  index={args.index_type}")
    print("  metrics:")
    for k, v in summary["metrics"].items():
        print(f"    {k:<12}{v:>8.4f}")
    print("\n  Healthy ranges for bge-small-en-v1.5 on SciFact:")
    print("    nDCG@10 ~0.65  recall@10 ~0.80  recall@20 ~0.90")

    out = Path(args.output) if args.output else (
        RESULTS_DIR / f"beir_{args.dataset}_{args.precision}_{args.index_type}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[beir] saved to {out}")


if __name__ == "__main__":
    main()
