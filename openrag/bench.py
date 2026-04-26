"""
Moss-compatible retrieval benchmark.

Matches the methodology from https://github.com/usemoss/moss/blob/main/benchmarks/README.md:
  - 100k FAQ-style documents (bench_100k_docs.json)
  - 15 diverse queries
  - 3 warmup rounds (excluded), then 50 measured rounds per query
  - top_k = 5, retrieval-only (embedding + ANN + rerank)
  - Reports p50/p95/p99 per-stage + total retrieval latency

OpenRAG's retrieval is local (CPU/GPU embed + FAISS flat/HNSW + cross-encoder
rerank), so this measures the "complete query cycle" just like Moss does —
embedding time is always counted.
"""
from __future__ import annotations
import json
import statistics
import time
import argparse
from pathlib import Path
from typing import Iterable
import httpx

from .config import settings
from .embed import Embedder
from .index import VectorIndex
from .rerank import Reranker
from .timer import PipelineTimer


def _make_embedder(backend: str, device: str | None, precision: str = "fp32"):
    """Construct the configured embedder backend.

    backend: "pytorch" | "onnx"
    precision: for onnx — "fp32" | "fp16" | "int8"
    """
    if backend == "pytorch":
        return Embedder(device=device)
    if backend == "onnx":
        from .embed_onnx import OnnxEmbedder
        return OnnxEmbedder(device=device, precision=precision)
    raise ValueError(f"unknown embed backend: {backend!r}")


DEFAULT_DOCS_URL = "https://raw.githubusercontent.com/usemoss/moss/main/benchmarks/bench_100k_docs.json"
DEFAULT_QUERIES = [
    "how do I reset my password",
    "refund policy for damaged items",
    "international shipping costs",
    "cancel subscription",
    "update billing information",
    "two factor authentication setup",
    "delete my account permanently",
    "export my data",
    "change email address",
    "bulk order discount",
    "warranty claim process",
    "track my package",
    "integrate with third party apps",
    "contact customer support by phone",
    "gift card balance check",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def fetch_docs(source: str | Path, limit: int | None = None) -> list[dict]:
    """Load docs from a local path or URL. Expected JSON shape: list of
    objects with at least {"id": str, "text": str, optional "category": str}.
    """
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        print(f"[bench] fetching {source} ...")
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            r = client.get(source)
            r.raise_for_status()
            data = r.json()
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"docs file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "documents" in data:
        data = data["documents"]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of documents")

    docs = []
    for i, d in enumerate(data):
        if isinstance(d, str):
            docs.append({"id": str(i), "text": d})
        elif isinstance(d, dict):
            text = d.get("text") or d.get("content") or d.get("body") or ""
            if not text:
                continue
            docs.append({
                "id": str(d.get("id", i)),
                "text": text,
                "category": d.get("category"),
            })
    if limit:
        docs = docs[:limit]
    return docs


def build_index(
    docs: list[dict],
    embedder: Embedder,
    batch_size: int = 128,
    index_type: str = "flat",
    hnsw_ef_search: int = 64,
) -> VectorIndex:
    print(f"[bench] embedding {len(docs)} docs (batch={batch_size})...")
    t0 = time.perf_counter()
    texts = [d["text"] for d in docs]
    vecs = embedder.encode_docs(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    print(f"[bench] embedded in {elapsed:.1f}s ({len(docs)/elapsed:.0f} docs/s)")

    index = VectorIndex(dim=embedder.dim, index_type=index_type, hnsw_ef_search=hnsw_ef_search)
    meta = [{"source": f"doc_{d['id']}", "text": d["text"], "category": d.get("category")} for d in docs]
    t0 = time.perf_counter()
    index.add(vecs, meta)
    build_elapsed = time.perf_counter() - t0
    print(f"[bench] index_type={index_type}  built in {build_elapsed:.1f}s")
    return index


def time_query(
    query: str,
    embedder: Embedder,
    index: VectorIndex,
    reranker: Reranker | None,
    top_k_ann: int,
    top_k_rerank: int,
) -> dict:
    timer = PipelineTimer(query=query)
    qvec = embedder.encode_query(query)
    timer.mark("embed_done")
    candidates = index.search(qvec, k=top_k_ann)
    timer.mark("ann_done")
    if reranker is not None:
        _ = reranker.rerank(query, candidates, top_k=top_k_rerank)
        timer.mark("rerank_done")
        return {
            "embed_ms": timer.stages["embed_done"],
            "ann_ms": timer.gap("embed_done", "ann_done"),
            "rerank_ms": timer.gap("ann_done", "rerank_done"),
            "total_ms": timer.stages["rerank_done"],
        }
    return {
        "embed_ms": timer.stages["embed_done"],
        "ann_ms": timer.gap("embed_done", "ann_done"),
        "total_ms": timer.stages["ann_done"],
    }


def _stats(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 2),
        "p50":  round(_percentile(values, 50), 2),
        "p95":  round(_percentile(values, 95), 2),
        "p99":  round(_percentile(values, 99), 2),
        "min":  round(min(values), 2),
        "max":  round(max(values), 2),
    }


def run_bench(
    docs_source: str | Path,
    queries: list[str] | None = None,
    doc_limit: int | None = None,
    warmup: int = 3,
    rounds: int = 50,
    top_k_ann: int | None = None,
    top_k_rerank: int = 5,
    output: Path | None = None,
    device: str | None = None,
    embed_device: str | None = None,
    rerank_device: str | None = None,
    no_rerank: bool = False,
    embed_batch: int = 128,
    index_type: str = "flat",
    hnsw_ef_search: int = 64,
    embed_backend: str = "pytorch",
    embed_precision: str = "fp32",
    reuse_index: bool = False,
    cache_dir: str = "data/index-cache",
) -> dict:
    queries = queries or DEFAULT_QUERIES
    top_k_ann = top_k_ann or settings.top_k_ann

    docs = fetch_docs(docs_source, limit=doc_limit)
    print(f"[bench] loaded {len(docs)} docs, {len(queries)} queries")

    embedder = _make_embedder(embed_backend, device=embed_device or device, precision=embed_precision)
    reranker = None if no_rerank else Reranker(device=rerank_device or device)
    backend_tag = f"{embed_backend}" if embed_backend == "pytorch" else f"onnx-{embed_precision}"
    if no_rerank:
        print(f"[bench] device={embedder.device}  backend={backend_tag}  embed_model={settings.embed_model}  rerank=DISABLED  embed_batch={embed_batch}")
    elif embedder.device == reranker.device:
        print(f"[bench] device={embedder.device}  backend={backend_tag}  embed_model={settings.embed_model}  rerank_model={settings.rerank_model}  embed_batch={embed_batch}")
    else:
        print(f"[bench] embed_device={embedder.device}  rerank_device={reranker.device}  backend={backend_tag}  embed_model={settings.embed_model}  rerank_model={settings.rerank_model}  embed_batch={embed_batch}")

    cache_key = f"{embed_backend}-{embed_precision}-docs{len(docs)}-{index_type}"
    cache_path = Path(cache_dir) / cache_key
    if reuse_index and (cache_path / "faiss.index").exists():
        print(f"[bench] reusing cached index: {cache_path}  (skipping embed)")
        index = VectorIndex.load(cache_path)
        if index_type == "hnsw" and hasattr(index.index, "hnsw"):
            index.index.hnsw.efSearch = hnsw_ef_search
    else:
        index = build_index(docs, embedder, batch_size=embed_batch, index_type=index_type, hnsw_ef_search=hnsw_ef_search)
        if reuse_index:
            print(f"[bench] saving index to {cache_path}")
            index.save(cache_path)
    print(f"[bench] index size: {index.size}")

    print(f"[bench] warmup {warmup} rounds × {len(queries)} queries")
    for _ in range(warmup):
        for q in queries:
            time_query(q, embedder, index, reranker, top_k_ann, top_k_rerank)

    print(f"[bench] measuring {rounds} rounds × {len(queries)} queries = {rounds * len(queries)} measurements")

    per_query: dict[str, dict] = {}
    all_embed, all_ann, all_rerank, all_total = [], [], [], []

    for q in queries:
        embed_ms, ann_ms, rerank_ms, total_ms = [], [], [], []
        for _ in range(rounds):
            r = time_query(q, embedder, index, reranker, top_k_ann, top_k_rerank)
            embed_ms.append(r["embed_ms"])
            ann_ms.append(r["ann_ms"])
            if "rerank_ms" in r:
                rerank_ms.append(r["rerank_ms"])
            total_ms.append(r["total_ms"])
        per_query[q] = {
            "embed": _stats(embed_ms),
            "ann":   _stats(ann_ms),
            "total": _stats(total_ms),
        }
        if rerank_ms:
            per_query[q]["rerank"] = _stats(rerank_ms)
        all_embed.extend(embed_ms); all_ann.extend(ann_ms)
        all_rerank.extend(rerank_ms); all_total.extend(total_ms)

    aggregate = {
        "embed":  _stats(all_embed),
        "ann":    _stats(all_ann),
        "total":  _stats(all_total),
    }
    if all_rerank:
        aggregate["rerank"] = _stats(all_rerank)

    if no_rerank:
        device_label = embedder.device
        rerank_device_label = None
    elif embedder.device == reranker.device:
        device_label = embedder.device
        rerank_device_label = reranker.device
    else:
        device_label = f"embed={embedder.device},rerank={reranker.device}"
        rerank_device_label = reranker.device

    report = {
        "system": "openrag",
        "device": device_label,
        "embed_device": embedder.device,
        "rerank_device": rerank_device_label,
        "rerank_enabled": not no_rerank,
        "embed_batch": embed_batch,
        "index_type": index_type,
        "hnsw_ef_search": hnsw_ef_search if index_type == "hnsw" else None,
        "embed_backend": embed_backend,
        "embed_precision": embed_precision if embed_backend == "onnx" else None,
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "docs": len(docs),
        "queries": len(queries),
        "warmup_rounds": warmup,
        "measured_rounds": rounds,
        "top_k_ann": top_k_ann,
        "top_k_rerank": top_k_rerank,
        "aggregate_ms": aggregate,
        "per_query_ms": per_query,
    }

    if output:
        Path(output).write_text(json.dumps(report, indent=2))
        print(f"[bench] wrote report → {output}")

    print_report(report)
    return report


def print_report(report: dict):
    print()
    print("=" * 70)
    print(f"  OpenRAG benchmark — device={report['device']}  docs={report['docs']}")
    print(f"  {report['measured_rounds']} rounds × {report['queries']} queries = {report['measured_rounds']*report['queries']} measurements")
    print("=" * 70)
    agg = report["aggregate_ms"]
    print(f"{'stage':<10} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8}")
    print("-" * 70)
    for stage in ("embed", "ann", "rerank", "total"):
        if stage not in agg:
            continue
        s = agg[stage]
        print(f"{stage:<10} {s['mean']:>8} {s['p50']:>8} {s['p95']:>8} {s['p99']:>8} {s['min']:>8} {s['max']:>8}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Moss-compatible OpenRAG benchmark")
    ap.add_argument("--docs", default=DEFAULT_DOCS_URL, help="Path or URL to bench docs JSON")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of docs (for quick tests)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--top-k-ann", type=int, default=None)
    ap.add_argument("--top-k-rerank", type=int, default=5)
    ap.add_argument("--queries-file", default=None, help="JSON file with list of query strings")
    ap.add_argument("--output", default="bench_report.json")
    args = ap.parse_args()

    queries = None
    if args.queries_file:
        queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))

    run_bench(
        docs_source=args.docs,
        queries=queries,
        doc_limit=args.limit,
        warmup=args.warmup,
        rounds=args.rounds,
        top_k_ann=args.top_k_ann,
        top_k_rerank=args.top_k_rerank,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
