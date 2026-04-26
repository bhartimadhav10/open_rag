"""LLM-as-judge retrieval quality evaluation.

For each query, retrieves top-k chunks from the production pipeline (ONNX +
HNSW), then asks Groq llama-3.1-8b-instant to score each chunk's relevance
to the query on a 1-5 scale. No labeled data required.

Why this exists: doc-id-based recall is broken on the Moss synthetic corpus
(20 templates × ~5,000 near-duplicates). LLM-judge sidesteps that — it scores
semantic relevance directly, so it works equally well on synthetic Moss data
or any real production corpus.

Usage (run from the project root):
  python scripts/eval_llm_judge.py                    # 15 default queries, INT8+HNSW, top-5
  python scripts/eval_llm_judge.py --top-k 5
  python scripts/eval_llm_judge.py --queries my_queries.txt
  python scripts/eval_llm_judge.py --precision fp32
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from groq import AsyncGroq

from openrag.config import settings
from openrag.embed_onnx import OnnxEmbedder
from openrag.bench import DEFAULT_DOCS_URL, DEFAULT_QUERIES, fetch_docs


RESULTS_DIR = Path("data/quality")
CACHE_DIR = Path("data/recall-cache")


JUDGE_SYSTEM = (
    "You are a strict relevance judge for a search system. You will be given "
    "a user query and one retrieved text chunk. Rate how relevant the chunk "
    "is to the query on this exact scale:\n"
    "  5 = directly answers the query\n"
    "  4 = highly relevant, partial answer\n"
    "  3 = related topic, indirect relevance\n"
    "  2 = same general domain, not useful\n"
    "  1 = unrelated\n"
    "Respond with ONLY a single digit 1-5. No words, no explanation."
)

JUDGE_USER = "Query: {query}\n\nChunk: {chunk}\n\nScore (1-5):"


def build_hnsw(vecs: np.ndarray) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    idx.add(vecs)
    return idx


_DIGIT_RE = re.compile(r"[1-5]")


def parse_score(text: str) -> Optional[int]:
    m = _DIGIT_RE.search(text or "")
    return int(m.group()) if m else None


async def judge_pair(client: AsyncGroq, model: str, query: str, chunk: str,
                     max_chunk_chars: int = 1500, retries: int = 2) -> Optional[int]:
    if len(chunk) > max_chunk_chars:
        chunk = chunk[:max_chunk_chars] + "..."
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(query=query, chunk=chunk)},
    ]
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=msgs, temperature=0.0, max_tokens=4,
            )
            txt = resp.choices[0].message.content or ""
            return parse_score(txt)
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.6 * (attempt + 1))
    print(f"  [judge] failed after {retries+1} attempts: {last_err}")
    return None


async def judge_all(queries: List[str], retrieved_texts: List[List[str]],
                    model: str, concurrency: int) -> List[List[Optional[int]]]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment/.env")
    client = AsyncGroq(api_key=settings.groq_api_key)
    sem = asyncio.Semaphore(concurrency)
    k = len(retrieved_texts[0]) if retrieved_texts else 0
    results: List[List[Optional[int]]] = [[None] * k for _ in queries]
    completed = [0]
    total = len(queries) * k

    async def task(qi: int, ci: int):
        async with sem:
            score = await judge_pair(client, model, queries[qi], retrieved_texts[qi][ci])
            results[qi][ci] = score
            completed[0] += 1
            if completed[0] % 10 == 0 or completed[0] == total:
                print(f"  [judge] {completed[0]}/{total}")

    coros = [task(qi, ci) for qi in range(len(queries)) for ci in range(k)]
    t0 = time.perf_counter()
    await asyncio.gather(*coros)
    print(f"[judge] {len(coros)} judgments in {time.perf_counter()-t0:.1f}s")
    return results


def get_or_embed_corpus(embedder: OnnxEmbedder, texts: List[str],
                        precision: str, n: int) -> np.ndarray:
    cache = CACHE_DIR / f"vecs_{precision}_{n}.npy"
    if cache.exists():
        print(f"[judge] using cached vectors {cache}")
        return np.load(cache)
    print(f"[judge] embedding {n:,} docs ({precision})...")
    vecs = embedder.encode_docs(texts, batch_size=256)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vecs)
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="int8", choices=["fp32", "int8"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--queries", default=None,
                    help="path to text file, one query per line (default: 15 Moss queries)")
    ap.add_argument("--model", default=None,
                    help="Groq model (default: settings.groq_model)")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="parallel Groq calls (default 2 — Groq free tier is 30 RPM)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            queries = [ln.strip() for ln in f if ln.strip()]
    else:
        queries = list(DEFAULT_QUERIES)

    print("[judge] loading corpus...")
    docs = fetch_docs(DEFAULT_DOCS_URL)
    texts = [d["text"] for d in docs]
    print(f"[judge] {len(docs):,} docs, {len(queries)} queries, top-{args.top_k}")

    embedder = OnnxEmbedder(device="cpu", precision=args.precision)
    doc_vecs = get_or_embed_corpus(embedder, texts, args.precision, len(docs))

    idx = build_hnsw(doc_vecs)
    q_vecs = np.vstack([embedder.encode_query(q) for q in queries])
    _, I = idx.search(q_vecs, args.top_k)

    retrieved_texts = [[texts[i] for i in row] for row in I]

    model = args.model or settings.groq_model
    print(f"[judge] judging with Groq model={model}, concurrency={args.concurrency}")
    scores = asyncio.run(judge_all(queries, retrieved_texts, model, args.concurrency))

    per_query = []
    all_scores: List[int] = []
    for q, qscores, retrieved in zip(queries, scores, retrieved_texts):
        clean = [s for s in qscores if s is not None]
        all_scores.extend(clean)
        per_query.append({
            "query": q,
            "scores": qscores,
            "mean": round(sum(clean) / len(clean), 2) if clean else None,
            "any_relevant@k": any(s is not None and s >= 4 for s in qscores),
            "all_relevant@k": (all(s is not None and s >= 4 for s in qscores)
                              if all(s is not None for s in qscores) else None),
            "top1_text_preview": (retrieved[0][:120] + "...") if retrieved and len(retrieved[0]) > 120 else (retrieved[0] if retrieved else None),
        })

    n = len(all_scores)
    summary = {
        "config": {
            "precision": args.precision,
            "index_type": "hnsw",
            "top_k": args.top_k,
            "judge_model": model,
            "n_queries": len(queries),
            "n_judgments": n,
            "n_failed": (len(queries) * args.top_k) - n,
        },
        "metrics": {
            "mean_relevance": round(sum(all_scores) / n, 3) if n else None,
            "frac_>=4": round(sum(1 for s in all_scores if s >= 4) / n, 3) if n else None,
            "frac_>=3": round(sum(1 for s in all_scores if s >= 3) / n, 3) if n else None,
            "any_relevant_rate": round(sum(1 for q in per_query if q["any_relevant@k"]) / len(per_query), 3),
        },
        "per_query": per_query,
    }

    out = Path(args.output) if args.output else (
        RESULTS_DIR / f"llm_judge_{args.precision}_top{args.top_k}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== LLM-judge quality summary ===")
    print(f"  config:     precision={args.precision}  judge={model}  top-k={args.top_k}")
    print(f"  judgments:  {n}/{len(queries)*args.top_k} ({summary['config']['n_failed']} failed)")
    print("  metrics:")
    for k, v in summary["metrics"].items():
        print(f"    {k:<22}{v}")
    print("\n  per-query results:")
    for q in per_query:
        qd = q["query"] if len(q["query"]) <= 38 else q["query"][:35] + "..."
        prev = q.get("top1_text_preview") or ""
        prev = prev if len(prev) <= 70 else prev[:67] + "..."
        print(f"    {qd:<40} mean={q['mean']}  scores={q['scores']}")
        print(f"      top1: {prev}")
    print("\n  Interpretation:")
    print("    mean_relevance >= 4.0    -> top-k chunks are mostly direct hits")
    print("    frac_>=4       >= 0.7    -> >=70% of returned chunks are useful")
    print("    any_relevant_rate ~ 1.0  -> at least one good chunk per query")
    print(f"\n[judge] saved to {out}")


if __name__ == "__main__":
    main()
