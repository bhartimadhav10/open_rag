"""High-level facade for embedding OpenRAG into a chatbot.

The lower-level building blocks (Pipeline, VectorIndex, Embedder, Reranker,
GroqLLM) remain available; this module just gives chatbot authors a small,
predictable surface so they don't have to wire components together.

Typical usage:

    from openrag import Engine

    # Build (or reload) an index from a folder of PDF/MD/TXT files:
    engine = Engine.from_directory("./my_docs")

    # Answer a question with retrieved context + LLM:
    result = await engine.chat("What's our refund policy?")
    print(result.answer)
    for src in result.sources:
        print(f"  [{src.rerank_score:.2f}] {src.source}: {src.preview}")

    # Or just retrieve, no LLM call:
    hits = engine.search("refund policy", top_k=5)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from .config import settings
from .ingest import ingest as _ingest
from .pipeline import Pipeline


@dataclass
class SearchResult:
    text: str
    source: str
    rerank_score: float
    ann_score: Optional[float] = None

    @property
    def preview(self) -> str:
        s = " ".join(self.text.split())
        return s if len(s) <= 100 else s[:97] + "..."


@dataclass
class ChatResult:
    query: str
    answer: str
    sources: list[SearchResult]
    session_id: str
    timing_ms: dict


class Engine:
    """High-level RAG engine — embed + retrieve + (optional) LLM answer.

    Construct via `Engine()` (loads an existing index from `data/index/`)
    or `Engine.from_directory(path)` to ingest a folder first.

    Rerank can be turned on/off at construction time:

        Engine()                          # uses RERANK_ENABLED from .env (default: on)
        Engine(rerank=False)              # skip cross-encoder rerank, ~80ms → ~25ms
        Engine.from_directory("./docs", rerank=False)
    """

    def __init__(self, rerank: bool | None = None):
        self._pipeline = Pipeline(rerank=rerank)

    @classmethod
    def from_directory(cls, docs_dir: str | Path, rerank: bool | None = None) -> "Engine":
        """Ingest all PDF/MD/TXT files under `docs_dir` and build an index,
        then return a ready-to-use Engine."""
        info = _ingest(Path(docs_dir))
        print(
            f"[openrag] ingested {info['chunks']} chunks "
            f"from {info['files']} files (dim={info['dim']}) → {info['index_dir']}"
        )
        return cls(rerank=rerank)

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve top-k chunks for a query. No LLM call. Honors the
        engine's rerank setting (set at construction)."""
        qvec = self._pipeline.embedder.encode_query(query)
        candidates = self._pipeline.index.search(qvec, k=settings.top_k_ann)
        k = top_k or settings.top_k_rerank
        if self._pipeline.reranker is not None:
            top = self._pipeline.reranker.rerank(query, candidates, top_k=k)
        else:
            top = candidates[:k]
            for c in top:
                c["rerank_score"] = c.get("ann_score", 0.0)
        return [
            SearchResult(
                text=c["text"],
                source=c.get("source", "?"),
                rerank_score=float(c["rerank_score"]),
                ann_score=c.get("ann_score"),
            )
            for c in top
        ]

    async def chat(self, query: str, session_id: str | None = None) -> ChatResult:
        """Run the full retrieval + LLM pipeline and return the final answer."""
        out = await self._pipeline.run_collect(query, session_id)
        return ChatResult(
            query=out["query"],
            answer=out["answer"],
            sources=[
                SearchResult(
                    text=c["text"],
                    source=c.get("source", "?"),
                    rerank_score=float(c["rerank_score"]),
                )
                for c in out["results"]
            ],
            session_id=out["session_id"],
            timing_ms=out["timing"],
        )

    async def stream(self, query: str, session_id: str | None = None) -> AsyncIterator[dict]:
        """Yield SSE-style events as the pipeline runs. Useful for streaming
        chat UIs. Each event is `{"event": str, "data": dict}`."""
        async for evt in self._pipeline.run(query, session_id):
            yield evt

    @property
    def index_size(self) -> int:
        return self._pipeline.index.size
