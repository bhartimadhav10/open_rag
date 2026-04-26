from __future__ import annotations
from typing import AsyncIterator
from .config import settings
from .embed import Embedder
from .index import VectorIndex
from .rerank import Reranker
from .llm import GroqLLM
from .session import SessionStore
from .timer import PipelineTimer


class Pipeline:
    def __init__(self, rerank: bool | None = None):
        """rerank: True | False to override settings.rerank_enabled.
        When False, the cross-encoder model is NOT loaded (saves ~200 MB RAM
        and a few seconds at startup) and the rerank stage is skipped at
        query time.
        """
        self.rerank_enabled = settings.rerank_enabled if rerank is None else rerank
        self.embedder = Embedder()
        self.index = VectorIndex.load()
        self.reranker = Reranker() if self.rerank_enabled else None
        self.llm = GroqLLM()
        self.sessions = SessionStore()

    def _select_top(self, query: str, candidates: list[dict]) -> list[dict]:
        """Return the final top-K with a `rerank_score` field on each chunk.
        When rerank is disabled, falls back to the FAISS ann_score so the
        downstream UI/JSON shape stays identical."""
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=settings.top_k_rerank)
        top = candidates[: settings.top_k_rerank]
        for c in top:
            c["rerank_score"] = c.get("ann_score", 0.0)
        return top

    async def run(self, query: str, session_id: str | None = None) -> AsyncIterator[dict]:
        """Yields SSE-shaped dicts: {event, data}."""
        sid = self.sessions.ensure(session_id)
        history = self.sessions.history(sid)
        timer = PipelineTimer(query=query)

        yield {"event": "start", "data": {"session_id": sid, "query": query, "latency_ms": 0.0}}

        qvec = self.embedder.encode_query(query)
        timer.mark("embed_done")
        yield {"event": "stage", "data": {"stage": "embed", "ms": timer.stages["embed_done"], "latency_ms": timer.elapsed_ms()}}

        candidates = self.index.search(qvec, k=settings.top_k_ann)
        timer.mark("ann_done")
        yield {"event": "stage", "data": {"stage": "ann", "ms": timer.gap("embed_done", "ann_done"), "hits": len(candidates), "latency_ms": timer.elapsed_ms()}}

        top = self._select_top(query, candidates)
        timer.mark("rerank_done")

        yield {
            "event": "retrieval_done",
            "data": {
                "embed_ms": timer.stages["embed_done"],
                "ann_ms": timer.gap("embed_done", "ann_done"),
                "rerank_ms": timer.gap("ann_done", "rerank_done"),
                "rerank_enabled": self.rerank_enabled,
                "retrieval_total_ms": timer.stages["rerank_done"],
                "chunks": [
                    {
                        "text": c["text"],
                        "source": c.get("source", "?"),
                        "rerank_score": round(c["rerank_score"], 4),
                    }
                    for c in top
                ],
                "latency_ms": timer.elapsed_ms(),
            },
        }

        answer_parts: list[str] = []
        first = True
        async for token in self.llm.stream(query, top, history):
            if first:
                timer.mark("first_token")
                yield {
                    "event": "first_token",
                    "data": {
                        "llm_ttft_ms": timer.gap("rerank_done", "first_token"),
                        "token": token,
                        "latency_ms": timer.elapsed_ms(),
                    },
                }
                first = False
            else:
                yield {"event": "token", "data": {"token": token, "latency_ms": timer.elapsed_ms()}}
            answer_parts.append(token)

        timer.mark("llm_done")
        answer = "".join(answer_parts)

        self.sessions.append_turn(sid, "user", query)
        self.sessions.append_turn(sid, "assistant", answer)

        summary = timer.summary()
        summary["session_id"] = sid
        summary["latency_ms"] = timer.elapsed_ms()
        summary["rerank_enabled"] = self.rerank_enabled
        yield {"event": "done", "data": summary}

    async def run_collect(self, query: str, session_id: str | None = None) -> dict:
        """Non-streaming: collect everything and return final dict."""
        chunks: list[dict] = []
        answer_parts: list[str] = []
        summary: dict = {}
        sid_out = session_id
        async for evt in self.run(query, session_id):
            if evt["event"] == "start":
                sid_out = evt["data"]["session_id"]
            elif evt["event"] == "retrieval_done":
                chunks = evt["data"]["chunks"]
            elif evt["event"] in ("first_token", "token"):
                answer_parts.append(evt["data"]["token"])
            elif evt["event"] == "done":
                summary = evt["data"]
        return {
            "session_id": sid_out,
            "query": query,
            "results": chunks,
            "answer": "".join(answer_parts),
            "timing": summary,
        }
