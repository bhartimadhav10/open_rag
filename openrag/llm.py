from __future__ import annotations
from typing import AsyncIterator
from groq import AsyncGroq
from .config import settings


SYSTEM_PROMPT = """You are a retrieval-augmented assistant powered by OpenRAG.

Answer the user's question using ONLY the provided context chunks. Cite sources
inline as [source_filename] after facts that depend on them. If the context
does not contain the answer, say so explicitly — do not invent.

Be concise. Prefer direct answers over preamble."""


def build_messages(
    query: str,
    chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    context = "\n\n".join(
        f"[chunk {i+1} — source: {c.get('source','?')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ) or "(no context retrieved)"

    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })
    return msgs


class GroqLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or settings.groq_api_key
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in environment/.env")
        self.client = AsyncGroq(api_key=key)
        self.model = model or settings.groq_model

    async def stream(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict],
    ) -> AsyncIterator[str]:
        messages = build_messages(query, chunks, history)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.2,
            max_tokens=1024,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
