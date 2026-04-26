"""Minimal RAG chatbot — drop-in starter for any project.

Prerequisites:
    1. pip install -e .
    2. python scripts/quantize_onnx.py    # one-time INT8 model build (~30s)
    3. cp .env.example .env  &&  set GROQ_API_KEY in .env
    4. Put some .pdf / .md / .txt files under ./my_docs/

Then:
    python examples/chatbot_minimal.py
"""
import asyncio
from openrag import Engine


async def main():
    # rerank=True (default) → ~80ms retrieval, best nDCG.
    # rerank=False → ~3-25ms retrieval, slightly lower nDCG, no cross-encoder loaded.
    engine = Engine.from_directory("./my_docs", rerank=True)
    print(f"Indexed {engine.index_size} chunks. Ask me anything (Ctrl+C to quit).\n")

    session_id = None
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue

        result = await engine.chat(q, session_id=session_id)
        session_id = result.session_id

        print(f"\n{result.answer}\n")
        if result.sources:
            print("Sources:")
            for s in result.sources:
                print(f"  [{s.rerank_score:.2f}] {s.source} — {s.preview}")
        print(f"  ({result.timing_ms.get('end_to_end_ms', '?')} ms)\n")


if __name__ == "__main__":
    asyncio.run(main())
