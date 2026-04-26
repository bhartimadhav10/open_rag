"""Streaming RAG chatbot — token-by-token output for live chat UIs.

Same prerequisites as chatbot_minimal.py.
"""
import asyncio
import sys
from openrag import Engine


async def main():
    engine = Engine.from_directory("./my_docs")
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

        async for evt in engine.stream(q, session_id=session_id):
            kind = evt["event"]
            data = evt["data"]
            if kind == "start":
                session_id = data["session_id"]
            elif kind == "retrieval_done":
                print(f"[retrieved {len(data['chunks'])} chunks "
                      f"in {data['retrieval_total_ms']} ms]")
            elif kind in ("first_token", "token"):
                sys.stdout.write(data["token"])
                sys.stdout.flush()
            elif kind == "done":
                print(f"\n[total {data.get('end_to_end_ms', '?')} ms]\n")


if __name__ == "__main__":
    asyncio.run(main())
