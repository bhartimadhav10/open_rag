"""OpenRAG — a fast, drop-in retrieval engine for RAG chatbots.

Quickstart:
    from openrag import Engine
    engine = Engine.from_directory("./my_docs")
    result = await engine.chat("What's our refund policy?")
    print(result.answer)
"""
__version__ = "0.1.0"
__all__ = ["Engine", "ChatResult", "SearchResult", "__version__"]


def __getattr__(name):
    # Lazy-load heavy deps (torch, sentence_transformers, faiss) on first use,
    # so `import openrag` itself stays fast and dependency-free.
    if name in __all__ and name != "__version__":
        from . import engine as _engine
        return getattr(_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
