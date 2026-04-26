# OpenRAG

A fast, drop-in retrieval engine for RAG chatbots — **3.7 ms p50 retrieval at 100k docs on a CPU laptop**, with quality matching or exceeding the published `bge-small-en-v1.5` baseline on BEIR SciFact.

- **CPU-first.** ONNX Runtime + INT8 dynamic quantization. No GPU required.
- **Quality preserved.** INT8 + HNSW give nDCG@10 = 0.7034 on SciFact (above the FP32 published baseline).
- **Batteries included.** Embedder, FAISS index (flat or HNSW), cross-encoder reranker, Groq LLM streaming, FastAPI server, CLI, and a high-level `Engine` facade.
- **Pluggable.** Swap LLM providers, swap embedding models, run with PyTorch or ONNX backends.

> The full benchmark methodology and results are in [BENCHMARKS.md](BENCHMARKS.md).

---

## Install

```bash
git clone <your-fork-url> open-rag
cd open-rag
pip install -e .
cp .env.example .env       # then add your GROQ_API_KEY
python scripts/quantize_onnx.py    # one-time INT8 model build (~30s)
```

---

## Use it in your chatbot — 5 lines

```python
import asyncio
from openrag import Engine

async def main():
    engine = Engine.from_directory("./my_docs", rerank=True)   # rerank=False for ~25ms retrieval
    result = await engine.chat("What's our refund policy?")
    print(result.answer)
    for s in result.sources:
        print(f"  [{s.rerank_score:.2f}] {s.source} — {s.preview}")

asyncio.run(main())
```

**`rerank=True`** (default) — cross-encoder rerank for higher nDCG, ~80 ms steady-state retrieval.
**`rerank=False`** — skip rerank, ~3-25 ms retrieval. Cross-encoder model isn't even loaded (saves ~200 MB RAM). Pick this when you have lots of high-quality chunks and care more about latency than the last few points of nDCG.

That's it. See [`examples/chatbot_minimal.py`](examples/chatbot_minimal.py) and [`examples/chatbot_streaming.py`](examples/chatbot_streaming.py) for runnable starters.

### Streaming variant (for live chat UIs)

```python
async for evt in engine.stream("how do I reset my password"):
    if evt["event"] in ("first_token", "token"):
        print(evt["data"]["token"], end="", flush=True)
```

### Just retrieve (no LLM)

```python
hits = engine.search("refund policy", top_k=5)
for h in hits:
    print(h.source, h.rerank_score, h.preview)
```

---

## Run the API server + UI

```bash
python -m openrag.cli serve
# → http://127.0.0.1:8000/ui          (live-timing chat UI)
# → http://127.0.0.1:8000/embed/demo  (the embeddable widget, in a demo page)
# → http://127.0.0.1:8000/search      (POST {"query": "..."})
# → http://127.0.0.1:8000/stream?q=…  (Server-Sent Events)
```

---

## Deploy on your website

One-line embed for any HTML page:

```html
<script src="https://api.your-site.com/embed.js"
        data-openrag-url="https://api.your-site.com"
        data-openrag-title="Ask our docs"
        data-openrag-color="#22d3ee"
        async></script>
```

That's a floating chat bubble that streams answers + cites sources. Host the backend with Docker:

```bash
docker compose up -d --build       # builds INT8 model on first start, persists data/
```

Full deployment guide (Render / Railway / Fly.io / self-hosted, plus widget config + CORS): **[DEPLOY.md](DEPLOY.md)**.

## CLI

```bash
python -m openrag.cli ingest --dir ./my_docs   # build index
python -m openrag.cli ask "your question"       # query the running server
python -m openrag.cli bench --device cpu --no-rerank \
    --index-type hnsw --embed-backend onnx --embed-precision int8 \
    --rounds 50 --reuse-index --output bench.json
```

---

## Architecture

```
query
  │
  ├─► Embedder (ONNX INT8 bge-small-en-v1.5)        ~1.5 ms
  ├─► FAISS HNSW (top-20 ANN)                       ~0.2 ms
  ├─► CrossEncoder rerank (ms-marco-MiniLM-L-6-v2)  ~2.0 ms  (optional)
  └─► GroqLLM stream → answer                        ~LLM TTFT
```

Source layout:

```
openrag/                 # the package
├── engine.py            # high-level facade (Engine, ChatResult, SearchResult)
├── pipeline.py          # full embed → ANN → rerank → LLM streaming pipeline
├── embed_onnx.py        # ONNX INT8 backend (default)
├── embed.py             # PyTorch backend (fallback / GPU)
├── index.py             # FAISS flat / HNSW
├── rerank.py            # CrossEncoder rerank
├── llm.py               # Groq streaming LLM
├── ingest.py            # PDF / MD / TXT loader + chunker
├── session.py           # SQLite-backed multi-turn history
├── timer.py             # per-stage latency markers
├── app.py               # FastAPI server
└── cli.py               # Typer CLI: ask / serve / ingest / bench / compare

scripts/
├── quantize_onnx.py     # one-time INT8 model build
├── eval_beir.py         # BEIR SciFact quality eval
└── eval_llm_judge.py    # Groq LLM-as-judge eval

benchmarks/reports/      # published 3-run INT8 + HNSW results
examples/                # chatbot starters
static/
├── index.html           # browser chat UI (live-timing dashboard)
├── embed.js             # drop-in widget: <script> tag for any website
└── embed-demo.html      # /embed/demo — preview page for the widget

Dockerfile               # production-ready container
docker-compose.yml       # one-command deploy
DEPLOY.md                # full deployment guide
```

---

## Configuration

All settings live in `.env` (see `.env.example`):

| var | default | what it does |
|---|---|---|
| `GROQ_API_KEY` | — | required for `chat()` / `stream()` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM model |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker model |
| `DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` |
| `TOP_K_ANN` | `20` | candidates fetched from FAISS |
| `TOP_K_RERANK` | `5` | final chunks after rerank |
| `MAX_HISTORY_TURNS` | `6` | how many prior turns the LLM sees |
| `RERANK_ENABLED` | `true` | cross-encoder rerank on/off; overridden by `Engine(rerank=...)` |
| `INDEX_DIR` | `./data/index` | FAISS index location |
| `DOCS_DIR` | `./data/docs` | corpus location for `ingest` |
| `CORS_ORIGINS` | (empty) | comma-separated origins allowed to call the API from a browser; set this when embedding the widget on your site |

---

## License

Apache-2.0. See [LICENSE](LICENSE).
