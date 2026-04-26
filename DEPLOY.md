# Deploying OpenRAG

This guide covers two things:

1. **Hosting the OpenRAG backend** — Docker is the recommended path; managed platforms (Render / Railway / Fly.io) work too.
2. **Embedding the chat into your website** — one `<script>` tag.

---

## 1. Run the backend

### Option A — Docker (recommended)

Prereqs: Docker, a `.env` with `GROQ_API_KEY`, and a folder of documents to index.

```bash
git clone <your-fork-url> open-rag
cd open-rag
cp .env.example .env       # then edit: set GROQ_API_KEY and CORS_ORIGINS
docker compose up -d --build
```

What happens on first start:

- The container builds a fresh INT8 ONNX model into `./data/onnx-bge-small-int8/` (~30 s).
- Your `data/` folder is mounted as a volume, so the model + FAISS index + session DB **persist across restarts**.

Then ingest your docs and verify:

```bash
docker compose exec openrag python -m openrag.cli ingest --dir /app/data/docs
curl http://localhost:8000/health
# → {"status":"ok","index_size":N,"device":"cpu"}
open http://localhost:8000/embed/demo   # live widget on a sample page
```

To upgrade later: `git pull && docker compose up -d --build`. The volume is preserved.

### Option B — Local Python (no Docker)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[onnx]"
python scripts/quantize_onnx.py  # one-time INT8 build (~30 s)
cp .env.example .env             # set GROQ_API_KEY, CORS_ORIGINS
python -m openrag.cli ingest --dir ./my_docs
python -m openrag.cli serve      # → http://127.0.0.1:8000
```

> If you hit `numpy.dtype size changed, may indicate binary incompatibility`,
> your current Python env has NumPy 2.x against modules built for NumPy 1.x.
> Fix: `pip install "numpy<2" --force-reinstall`, or use a fresh venv.

---

## 2. Deploy to a managed host

These all build the same `Dockerfile`. Pick one.

### Render

1. **New → Web Service**, connect your repo.
2. Runtime: **Docker**. Plan: Standard or above (the INT8 model needs ~600 MB RAM).
3. Add a **Disk** mounted at `/app/data` (size ≥ 2 GB) — survives deploys.
4. Add env vars: `GROQ_API_KEY`, `CORS_ORIGINS=https://your-site.com`.
5. Deploy. Render injects `PORT` automatically; the entrypoint honors it.

### Railway

1. **New Project → Deploy from GitHub**.
2. Railway detects the `Dockerfile`. Add a **Volume** mounted at `/app/data`.
3. Variables: `GROQ_API_KEY`, `CORS_ORIGINS`. Railway injects `PORT`.
4. Deploy.

### Fly.io

```bash
fly launch                               # accept Dockerfile, decline Postgres
fly volumes create openrag_data --size 2 --region <your-region>
```

Edit `fly.toml` to mount the volume:

```toml
[mounts]
  source = "openrag_data"
  destination = "/app/data"

[env]
  OPENRAG_HOST = "0.0.0.0"
```

```bash
fly secrets set GROQ_API_KEY=gsk_xxx CORS_ORIGINS=https://your-site.com
fly deploy
```

### Self-hosted VPS (Caddy + docker-compose)

Put OpenRAG behind a reverse proxy that handles HTTPS:

```caddy
api.your-site.com {
    reverse_proxy localhost:8000 {
        flush_interval -1            # important for SSE streaming
    }
}
```

Then `docker compose up -d` and Caddy auto-issues a cert. The `flush_interval -1` line is critical — without it, token streaming will buffer and feel broken.

> **Nginx note:** if you use Nginx instead, set `proxy_buffering off;` on the `/stream` location for the same reason.

---

## 3. Embed the chat into your website

Once the backend is running at, say, `https://api.your-site.com`, drop this into any HTML page:

```html
<script src="https://api.your-site.com/embed.js"
        data-openrag-url="https://api.your-site.com"
        data-openrag-title="Ask our docs"
        data-openrag-color="#22d3ee"
        data-openrag-position="bottom-right"
        data-openrag-greeting="Hi! Ask me anything about our product."
        async></script>
```

That's it — a floating chat bubble appears on every page that includes the script.

### Configurable attributes

| attribute | default | what it does |
|---|---|---|
| `data-openrag-url`      | (required) | base URL of your OpenRAG backend |
| `data-openrag-title`    | `Ask anything` | header text in the chat panel |
| `data-openrag-color`    | `#22d3ee` | accent color (button + user messages) |
| `data-openrag-position` | `bottom-right` | `bottom-right` \| `bottom-left` \| `top-right` \| `top-left` |
| `data-openrag-greeting` | `Hi! Ask me anything…` | first bot message; empty = no greeting |

### CORS — what to set

The browser will block requests from `your-site.com` to `api.your-site.com` unless the backend allows the origin. In your **backend `.env`**:

```
CORS_ORIGINS=https://your-site.com,https://www.your-site.com
```

(Comma-separated. No trailing slash. Must match the page origin exactly.) Restart the backend.

To verify: open the demo page at `https://api.your-site.com/embed/demo` — the widget there always works because it's same-origin. Then open your real site with the `<script>` tag and check the browser DevTools → Network tab for any CORS errors on `/stream`.

### Programmatic control

The widget exposes a small global API:

```js
OpenRAG.open();          // open the panel
OpenRAG.close();         // close it
OpenRAG.toggle();
OpenRAG.reset();         // clear conversation + start a fresh session
```

Wire it to your own button:

```html
<button onclick="OpenRAG.open()">Need help?</button>
```

---

## 4. Production checklist

- [ ] `.env` has `GROQ_API_KEY` set, **and `.env` is gitignored** (it is by default — verify).
- [ ] `CORS_ORIGINS` lists every origin that will load the widget. No `*` in production.
- [ ] `data/` is mounted as a persistent volume (Docker volume, Render Disk, Fly Volume…).
- [ ] You ran an `ingest` so `index_size > 0` at `/health`.
- [ ] Your reverse proxy disables response buffering on `/stream` (see Caddy/Nginx notes above).
- [ ] If you run multiple replicas: only one should hold the writer for `data/sessions.sqlite`. SQLite is single-writer; for horizontal scaling, swap `session.py`'s `SessionStore` for Postgres or Redis (out of scope for this guide).

---

## 5. Common issues

| symptom | cause | fix |
|---|---|---|
| `GROQ_API_KEY not set` at startup | `.env` not loaded | confirm `.env` is in the working directory; in Docker, confirm `env_file: .env` in compose |
| Widget shows but answers never arrive | reverse proxy buffering SSE | set `flush_interval -1` (Caddy) or `proxy_buffering off;` (Nginx) on `/stream` |
| `Access-Control-Allow-Origin` error in DevTools | origin not in `CORS_ORIGINS` | add the exact origin (scheme + host, no path) and restart |
| `numpy.dtype size changed` on start | NumPy 1.x/2.x mismatch in your env | `pip install "numpy<2" --force-reinstall` or use a fresh venv / Docker |
| Empty answers, fast response | index is empty | run `python -m openrag.cli ingest --dir <your-docs>` |
| Slow first request, fast after | cold-start: model + index loading | call `/health` once at deploy time to warm up |

---

For benchmarks and quality numbers, see [BENCHMARKS.md](BENCHMARKS.md).
For the API reference, see [README.md](README.md).
