from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .pipeline import Pipeline


app = FastAPI(title="OpenRAG")

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@app.on_event("startup")
async def _startup():
    get_pipeline()


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


@app.get("/ui")
async def ui():
    path = STATIC_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(path))


@app.get("/embed.js")
async def embed_js():
    """Serves the embeddable widget. Drop into any site:
        <script src="https://your-host/embed.js"
                data-openrag-url="https://your-host"></script>
    """
    path = STATIC_DIR / "embed.js"
    if not path.exists():
        raise HTTPException(status_code=404, detail="embed.js not found")
    return FileResponse(str(path), media_type="application/javascript")


@app.get("/embed/demo")
async def embed_demo():
    """A minimal HTML page that loads the widget — for testing the embed."""
    path = STATIC_DIR / "embed-demo.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="embed demo not found")
    return FileResponse(str(path))


@app.get("/health")
async def health():
    p = get_pipeline()
    return {
        "status": "ok",
        "index_size": p.index.size,
        "device": p.embedder.device,
        "rerank_enabled": p.rerank_enabled,
    }


@app.post("/search")
async def search(req: SearchRequest):
    p = get_pipeline()
    return await p.run_collect(req.query, req.session_id)


@app.get("/stream")
async def stream(
    q: str = Query(..., description="Query text"),
    session_id: Optional[str] = Query(None),
):
    p = get_pipeline()

    async def event_gen():
        async for evt in p.run(q, session_id):
            yield {
                "event": evt["event"],
                "data": json.dumps(evt["data"]),
            }

    return EventSourceResponse(event_gen())


@app.get("/sessions")
async def sessions():
    p = get_pipeline()
    return {"sessions": p.sessions.list_sessions()}


@app.get("/sessions/{session_id}")
async def session_detail(session_id: str):
    p = get_pipeline()
    return {"session_id": session_id, "history": p.sessions.history(session_id, max_turns=50)}


@app.post("/sessions/{session_id}/reset")
async def session_reset(session_id: str):
    p = get_pipeline()
    p.sessions.reset(session_id)
    return {"session_id": session_id, "reset": True}
