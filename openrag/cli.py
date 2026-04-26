from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from typing import Optional
import typer
import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from .config import settings


app = typer.Typer(help="OpenRAG — stateful RAG with live timing")
console = Console()

STAGES = [
    ("embed",   "Embedding query"),
    ("ann",     "ANN search (FAISS)"),
    ("rerank",  "Reranking top-20"),
]


def _session_file() -> Path:
    return Path.home() / ".openrag_session"


def _load_session() -> Optional[str]:
    f = _session_file()
    return f.read_text().strip() if f.exists() else None


def _save_session(sid: str):
    _session_file().write_text(sid)


def _bar(frac: float, width: int = 11) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _render(state: dict, query: str, answer: str, chunks: list) -> Panel:
    retrieval = Table.grid(padding=(0, 1))
    retrieval.add_column(justify="left", width=24)
    retrieval.add_column(justify="left", width=13)
    retrieval.add_column(justify="right", width=10)
    retrieval.add_column(justify="left")

    total = state.get("retrieval_total_ms") or 0.0
    for key, label in STAGES:
        ms = state.get(f"{key}_ms")
        if ms is None:
            retrieval.add_row(f"{label}...", Text("░" * 11, style="dim"), Text("waiting", style="dim"), "")
        else:
            frac = (ms / total) if total > 0 else 0.0
            retrieval.add_row(
                f"{label}...",
                Text(_bar(frac), style="green"),
                Text(f"{ms} ms", style="green"),
                Text("✓", style="green"),
            )

    retrieval_total_line = (
        Text(f"Retrieval complete: {total} ms", style="bold green")
        if "rerank_ms" in state
        else Text("Retrieval in progress...", style="dim")
    )

    llm_tbl = Table.grid(padding=(0, 1))
    llm_tbl.add_column(width=24)
    llm_tbl.add_column(justify="right")
    ttft = state.get("llm_ttft_ms")
    total_gen = state.get("llm_total_ms")
    llm_tbl.add_row("Time to first token:", Text(f"{ttft} ms", style="green") if ttft is not None else Text("waiting", style="dim"))
    llm_tbl.add_row("Total generation:",    Text(f"{total_gen} ms", style="green") if total_gen is not None else Text("waiting", style="dim"))

    e2e = state.get("end_to_end_ms")
    e2e_line = Text(f"END TO END: {e2e} ms", style="bold green") if e2e is not None else Text("END TO END: —", style="dim")

    header = Text(f'Query: "{query}"', style="bold cyan")

    body = Group(
        header,
        Text(""),
        Text("RETRIEVAL PIPELINE", style="bold"),
        retrieval,
        retrieval_total_line,
        Text(""),
        Text("LLM GENERATION", style="bold"),
        llm_tbl,
        Text(""),
        e2e_line,
    )
    return Panel(body, title="OpenRAG", border_style="cyan", expand=False)


async def _ask(query: str, session_id: Optional[str], base_url: str):
    state: dict = {}
    answer_parts: list[str] = []
    chunks: list = []
    final_session = session_id

    params = {"q": query}
    if session_id:
        params["session_id"] = session_id

    with Live(_render(state, query, "", chunks), console=console, refresh_per_second=20) as live:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"{base_url}/stream", params=params) as resp:
                event_type = "message"
                data_buf: list[str] = []
                async for raw in resp.aiter_lines():
                    if raw == "":
                        if data_buf:
                            payload = json.loads("\n".join(data_buf))
                            _apply_event(event_type, payload, state, answer_parts, chunks)
                            if event_type == "start":
                                final_session = payload.get("session_id", final_session)
                            live.update(_render(state, query, "".join(answer_parts), chunks))
                        event_type, data_buf = "message", []
                    elif raw.startswith("event:"):
                        event_type = raw[6:].strip()
                    elif raw.startswith("data:"):
                        data_buf.append(raw[5:].lstrip())

    if final_session:
        _save_session(final_session)

    if chunks:
        tbl = Table(title="Retrieved sources", show_lines=False, header_style="bold cyan")
        tbl.add_column("Rank", justify="right")
        tbl.add_column("Source")
        tbl.add_column("Score", justify="right")
        tbl.add_column("Preview")
        for i, c in enumerate(chunks, 1):
            preview = c["text"].strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            tbl.add_row(str(i), c.get("source", "?"), f"{c['rerank_score']:.3f}", preview)
        console.print(tbl)

    console.print(Panel(Markdown("".join(answer_parts) or "_(no answer)_"), title="Answer", border_style="green"))
    if final_session:
        console.print(f"[dim]session: {final_session}[/dim]")


def _apply_event(event_type: str, data: dict, state: dict, answer_parts: list[str], chunks: list):
    if event_type == "stage":
        state[f"{data['stage']}_ms"] = data["ms"]
    elif event_type == "retrieval_done":
        state["embed_ms"] = data["embed_ms"]
        state["ann_ms"] = data["ann_ms"]
        state["rerank_ms"] = data["rerank_ms"]
        state["retrieval_total_ms"] = data["retrieval_total_ms"]
        chunks.clear()
        chunks.extend(data["chunks"])
    elif event_type == "first_token":
        state["llm_ttft_ms"] = data["llm_ttft_ms"]
        answer_parts.append(data["token"])
    elif event_type == "token":
        answer_parts.append(data["token"])
    elif event_type == "done":
        state.update({k: v for k, v in data.items() if v is not None})


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session id (default: last used)"),
    new: bool = typer.Option(False, "--new", help="Start a fresh session"),
    base_url: str = typer.Option(None, "--url", help="Server base URL"),
):
    url = base_url or os.environ.get("OPENRAG_URL") or f"http://{settings.host}:{settings.port}"
    sid = None if new else (session or _load_session())
    asyncio.run(_ask(query, sid, url))


@app.command()
def reset(session: Optional[str] = typer.Option(None, "--session", "-s")):
    sid = session or _load_session()
    if not sid:
        console.print("[yellow]No active session to reset[/yellow]")
        raise typer.Exit(1)
    url = os.environ.get("OPENRAG_URL") or f"http://{settings.host}:{settings.port}"
    r = httpx.post(f"{url}/sessions/{sid}/reset", timeout=10)
    r.raise_for_status()
    console.print(f"[green]Session {sid} reset[/green]")


@app.command()
def ingest(docs_dir: Optional[Path] = typer.Option(None, "--dir", "-d")):
    from .ingest import ingest as do_ingest
    with console.status("[cyan]Ingesting documents..."):
        info = do_ingest(docs_dir)
    console.print(f"[green]Ingested {info['chunks']} chunks from {info['files']} files (dim={info['dim']})[/green]")
    console.print(f"[dim]Index saved to {info['index_dir']}[/dim]")


@app.command()
def bench(
    docs: str = typer.Option(None, "--docs", help="Path or URL to bench docs JSON (default: Moss 100k)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap number of docs for quick tests"),
    warmup: int = typer.Option(3, "--warmup"),
    rounds: int = typer.Option(50, "--rounds"),
    top_k_ann: Optional[int] = typer.Option(None, "--top-k-ann"),
    top_k_rerank: int = typer.Option(5, "--top-k-rerank"),
    queries_file: Optional[Path] = typer.Option(None, "--queries-file"),
    device: Optional[str] = typer.Option(None, "--device", help="auto | cuda | cpu  (overrides .env DEVICE)"),
    embed_device: Optional[str] = typer.Option(None, "--embed-device", help="Device for embedder (overrides --device)"),
    rerank_device: Optional[str] = typer.Option(None, "--rerank-device", help="Device for reranker (overrides --device)"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip cross-encoder rerank (measures embed + ANN only)"),
    embed_batch: int = typer.Option(128, "--embed-batch", help="Batch size for doc embedding during index build"),
    index_type: str = typer.Option("flat", "--index-type", help="flat | hnsw  (HNSW = fast approximate ANN, 10× faster at 100k)"),
    hnsw_ef_search: int = typer.Option(64, "--hnsw-ef-search", help="HNSW efSearch (higher = better recall, slower)"),
    embed_backend: str = typer.Option("pytorch", "--embed-backend", help="pytorch | onnx  (ONNX Runtime usually faster on CPU)"),
    embed_precision: str = typer.Option("fp32", "--embed-precision", help="fp32 | fp16 | int8  (ONNX only; int8 ~2x faster on CPU)"),
    reuse_index: bool = typer.Option(False, "--reuse-index", help="Reuse cached embeddings+index from prior run (skip embedding step)"),
    cache_dir: str = typer.Option("data/index-cache", "--cache-dir", help="Where to store/find cached indexes"),
    output: Path = typer.Option(Path("bench_report.json"), "--output"),
):
    """Run Moss-compatible retrieval benchmark."""
    from .bench import run_bench, DEFAULT_DOCS_URL
    qs = None
    if queries_file:
        qs = json.loads(queries_file.read_text(encoding="utf-8"))
    run_bench(
        docs_source=docs or DEFAULT_DOCS_URL,
        queries=qs,
        doc_limit=limit,
        warmup=warmup,
        rounds=rounds,
        top_k_ann=top_k_ann,
        top_k_rerank=top_k_rerank,
        output=output,
        device=device,
        embed_device=embed_device,
        rerank_device=rerank_device,
        no_rerank=no_rerank,
        embed_batch=embed_batch,
        index_type=index_type,
        hnsw_ef_search=hnsw_ef_search,
        embed_backend=embed_backend,
        embed_precision=embed_precision,
        reuse_index=reuse_index,
        cache_dir=cache_dir,
    )


MOSS_BASELINE = {
    "label": "Moss (published, M4 Pro)",
    "docs": 100000,
    "device": "M4 Pro",
    "rerank_enabled": False,
    "total": {"mean": 3.3, "p50": 3.1, "p95": 4.3, "p99": 5.4},
}


@app.command()
def compare(
    reports: list[Path] = typer.Argument(..., help="Paths to bench_report.json files"),
    no_moss: bool = typer.Option(False, "--no-moss", help="Exclude Moss published baseline"),
    output: Optional[Path] = typer.Option(None, "--output", help="Write comparison table to a markdown file"),
):
    """Compare multiple bench_report.json files side-by-side."""
    rows: list[dict] = []
    for p in reports:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[red]skipping {p}: {e}[/red]")
            continue
        agg = data.get("aggregate_ms", {})
        total = agg.get("total") or {}
        rerank_on = data.get("rerank_enabled", "rerank" in agg)
        label = f"OpenRAG {data.get('device','?')} · {'rerank' if rerank_on else 'no-rerank'} · {data.get('docs','?')} docs"
        rows.append({
            "label": label,
            "docs": data.get("docs", "?"),
            "device": data.get("device", "?"),
            "rerank_enabled": rerank_on,
            "total": total,
            "embed": agg.get("embed") or {},
            "ann": agg.get("ann") or {},
            "rerank": agg.get("rerank") or {},
        })

    if not no_moss:
        rows.append(MOSS_BASELINE)

    tbl = Table(title="Retrieval latency comparison (total_ms)", header_style="bold cyan", show_lines=False)
    tbl.add_column("Run", style="cyan")
    tbl.add_column("docs", justify="right")
    tbl.add_column("rerank", justify="center")
    tbl.add_column("mean", justify="right")
    tbl.add_column("p50", justify="right")
    tbl.add_column("p95", justify="right")
    tbl.add_column("p99", justify="right")

    def _fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "—"

    for r in rows:
        t = r.get("total", {})
        rerank_cell = "✓" if r.get("rerank_enabled") else "—"
        tbl.add_row(
            r["label"],
            str(r.get("docs", "?")),
            rerank_cell,
            _fmt(t.get("mean")),
            _fmt(t.get("p50")),
            _fmt(t.get("p95")),
            _fmt(t.get("p99")),
        )
    console.print(tbl)

    openrag_rows = [r for r in rows if r["label"] != MOSS_BASELINE["label"]]
    if openrag_rows:
        stage_tbl = Table(title="Per-stage breakdown (mean ms)", header_style="bold magenta")
        stage_tbl.add_column("Run", style="magenta")
        stage_tbl.add_column("embed", justify="right")
        stage_tbl.add_column("ann", justify="right")
        stage_tbl.add_column("rerank", justify="right")
        stage_tbl.add_column("total", justify="right")
        for r in openrag_rows:
            stage_tbl.add_row(
                r["label"],
                _fmt(r.get("embed", {}).get("mean")),
                _fmt(r.get("ann", {}).get("mean")),
                _fmt(r.get("rerank", {}).get("mean")) if r.get("rerank") else "—",
                _fmt(r.get("total", {}).get("mean")),
            )
        console.print(stage_tbl)

    if output:
        lines = [
            "# Retrieval latency comparison",
            "",
            "| Run | docs | rerank | mean | p50 | p95 | p99 |",
            "|---|---:|:-:|---:|---:|---:|---:|",
        ]
        for r in rows:
            t = r.get("total", {})
            rerank_cell = "✓" if r.get("rerank_enabled") else "—"
            lines.append(
                f"| {r['label']} | {r.get('docs','?')} | {rerank_cell} | "
                f"{_fmt(t.get('mean'))} | {_fmt(t.get('p50'))} | {_fmt(t.get('p95'))} | {_fmt(t.get('p99'))} |"
            )
        if openrag_rows:
            lines += ["", "## Per-stage breakdown (mean ms)", "",
                      "| Run | embed | ann | rerank | total |",
                      "|---|---:|---:|---:|---:|"]
            for r in openrag_rows:
                lines.append(
                    f"| {r['label']} | {_fmt(r.get('embed', {}).get('mean'))} | "
                    f"{_fmt(r.get('ann', {}).get('mean'))} | "
                    f"{_fmt(r.get('rerank', {}).get('mean')) if r.get('rerank') else '—'} | "
                    f"{_fmt(r.get('total', {}).get('mean'))} |"
                )
        Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"[green]wrote comparison → {output}[/green]")


@app.command()
def serve(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
    reload: bool = typer.Option(False),
):
    import uvicorn
    uvicorn.run(
        "openrag.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
