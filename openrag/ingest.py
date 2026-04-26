from __future__ import annotations
import re
from pathlib import Path
from pypdf import PdfReader
from .config import settings
from .embed import Embedder
from .index import VectorIndex


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        piece = text[i : i + chunk_size]
        if piece.strip():
            chunks.append(piece)
        if i + chunk_size >= len(text):
            break
    return chunks


def ingest(docs_dir: Path | None = None) -> dict:
    docs = Path(docs_dir or settings.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    files = [p for p in docs.rglob("*") if p.suffix.lower() in (".pdf", ".txt", ".md", ".markdown") and p.is_file()]
    if not files:
        raise RuntimeError(f"No supported documents found under {docs}")

    embedder = Embedder()
    index = VectorIndex(dim=embedder.dim)

    all_chunks: list[str] = []
    all_meta: list[dict] = []
    for f in files:
        text = load_text(f)
        for i, ch in enumerate(chunk_text(text)):
            all_chunks.append(ch)
            all_meta.append({"source": f.name, "path": str(f), "chunk_idx": i, "text": ch})

    if not all_chunks:
        raise RuntimeError("No text extracted from documents")

    vecs = embedder.encode_docs(all_chunks)
    index.add(vecs, all_meta)
    index.save()
    return {"files": len(files), "chunks": len(all_chunks), "dim": embedder.dim, "index_dir": str(settings.index_dir)}
