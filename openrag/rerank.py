from __future__ import annotations
from sentence_transformers import CrossEncoder
from .config import settings


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.device = device or settings.resolve_device()
        self.model = CrossEncoder(
            model_name or settings.rerank_model,
            device=self.device,
            max_length=512,
        )
        self._warmup()

    def _warmup(self):
        self.model.predict([("warmup", "warmup")], show_progress_bar=False)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
