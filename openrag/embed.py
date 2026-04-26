from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import settings


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.device = device or settings.resolve_device()
        self.model = SentenceTransformer(
            model_name or settings.embed_model,
            device=self.device,
        )
        if self.device == "cuda":
            self.model.half()
        self.dim = self.model.get_sentence_embedding_dimension()
        self._warmup()

    def _warmup(self):
        self.encode_query("warmup")

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vec = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vec.astype(np.float32)
