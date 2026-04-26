from __future__ import annotations
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import settings


LOCAL_INT8_DIR = Path("data/onnx-bge-small-int8")


class OnnxEmbedder:
    """ONNX Runtime backend for sentence-transformers embedders.

    precision: "fp32" | "int8"
      - fp32: loads BAAI/bge-small-en-v1.5 pre-exported ONNX from HF hub
      - int8: loads locally-quantized model from ./data/onnx-bge-small-int8/
              (run `python quantize_onnx.py` once to build it)
    """
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        precision: str = "fp32",
    ):
        self.device = device or settings.resolve_device()
        self.precision = precision

        if precision == "int8":
            if not LOCAL_INT8_DIR.exists():
                raise FileNotFoundError(
                    f"INT8 model not built yet. Run: python quantize_onnx.py\n"
                    f"(expected at {LOCAL_INT8_DIR.resolve()})"
                )
            load_path = str(LOCAL_INT8_DIR)
            model_kwargs: dict = {"file_name": "onnx/model_qint8_avx512_vnni.onnx"}
        elif precision == "fp32":
            load_path = model_name or settings.embed_model
            model_kwargs = {}  # default onnx/model.onnx
        else:
            raise ValueError(f"precision must be fp32 or int8, got {precision!r}")

        self.model = SentenceTransformer(
            load_path,
            backend="onnx",
            device=self.device,
            model_kwargs=model_kwargs,
        )
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
