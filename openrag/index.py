from __future__ import annotations
import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from .config import settings


class VectorIndex:
    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ):
        self.dim = dim
        self.index_type = index_type
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = hnsw_ef_construction
            self.index.hnsw.efSearch = hnsw_ef_search
        else:
            raise ValueError(f"unknown index_type: {index_type!r} (want 'flat' or 'hnsw')")
        self.meta: list[dict] = []

    def add(self, vecs: np.ndarray, meta: list[dict]):
        assert vecs.shape[1] == self.dim, f"dim mismatch {vecs.shape[1]} != {self.dim}"
        assert len(meta) == vecs.shape[0]
        self.index.add(vecs)
        self.meta.extend(meta)

    def search(self, query: np.ndarray, k: int) -> list[dict]:
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, idxs = self.index.search(query, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hit = dict(self.meta[idx])
            hit["ann_score"] = float(score)
            out.append(hit)
        return out

    def save(self, dir_path: Path | None = None):
        d = Path(dir_path or settings.index_dir)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        with open(d / "meta.pkl", "wb") as f:
            pickle.dump(self.meta, f)
        (d / "info.json").write_text(json.dumps({
            "dim": self.dim,
            "count": self.index.ntotal,
            "index_type": getattr(self, "index_type", "flat"),
        }))

    @classmethod
    def load(cls, dir_path: Path | None = None) -> "VectorIndex":
        d = Path(dir_path or settings.index_dir)
        info = json.loads((d / "info.json").read_text())
        obj = cls(dim=info["dim"])
        obj.index = faiss.read_index(str(d / "faiss.index"))
        with open(d / "meta.pkl", "rb") as f:
            obj.meta = pickle.load(f)
        return obj

    @property
    def size(self) -> int:
        return self.index.ntotal
