import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineTimer:
    query: str
    _start: float = field(default_factory=time.perf_counter)
    stages: dict = field(default_factory=dict)

    def mark(self, stage: str) -> float:
        elapsed = round((time.perf_counter() - self._start) * 1000, 1)
        self.stages[stage] = elapsed
        return elapsed

    def gap(self, from_stage: str, to_stage: str) -> float:
        return round(self.stages[to_stage] - self.stages[from_stage], 1)

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 1)

    def summary(self) -> dict:
        s = self.stages
        return {
            "embed_ms":           s.get("embed_done", 0),
            "ann_ms":             self.gap("embed_done", "ann_done") if "ann_done" in s else 0,
            "rerank_ms":          self.gap("ann_done", "rerank_done") if "rerank_done" in s else 0,
            "retrieval_total_ms": s.get("rerank_done", 0),
            "llm_ttft_ms":        self.gap("rerank_done", "first_token") if "first_token" in s else None,
            "llm_total_ms":       self.gap("rerank_done", "llm_done") if "llm_done" in s else None,
            "end_to_end_ms":      s.get("llm_done", s.get("rerank_done", 0)),
        }
