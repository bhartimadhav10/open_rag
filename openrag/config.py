from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    device: str = "auto"
    index_dir: Path = Path("./data/index")
    docs_dir: Path = Path("./data/docs")
    session_db: Path = Path("./data/sessions.sqlite")

    top_k_ann: int = 20
    top_k_rerank: int = 5
    max_history_turns: int = 6

    # Cross-encoder reranking. Default ON — better quality.
    # Set RERANK_ENABLED=false to skip rerank stage (~3-25ms steady-state
    # vs ~80ms with rerank), at the cost of slightly lower nDCG.
    # Engine(rerank=True/False) overrides this at construction time.
    rerank_enabled: bool = True

    host: str = "127.0.0.1"
    port: int = 8000

    # Comma-separated origins allowed to call the API from a browser.
    # Empty (default) = same-origin only. Use "*" for any origin (dev only).
    # Example for prod: "https://your-site.com,https://www.your-site.com"
    cors_origins: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


settings = Settings()
