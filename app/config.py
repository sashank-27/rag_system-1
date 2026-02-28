"""
Centralized configuration loaded from environment variables.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — loaded from .env file or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Qdrant ──────────────────────────────────────────────
    qdrant_path: str = "./vector_db"
    qdrant_collection: str = "rag_documents"

    # ── Embedding ───────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"

    # ── LLM ─────────────────────────────────────────────────
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # ── Reranker ────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-base"

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = 600
    chunk_overlap: int = 100

    # ── Retrieval ───────────────────────────────────────────
    top_k: int = 5

    # ── Upload ──────────────────────────────────────────────
    max_file_size_mb: int = 50

    # ── Tesseract ───────────────────────────────────────────
    tesseract_cmd: str = "tesseract"

    # ── ServiceNow ──────────────────────────────────────────
    servicenow_instance: str = ""
    servicenow_username: str = ""
    servicenow_password: str = ""

    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def vector_db_path(self) -> Path:
        return Path(self.qdrant_path)

    @property
    def servicenow_base_url(self) -> str:
        return f"https://{self.servicenow_instance}.service-now.com"


settings = Settings()
