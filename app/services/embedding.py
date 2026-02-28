"""
Multilingual embedding service using BAAI/bge-m3 via sentence-transformers.
"""

from __future__ import annotations

import structlog
import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    Wraps ``sentence-transformers`` to produce dense vectors for text.
    The model is loaded lazily on first call and cached for reuse.
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    # ── lazy loading ────────────────────────────────────────────────────

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "loading_embedding_model",
                model=settings.embedding_model,
                device=device,
            )
            self._model = SentenceTransformer(
                settings.embedding_model,
                device=device,
            )
            logger.info("embedding_model_loaded")
        return self._model

    # ── public API ──────────────────────────────────────────────────────

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        """
        Encode one or more texts into dense vectors.

        Returns a list of vectors (even for a single input).
        Normalizes Unicode and strips invisible characters before encoding.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Normalize text for consistent embeddings
        texts = [self._clean(t) for t in texts]

        model = self._load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @staticmethod
    def _clean(text: str) -> str:
        """Strip invisible Unicode chars that hurt embedding quality."""
        import re
        import unicodedata
        text = unicodedata.normalize("NFC", text)
        # Remove zero-width and format characters
        text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
        return text

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


# Module-level singleton
embedding_service = EmbeddingService()
