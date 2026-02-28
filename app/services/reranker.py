"""
Reranker service using a cross-encoder model.

Re-scores (query, passage) pairs so the most relevant chunks rise to the top.
"""

from __future__ import annotations

import structlog
import torch
from sentence_transformers import CrossEncoder

from app.config import settings

logger = structlog.get_logger(__name__)


class RerankerService:
    """Lazy-loaded cross-encoder reranker."""

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "loading_reranker",
                model=settings.reranker_model,
                device=device,
            )
            self._model = CrossEncoder(
                settings.reranker_model,
                device=device,
            )
            logger.info("reranker_loaded")
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank *documents* against *query*.

        Each document dict must have a ``"text"`` key.
        Returns the top-k documents sorted by relevance score (descending).
        """
        if not documents:
            return []

        k = top_k or settings.top_k
        model = self._load_model()

        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:k]


# Module-level singleton
reranker_service = RerankerService()
