"""
Qdrant vector-store service (local / embedded mode — no Docker).

Provides CRUD operations for document chunks.
"""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.models.schemas import ChunkMetadata
from app.services.embedding import embedding_service

logger = structlog.get_logger(__name__)


class RetrievalService:
    """Manages the Qdrant collection for RAG document chunks."""

    def __init__(self) -> None:
        self._client: QdrantClient | None = None

    # ── lifecycle ───────────────────────────────────────────────────────

    def init(self) -> None:
        """Open (or create) the local Qdrant database."""
        path = str(settings.vector_db_path)
        logger.info("initializing_qdrant", path=path)
        self._client = QdrantClient(path=path)
        self._ensure_collection()

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            logger.info("qdrant_closed")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("RetrievalService not initialised — call .init() first")
        return self._client

    # ── collection management ───────────────────────────────────────────

    def _ensure_collection(self) -> None:
        name = settings.qdrant_collection
        collections = [c.name for c in self.client.get_collections().collections]
        if name not in collections:
            dim = embedding_service.dimension
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("collection_created", name=name, dimension=dim)
        else:
            logger.info("collection_exists", name=name)

    # ── write ───────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[ChunkMetadata]) -> None:
        """Embed and upsert a batch of chunks into Qdrant."""
        texts = [c.text for c in chunks]
        embeddings = embedding_service.encode(texts)

        points = [
            PointStruct(
                id=idx,
                vector=vec,
                payload=chunk.model_dump(),
            )
            for idx, (chunk, vec) in enumerate(
                zip(chunks, embeddings), start=self._next_id()
            )
        ]

        self.client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        logger.info("chunks_stored", count=len(points))

    # ── read ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the top-k most similar chunks for *query*."""
        k = top_k or settings.top_k
        query_vec = embedding_service.encode(query)[0]

        results = self.client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vec,
            limit=k,
        )

        return [
            {
                "score": hit.score,
                **hit.payload,
            }
            for hit in results
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique documents stored in the collection."""
        # Scroll all points (lightweight — payloads only, no vectors)
        records, _ = self.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )

        docs: dict[str, dict] = {}
        for rec in records:
            payload = rec.payload or {}
            doc_id = payload.get("document_id", "")
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": payload.get("filename", ""),
                    "detected_language": payload.get("detected_language", ""),
                    "upload_timestamp": payload.get("upload_timestamp", ""),
                    "chunk_count": 0,
                }
            if doc_id in docs:
                docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a given document_id. Returns count deleted."""
        self.client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info("document_deleted", document_id=document_id)
        return 1  # Qdrant delete is idempotent

    # ── helpers ─────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        """Simple auto-increment ID based on current point count."""
        info = self.client.get_collection(settings.qdrant_collection)
        return info.points_count


# Module-level singleton
retrieval_service = RetrievalService()
