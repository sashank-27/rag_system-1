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
        """
        Return the top-k most similar chunks for *query* with document diversity.
        
        Fetches extra results and interleaves across documents so that
        chunks from multiple PDFs are included in the final set.
        """
        k = top_k or settings.top_k
        # Fetch 3x to have enough for diversity re-ordering
        fetch_k = min(k * 3, 50)
        query_vec = embedding_service.encode(query)[0]

        results = self.client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vec,
            limit=fetch_k,
        )

        if not results:
            return []

        # Group hits by document_id
        from collections import OrderedDict
        doc_buckets: OrderedDict[str, list] = OrderedDict()
        for hit in results:
            doc_id = (hit.payload or {}).get("document_id", "unknown")
            if doc_id not in doc_buckets:
                doc_buckets[doc_id] = []
            doc_buckets[doc_id].append(hit)

        # Round-robin interleave across documents for diversity
        diverse_hits = []
        bucket_iters = [iter(bucket) for bucket in doc_buckets.values()]
        while len(diverse_hits) < k and bucket_iters:
            exhausted = []
            for i, it in enumerate(bucket_iters):
                if len(diverse_hits) >= k:
                    break
                try:
                    diverse_hits.append(next(it))
                except StopIteration:
                    exhausted.append(i)
            for i in reversed(exhausted):
                bucket_iters.pop(i)

        logger.debug(
            "search_diversity",
            total_fetched=len(results),
            documents_found=len(doc_buckets),
            returned=len(diverse_hits),
        )

        return [
            {
                "score": hit.score,
                **hit.payload,
            }
            for hit in diverse_hits
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
