"""
Document management routes: upload, list, delete.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    UploadResponse,
)
from app.services.pdf_processor import chunk_document, extract_text_from_pdf
from app.services.retrieval import retrieval_service

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Documents"])


# ── POST /upload ────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a PDF, extract text, chunk, embed, and store in Qdrant.
    """
    # ── Validate content type ───────────────────────────────────────────
    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    # ── Validate file size ──────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb} MB.",
        )

    # Reset the file cursor for processing
    await file.seek(0)

    filename = file.filename or "unnamed.pdf"
    document_id = uuid.uuid4().hex

    logger.info("upload_started", filename=filename, document_id=document_id)

    try:
        # ── Extract text ────────────────────────────────────────────────
        pages = extract_text_from_pdf(file.file, filename)

        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the PDF.",
            )

        # ── Chunk ───────────────────────────────────────────────────────
        chunks = chunk_document(pages, filename, document_id)

        # ── Store embeddings ────────────────────────────────────────────
        retrieval_service.add_documents(chunks)

        detected_lang = chunks[0].detected_language if chunks else "unknown"

        logger.info(
            "upload_complete",
            document_id=document_id,
            total_chunks=len(chunks),
        )

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            total_chunks=len(chunks),
            detected_language=detected_lang,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("upload_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")


# ── GET /documents ──────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """Return all indexed documents."""
    try:
        docs = retrieval_service.list_documents()
        return DocumentListResponse(documents=docs, total=len(docs))
    except Exception as exc:
        logger.exception("list_documents_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── DELETE /documents/{document_id} ────────────────────────────────────────

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete all chunks belonging to a document."""
    try:
        retrieval_service.delete_document(document_id)
        return DeleteResponse(document_id=document_id)
    except Exception as exc:
        logger.exception("delete_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
