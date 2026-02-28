"""
PDF text extraction and chunking service.

- Uses PyMuPDF (fitz) for text-based PDFs.
- Falls back to Tesseract OCR for scanned / image-based pages.
- Chunks text with configurable size and overlap.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import BinaryIO

import fitz  # PyMuPDF
import pytesseract
import structlog
from PIL import Image

from app.config import settings
from app.models.schemas import ChunkMetadata
from app.services.language_detector import language_detector

logger = structlog.get_logger(__name__)

# Point pytesseract at the configured binary
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


# ── Text extraction ─────────────────────────────────────────────────────────

def _extract_text_from_page(page: fitz.Page) -> str:
    """Extract text from a page; fall back to OCR if empty."""
    text = page.get_text("text").strip()
    if text:
        return text

    # OCR fallback: render page to image then run Tesseract
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img, lang="eng+hin+guj")
        logger.debug("ocr_applied", page=page.number)
        return ocr_text.strip()
    except Exception as exc:
        logger.warning("ocr_failed", page=page.number, error=str(exc))
        return ""


def extract_text_from_pdf(file: BinaryIO, filename: str) -> list[dict]:
    """
    Read a PDF and return a list of ``{page_number, text}`` dicts.
    """
    data = file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[dict] = []
    for page in doc:
        text = _extract_text_from_page(page)
        if text:
            pages.append({"page_number": page.number + 1, "text": text})
    doc.close()
    logger.info("pdf_extracted", filename=filename, pages=len(pages))
    return pages


# ── Chunking ────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split *text* into chunks of roughly *chunk_size* characters with
    *overlap* character overlap.  Splits on paragraph/sentence boundaries
    when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a clean break point
        split_pos = end
        for sep in separators:
            pos = text.rfind(sep, start, end)
            if pos > start:
                split_pos = pos + len(sep)
                break

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Move start back by overlap
        start = max(start + 1, split_pos - overlap)

    return chunks


def chunk_document(
    pages: list[dict],
    filename: str,
    document_id: str | None = None,
) -> list[ChunkMetadata]:
    """
    Chunk extracted pages and attach metadata to each chunk.
    """
    if document_id is None:
        document_id = uuid.uuid4().hex

    upload_ts = datetime.now(timezone.utc).isoformat()

    # Detect dominant language from first available text
    sample_text = " ".join(p["text"][:500] for p in pages[:3])
    detected_lang = language_detector.detect(sample_text)

    chunks: list[ChunkMetadata] = []
    global_idx = 0

    for page in pages:
        page_chunks = _split_text(
            page["text"],
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for text in page_chunks:
            chunks.append(
                ChunkMetadata(
                    document_id=document_id,
                    filename=filename,
                    page_number=page["page_number"],
                    detected_language=detected_lang,
                    upload_timestamp=upload_ts,
                    chunk_index=global_idx,
                    text=text,
                )
            )
            global_idx += 1

    logger.info(
        "document_chunked",
        document_id=document_id,
        total_chunks=len(chunks),
        language=detected_lang,
    )
    return chunks
