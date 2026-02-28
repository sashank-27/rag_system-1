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

def _is_garbled(text: str) -> bool:
    """
    Heuristic: text is likely garbled if it has a high ratio of
    non-alphanumeric, non-whitespace, non-common-punctuation characters.
    This catches custom-font Hindi PDFs where PyMuPDF extracts gibberish.
    
    Correctly handles Hindi/Devanagari text with zero-width joiners (ZWJ)
    and combining marks which are normal in Indic scripts.
    """
    if not text or len(text) < 20:
        return True

    import unicodedata
    good = 0
    for ch in text:
        cat = unicodedata.category(ch)
        # L = letters (any script), N = numbers, Z = separators,
        # P = punctuation, M = combining marks (matras/vowel signs),
        # Cf = format chars (zero-width joiners, common in Hindi)
        if cat[0] in ("L", "N", "Z", "P", "M") or cat == "Cf":
            good += 1

    ratio = good / len(text)
    # If less than 50% of characters are "normal", it's garbled
    return ratio < 0.50


def _ocr_page(page: fitz.Page) -> str:
    """Render page to image and run Tesseract OCR."""
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img, lang="eng+hin+guj")
        logger.debug("ocr_applied", page=page.number)
        return ocr_text.strip()
    except Exception as exc:
        logger.warning("ocr_failed", page=page.number, error=str(exc))
        return ""


def _extract_text_from_page(page: fitz.Page) -> str:
    """
    Extract text from a page.
    
    Flow: extract → normalize → check garbled → OCR fallback if needed.
    Normalization happens BEFORE the garbled check so that valid Hindi 
    text with zero-width joiners isn't incorrectly flagged.
    """
    raw_text = page.get_text("text").strip()

    # Normalize first — strip ZWJ, invisible chars, normalize Unicode
    text = _normalize_text(raw_text) if raw_text else ""

    # If normalized text is good, return it
    if text and not _is_garbled(text):
        return text

    if text and _is_garbled(text):
        logger.debug("garbled_text_detected", page=page.number, snippet=text[:60])

    # OCR fallback
    ocr_text = _ocr_page(page)
    if ocr_text:
        return _normalize_text(ocr_text)

    # If OCR also failed, return whatever we have (even if garbled)
    return text


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


# ── Text normalization ──────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """
    Normalize Unicode text for better embedding quality.
    
    - Strips zero-width joiners/non-joiners (common in Hindi PDFs)
    - Normalizes to NFC form (canonical composition)
    - Collapses excess whitespace
    - Removes other invisible Unicode characters
    - Filters out footnotes and amendment notes
    """
    import re
    import unicodedata

    # Normalize to NFC (canonical composition)
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters that break embeddings
    # ZWJ (\u200d), ZWNJ (\u200c), Zero-width space (\u200b),
    # Word joiner (\u2060), Zero-width no-break space (\ufeff)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)

    # Remove other invisible format characters (category Cf) except common ones
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue  # Skip format characters
        cleaned.append(ch)
    text = "".join(cleaned)

    # Collapse multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Filter footnotes and amendment notes
    text = _filter_footnotes(text)

    return text.strip()


def _filter_footnotes(text: str) -> str:
    """
    Remove footnotes and amendment/editorial notes common in Indian legal PDFs.
    
    These include lines like:
    - "1[...द्वारा प्रतिस्थापित।" (substituted by Act...)
    - "2[...द्वारा अंतःस्थापित।" (inserted by Act...)
    - "Subs. by Act 3 of 1951, s. 3"
    - Numbered footnote markers at start of lines
    """
    import re

    lines = text.split("\n")
    filtered = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            filtered.append(line)
            continue

        # Skip footnote-style lines (start with digit + bracket/period then short ref)
        # e.g., "1. Subs. by Act 3 of 1951, s. 3, for..."
        if re.match(r"^\d+\.\s*(Subs|Ins|Added|Omitted|Rep|See)\b", stripped, re.IGNORECASE):
            continue

        # Skip Hindi amendment references
        # "द्वारा प्रतिस्थापित" = substituted by
        # "द्वारा अंतःस्थापित" = inserted by
        # "द्वारा लोप किया गया" = omitted by
        if re.search(r"द्वारा\s*(मूल\s*)?(धारा|शब्दों?|अंक).*?(प्रतिस्थापित|अंतःस्थापित|लोप)", stripped):
            continue

        # Skip English amendment short references
        if re.match(r"^\d+\[", stripped) and len(stripped) < 80:
            # Short parenthetical references like "1[substituted...]"
            if re.search(r"(substitut|insert|omit|repeal|amend)", stripped, re.IGNORECASE):
                continue

        # Skip lines that are purely citation markers like "1.", "2.", etc.
        if re.match(r"^\d+[.\]]\s*$", stripped):
            continue

        filtered.append(line)

    return "\n".join(filtered)


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

    # Detect dominant language — sample from multiple parts of the document
    # (first pages often have English headers, so include middle pages too)
    sample_pages = []
    if len(pages) <= 5:
        sample_pages = pages
    else:
        mid = len(pages) // 2
        sample_pages = [pages[0], pages[mid - 1], pages[mid], pages[mid + 1], pages[-1]]
    sample_text = " ".join(p["text"][:500] for p in sample_pages)
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
