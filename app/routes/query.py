"""
Query route with intent routing (RAG vs ServiceNow).
"""

from __future__ import annotations

import re

import structlog
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import AskRequest, AskResponse
from app.services.language_detector import language_detector
from app.services.llm import llm_service
from app.services.reranker import reranker_service
from app.services.retrieval import retrieval_service
from app.services.servicenow import servicenow_client

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Query"])

# Keywords that trigger ServiceNow routing
_SERVICENOW_KEYWORDS = re.compile(
    r"\b(host|server|cmdb|servicenow)\b", re.IGNORECASE
)


def _should_route_to_servicenow(question: str) -> bool:
    """Return True if the question should be handled by ServiceNow."""
    return bool(_SERVICENOW_KEYWORDS.search(question))


def _extract_hostname(question: str) -> str | None:
    """
    Try to extract a hostname/server name from the question.

    Looks for patterns like:
        "host server01", "server myhost", "details of vm-prod-01"
    """
    # Try "host <name>" or "server <name>" patterns
    match = re.search(
        r"(?:host|server|cmdb)\s+(\S+)", question, re.IGNORECASE
    )
    if match:
        return match.group(1).strip("?.,!\"'")
    return None


@router.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    """
    Accept a question, detect language, route to RAG or ServiceNow,
    and return an answer.
    """
    question = body.question.strip()
    detected_lang = language_detector.detect(question)
    lang_name = language_detector.get_language_name(detected_lang)

    logger.info(
        "question_received",
        question=question[:120],
        language=detected_lang,
    )

    # ── Intent routing ──────────────────────────────────────────────────
    if _should_route_to_servicenow(question):
        return await _handle_servicenow(question, detected_lang, lang_name)

    return await _handle_rag(question, detected_lang, lang_name)


async def _handle_servicenow(
    question: str, lang_code: str, lang_name: str
) -> AskResponse:
    """Handle ServiceNow-routed queries."""
    hostname = _extract_hostname(question)
    if not hostname:
        return AskResponse(
            question=question,
            answer="Please specify a hostname or server name. "
            "Example: 'Tell me about host server01'.",
            detected_language=lang_code,
            routed_to="servicenow",
        )

    result = await servicenow_client.lookup_host(hostname)

    if isinstance(result, str):
        # Error or "not found" message
        answer = result
    else:
        answer = (
            f"**Host:** {result.name}\n"
            f"**IP Address:** {result.ip_address}\n"
            f"**OS:** {result.os}\n"
            f"**Location:** {result.location}\n"
            f"**Install Status:** {result.install_status}"
        )

    return AskResponse(
        question=question,
        answer=answer,
        detected_language=lang_code,
        routed_to="servicenow",
    )


def _get_document_languages() -> set[str]:
    """Get the set of languages present in indexed documents."""
    try:
        docs = retrieval_service.list_documents()
        return {d.get("detected_language", "en") for d in docs if d.get("detected_language")}
    except Exception:
        return set()


def _translate_query_for_search(question: str, target_lang: str) -> str | None:
    """
    Use the LLM to translate a query to the target language for cross-lingual search.
    Returns None if translation fails.
    """
    lang_map = {
        "hi": "Hindi", "gu": "Gujarati", "ta": "Tamil", "te": "Telugu",
        "bn": "Bengali", "mr": "Marathi", "kn": "Kannada", "ml": "Malayalam",
        "pa": "Punjabi", "ur": "Urdu", "en": "English", "fr": "French",
        "de": "German", "es": "Spanish", "zh": "Chinese", "ja": "Japanese",
        "ko": "Korean", "ar": "Arabic", "ru": "Russian", "pt": "Portuguese",
    }
    target_name = lang_map.get(target_lang, target_lang)

    try:
        # Use a lightweight prompt — just translate, no explanation
        result = llm_service.generate(
            question=f"Translate this to {target_name}. Output ONLY the translation, nothing else: {question}",
            chunks=[],  # No context needed
            language=target_name,
            max_new_tokens=128,
        )
        translated = result.strip().strip('"').strip("'")
        if translated and translated.lower() != question.lower():
            logger.info("query_translated", original=question, translated=translated[:100], target=target_lang)
            return translated
    except Exception as exc:
        logger.warning("query_translation_failed", error=str(exc))

    return None


async def _handle_rag(
    question: str, lang_code: str, lang_name: str
) -> AskResponse:
    """Handle RAG pipeline queries with cross-lingual support."""
    try:
        # ── Cross-lingual query expansion ───────────────────────────────
        # Check if documents exist in languages different from the query
        doc_languages = _get_document_languages()
        queries_to_search = [question]

        if doc_languages and lang_code not in doc_languages:
            # Query language differs from document languages — translate
            for doc_lang in doc_languages:
                if doc_lang != lang_code:
                    translated = _translate_query_for_search(question, doc_lang)
                    if translated:
                        queries_to_search.append(translated)
                    break  # Translate to the first different language only

        logger.info(
            "cross_lingual_search",
            query_lang=lang_code,
            doc_langs=list(doc_languages),
            num_queries=len(queries_to_search),
        )

        # ── Retrieve with all queries ───────────────────────────────────
        all_results = []
        seen_texts = set()
        for q in queries_to_search:
            results = retrieval_service.search(q, top_k=settings.top_k * 2)
            for r in results:
                # Deduplicate by text content
                text_key = r.get("text", "")[:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_results.append(r)

        if not all_results:
            return AskResponse(
                question=question,
                answer="The information is not available in the provided documents.",
                detected_language=lang_code,
            )

        # ── Rerank ──────────────────────────────────────────────────────
        reranked = reranker_service.rerank(question, all_results, top_k=settings.top_k)

        # ── Generate answer ─────────────────────────────────────────────
        answer = llm_service.generate(
            question=question,
            chunks=reranked,
            language=lang_name,
        )

        # Build source references
        sources = [
            {
                "filename": doc.get("filename", ""),
                "page_number": doc.get("page_number", 0),
                "score": round(doc.get("rerank_score", doc.get("score", 0)), 4),
                "snippet": doc.get("text", "")[:200],
            }
            for doc in reranked
        ]

        return AskResponse(
            question=question,
            answer=answer,
            detected_language=lang_code,
            source_documents=sources,
        )

    except Exception as exc:
        logger.exception("rag_query_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
