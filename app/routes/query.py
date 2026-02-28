"""
Query route with intent routing (RAG vs ServiceNow).
"""

from __future__ import annotations

import re

import structlog
from fastapi import APIRouter, HTTPException

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


async def _handle_rag(
    question: str, lang_code: str, lang_name: str
) -> AskResponse:
    """Handle RAG pipeline queries."""
    try:
        # ── Retrieve ────────────────────────────────────────────────────
        results = retrieval_service.search(question)

        if not results:
            return AskResponse(
                question=question,
                answer="The information is not available in the provided documents.",
                detected_language=lang_code,
            )

        # ── Rerank ──────────────────────────────────────────────────────
        reranked = reranker_service.rerank(question, results)

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
