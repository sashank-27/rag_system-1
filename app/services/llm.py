"""
LLM service — loads an open-source instruct model via HuggingFace Transformers.

Constructs the RAG prompt with retrieved context and enforces same-language
answering plus anti-hallucination guardrails.
"""

from __future__ import annotations

import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.config import settings

logger = structlog.get_logger(__name__)

# ── System prompt (constant) ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an advanced multilingual document assistant.\n"
    "Use ONLY the provided context to answer.\n"
    "If the answer is not present, say:\n"
    "'The information is not available in the provided documents.'\n"
    "Always respond in the same language as the user's question.\n"
    "Do not hallucinate."
)


class LLMService:
    """Lazy-loaded HuggingFace text-generation pipeline."""

    def __init__(self) -> None:
        self._pipe = None
        self._tokenizer = None

    # ── model loading ───────────────────────────────────────────────────

    def _load(self) -> None:
        if self._pipe is not None:
            return

        model_id = settings.llm_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("loading_llm", model=model_id, device=device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            device_map="auto" if device == "cuda" else None,
        )
        logger.info("llm_loaded")

    # ── prompt construction ─────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("filename", "unknown")
            page = chunk.get("page_number", "?")
            parts.append(
                f"[Source {i}: {source}, Page {page}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self, question: str, context: str, language: str
    ) -> list[dict[str, str]]:
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question ({language}): {question}\n\n"
            f"Answer in {language}:"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ── generation ──────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        chunks: list[dict],
        language: str = "English",
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate an answer grounded in *chunks* for *question*.
        """
        self._load()

        context = self._build_context(chunks)
        messages = self._build_messages(question, context, language)

        output = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False,
        )

        answer = output[0]["generated_text"].strip()
        logger.info("llm_generated", question_len=len(question), answer_len=len(answer))
        return answer


# Module-level singleton
llm_service = LLMService()
