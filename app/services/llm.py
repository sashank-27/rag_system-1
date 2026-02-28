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
    "You are a precise multilingual document assistant.\n"
    "Your task is to answer the user's question using ONLY the provided context.\n"
    "\n"
    "STRICT RULES — FOLLOW EXACTLY:\n"
    "1. Read ALL the context carefully before answering.\n"
    "2. The context may be in a DIFFERENT language than the question. Read and understand it anyway.\n"
    "3. Answer ONLY from what is explicitly written in the context. Do NOT add any external knowledge.\n"
    "4. When possible, QUOTE or closely paraphrase the exact words from the context.\n"
    "5. Do NOT invent details, terms, conditions, or clauses that are not in the context.\n"
    "6. If the question asks about a specific section/article, find it in the context and state exactly what it says.\n"
    "7. If only partial information is available, provide what you can find and say the rest is not in the documents.\n"
    "8. ONLY say 'The information is not available in the provided documents' if you truly cannot find ANY relevant information.\n"
    "9. Always respond in the SAME language as the user's question.\n"
    "10. NEVER hallucinate or fabricate information. Accuracy is more important than completeness.\n"
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
        if context:
            user_content = (
                f"Context:\n{context}\n\n"
                f"Question ({language}): {question}\n\n"
                f"Answer in {language}:"
            )
        else:
            # No context — used for translation or direct prompts
            user_content = question

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
            temperature=0.1,
            top_p=0.85,
            repetition_penalty=1.1,
            return_full_text=False,
        )

        answer = output[0]["generated_text"].strip()
        logger.info("llm_generated", question_len=len(question), answer_len=len(answer))
        return answer


# Module-level singleton
llm_service = LLMService()
