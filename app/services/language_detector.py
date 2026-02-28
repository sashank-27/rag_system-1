"""
Language detection service using langdetect.
"""

import structlog
from langdetect import detect, DetectorFactory

# Make detection deterministic
DetectorFactory.seed = 0

logger = structlog.get_logger(__name__)


class LanguageDetector:
    """Detects the language of a given text string."""

    # Friendly display names for common ISO-639-1 codes
    LANGUAGE_NAMES: dict[str, str] = {
        "en": "English",
        "hi": "Hindi",
        "gu": "Gujarati",
        "mr": "Marathi",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
        "kn": "Kannada",
        "ml": "Malayalam",
        "pa": "Punjabi",
        "ur": "Urdu",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "ru": "Russian",
        "pt": "Portuguese",
    }

    def detect(self, text: str) -> str:
        """
        Detect the ISO-639-1 language code of *text*.

        Returns ``"en"`` as fallback if detection fails.
        """
        if not text or not text.strip():
            return "en"
        try:
            lang = detect(text)
            logger.debug("language_detected", language=lang, snippet=text[:80])
            return lang
        except Exception as exc:
            logger.warning("language_detection_failed", error=str(exc))
            return "en"

    def get_language_name(self, code: str) -> str:
        return self.LANGUAGE_NAMES.get(code, code)


# Module-level singleton
language_detector = LanguageDetector()
