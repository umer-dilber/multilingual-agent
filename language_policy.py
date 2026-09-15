from __future__ import annotations

import re
from typing import Iterable, Literal

TurnLanguage = Literal["en", "es", "mixed"]

_SPANISH_CHARS = re.compile(r"[áéíóúüñ¿¡ÁÉÍÓÚÜÑ]")
_WORD_RE = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)

_SPANISH_WORDS = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "una",
        "uno",
        "unos",
        "unas",
        "del",
        "al",
        "que",
        "qué",
        "por",
        "para",
        "con",
        "está",
        "están",
        "estoy",
        "estás",
        "hola",
        "gracias",
        "quiero",
        "quieres",
        "puede",
        "puedes",
        "puedo",
        "favor",
        "cómo",
        "cuando",
        "cuándo",
        "donde",
        "dónde",
        "también",
        "pero",
        "porque",
        "entonces",
        "ahora",
        "después",
        "necesito",
        "tengo",
        "tiene",
        "hacer",
        "ayuda",
        "cuenta",
        "factura",
        "pago",
        "llamar",
        "habla",
        "hablar",
        "español",
        "buenos",
        "buenas",
        "días",
        "tardes",
        "noches",
        "señor",
        "señora",
        "porfa",
        "vale",
        "este",
        "esta",
        "esto",
        "eso",
        "aquí",
        "allí",
        "sí",
        "así",
        "muy",
        "más",
        "menos",
        "todo",
        "nada",
        "bien",
        "mal",
        "recuerdo",
        "olvidé",
        "olvidar",
        "clave",
        "contraseña",
    }
)

_ENGLISH_WORDS = frozenset(
    {
        "the",
        "is",
        "are",
        "this",
        "that",
        "with",
        "from",
        "have",
        "has",
        "please",
        "thanks",
        "thank",
        "hello",
        "what",
        "how",
        "can",
        "could",
        "would",
        "should",
        "password",
        "bank",
        "account",
        "invoice",
        "help",
        "need",
        "want",
        "call",
        "speak",
        "english",
        "your",
        "and",
        "but",
        "because",
        "then",
        "now",
        "after",
        "before",
        "yes",
        "okay",
        "hey",
        "sorry",
        "just",
        "really",
        "about",
        "know",
        "forgot",
        "remember",
        "my",
    }
)

_IGNORED_LANG_CODES = frozenset({"", "multi", "und", "unknown", "auto"})

_TURN_INSTRUCTIONS = {
    "en": (
        "The user spoke English this turn. Answer entirely in English. "
        "Do not switch into Spanish unless the user used a Spanish proper noun."
    ),
    "es": (
        "The user spoke Spanish this turn. Answer entirely in Spanish. "
        "Do not switch into English unless the user used an English proper noun, "
        "product name, or identifier."
    ),
    "mixed": (
        "The user mixed Spanish and English this turn. Answer in mixed Spanish "
        "and English as well: keep the same blend and register. Do not flatten "
        "the reply into only one language."
    ),
}


def normalize_lang_code(code: object) -> str | None:
    if code is None:
        return None
    text = str(code).strip()
    if not text:
        return None
    base = text.replace("_", "-").split("-", 1)[0].lower()
    if base in _IGNORED_LANG_CODES:
        return None
    return base


def codes_from_speech_fields(
    language: object = None,
    source_languages: Iterable[object] | None = None,
    metadata: dict | None = None,
) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        normalized = normalize_lang_code(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            codes.append(normalized)

    add(language)
    if source_languages:
        for item in source_languages:
            add(item)

    extra = metadata or {}
    add(extra.get("language_code") or extra.get("language"))
    for key in ("languages", "language_codes", "source_languages"):
        values = extra.get(key)
        if isinstance(values, str):
            add(values)
        elif isinstance(values, Iterable):
            for item in values:
                add(item)

    return codes


def classify_transcript(transcript: str) -> TurnLanguage | None:
    text = (transcript or "").strip()
    if not text:
        return None

    words = {match.group(0).lower() for match in _WORD_RE.finditer(text)}
    spanish_hits = len(words & _SPANISH_WORDS)
    english_hits = len(words & _ENGLISH_WORDS)
    has_spanish_chars = bool(_SPANISH_CHARS.search(text))

    spanish = spanish_hits >= 1 or has_spanish_chars
    english = english_hits >= 1

    if spanish and english:
        return "mixed"
    if spanish:
        return "es"
    if english:
        return "en"
    return None


def classify_turn(
    transcript: str = "",
    stt_codes: Iterable[str] | None = None,
) -> TurnLanguage:
    spoken = {code for code in (stt_codes or []) if code in {"en", "es"}}
    heuristic = classify_transcript(transcript)

    if spoken == {"en", "es"} or heuristic == "mixed":
        return "mixed"
    if heuristic in {"en", "es"}:
        return heuristic
    if spoken == {"es"}:
        return "es"
    if spoken == {"en"}:
        return "en"
    return "en"


def language_instruction(mode: TurnLanguage) -> str:
    return (
        "# Language for this turn\n"
        f"{_TURN_INSTRUCTIONS[mode]}\n"
        "Follow these language rules for this reply even if earlier turns used a different language."
    )


def instructions_with_language(base_instructions: str, mode: TurnLanguage) -> str:
    return f"{base_instructions.rstrip()}\n\n{language_instruction(mode)}"
