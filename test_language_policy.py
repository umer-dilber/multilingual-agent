from language_policy import (
    classify_transcript,
    classify_turn,
    codes_from_speech_fields,
    instructions_with_language,
    language_instruction,
)


def test_codes_from_primary_and_source_languages() -> None:
    assert codes_from_speech_fields("es-MX", ["en", "es"]) == ["es", "en"]


def test_codes_ignore_multi_and_read_metadata() -> None:
    assert codes_from_speech_fields(
        "multi",
        metadata={"language_code": "es", "languages": ["es", "en-US"]},
    ) == ["es", "en"]


def test_classify_spanish_and_english() -> None:
    assert classify_transcript("Hola, ¿cómo estás?") == "es"
    assert classify_transcript("Can you help me with my account?") == "en"


def test_classify_mixed_spanglish() -> None:
    assert classify_transcript("No recuerdo mi bank password") == "mixed"
    assert classify_transcript("Can you help me with mi factura") == "mixed"


def test_classify_turn_prefers_mixed_from_stt_or_text() -> None:
    assert classify_turn("okay thanks", stt_codes=["en", "es"]) == "mixed"
    assert classify_turn("No recuerdo mi bank password", stt_codes=["es"]) == "mixed"
    assert classify_turn("Hola", stt_codes=["es"]) == "es"
    assert classify_turn("", stt_codes=["en"]) == "en"


def test_instructions_name_the_spoken_language() -> None:
    mixed = language_instruction("mixed")
    assert "mixed Spanish and English" in mixed
    assert "Do not flatten" in mixed

    combined = instructions_with_language("You are a voice assistant.", "es")
    assert combined.startswith("You are a voice assistant.")
    assert "Answer entirely in Spanish" in combined
