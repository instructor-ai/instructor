"""Models, modes, and helpers for Sarvam provider integration tests."""

from __future__ import annotations

import unicodedata

import instructor

models: list[str] = ["sarvam-30b"]
modes: list[instructor.Mode] = [instructor.Mode.TOOLS]


def normalize_name(value: str) -> str:
    """Normalize names for cross-script comparison."""
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return "".join(ch for ch in normalized if not ch.isspace())


def name_matches(actual: str, acceptable_names: tuple[str, ...]) -> bool:
    """Return True when actual matches any acceptable romanized or native name."""
    actual_norm = normalize_name(actual)
    return any(normalize_name(candidate) == actual_norm for candidate in acceptable_names)
