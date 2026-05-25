from __future__ import annotations
from pydantic import BeforeValidator
import re
from typing import Any

from pydantic import model_validator


#patterns that strongly indicate prompt injection attempts
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?!a\s+helpful)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[\s*system\s*\]", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)|a\s+different)", re.IGNORECASE),
    re.compile(r"override\s+(previous\s+)?instructions?", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"prompt\s+injection", re.IGNORECASE),
]


def _scan_value(value: Any, field_path: str = "") -> list[str]:
    violations: list[str] = []

    if isinstance(value, str):
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(value):
                violations.append(
                    f"Field '{field_path}': matched injection pattern '{pattern.pattern}'"
                )
                break  # one violation per field is enough

    elif isinstance(value, dict):
        for k, v in value.items():
            path = f"{field_path}.{k}" if field_path else k
            violations.extend(_scan_value(v, path))

    elif isinstance(value, (list, tuple, set)):
        for i, item in enumerate(value):
            path = f"{field_path}[{i}]"
            violations.extend(_scan_value(item, path))

    return violations


class MemoryGuard:

    @model_validator(mode="after")
    def _check_for_memory_poisoning(self) -> "MemoryGuard":
        data = self.model_dump()
        violations = _scan_value(data)

        if violations:
            violation_details = "\n".join(f"  - {v}" for v in violations)
            raise ValueError(
                f"MemoryGuard: Potential prompt injection detected in structured output.\n"
                f"Violations:\n{violation_details}\n"
                f"This may indicate a memory poisoning attempt (OWASP ASI06)."
            )

        return self


def sensitive_field_guard(
    *suspicious_patterns: str,
    case_sensitive: bool = False,
) -> Any:
    

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = [re.compile(re.escape(p), flags) for p in suspicious_patterns]

    def _guard(v: Any) -> Any:
        if isinstance(v, str):
            for pattern in compiled:
                if pattern.search(v):
                    raise ValueError(
                        f"sensitive_field_guard: rejected value matching '{pattern.pattern}'. "
                        f"Possible injection attempt."
                    )
        return v

    return BeforeValidator(_guard)