"""Tests for CitationMixin fuzzy span matching tolerance."""

from instructor.v2.dsl.citation import CitationMixin


class _Answer(CitationMixin):
    pass


def test_quote_within_error_tolerance_matches():
    # 3 substitutions from the 10-char source (<= errs default of 5): should match.
    context = "0123456789"
    result = _Answer.model_validate(
        {"substring_quotes": ["0123456ZZZ"]}, context={"context": context}
    )
    assert result.substring_quotes == ["0123456789"]


def test_quote_beyond_error_tolerance_is_dropped():
    # 6 substitutions from the 10-char source (> errs default of 5): should be
    # dropped as "not found", not accepted and rewritten to an unrelated span.
    context = "0123456789"
    result = _Answer.model_validate(
        {"substring_quotes": ["0123ZZZZZZ"]}, context={"context": context}
    )
    assert result.substring_quotes == []
