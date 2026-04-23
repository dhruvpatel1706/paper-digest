"""Tests for the summarize entry point (input validation only; live API calls not tested)."""

from __future__ import annotations

import pytest

from paper_digest.summarize import summarize


def test_summarize_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="Empty input text"):
        summarize("")


def test_summarize_rejects_whitespace_only() -> None:
    with pytest.raises(ValueError, match="Empty input text"):
        summarize("   \n\t  ")
