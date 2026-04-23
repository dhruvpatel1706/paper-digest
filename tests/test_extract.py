"""Tests for PDF text extraction."""

from __future__ import annotations

import pytest

from paper_digest.extract import ExtractError, extract_text


def test_extract_rejects_non_pdf() -> None:
    with pytest.raises(ExtractError):
        extract_text(b"this is not a pdf")


def test_extract_rejects_empty_pdf() -> None:
    # Minimal-but-empty PDF (no extractable text)
    minimal_pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"xref\n0 3\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000054 00000 n \n"
        b"trailer<</Size 3/Root 1 0 R>>\nstartxref\n94\n%%EOF\n"
    )
    with pytest.raises(ExtractError):
        extract_text(minimal_pdf)
