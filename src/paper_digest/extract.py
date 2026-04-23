"""Extract text from a PDF byte stream."""

from __future__ import annotations

import io

from pypdf import PdfReader


class ExtractError(RuntimeError):
    """Raised when text extraction fails."""


def extract_text(pdf_bytes: bytes, *, max_pages: int | None = None) -> str:
    """Extract text from `pdf_bytes`, optionally capped at `max_pages` pages.

    Pages are joined with double newlines. Empty pages are skipped.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ExtractError(f"Could not parse PDF: {exc}") from exc

    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    chunks: list[str] = []
    for i, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            chunks.append(f"[page {i}]\n{text}")

    if not chunks:
        raise ExtractError("PDF contains no extractable text (may be scanned images).")

    return "\n\n".join(chunks)
