"""Fetch a paper's PDF from arXiv, an arbitrary URL, or a local path."""

from __future__ import annotations

import re
from pathlib import Path

import httpx

ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_ABS_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)")


class FetchError(RuntimeError):
    """Raised when a paper can't be retrieved."""


def _to_arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def resolve_input(source: str) -> tuple[str, str]:
    """Classify `source` as arxiv_id / url / local_path and return (kind, normalized).

    Returns:
        (kind, normalized) where kind is one of {"arxiv", "url", "path"}.
    """
    if ARXIV_ID_PATTERN.match(source):
        return "arxiv", source
    m = ARXIV_ABS_URL.search(source)
    if m:
        return "arxiv", m.group(1)
    if source.startswith(("http://", "https://")):
        return "url", source
    if Path(source).exists():
        return "path", str(Path(source).resolve())
    raise FetchError(
        f"Could not interpret input: {source!r}. "
        "Expected an arXiv ID (e.g. 2305.12345), an arxiv.org URL, a direct PDF URL, "
        "or a local file path."
    )


def fetch_pdf_bytes(source: str, *, timeout: float = 30.0) -> bytes:
    """Return raw PDF bytes for `source`. Accepts arXiv IDs, URLs, or local paths."""
    kind, normalized = resolve_input(source)

    if kind == "path":
        data = Path(normalized).read_bytes()
        if not data.startswith(b"%PDF"):
            raise FetchError(f"{normalized} does not look like a PDF.")
        return data

    url = _to_arxiv_pdf_url(normalized) if kind == "arxiv" else normalized

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "paper-digest/0.1.0"})
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc

    data = response.content
    if not data.startswith(b"%PDF"):
        raise FetchError(
            f"{url} returned {len(data)} bytes but it isn't a PDF "
            "(the server may have redirected to an HTML page)."
        )
    return data
