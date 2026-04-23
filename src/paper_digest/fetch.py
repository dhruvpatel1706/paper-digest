"""Fetch a paper's PDF from arXiv, OpenReview, ACL Anthology, any PDF URL, or a local path."""

from __future__ import annotations

import re
from pathlib import Path

import httpx

ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_ABS_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)")

# OpenReview uses an 'id' query param both on /forum and /pdf pages.
# Forum URL: https://openreview.net/forum?id=XYZ
# Direct PDF: https://openreview.net/pdf?id=XYZ
OPENREVIEW_URL = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_-]+)")

# ACL Anthology paper landing pages live at https://aclanthology.org/2023.acl-long.1/
# The matching PDF is the same path + .pdf
ACL_LANDING_URL = re.compile(r"aclanthology\.org/(\d{4}\.[^/]+)/?$")
ACL_PDF_URL = re.compile(r"aclanthology\.org/(\d{4}\.[^/]+)\.pdf$")


class FetchError(RuntimeError):
    """Raised when a paper can't be retrieved."""


def _to_arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _to_openreview_pdf_url(paper_id: str) -> str:
    return f"https://openreview.net/pdf?id={paper_id}"


def _to_acl_pdf_url(paper_id: str) -> str:
    # Anthology serves PDFs at the landing-page path + .pdf
    return f"https://aclanthology.org/{paper_id}.pdf"


def resolve_input(source: str) -> tuple[str, str]:
    """Classify `source` and return (kind, normalized).

    `kind` is one of: ``arxiv``, ``openreview``, ``acl``, ``url``, ``path``.
    The caller uses the kind to decide how to translate `normalized` into a
    fetchable URL; ``path`` and ``url`` are used as-is.
    """
    # Bare arXiv IDs
    if ARXIV_ID_PATTERN.match(source):
        return "arxiv", source

    # arXiv URLs
    m = ARXIV_ABS_URL.search(source)
    if m:
        return "arxiv", m.group(1)

    # OpenReview (both forum + direct pdf URLs)
    m = OPENREVIEW_URL.search(source)
    if m:
        return "openreview", m.group(1)

    # ACL Anthology — try PDF form first, then landing page
    m = ACL_PDF_URL.search(source) or ACL_LANDING_URL.search(source)
    if m:
        return "acl", m.group(1)

    # Generic URL — assume the caller is pointing at a PDF directly
    if source.startswith(("http://", "https://")):
        return "url", source

    # Local file
    if Path(source).exists():
        return "path", str(Path(source).resolve())

    raise FetchError(
        f"Could not interpret input: {source!r}. Expected: an arXiv ID "
        "(e.g. 2305.12345), an arxiv.org / openreview.net / aclanthology.org URL, "
        "a direct PDF URL, or a local file path."
    )


def fetch_pdf_bytes(source: str, *, timeout: float = 30.0) -> bytes:
    """Return raw PDF bytes for `source`. Handles arXiv, OpenReview, ACL, URLs, or paths."""
    kind, normalized = resolve_input(source)

    if kind == "path":
        data = Path(normalized).read_bytes()
        if not data.startswith(b"%PDF"):
            raise FetchError(f"{normalized} does not look like a PDF.")
        return data

    url = {
        "arxiv": _to_arxiv_pdf_url,
        "openreview": _to_openreview_pdf_url,
        "acl": _to_acl_pdf_url,
        "url": lambda u: u,
    }[kind](normalized)

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "paper-digest/0.3.0"})
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
