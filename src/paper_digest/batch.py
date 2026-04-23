"""Process a reading list of paper references and emit a combined markdown digest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from paper_digest.extract import ExtractError, extract_text
from paper_digest.fetch import FetchError, fetch_pdf_bytes
from paper_digest.models import Summary
from paper_digest.summarize import summarize


@dataclass
class BatchItem:
    source: str
    summary: Summary | None = None
    error: str | None = None


def parse_reading_list(path: Path) -> list[str]:
    """Read one reference per line. Skips blanks and `#`-prefixed comments."""
    refs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        refs.append(line)
    return refs


def _run_one(
    ref: str,
    model: str,
    max_pages: int,
    summarize_fn: Callable[..., Summary],
) -> BatchItem:
    try:
        pdf = fetch_pdf_bytes(ref)
        text = extract_text(pdf, max_pages=max_pages)
        summary = summarize_fn(text, model=model)
        return BatchItem(source=ref, summary=summary)
    except (FetchError, ExtractError) as exc:
        return BatchItem(source=ref, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 — surface SDK / network errors cleanly
        return BatchItem(source=ref, error=f"{type(exc).__name__}: {exc}")


def run_batch(
    refs: list[str],
    *,
    model: str,
    max_pages: int = 50,
    progress: Callable[[int, int, str], None] | None = None,
    summarize_fn: Callable[..., Summary] = summarize,
) -> list[BatchItem]:
    """Fetch + summarize each reference. Doesn't stop on individual failures."""
    results: list[BatchItem] = []
    total = len(refs)
    for i, ref in enumerate(refs, 1):
        if progress is not None:
            progress(i, total, ref)
        results.append(_run_one(ref, model, max_pages, summarize_fn))
    return results


def render_digest(items: list[BatchItem]) -> str:
    """Render a multi-paper digest as markdown."""
    if not items:
        return "# Reading list digest\n\n_No papers processed._\n"

    succeeded = [it for it in items if it.summary is not None]
    failed = [it for it in items if it.error is not None]

    parts: list[str] = [
        "# Reading list digest",
        "",
        f"_{len(succeeded)} summarized_" + (f", _{len(failed)} failed_" if failed else ""),
        "",
    ]

    for it in succeeded:
        s = it.summary
        assert s is not None
        parts.append(f"## {s.title}")
        parts.append(f"*{', '.join(s.authors)}* — `{it.source}`")
        parts.append("")
        parts.append(f"**Problem.** {s.problem}")
        parts.append("")
        parts.append(f"**Method.** {s.method}")
        parts.append("")
        parts.append(f"**Key insight.** {s.key_insight}")
        parts.append("")
        parts.append(f"**Results.** {s.results}")
        parts.append("")
        parts.append(f"**Limitations.** {s.limitations}")
        parts.append("")
        parts.append(f"**Tags.** {' '.join(f'`{t}`' for t in s.tags)}")
        parts.append("")
        parts.append("---")
        parts.append("")

    if failed:
        parts.append("## Failures")
        parts.append("")
        for it in failed:
            parts.append(f"- `{it.source}` — {it.error}")
        parts.append("")

    return "\n".join(parts)
