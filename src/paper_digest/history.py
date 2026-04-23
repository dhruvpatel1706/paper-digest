"""Local history of past summaries so 'find the paper where X was proposed' works.

Every successful `digest` run writes a JSON file with the source reference,
the structured Summary, and a UTC timestamp. Search is plain keyword — weighted
across title, tags, key_insight, and problem. Good enough at the scale this is
designed for (personal reading lists up to the low thousands); swap in
embeddings later if recall becomes an issue.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paper_digest.models import Summary

DEFAULT_HISTORY_DIR = Path(
    os.environ.get("PAPER_DIGEST_HISTORY", Path.home() / ".paper-digest" / "history")
)

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class HistoryEntry:
    source: str
    summary: Summary
    summarized_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "summarized_at": self.summarized_at.isoformat(),
                "summary": self.summary.model_dump(),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> "HistoryEntry":
        data = json.loads(raw)
        return cls(
            source=data["source"],
            summary=Summary.model_validate(data["summary"]),
            summarized_at=datetime.fromisoformat(data["summarized_at"]),
        )


def _slug(source: str) -> str:
    """Stable, filesystem-safe short id derived from the source reference."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]


def save(
    source: str,
    summary: Summary,
    *,
    history_dir: Path | None = None,
    when: datetime | None = None,
) -> Path:
    """Persist one summary. Re-summarizing the same source overwrites the entry."""
    d = history_dir or DEFAULT_HISTORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    entry = HistoryEntry(
        source=source,
        summary=summary,
        summarized_at=when or datetime.now(timezone.utc),
    )
    path = d / f"{_slug(source)}.json"
    path.write_text(entry.to_json(), encoding="utf-8")
    return path


def load_all(history_dir: Path | None = None) -> list[HistoryEntry]:
    d = history_dir or DEFAULT_HISTORY_DIR
    if not d.exists():
        return []
    entries: list[HistoryEntry] = []
    for p in sorted(d.glob("*.json")):
        try:
            entries.append(HistoryEntry.from_json(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            # Tolerate stale or partial files rather than crash the CLI.
            continue
    entries.sort(key=lambda e: e.summarized_at, reverse=True)
    return entries


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _score(entry: HistoryEntry, query_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    s = entry.summary
    title_t = _tokens(s.title)
    tag_t = _tokens(" ".join(s.tags))
    insight_t = _tokens(s.key_insight)
    problem_t = _tokens(s.problem)
    score = 0.0
    for q in query_tokens:
        score += 4.0 * title_t.count(q)
        score += 3.0 * tag_t.count(q)
        score += 2.0 * insight_t.count(q)
        score += 1.0 * problem_t.count(q)
    return score


def search(query: str, *, history_dir: Path | None = None, limit: int = 5) -> list[HistoryEntry]:
    """Return up to `limit` past summaries most relevant to `query`."""
    if not query.strip():
        return []
    q = _tokens(query)
    scored: list[tuple[float, HistoryEntry]] = []
    for entry in load_all(history_dir):
        s = _score(entry, q)
        if s > 0:
            scored.append((s, entry))
    scored.sort(key=lambda x: (-x[0], -x[1].summarized_at.timestamp()))
    return [e for _, e in scored[:limit]]
