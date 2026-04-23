"""Tests for the summary history store."""

from __future__ import annotations

from datetime import datetime, timezone

from paper_digest.history import HistoryEntry, load_all, save, search
from paper_digest.models import Summary


def _summary(title: str = "A paper", tags: list[str] | None = None, insight: str = "ki") -> Summary:
    return Summary(
        title=title,
        authors=["Doe et al."],
        problem="p",
        method="m",
        key_insight=insight,
        results="r",
        limitations="l",
        tags=tags or ["a", "b", "c"],
    )


def test_save_and_reload_roundtrip(tmp_path):
    path = save("2305.13048", _summary(title="Hello"), history_dir=tmp_path)
    assert path.exists()
    back = load_all(tmp_path)
    assert len(back) == 1
    assert back[0].source == "2305.13048"
    assert back[0].summary.title == "Hello"


def test_resumarizing_same_source_overwrites(tmp_path):
    save("2305.13048", _summary(title="Old"), history_dir=tmp_path)
    save("2305.13048", _summary(title="New"), history_dir=tmp_path)
    entries = load_all(tmp_path)
    assert len(entries) == 1
    assert entries[0].summary.title == "New"


def test_load_all_sorts_newest_first(tmp_path):
    save(
        "one",
        _summary(title="First"),
        history_dir=tmp_path,
        when=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    save(
        "two",
        _summary(title="Second"),
        history_dir=tmp_path,
        when=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    entries = load_all(tmp_path)
    assert [e.summary.title for e in entries] == ["Second", "First"]


def test_load_all_skips_corrupt_files(tmp_path):
    save("good", _summary(title="OK"), history_dir=tmp_path)
    (tmp_path / "trash.json").write_text("not valid json", encoding="utf-8")
    entries = load_all(tmp_path)
    assert [e.summary.title for e in entries] == ["OK"]


def test_search_matches_titles_tags_and_insights(tmp_path):
    save(
        "p1",
        _summary(title="Retrieval augmented generation", tags=["rag", "nlp", "dense"]),
        history_dir=tmp_path,
    )
    save(
        "p2",
        _summary(
            title="Transformer scaling laws",
            tags=["scaling", "training", "llm"],
            insight="RAG was not tested",
        ),
        history_dir=tmp_path,
    )
    save(
        "p3",
        _summary(title="Kubernetes operator", tags=["k8s", "ops", "infra"]),
        history_dir=tmp_path,
    )

    hits = search("rag", history_dir=tmp_path)
    # Both p1 and p2 mention "rag" (p1 in title+tags, p2 in insight).
    # p1 should rank higher because title + tag hits weigh more than insight.
    ids = [h.source for h in hits]
    assert ids[0] == "p1"
    assert "p2" in ids
    assert "p3" not in ids


def test_search_empty_query_returns_empty(tmp_path):
    save("p1", _summary(title="Whatever"), history_dir=tmp_path)
    assert search("", history_dir=tmp_path) == []
    assert search("  ", history_dir=tmp_path) == []


def test_history_entry_json_round_trip():
    e = HistoryEntry(
        source="src",
        summary=_summary(title="X"),
        summarized_at=datetime(2026, 2, 14, 9, 30, tzinfo=timezone.utc),
    )
    back = HistoryEntry.from_json(e.to_json())
    assert back.source == e.source
    assert back.summary.title == "X"
    assert back.summarized_at == e.summarized_at
