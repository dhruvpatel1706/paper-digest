"""Tests for the batch mode reading-list processor."""

from __future__ import annotations

from paper_digest.batch import BatchItem, parse_reading_list, render_digest, run_batch
from paper_digest.models import Summary


def _summary(title: str = "A paper") -> Summary:
    return Summary(
        title=title,
        authors=["Alice", "Bob"],
        problem="p",
        method="m",
        key_insight="ki",
        results="r",
        limitations="l",
        tags=["a", "b", "c"],
    )


def test_parse_reading_list_skips_blanks_and_comments(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text(
        "\n".join(
            [
                "# reading list for Thursday",
                "",
                "2305.13048",
                "  https://arxiv.org/abs/2402.00001  ",
                "",
                "# section header",
                "https://openreview.net/forum?id=ABC",
            ]
        ),
        encoding="utf-8",
    )
    refs = parse_reading_list(p)
    assert refs == [
        "2305.13048",
        "https://arxiv.org/abs/2402.00001",
        "https://openreview.net/forum?id=ABC",
    ]


def test_run_batch_continues_past_individual_failures(monkeypatch):
    calls = []

    def fake_fetch(src, **kwargs):
        calls.append(("fetch", src))
        if src == "bad":
            raise __import__("paper_digest.fetch", fromlist=["FetchError"]).FetchError("boom")
        return b"%PDF-1.4\nfake"

    def fake_extract(pdf, **kwargs):
        return "pretend paper text"

    def fake_summarize(text, **kwargs):
        return _summary(title="ok")

    import paper_digest.batch as b

    monkeypatch.setattr(b, "fetch_pdf_bytes", fake_fetch)
    monkeypatch.setattr(b, "extract_text", fake_extract)

    results = run_batch(
        ["good1", "bad", "good2"],
        model="claude-opus-4-7",
        summarize_fn=fake_summarize,
    )
    assert len(results) == 3
    assert results[0].summary is not None
    assert results[1].error is not None
    assert "boom" in results[1].error
    assert results[2].summary is not None


def test_render_digest_empty():
    out = render_digest([])
    assert "# Reading list digest" in out
    assert "No papers processed" in out


def test_render_digest_groups_successes_and_failures():
    items = [
        BatchItem(source="ok-1", summary=_summary(title="One")),
        BatchItem(source="fail-1", error="FetchError: 404"),
        BatchItem(source="ok-2", summary=_summary(title="Two")),
    ]
    out = render_digest(items)
    # Both successful titles appear
    assert "## One" in out
    assert "## Two" in out
    # Failure section exists with the bad source listed
    assert "## Failures" in out
    assert "fail-1" in out
    assert "FetchError: 404" in out
    # Succeeded count reflected in header
    assert "2 summarized" in out
    assert "1 failed" in out
