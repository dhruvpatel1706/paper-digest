"""Tests for the watch-a-folder auto-digest loop.

We don't spin up a real `watchdog.Observer` — that would make tests slow
and flaky. Instead we exercise the pieces directly: the debounce queue,
_process_one, and _seen_sources_in_history.
"""

from __future__ import annotations

import threading
import time

from paper_digest import history as history_store
from paper_digest.models import Summary
from paper_digest.watch import (
    _DebounceQueue,
    _process_one,
    _seen_sources_in_history,
)


def _summary(title: str = "Test Paper") -> Summary:
    return Summary(
        title=title,
        authors=["A", "B"],
        problem="p",
        method="m",
        key_insight="ki",
        results="r",
        limitations="l",
        tags=["a", "b", "c"],
    )


def _stub_summarize(text, *, model):
    return _summary(title=f"summary of len {len(text)}")


def _stub_extract(pdf_bytes, *, max_pages=None):
    # Pretend the PDF parsed; return some text. The real extract uses pypdf
    # which would choke on our fake bytes.
    return f"extracted {len(pdf_bytes)} bytes"


def test_debounce_queue_coalesces_rapid_pushes(tmp_path):
    # Six pushes on the same path inside ~50ms should result in exactly one
    # callback once the delay elapses.
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4")

    hits: list = []
    q = _DebounceQueue(delay=0.15, on_ready=lambda p: hits.append(p))
    q.start()
    try:
        for _ in range(6):
            q.push(path)
            time.sleep(0.01)
        # Wait safely past the debounce window
        time.sleep(0.5)
    finally:
        q.stop()

    assert hits == [path], f"expected exactly one hit, got {hits!r}"


def test_debounce_queue_ignores_non_pdf(tmp_path):
    not_pdf = tmp_path / "readme.md"
    not_pdf.write_text("hi")
    hits: list = []
    q = _DebounceQueue(delay=0.05, on_ready=lambda p: hits.append(p))
    q.start()
    try:
        q.push(not_pdf)
        time.sleep(0.2)
    finally:
        q.stop()
    assert hits == []


def test_debounce_queue_pdf_suffix_is_case_insensitive(tmp_path):
    path = tmp_path / "paper.PDF"
    path.write_bytes(b"x")
    hits: list = []
    q = _DebounceQueue(delay=0.05, on_ready=lambda p: hits.append(p))
    q.start()
    try:
        q.push(path)
        time.sleep(0.2)
    finally:
        q.stop()
    assert hits == [path]


def test_process_one_saves_to_history(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"dummy-pdf-bytes")
    hist_dir = tmp_path / "hist"

    monkeypatch.setattr("paper_digest.watch.extract_text", _stub_extract)

    logs: list[str] = []
    seen: set[str] = set()
    _process_one(
        pdf,
        model="claude-opus-4-7",
        max_pages=50,
        seen=seen,
        history_dir=hist_dir,
        summarize_fn=_stub_summarize,
        on_log=logs.append,
    )

    entries = history_store.load_all(hist_dir)
    assert len(entries) == 1
    assert entries[0].source == str(pdf.resolve())
    assert str(pdf.resolve()) in seen
    # A human-facing line was logged
    assert any("paper.pdf" in line for line in logs)


def test_process_one_skips_already_seen(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"x")
    monkeypatch.setattr("paper_digest.watch.extract_text", _stub_extract)

    seen = {str(pdf.resolve())}
    calls: list = []

    def fake_summarize(text, *, model):
        calls.append(text)
        return _summary()

    _process_one(
        pdf,
        model="m",
        max_pages=50,
        seen=seen,
        history_dir=tmp_path / "hist",
        summarize_fn=fake_summarize,
        on_log=lambda _m: None,
    )
    assert calls == []  # summarize was never called


def test_process_one_handles_extract_error(tmp_path, monkeypatch):
    from paper_digest.extract import ExtractError

    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not actually a pdf")

    def boom(*args, **kwargs):
        raise ExtractError("corrupt")

    monkeypatch.setattr("paper_digest.watch.extract_text", boom)

    logs: list[str] = []
    _process_one(
        pdf,
        model="m",
        max_pages=50,
        seen=set(),
        history_dir=tmp_path / "hist",
        summarize_fn=_stub_summarize,
        on_log=logs.append,
    )
    assert any("[error]" in line and "extract failed" in line for line in logs)
    # Nothing should have been persisted
    assert history_store.load_all(tmp_path / "hist") == []


def test_process_one_swallows_summarize_failure(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"x")
    monkeypatch.setattr("paper_digest.watch.extract_text", _stub_extract)

    def fails(text, *, model):
        raise RuntimeError("rate limited")

    logs: list[str] = []
    _process_one(
        pdf,
        model="m",
        max_pages=50,
        seen=set(),
        history_dir=tmp_path / "hist",
        summarize_fn=fails,
        on_log=logs.append,
    )
    assert any("summarize failed" in line for line in logs)


def test_process_one_skips_vanished_file(tmp_path, monkeypatch):
    # File existed long enough to make the queue, but was deleted before drain
    pdf = tmp_path / "gone.pdf"
    monkeypatch.setattr("paper_digest.watch.extract_text", _stub_extract)
    logs: list[str] = []
    _process_one(
        pdf,
        model="m",
        max_pages=50,
        seen=set(),
        history_dir=tmp_path / "hist",
        summarize_fn=_stub_summarize,
        on_log=logs.append,
    )
    assert logs == []  # silent skip


def test_seen_sources_dedupes_across_restarts(tmp_path):
    # Seed a history file for an absolute-path source; URL-style sources shouldn't
    # be picked up (watcher never sees them, so including them would cause false
    # skips if you later run batch on the same URL).
    hist = tmp_path / "hist"
    summ = _summary()
    history_store.save("/tmp/local.pdf", summ, history_dir=hist)
    history_store.save("https://arxiv.org/abs/1234", summ, history_dir=hist)

    seen = _seen_sources_in_history(hist)
    assert "/tmp/local.pdf" in seen
    assert "https://arxiv.org/abs/1234" not in seen


def test_watch_rejects_nonexistent_dir(tmp_path):
    import pytest

    from paper_digest.watch import watch

    # Using a file, not a dir
    f = tmp_path / "not-a-dir.pdf"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="existing directory"):
        watch(f, model="m", summarize_fn=_stub_summarize)


def test_watch_picks_up_dropped_pdf(tmp_path, monkeypatch):
    """Full integration: spin up the watcher on a real folder and drop a file.

    This is slower than the unit tests above — we give the observer ~3s
    to notice the new file, debounce, and run the callback.
    """
    monkeypatch.setattr("paper_digest.watch.extract_text", _stub_extract)

    folder = tmp_path / "watched"
    folder.mkdir()
    hist_dir = tmp_path / "hist"

    from paper_digest.watch import watch

    stop = threading.Event()
    logs: list[str] = []

    def _run():
        watch(
            folder,
            model="m",
            max_pages=5,
            debounce_s=0.2,
            history_dir=hist_dir,
            summarize_fn=_stub_summarize,
            on_log=logs.append,
            stop_event=stop,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        # Give the observer a moment to attach
        time.sleep(0.3)
        (folder / "dropped.pdf").write_bytes(b"%PDF-1.4 fake")
        # Wait for debounce + processing
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if history_store.load_all(hist_dir):
                break
            time.sleep(0.1)
    finally:
        stop.set()
        t.join(timeout=3.0)

    entries = history_store.load_all(hist_dir)
    assert len(entries) == 1
    assert entries[0].source.endswith("dropped.pdf")
