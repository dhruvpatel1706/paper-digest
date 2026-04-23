"""Tests for PaperChat. We stub the Anthropic client — no network calls."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from paper_digest.chat import CHAT_SYSTEM_PROMPT, PaperChat, _build_system_blocks
from paper_digest.models import Summary


@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Response:
    content: list


class _StubClient:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.messages = self  # type: ignore[assignment]

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        # Snapshot the messages list — the caller mutates the same list between
        # turns, so we need a frozen copy for meaningful test assertions.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.calls.append(snapshot)
        reply = self._replies.pop(0) if self._replies else "…"
        return _Response(content=[_Block(type="text", text=reply)])


def _summary() -> Summary:
    return Summary(
        title="A Paper",
        authors=["Doe et al."],
        problem="p",
        method="m",
        key_insight="ki",
        results="r",
        limitations="l",
        tags=["a", "b", "c"],
    )


def test_system_blocks_shape():
    blocks = _build_system_blocks("paper text", _summary())
    assert len(blocks) == 2
    assert blocks[0]["text"] == CHAT_SYSTEM_PROMPT
    # Second block is cached and contains both the summary and the paper text.
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert "Pre-computed summary" in blocks[1]["text"]
    assert "paper text" in blocks[1]["text"]


def test_system_blocks_truncation_note_when_paper_too_long():
    big_paper = "x" * 250_000
    blocks = _build_system_blocks(big_paper, _summary())
    assert "truncated" in blocks[1]["text"].lower()


def test_empty_paper_text_raises():
    with pytest.raises(ValueError, match="Empty paper text"):
        PaperChat(paper_text="", summary=_summary(), client=_StubClient([]))


def test_empty_question_raises():
    chat = PaperChat("some body", _summary(), client=_StubClient([]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Empty question"):
        chat.ask("")


def test_ask_roundtrip_records_turns_and_uses_cached_system():
    client = _StubClient(["First answer.", "Second answer."])
    chat = PaperChat("body text", _summary(), client=client, model="claude-opus-4-7")  # type: ignore[arg-type]

    r1 = chat.ask("what's the main result?")
    assert r1 == "First answer."
    assert chat.turn_count == 1

    r2 = chat.ask("how was it measured?")
    assert r2 == "Second answer."
    assert chat.turn_count == 2

    # Both calls share the SAME system-blocks object (cache prefix is stable).
    assert client.calls[0]["system"] is client.calls[1]["system"]
    # Second call sends a longer messages array (history grows).
    assert len(client.calls[1]["messages"]) > len(client.calls[0]["messages"])
    # Both requests carry adaptive thinking and the requested model.
    assert client.calls[0]["thinking"] == {"type": "adaptive"}
    assert client.calls[0]["model"] == "claude-opus-4-7"
