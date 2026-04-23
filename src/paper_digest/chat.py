"""Multi-turn Q&A against a single paper's text.

The paper text is sent as a cached system prompt on the first call, so every
follow-up question in the same chat session reuses the cache — cheap and fast.
"""

from __future__ import annotations

import os

import anthropic

from paper_digest.models import Summary

DEFAULT_CHAT_MODEL = "claude-opus-4-7"
MAX_PAPER_CHARS = 200_000

CHAT_SYSTEM_PROMPT = """You are a research assistant answering follow-up questions \
about an academic paper. The full paper text and its structured summary are in the \
system context. Follow these rules:

- Quote the paper's own numbers, never invent values.
- If the paper doesn't address a question, say so plainly instead of speculating.
- For questions that require math or reasoning, do it step-by-step.
- Keep answers under 4 sentences unless the user asks for depth.
- When citing, reference sections or figure numbers as the paper names them."""


def _build_system_blocks(paper_text: str, summary: Summary) -> list[dict]:
    """Two system blocks — an instruction block and a cached paper-text block."""
    truncated_note = ""
    if len(paper_text) > MAX_PAPER_CHARS:
        paper_text = paper_text[:MAX_PAPER_CHARS]
        truncated_note = (
            "\n\n(Note: paper text was truncated to fit the context window. "
            "Flag this in any answer that might depend on the cut portion.)"
        )

    summary_block = (
        f"## Pre-computed summary\n\n"
        f"**Title:** {summary.title}\n"
        f"**Authors:** {', '.join(summary.authors)}\n"
        f"**Problem:** {summary.problem}\n"
        f"**Method:** {summary.method}\n"
        f"**Key insight:** {summary.key_insight}\n"
        f"**Results:** {summary.results}\n"
        f"**Limitations:** {summary.limitations}\n"
        f"**Tags:** {', '.join(summary.tags)}"
    )

    paper_block = f"## Full paper text\n\n{paper_text}{truncated_note}"

    return [
        {"type": "text", "text": CHAT_SYSTEM_PROMPT},
        {
            "type": "text",
            "text": f"{summary_block}\n\n---\n\n{paper_block}",
            "cache_control": {"type": "ephemeral"},
        },
    ]


class PaperChat:
    """Stateful chat session grounded in a single paper."""

    def __init__(
        self,
        paper_text: str,
        summary: Summary,
        *,
        model: str = DEFAULT_CHAT_MODEL,
        client: anthropic.Anthropic | None = None,
    ):
        if not paper_text.strip():
            raise ValueError("Empty paper text.")
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example to .env.")
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.messages: list[dict] = []
        self._system_blocks = _build_system_blocks(paper_text, summary)

    def ask(self, question: str) -> str:
        """Send `question` as the next user turn; return the assistant's text reply."""
        if not question.strip():
            raise ValueError("Empty question.")
        self.messages.append({"role": "user", "content": question})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            thinking={"type": "adaptive"},
            system=self._system_blocks,
            messages=self.messages,
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        self.messages.append({"role": "assistant", "content": text})
        return text

    @property
    def turn_count(self) -> int:
        """Number of (user, assistant) exchanges so far."""
        return len(self.messages) // 2
