"""Send extracted paper text to Claude and parse a structured Summary."""

from __future__ import annotations

import os

import anthropic

from paper_digest.models import Summary

DEFAULT_MODEL = "claude-opus-4-7"
MAX_INPUT_CHARS = 200_000

SYSTEM_PROMPT = """You are a senior research assistant who digests academic papers for \
busy engineers and researchers. For each paper you are given, return a structured \
summary that is concise, honest about limitations, and useful for someone deciding \
whether to read the full paper.

Rules:
- Write for a technical reader, not a specialist in this subfield.
- Quote the paper's own numbers for results (e.g. "58.2% on MMLU"), never invent numbers.
- For `limitations`, prefer what the authors stated; if they skipped it, note one obvious \
limitation that an experienced reader would spot.
- Tags are lowercase-hyphenated topics (e.g. "retrieval-augmented-generation", "vision-transformers").
- If the paper text is truncated, summarize what's available and set `limitations` to \
include "summary is based on partial text".
"""


def summarize(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
) -> Summary:
    """Summarize extracted paper `text` into a structured `Summary`.

    Requires ANTHROPIC_API_KEY in the environment unless `client` is provided.
    """
    if not text.strip():
        raise ValueError("Empty input text.")

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Get a key at "
                "https://console.anthropic.com/settings/keys and export it, "
                "or copy .env.example to .env."
            )
        client = anthropic.Anthropic()

    truncated = False
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        truncated = True

    user_note = "Summarize the following paper.\n"
    if truncated:
        user_note += (
            f"NOTE: The paper was truncated to {MAX_INPUT_CHARS:,} characters. "
            "Say so in `limitations`.\n"
        )

    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"{user_note}\n---\n{text}",
            }
        ],
        output_format=Summary,
    )

    if response.parsed_output is None:
        raise RuntimeError(
            "Claude returned a response but structured parsing failed. "
            f"stop_reason={response.stop_reason}"
        )
    return response.parsed_output
