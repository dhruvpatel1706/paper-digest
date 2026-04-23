"""Pydantic models for structured summary output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Summary(BaseModel):
    """Structured summary of an academic paper."""

    title: str = Field(description="The paper's title, as written on the first page.")
    authors: list[str] = Field(
        description="First-author last name followed by 'et al.' if more than 3 authors; "
        "otherwise comma-separated last names."
    )
    problem: str = Field(
        description="What problem or question the paper addresses. 2-3 sentences. "
        "Avoid jargon a non-specialist couldn't parse."
    )
    method: str = Field(
        description="How the authors approach the problem. 2-4 sentences naming the "
        "key technique(s), dataset(s), and baseline(s)."
    )
    key_insight: str = Field(
        description="The single most important contribution or finding — what a reader "
        "takes away in one sentence."
    )
    results: str = Field(
        description="Quantitative and qualitative outcomes. Include the headline number(s) "
        "if reported. 2-4 sentences."
    )
    limitations: str = Field(
        description="Honest limitations or failure modes the authors acknowledge (or obvious "
        "ones they don't). 1-3 sentences. If none stated and none obvious, say so explicitly."
    )
    tags: list[str] = Field(
        description="3-5 lowercase topic tags that classify the paper (e.g. 'rag', "
        "'diffusion', 'reinforcement-learning', 'nlp-benchmarks').",
        min_length=3,
        max_length=5,
    )
