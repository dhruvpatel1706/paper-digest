"""Tests for source resolution. Network-dependent tests are skipped by default."""

from __future__ import annotations

import pytest

from paper_digest.fetch import FetchError, resolve_input


def test_resolve_bare_arxiv_id() -> None:
    kind, normalized = resolve_input("2305.12345")
    assert kind == "arxiv"
    assert normalized == "2305.12345"


def test_resolve_arxiv_id_with_version() -> None:
    kind, normalized = resolve_input("2305.12345v2")
    assert kind == "arxiv"
    assert normalized == "2305.12345v2"


def test_resolve_arxiv_abs_url() -> None:
    kind, normalized = resolve_input("https://arxiv.org/abs/2305.12345")
    assert kind == "arxiv"
    assert normalized == "2305.12345"


def test_resolve_arxiv_pdf_url() -> None:
    kind, normalized = resolve_input("https://arxiv.org/pdf/2305.12345v2.pdf")
    assert kind == "arxiv"
    assert normalized == "2305.12345v2"


def test_resolve_generic_pdf_url() -> None:
    kind, normalized = resolve_input("https://example.com/paper.pdf")
    assert kind == "url"
    assert normalized == "https://example.com/paper.pdf"


def test_resolve_local_path(tmp_path) -> None:
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    kind, normalized = resolve_input(str(p))
    assert kind == "path"
    assert normalized == str(p.resolve())


def test_resolve_garbage() -> None:
    with pytest.raises(FetchError):
        resolve_input("this is not a paper")


def test_resolve_openreview_forum_url() -> None:
    kind, normalized = resolve_input("https://openreview.net/forum?id=AbCdEf_123-XYZ")
    assert kind == "openreview"
    assert normalized == "AbCdEf_123-XYZ"


def test_resolve_openreview_pdf_url() -> None:
    kind, normalized = resolve_input("https://openreview.net/pdf?id=X9Y8Z7")
    assert kind == "openreview"
    assert normalized == "X9Y8Z7"


def test_resolve_acl_landing_url() -> None:
    kind, normalized = resolve_input("https://aclanthology.org/2023.acl-long.42/")
    assert kind == "acl"
    assert normalized == "2023.acl-long.42"


def test_resolve_acl_pdf_url() -> None:
    kind, normalized = resolve_input("https://aclanthology.org/2023.emnlp-main.7.pdf")
    assert kind == "acl"
    assert normalized == "2023.emnlp-main.7"
