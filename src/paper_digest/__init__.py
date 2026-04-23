"""paper-digest — arXiv URL or PDF → structured summary via Claude."""

__version__ = "0.1.0"

from paper_digest.models import Summary
from paper_digest.summarize import summarize

__all__ = ["Summary", "summarize", "__version__"]
