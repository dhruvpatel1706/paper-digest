"""Typer CLI for paper-digest."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from paper_digest import __version__
from paper_digest.extract import ExtractError, extract_text
from paper_digest.fetch import FetchError, fetch_pdf_bytes
from paper_digest.summarize import DEFAULT_MODEL, summarize

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn an arXiv URL/ID or PDF file into a structured summary.",
)

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"paper-digest {__version__}")
        raise typer.Exit()


@app.command()
def digest(
    source: str = typer.Argument(
        ...,
        help="arXiv ID (e.g. 2305.12345), arxiv.org URL, PDF URL, or local PDF path.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Claude model to use.",
    ),
    max_pages: int = typer.Option(
        50,
        "--max-pages",
        help="Cap on how many PDF pages to parse (for long papers).",
        min=1,
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Print the raw JSON summary instead of the pretty card.",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Summarize a paper."""
    with console.status("[cyan]Fetching paper...", spinner="dots"):
        try:
            pdf = fetch_pdf_bytes(source)
        except FetchError as exc:
            err_console.print(f"[red]Fetch failed:[/red] {exc}")
            raise typer.Exit(1)

    with console.status("[cyan]Extracting text...", spinner="dots"):
        try:
            text = extract_text(pdf, max_pages=max_pages)
        except ExtractError as exc:
            err_console.print(f"[red]Extraction failed:[/red] {exc}")
            raise typer.Exit(1)

    with console.status(f"[cyan]Summarizing with {model}...", spinner="dots"):
        try:
            summary = summarize(text, model=model)
        except Exception as exc:  # broad: surface anthropic SDK errors cleanly
            err_console.print(f"[red]Summarize failed:[/red] {exc}")
            raise typer.Exit(1)

    if output_json:
        json.dump(summary.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    _print_pretty(summary)


def _print_pretty(summary) -> None:  # type: ignore[no-untyped-def]
    header = f"[bold]{summary.title}[/bold]\n[dim]{', '.join(summary.authors)}[/dim]"
    console.print(Panel(header, border_style="cyan"))

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("field", style="bold cyan", no_wrap=True)
    table.add_column("value")
    table.add_row("Problem", summary.problem)
    table.add_row("Method", summary.method)
    table.add_row("Key insight", f"[yellow]{summary.key_insight}[/yellow]")
    table.add_row("Results", summary.results)
    table.add_row("Limitations", summary.limitations)
    table.add_row("Tags", " ".join(f"[dim]#{t}[/dim]" for t in summary.tags))
    console.print(table)


if __name__ == "__main__":
    app()
