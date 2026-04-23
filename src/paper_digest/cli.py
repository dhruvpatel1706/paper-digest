"""Typer CLI for paper-digest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from paper_digest import __version__
from paper_digest import history as history_store
from paper_digest.batch import parse_reading_list, render_digest, run_batch
from paper_digest.chat import PaperChat
from paper_digest.extract import ExtractError, extract_text
from paper_digest.fetch import FetchError, fetch_pdf_bytes
from paper_digest.summarize import DEFAULT_MODEL, summarize

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn an arXiv URL/ID or PDF file into a structured summary (with optional interactive Q&A).",
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
    chat: bool = typer.Option(
        False,
        "--chat",
        help=(
            "After the summary, enter an interactive follow-up Q&A loop. "
            "The full paper text is prompt-cached so every question after the first "
            "is cheap and fast."
        ),
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
    """Summarize a paper. Optional --chat enters an interactive follow-up Q&A session."""
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

    # Persist to history (v0.5). Best-effort — don't fail the command just
    # because the history dir was unwritable for some reason.
    try:
        history_store.save(source, summary)
    except OSError as exc:
        err_console.print(f"[dim](history save skipped: {exc})[/dim]")

    if output_json:
        json.dump(summary.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    _print_pretty(summary)

    if chat:
        _run_chat(text, summary, model)


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


def _run_chat(paper_text, summary, model: str) -> None:  # type: ignore[no-untyped-def]
    console.print()
    console.print(
        Panel.fit(
            "[dim]Ask follow-up questions about this paper. "
            "Type /quit or press Ctrl-D to exit.[/dim]",
            border_style="cyan",
        )
    )
    try:
        chat = PaperChat(paper_text, summary, model=model)
    except Exception as exc:
        err_console.print(f"[red]Could not start chat:[/red] {exc}")
        raise typer.Exit(1)

    while True:
        try:
            question = Prompt.ask("[yellow]You[/yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim](ended)[/dim]")
            break
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "/q"}:
            break
        with console.status("[cyan]Thinking...", spinner="dots"):
            try:
                reply = chat.ask(question)
            except Exception as exc:
                err_console.print(f"[red]Error:[/red] {exc}")
                continue
        console.print(f"\n[cyan]Claude:[/cyan] {reply}\n")


@app.command("batch")
def batch_cmd(
    list_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="File with one paper reference per line. Comments (#) and blank lines ignored.",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    max_pages: int = typer.Option(50, "--max-pages", min=1),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the combined digest to this path (markdown). Default: stdout.",
    ),
) -> None:
    """Summarize every paper in a reading list and emit a single markdown digest.

    Failures are collected and listed at the bottom of the digest — one paper's
    timeout or bad URL won't abort the run.
    """
    refs = parse_reading_list(list_file)
    if not refs:
        err_console.print(f"[yellow]No references found in {list_file}[/yellow]")
        raise typer.Exit(0)

    total = len(refs)
    console.print(f"[dim]Processing {total} paper{'s' if total != 1 else ''}...[/dim]")

    def _progress(i: int, n: int, ref: str) -> None:
        console.print(f"[dim]  [{i}/{n}][/dim] {ref}")

    items = run_batch(refs, model=model, max_pages=max_pages, progress=_progress)
    digest = render_digest(items)

    # Persist each successful summary to the local history (v0.5)
    for item in items:
        if item.summary is not None:
            try:
                history_store.save(item.source, item.summary)
            except OSError:
                pass  # tolerate history failures in batch

    if output:
        output.write_text(digest, encoding="utf-8")
        succeeded = sum(1 for it in items if it.summary is not None)
        console.print(f"\n[green]Wrote digest to {output}[/green] ({succeeded}/{total} succeeded)")
    else:
        sys.stdout.write(digest)


# ---- v0.5: history subcommands ---------------------------------------------

history_app = typer.Typer(help="Browse and search previously-summarized papers.")
app.add_typer(history_app, name="history")


@history_app.command("list")
def history_list_cmd(
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="How many entries to show."),
) -> None:
    """Show the most recent summaries (newest first)."""
    entries = history_store.load_all()
    if not entries:
        console.print("[dim]No history yet — run `paper-digest digest ...` first.[/dim]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("when", no_wrap=True)
    table.add_column("title")
    table.add_column("source")
    for e in entries[:limit]:
        when = e.summarized_at.strftime("%Y-%m-%d")
        table.add_row(when, e.summary.title, e.source)
    console.print(table)


@history_app.command("search")
def history_search_cmd(
    query: str = typer.Argument(..., help="Keywords to search your past summaries."),
    limit: int = typer.Option(5, "--limit", "-n", min=1),
) -> None:
    """Find past summaries by title/tag/insight/problem match."""
    hits = history_store.search(query, limit=limit)
    if not hits:
        console.print(f"[dim]No past summaries matched {query!r}.[/dim]")
        return
    for e in hits:
        s = e.summary
        console.print(f"\n[bold]{s.title}[/bold]  [dim]({e.source})[/dim]")
        console.print(f"[dim]{e.summarized_at.strftime('%Y-%m-%d')} · {', '.join(s.tags)}[/dim]")
        console.print(f"[yellow]insight:[/yellow] {s.key_insight}")


if __name__ == "__main__":
    app()
