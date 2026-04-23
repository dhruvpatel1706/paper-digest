"""Watch a folder for new PDFs and digest them as they land.

The loop I actually wanted: drop an arXiv PDF into ~/Downloads/papers,
come back five minutes later, the summary is already in `paper-digest
history`. Works offline once the PDF is on disk — the LLM summarize
call still needs the network, but no arXiv fetch.

Reuses the debounce pattern from personal-rag's watcher: PDFs arrive in
chunks over a slow network, and `on_modified` can fire dozens of times
per file. We wait for the path to go quiet for `debounce_s` before
handing it off, and we skip anything that's already in the local
history (matched by absolute path) so restarting the watcher doesn't
re-summarize the whole folder.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from paper_digest import history as history_store
from paper_digest.extract import ExtractError, extract_text
from paper_digest.models import Summary
from paper_digest.summarize import summarize as _summarize

log = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_S = 2.0
PDF_SUFFIX = ".pdf"


@dataclass
class _Pending:
    path: Path
    at: float  # monotonic time of the last observed event


class _Handler(FileSystemEventHandler):
    def __init__(self, queue: "_DebounceQueue"):
        super().__init__()
        self.queue = queue

    def on_created(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self.queue.push(Path(event.src_path))

    def on_modified(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self.queue.push(Path(event.src_path))

    def on_moved(self, event):  # type: ignore[no-untyped-def]
        # A rename into the watched dir looks like a move on some platforms.
        if not event.is_directory:
            self.queue.push(Path(event.dest_path))


class _DebounceQueue:
    """Coalesce rapid events on the same path into a single digest call.

    One background thread drains the pending dict. `push` is cheap; the
    actual summarize work runs on the drain thread so we don't block the
    watchdog observer.
    """

    def __init__(
        self,
        *,
        delay: float,
        on_ready: Callable[[Path], None],
    ):
        self.delay = delay
        self.on_ready = on_ready
        self._pending: dict[Path, _Pending] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def push(self, path: Path) -> None:
        if path.suffix.lower() != PDF_SUFFIX:
            return
        with self._lock:
            self._pending[path] = _Pending(path=path, at=time.monotonic())

    def _drain(self) -> None:
        while not self._stop.is_set():
            time.sleep(min(0.25, self.delay / 4))
            now = time.monotonic()
            ready: list[Path] = []
            with self._lock:
                for path, evt in list(self._pending.items()):
                    if now - evt.at >= self.delay:
                        ready.append(path)
                        del self._pending[path]
            for path in ready:
                try:
                    self.on_ready(path)
                except Exception:  # noqa: BLE001
                    log.exception("watcher callback failed for %s", path)


def _seen_sources_in_history(history_dir: Path | None) -> set[str]:
    """Absolute paths we've already summarized, so we don't redo them on startup."""
    out: set[str] = set()
    for entry in history_store.load_all(history_dir):
        src = entry.source
        # Only dedupe by absolute path — URLs are fine to re-summarize
        # manually via `digest`, but the watcher only ever sees files.
        if src.startswith("/"):
            out.add(src)
    return out


def _process_one(
    path: Path,
    *,
    model: str,
    max_pages: int,
    seen: set[str],
    history_dir: Path | None,
    summarize_fn: Callable[..., Summary],
    on_log: Callable[[str], None],
) -> None:
    resolved = str(path.resolve())
    if resolved in seen:
        return
    if not path.exists():
        # Lost the race — file was deleted between event and drain.
        return

    try:
        pdf_bytes = path.read_bytes()
    except OSError as exc:
        on_log(f"[error] {path.name}: couldn't read — {exc}")
        return

    try:
        text = extract_text(pdf_bytes, max_pages=max_pages)
    except ExtractError as exc:
        on_log(f"[error] {path.name}: extract failed — {exc}")
        return

    try:
        summary = summarize_fn(text, model=model)
    except Exception as exc:  # noqa: BLE001 — SDK errors, rate limits, etc.
        on_log(f"[error] {path.name}: summarize failed — {exc}")
        return

    try:
        history_store.save(resolved, summary, history_dir=history_dir)
    except OSError as exc:
        on_log(f"[warn] saved summary but history write failed: {exc}")

    seen.add(resolved)
    on_log(f"✓ {path.name}: {summary.title}")
    on_log(f"  → {summary.key_insight}")


def watch(
    target: Path,
    *,
    model: str,
    max_pages: int = 50,
    debounce_s: float = DEFAULT_DEBOUNCE_S,
    history_dir: Path | None = None,
    summarize_fn: Callable[..., Summary] = _summarize,
    on_log: Callable[[str], None] = print,
    stop_event: threading.Event | None = None,
) -> None:
    """Block on a filesystem watch of `target`, digesting any new PDFs.

    `stop_event` is an escape hatch for tests and embedding callers — passing
    one lets you stop the loop without raising KeyboardInterrupt. In the CLI
    it's None and Ctrl-C ends the loop.
    """
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Watch target must be an existing directory: {target}")

    seen = _seen_sources_in_history(history_dir)

    queue = _DebounceQueue(
        delay=debounce_s,
        on_ready=lambda path: _process_one(
            path,
            model=model,
            max_pages=max_pages,
            seen=seen,
            history_dir=history_dir,
            summarize_fn=summarize_fn,
            on_log=on_log,
        ),
    )
    handler = _Handler(queue)

    observer = Observer()
    observer.schedule(handler, str(target), recursive=False)
    observer.start()
    queue.start()

    on_log(
        f"watching {target} for new PDFs (debounce {debounce_s:.1f}s, "
        f"{len(seen)} already in history). Ctrl-C to stop."
    )

    try:
        while observer.is_alive():
            if stop_event is not None and stop_event.is_set():
                break
            observer.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        queue.stop()
        observer.stop()
        observer.join(timeout=2.0)
        on_log("stopped.")
