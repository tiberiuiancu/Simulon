from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Generator


@contextmanager
def log_progress(
    description: str,
    total: int,
    logger: logging.Logger,
) -> Generator[Callable[[], None], None, None]:
    """Context manager that yields an advance() callable.

    When the given logger is at INFO level, displays a rich progress bar.
    Falls back to a no-op if rich is not installed or logging is not enabled.

    Usage::

        with log_progress("  building DAG", n_gpus, logger) as advance:
            for gpu in gpus:
                ...
                advance()
    """
    if not logger.isEnabledFor(logging.INFO):
        yield lambda: None
        return

    try:
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(description, total=total)
            yield lambda: progress.advance(task)
    except ImportError:
        yield lambda: None
