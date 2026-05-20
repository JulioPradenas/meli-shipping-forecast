"""Structured logging configuration for the API service.

Uses structlog to emit logs that are:
  - JSON in production (stdout is not a TTY): one JSON object per line,
    ready for ingestion by Datadog / CloudWatch / Loki without parsing.
  - Human-readable with colors in development (stdout is a TTY): the
    ConsoleRenderer is far easier to scan than raw JSON.

The format is auto-detected via sys.stderr.isatty(), so no manual env
var is needed to switch between the two.

A request_id is bound to the logging context by the middleware in app.py
(bind_contextvars), so every log line emitted while handling a request
automatically carries that request_id without it being passed explicitly.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog for the process.

    Idempotent enough for app startup: calling it more than once just
    reconfigures with the same settings.

    Args:
        level: The minimum log level to emit. Defaults to INFO.
    """
    is_tty = sys.stderr.isatty()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if is_tty:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger, optionally namespaced.

    Args:
        name: Optional logger name (typically __name__ of the caller).

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
