"""Structured logging.

Human-readable console rendering when attached to a TTY, JSON lines otherwise, so
the same code path is usable interactively and in CI/batch logs.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    """Configure ``structlog`` once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if json_logs is None:
        json_logs = not sys.stderr.isatty()

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]
