"""Central observability setup (Pydantic Logfire).

Single initialisation point that replaces the scattered ``logging.basicConfig``
calls of the prototype. Routes the stdlib ``logging`` through Logfire so every
module logs consistently. Idempotent: safe to call more than once.

Note: without a ``LOGFIRE_TOKEN`` Logfire runs locally (nothing is sent to the
cloud), which keeps tests and local runs friction-free.

TODO (Phase 1/3): wire framework instrumentation once those exist
(``logfire.instrument_fastapi(app)``, ``logfire.instrument_pydantic_ai(...)``).
Confirm the exact Logfire API against the official docs when adding `logfire`
via `uv add`.
"""

from __future__ import annotations

import logging

import logfire

from .config import get_settings

_configured = False


def setup_observability(service_name: str = "sportsee-rag") -> None:
    """Initialise Logfire and bridge stdlib logging into it (once)."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    logfire.configure(
        service_name=service_name,
        token=settings.logfire_token,  # None -> local mode
        send_to_logfire="if-token-present",
    )

    # Route the standard logging module through Logfire (single handler).
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logfire.LogfireLoggingHandler()],
        force=True,
    )

    _configured = True
