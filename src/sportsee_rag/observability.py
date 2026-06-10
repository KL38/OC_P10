"""Central observability setup (Pydantic Logfire).

Single initialisation point that replaces the scattered ``logging.basicConfig``
calls of the prototype. Routes the stdlib ``logging`` through Logfire so every
module logs consistently. Idempotent: safe to call more than once.

En fait ici c'est juste pour valider que le LOGFIRE est configuré correctement, 
et que les logs sont envoyés à Logfire (ou affichés localement si pas de token). 
Si configuré alors return, sinon, on le configure et configured passe True
Pas besoin de faire du logging dans ce module, c'est juste pour la configuration.  

TODO (bonus FastAPI): add ``logfire.instrument_fastapi(app)`` if the REST API
phase happens.
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

    # Every agent run becomes a full trace: one span per model request
    # (token counts, latency) and per tool call (name, args, return value).
    logfire.instrument_pydantic_ai()

    # Route the standard logging module through Logfire (single handler).
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logfire.LogfireLoggingHandler()],
        force=True,
    )

    _configured = True
