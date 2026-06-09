"""Thin wrapper around the Mistral Python SDK (chat + embeddings).

Single place that talks to Mistral, so resilience (throttle + retry/backoff
against the free-tier rate limits) lives in one spot. This replaces the
prototype's direct ``MistralClient`` calls, which were on the deprecated
``mistralai 0.4.x`` API and injected *null vectors* on failure (corrupting the
index). Here a failed call fails cleanly after bounded retries.

SDK migration (0.4.x -> v1.x), method-for-method:
    MistralClient(api_key=...)               -> Mistral(api_key=...)
    client.embeddings(model=, input=)        -> client.embeddings.create(model=, inputs=)
    client.chat(model=, messages=)           -> client.chat.complete(model=, messages=)

We target **v1.x** (``>=1.5.1,<2``) because ragas -> instructor caps mistralai
to ``<2``, while pydantic-ai's native ``mistral`` extra wants ``>=2``; v1.x lets
the whole stack share one environment. The import below is version-agnostic so
the wrapper keeps working on either layout.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

try:
    from mistralai import Mistral  # mistralai v1.x: top-level export
except ImportError:  # pragma: no cover - mistralai v2.x layout
    from mistralai.client import Mistral

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MistralError(RuntimeError):
    """Raised when a Mistral call keeps failing after all retries."""


class MistralLLM:
    """Resilient facade over the Mistral chat and embeddings endpoints.

    The underlying client can be injected for testing (no network, no key).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: Mistral | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or Mistral(api_key=self._settings.mistral_api_key)

    # --- public API -----------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch of texts. Raises ``MistralError`` on final failure."""
        if not texts:
            return []
        response = self._with_retries(
            lambda: self._client.embeddings.create(
                model=self._settings.embedding_model,
                inputs=texts,
            ),
            label="embeddings.create",
        )
        return [item.embedding for item in response.data]

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> str:
        """Run a chat completion and return the assistant message content."""
        response = self._with_retries(
            lambda: self._client.chat.complete(
                model=self._settings.chat_model,
                messages=messages,
                temperature=temperature,
            ),
            label="chat.complete",
        )
        if not response.choices:
            raise MistralError("chat.complete returned no choices")
        return response.choices[0].message.content

    # --- internals ------------------------------------------------------

    def _with_retries(self, call: Callable[[], T], *, label: str) -> T:
        """Call ``call`` with bounded exponential backoff on any exception.

        Mistral's exact exception types vary across SDK versions, so we retry
        broadly and surface a clean ``MistralError`` once retries are exhausted
        rather than silently degrading (the prototype's null-vector bug).
        """
        max_retries = self._settings.embedding_max_retries
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - SDK error types vary
                last_exc = exc
                if attempt == max_retries:
                    break
                wait = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Mistral %s failed (attempt %d/%d): %s — retrying in %ds",
                    label, attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
        raise MistralError(
            f"Mistral {label} failed after {max_retries} attempts"
        ) from last_exc
