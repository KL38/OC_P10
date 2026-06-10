"""Tests for the Pydantic AI agent plumbing (routing -> trace -> RagAnswer).

``TestModel`` (pydantic-ai) is scripted to call one specific tool per test, so
we verify the wiring around the LLM decision — tool execution, trace
accumulation, ``used_tool`` and contexts in the final ``RagAnswer`` — without
any network call. The actual routing *quality* is judged by RAGAS in Phase 4.
"""

from __future__ import annotations

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider

from sportsee_rag.agent.agent import MistralCompatChatModel, build_agent, ask_agent
from sportsee_rag.models import AskRequest, RetrievedChunk, SqlResult


class FakeManager:
    """Stands in for ``VectorStoreManager`` (deterministic search results)."""

    def search(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="Paolo et Franz portent le Magic.",
                score=87.5,
                raw_score=0.875,
                metadata={"source": "Match 1.pdf"},
            )
        ]


class FakeSqlTool:
    """Stands in for ``SqlTool`` (deterministic SQL round-trip)."""

    def __init__(self, result: SqlResult | None = None) -> None:
        self._result = result or SqlResult(
            query="SELECT name, reb FROM players ORDER BY reb DESC LIMIT 1",
            result="[('Ivica Zubac', 1008)]",
        )

    def run(self, question: str) -> SqlResult:
        return self._result


@pytest.fixture
def fake_manager() -> FakeManager:
    return FakeManager()


@pytest.fixture
def fake_sql_tool() -> FakeSqlTool:
    return FakeSqlTool()


def _ask(settings, fake_manager, fake_sql_tool, *, call_tools: list[str]) -> object:
    agent = build_agent(model=TestModel(call_tools=call_tools), settings=settings)
    return ask_agent(
        AskRequest(question="Question de test"),
        agent=agent,
        manager=fake_manager,
        sql_tool=fake_sql_tool,
        settings=settings,
    )


def test_rag_route_fills_trace(settings, fake_manager, fake_sql_tool):
    answer = _ask(settings, fake_manager, fake_sql_tool, call_tools=["search_match_commentary"])
    assert answer.used_tool == "rag"
    assert answer.contexts == ["Paolo et Franz portent le Magic."]
    assert answer.sources == ["Match 1.pdf"]
    assert answer.answer  # the model produced a final text

def test_sql_route_fills_trace(settings, fake_manager, fake_sql_tool):
    answer = _ask(settings, fake_manager, fake_sql_tool, call_tools=["query_player_stats"])
    assert answer.used_tool == "sql"
    assert len(answer.contexts) == 1
    assert "Zubac" in answer.contexts[0]
    assert answer.sources[0].startswith("SQL:")

def test_sql_error_degrades_gracefully(settings, fake_manager):
    failing = FakeSqlTool(result=SqlResult(query="SELECT x", error="boom"))
    answer = _ask(settings, fake_manager, failing, call_tools=["query_player_stats"])
    # Tool was used but produced no context — the agent still answers.
    assert answer.used_tool == "sql"
    assert answer.contexts == []
    assert answer.answer

def test_both_tools_used_reports_both(settings, fake_manager, fake_sql_tool):
    answer = _ask(
        settings, fake_manager, fake_sql_tool,
        call_tools=["search_match_commentary", "query_player_stats"],
    )
    assert answer.used_tool == "both"
    assert len(answer.contexts) == 2

def test_no_tool_reports_none(settings, fake_manager, fake_sql_tool):
    # The agent answered from model weights alone: zero evidence, flagged.
    answer = _ask(settings, fake_manager, fake_sql_tool, call_tools=[])
    assert answer.used_tool == "none"
    assert answer.contexts == []
    assert answer.answer


class TestMistralCompatChatModel:
    """Mistral's OpenAI-compat endpoint may return ``content`` as a chunk list
    instead of the spec-mandated string; the adapter must flatten it so the
    strict pydantic-ai validation accepts the completion."""

    @staticmethod
    def _completion(content) -> ChatCompletion:
        # model_construct bypasses the SDK validation, mirroring how the
        # openai client materialises whatever JSON the server actually sent.
        message = ChatCompletionMessage.model_construct(role="assistant", content=content)
        choice = Choice.model_construct(index=0, message=message, finish_reason="stop")
        return ChatCompletion.model_construct(
            id="cmpl-test",
            choices=[choice],
            created=1,
            model="mistral-small-latest",
            object="chat.completion",
        )

    @staticmethod
    def _model() -> MistralCompatChatModel:
        return MistralCompatChatModel(
            "mistral-small-latest",
            provider=OpenAIProvider(base_url="http://localhost:1/v1", api_key="test-key"),
        )

    def test_chunk_list_content_is_flattened(self):
        completion = self._completion(
            [{"type": "text", "text": "Les fans reprochent "}, {"type": "text", "text": "le marketing."}]
        )
        validated = self._model()._validate_completion(completion)
        assert validated.choices[0].message.content == "Les fans reprochent le marketing."

    def test_non_text_chunks_are_dropped(self):
        completion = self._completion(
            [{"type": "text", "text": "Réponse."}, {"type": "reference", "ids": [1]}]
        )
        validated = self._model()._validate_completion(completion)
        assert validated.choices[0].message.content == "Réponse."

    def test_plain_string_content_is_untouched(self):
        completion = self._completion("Réponse classique.")
        validated = self._model()._validate_completion(completion)
        assert validated.choices[0].message.content == "Réponse classique."
