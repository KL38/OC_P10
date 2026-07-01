"""Pydantic AI agent routing questions to the RAG or SQL tool (Phase 3).

The agent is the new entrypoint of the enriched system: it reads the question,
decides between
    - ``search_match_commentary``  (FAISS retrieval over the Reddit-match
      corpus — the *unchanged* baseline retriever), and
    - ``query_player_stats``       (NL->SQL over the NBA stats database),
then writes the final answer itself from the tool output.

Mistral is driven through its OpenAI-compatible endpoint: ``pydantic-ai-slim``
is installed without the ``mistral`` extra because that extra requires
``mistralai>=2`` while ragas/instructor pin ``<2`` (see plan, § conflits).

Everything is injectable (model, retriever, SQL tool) so tests can run the
full agent loop offline with ``TestModel``/``FunctionModel``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, cast

from openai.types import chat
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..config import Settings, get_settings
from ..models import AskRequest, RagAnswer
from ..retrieval.vector_store import VectorStoreManager
from ..sql.sql_tool import SqlTool

logger = logging.getLogger(__name__)

AGENT_INSTRUCTIONS = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

Périmètre des données (important) :
- Les statistiques couvrent UNE SEULE saison régulière NBA (totaux par joueur),
  pas l'histoire de la ligue. Toute question chiffrée sans précision temporelle
  porte sur CETTE saison : réponds dans ce cadre, sans extrapoler à la carrière
  ou à l'histoire NBA.
- Les commentaires de match couvrent quelques discussions de fans (Reddit).

Choisis le bon outil :
- `search_match_commentary` pour les questions sur les discussions de fans, les
  opinions, les débats et les commentaires de match (contenu textuel).
- `query_player_stats` pour toute question chiffrée ou statistique : totaux,
  classements, comparaisons, pourcentages, agrégats par joueur ou par équipe.

Règles de réponse :
- Fonde ta réponse UNIQUEMENT sur le résultat des outils ; cite les chiffres
  exactement tels qu'ils sont retournés, sans les recalculer ni en inventer.
- Ne compare JAMAIS le résultat d'un outil avec tes connaissances d'entraînement :
  pas de note de doute ni de correction « de mémoire » — la base fait foi.
- Si l'outil ne retourne rien d'utile (ou une erreur), dis clairement que
  l'information n'est pas disponible dans la base — ne réponds pas de mémoire.
- Réponds en français, de façon concise et factuelle."""

_NO_CONTEXT = "Aucune information pertinente trouvée dans les commentaires de match pour cette question."


class MistralCompatChatModel(OpenAIChatModel):
    """``OpenAIChatModel`` adapted to Mistral's OpenAI-compatible endpoint.

    Mistral occasionally returns ``message.content`` as a list of chunks
    (``[{'type': 'text', 'text': ...}]``) where the OpenAI spec mandates a
    plain string; pydantic-ai's strict response validation then rejects the
    whole completion (``UnexpectedModelBehavior``). ``_validate_completion``
    is the hook documented for custom completion validation: we flatten the
    text chunks into a string, then delegate to the standard validation.

    Version-sensitive: relies on a pydantic-ai internal hook (present in
    1.106.0, still on main as of 2026-06) — re-check on upgrade.
    """

    def _validate_completion(self, response: chat.ChatCompletion) -> chat.ChatCompletion:
        for choice in response.choices:
            content = choice.message.content
            if isinstance(content, list):
                choice.message.content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
        return super()._validate_completion(response)


@dataclass
class AgentTrace:
    """Mutable per-run accumulator filled by the tools.

    Deliberately not frozen: this is run-scoped scratch state the tools write
    into so ``ask_agent`` can assemble the typed ``RagAnswer`` afterwards
    (contexts feed RAGAS in Phase 4).
    """

    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)


@dataclass
class AgentDeps:
    """Dependencies injected into every agent run."""

    question: str
    manager: VectorStoreManager
    sql_tool: SqlTool
    search_k: int
    trace: AgentTrace


def build_agent(model: Model | None = None, settings: Settings | None = None) -> Agent[AgentDeps, str]:
    """Build the routing agent. ``model`` is injectable for offline tests."""
    settings = settings or get_settings()
    if model is None:
        model = MistralCompatChatModel(
            settings.chat_model,
            provider=OpenAIProvider(
                base_url=settings.mistral_openai_base_url,
                api_key=settings.mistral_api_key,
            ),
        )

    agent: Agent[AgentDeps, str] = Agent(
        model,
        deps_type=AgentDeps,
        instructions=AGENT_INSTRUCTIONS,
        model_settings={"temperature": 0.1},
    )

    @agent.tool
    def search_match_commentary(ctx: RunContext[AgentDeps]) -> str:
        """Search the indexed match commentary (Reddit threads) for passages
        relevant to a textual/opinion question. Returns the retrieved passages."""
        # No query argument on purpose (reco #1): retrieval runs on the *verbatim*
        # user question (ctx.deps.question), not an LLM-reformulated one. The model
        # keeps the routing choice (RAG vs SQL) but loses control of the query text
        # — LLM keyword-compression flattened the FAISS scores and pulled wrong
        # chunks (proven in Logfire on Q1).
        results = ctx.deps.manager.search(ctx.deps.question, k=ctx.deps.search_k)
        ctx.deps.trace.tools_used.add("rag")
        if not results:
            return _NO_CONTEXT
        ctx.deps.trace.contexts.extend(r.text for r in results)
        ctx.deps.trace.sources.extend(r.metadata.get("source", "Inconnue") for r in results)
        # Same context block formatting as the baseline pipeline.
        return "\n\n---\n\n".join(
            f"Source: {r.metadata.get('source', 'Inconnue')} (Score: {r.score:.1f}%)\n"
            f"Contenu: {r.text}"
            for r in results
        )

    @agent.tool
    def query_player_stats(ctx: RunContext[AgentDeps], question: str) -> str:
        """Answer a numeric/statistical question (totals, rankings, comparisons,
        percentages) by querying the NBA stats SQL database."""
        result = ctx.deps.sql_tool.run(question)
        ctx.deps.trace.tools_used.add("sql")
        if result.error:
            return f"La requête SQL a échoué : {result.error}"
        ctx.deps.trace.contexts.append(f"Requête SQL : {result.query}\nRésultat : {result.result}")
        ctx.deps.trace.sources.append(f"SQL: {result.query}")
        return f"Requête exécutée : {result.query}\nRésultat : {result.result}"

    return agent


def ask_agent(
    request: AskRequest,
    *,
    agent: Agent[AgentDeps, str] | None = None,
    manager: VectorStoreManager | None = None,
    sql_tool: SqlTool | None = None,
    settings: Settings | None = None,
) -> RagAnswer:
    """Run the agent on one question and return the typed answer.

    Same contract as ``rag.pipeline.answer_question`` so Streamlit and the
    Phase 4 RAGAS harness can swap between baseline and enriched system.
    """
    settings = settings or get_settings()
    agent = agent or build_agent(settings=settings)
    manager = manager or VectorStoreManager(settings=settings)
    sql_tool = sql_tool or SqlTool(settings=settings)

    trace = AgentTrace()
    deps = AgentDeps(
        question=request.question,
        manager=manager,
        sql_tool=sql_tool,
        search_k=request.k or settings.search_k,
        trace=trace,
    )
    result = agent.run_sync(request.question, deps=deps)

    used_tool: Literal["rag", "sql", "both", "none"]
    if not trace.tools_used:
        used_tool = "none"
        # No tool = no evidence: the answer came from model weights alone.
        logger.warning("Agent answered without calling any tool: %s", request.question)
    elif trace.tools_used == {"rag", "sql"}:
        used_tool = "both"
    else:
        # The set holds exactly one of the two tool names here.
        used_tool = cast(Literal["rag", "sql"], next(iter(trace.tools_used)))
    logger.info("Agent answered (tool=%s, %d contexts)", used_tool, len(trace.contexts))
    return RagAnswer(
        answer=result.output,
        contexts=list(trace.contexts),
        sources=list(trace.sources),
        used_tool=used_tool,
    )
