"""Minimal Streamlit chat UI — thin client over the Pydantic AI agent.

Port of the prototype's ``MistralChat.py``. Since Phase 3 the call site routes
through the agent (``agent/agent.py``), which picks between the RAG tool
(match commentary) and the SQL tool (NBA stats) — the UI only renders the
typed ``RagAnswer``.

Run:
    uv run streamlit run streamlit_app.py
(Build the index first: ``uv run python scripts/build_index.py``,
 and load the DB: ``uv run python scripts/load_excel_to_db.py``.)
"""

from __future__ import annotations

import logging

import streamlit as st

from sportsee_rag.agent.agent import AgentDeps, Agent, ask_agent, build_agent
from sportsee_rag.config import get_settings
from sportsee_rag.models import AskRequest
from sportsee_rag.observability import setup_observability
from sportsee_rag.retrieval.vector_store import VectorStoreManager
from sportsee_rag.sql.sql_tool import SqlTool

setup_observability()
settings = get_settings()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="NBA Analyst AI", page_icon="🏀")

_TOOL_LABELS = {
    "rag": "🔎 RAG (commentaires de match)",
    "sql": "🗃️ SQL (stats NBA)",
    "both": "🔎+🗃️ RAG + SQL",
    "none": "⚠️ aucun outil (réponse du modèle seul)",
}


@st.cache_resource
def get_manager() -> VectorStoreManager:
    """Load the vector store once per session (heavy: FAISS + chunks)."""
    return VectorStoreManager()


@st.cache_resource
def get_agent() -> Agent[AgentDeps, str]:
    """Build the routing agent once per session."""
    return build_agent()


@st.cache_resource
def get_sql_tool() -> SqlTool:
    """Open the SQL database handle once per session."""
    return SqlTool()


st.title("🏀 NBA Analyst AI")
st.caption(f"Assistant RAG + SQL SportSee · modèle : {settings.chat_model}")

with st.sidebar:
    k = st.slider("Chunks récupérés (k)", min_value=1, max_value=10, value=settings.search_k)

manager = get_manager()
if manager.index is None:
    st.error(
        "Index vectoriel introuvable. Construis-le d'abord :\n\n"
        "`uv run python scripts/build_index.py`"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Posez vos questions sur la NBA — commentaires de match "
            "(débats de fans) ou statistiques chiffrées des joueurs.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Posez votre question NBA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Routing + recherche + génération..."):
                answer = ask_agent(
                    AskRequest(question=prompt, k=k),
                    agent=get_agent(),
                    manager=manager,
                    sql_tool=get_sql_tool(),
                )
            st.write(answer.answer)
            st.caption(f"Outil utilisé : {_TOOL_LABELS[answer.used_tool]}")
            if answer.sources:
                with st.expander(f"Sources ({len(answer.sources)})"):
                    for source in dict.fromkeys(answer.sources):  # de-duplicated
                        st.markdown(f"- {source}")
            content = answer.answer
        except Exception:  # noqa: BLE001 - any backend failure: log details, show generic
            # Driver/SDK exceptions can embed the connection string or the
            # failing query — never surface them to the user.
            logger.exception("Agent run failed")
            content = "⚠️ Erreur technique. Réessayez dans un instant."
            st.error(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
