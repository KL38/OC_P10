"""Minimal Streamlit chat UI — thin client over the RAG pipeline.

Port of the prototype's ``MistralChat.py``, but the RAG logic now lives in
``rag/pipeline.py``: this file only handles the UI and delegates to
``answer_question``. In Phase 3 the same call site will route through the
Pydantic AI agent (RAG vs SQL) instead of the RAG pipeline directly.

Run:
    uv run streamlit run streamlit_app.py
(Build the index first: ``uv run python scripts/build_index.py``.)
"""

from __future__ import annotations

import streamlit as st

from sportsee_rag.config import get_settings
from sportsee_rag.llm.client import MistralError
from sportsee_rag.models import AskRequest
from sportsee_rag.observability import setup_observability
from sportsee_rag.rag.pipeline import answer_question
from sportsee_rag.retrieval.vector_store import VectorStoreManager

setup_observability()
settings = get_settings()

st.set_page_config(page_title="NBA Analyst AI", page_icon="🏀")


@st.cache_resource
def get_manager() -> VectorStoreManager:
    """Load the vector store once per session (heavy: FAISS + chunks)."""
    return VectorStoreManager()


st.title("🏀 NBA Analyst AI")
st.caption(f"Assistant RAG SportSee · modèle : {settings.chat_model}")

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
            "content": "Bonjour ! Posez vos questions sur la NBA — je réponds à partir des "
            "commentaires de match indexés.",
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
            with st.spinner("Recherche + génération..."):
                answer = answer_question(AskRequest(question=prompt, k=k), manager=manager)
            st.write(answer.answer)
            if answer.sources:
                with st.expander(f"Sources ({len(answer.sources)})"):
                    for source in dict.fromkeys(answer.sources):  # de-duplicated
                        st.markdown(f"- {source}")
            content = answer.answer
        except MistralError as exc:
            content = f"⚠️ Erreur technique (Mistral) : {exc}. Réessayez dans un instant."
            st.error(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
