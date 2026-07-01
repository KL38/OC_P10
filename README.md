# SportSee — Assistant NBA (RAG + SQL) · Projet OC P10

Assistant conversationnel qui répond à des questions sur la NBA à partir de deux sources :

- des **commentaires de match** (threads de fans, PDF scannés) → recherche sémantique (**RAG**) ;
- des **statistiques de joueurs** (classeur Excel d'une saison) → requêtes **SQL** générées à la volée.

Un **agent** décide seul, pour chaque question, quel outil interroger (RAG, SQL, ou les deux). Le projet
livre aussi une **évaluation objective** de la qualité des réponses avec [RAGAS](https://docs.ragas.io),
qui compare le prototype RAG d'origine (« baseline ») à la version enrichie du SQL.

> 📊 **Résultats** : le passage baseline → enrichi fait bondir la fidélité des réponses
> (`faithfulness` 0.36 → 0.88) et la pertinence du contexte récupéré (`context_precision` 0.17 → 0.71).
> Analyse détaillée : [`eval/reports/comparatif_baseline_v2.html`](eval/reports/comparatif_baseline_v2.html).

---

## Architecture

```mermaid
flowchart TB
    subgraph SRC["📁 data/"]
        PDF["Match 1-4.pdf<br/>threads Reddit (OCR)"]
        XLS["regular+NBA.xlsx<br/>stats de saison"]
    end

    PDF -->|"build_index.py<br/>OCR + chunking"| FAISS[("FAISS<br/>index vectoriel")]
    XLS -.->|"aplati (baseline)"| FAISS
    XLS -->|"load_excel_to_db.py<br/>validation Pydantic"| DB[("SQL<br/>PostgreSQL / SQLite<br/>teams · players")]

    Q(["❓ Question"]) --> AGENT

    subgraph AGENT["🤖 Agent Pydantic AI — routing"]
        RAGT["🔎 tool RAG<br/>commentaires"]
        SQLT["🗃️ tool SQL<br/>SELECT only"]
    end

    FAISS --> RAGT
    DB --> SQLT
    AGENT --> ANS["✅ RagAnswer<br/>réponse + sources + outil utilisé"]
    ANS --> UI["💬 Streamlit"]

    QSET["questions.yaml<br/>(60 questions)"] -.->|"evaluate_ragas.py"| RAGAS["📏 RAGAS<br/>4 métriques"]
    AGENT -.-> RAGAS
    RAGAS -.-> REP["eval/reports/<br/>comparatif HTML"]

    MISTRAL["Mistral<br/>génération + juge"] -.- AGENT
    MISTRAL -.- RAGAS
    LOGFIRE["Logfire<br/>traces"] -.- AGENT
```

- **Trois points de mesure** dans le comparatif RAGAS :
  - `baseline` — RAG texte seul, Excel *aplati* inclus dans l'index (reproduit le bug du prototype) ;
  - `enriched` — agent RAG + SQL sur l'index **complet** (PDF + Excel aplati) : isole l'apport du SQL ;
  - `enriched_v2` — même agent sur un index **PDF-only** (l'Excel ne passe plus que par SQL) **+ garde-fou
    de périmètre** ; c'est le run de référence du comparatif [avant/après](eval/reports/comparatif_baseline_v2.html).
- **Sécurité DB** : l'application tourne avec un rôle **SELECT-only** (`DATABASE_URL_READONLY`) ;
  seul le script de chargement utilise le rôle admin. Défense en profondeur, vérifiée par
  `scripts/check_db_readonly.py`.

---

## Prérequis

- **Python 3.12** et **[UV](https://docs.astral.sh/uv/)** (gestion d'environnement et de dépendances)
- Une **clé API Mistral** (gratuite) — <https://console.mistral.ai/>
- *(optionnel)* un **GPU CUDA** : accélère l'OCR EasyOCR à l'indexation (sinon CPU, plus lent)
- *(optionnel)* un projet **Supabase / PostgreSQL** : sinon repli automatique sur SQLite local

## Installation

```powershell
# 1. Dépendances (environnement reproductible)
uv sync

# 2. Secrets : copier le modèle et renseigner la clé Mistral
Copy-Item .env.example .env
# puis éditer .env → MISTRAL_API_KEY=...  (les autres variables sont optionnelles)
```

Variables d'environnement (`.env`) :

| Variable | Requis | Rôle |
|---|---|---|
| `MISTRAL_API_KEY` | ✅ | génération, embeddings, juge RAGAS |
| `DATABASE_URL` | ⚪ | rôle **admin** — chargement Excel → SQL uniquement |
| `DATABASE_URL_READONLY` | ⚪ | rôle **SELECT-only** — runtime de l'app |
| `LOGFIRE_TOKEN` | ⚪ | traces cloud (sinon Logfire tourne en local) |

> Sans `DATABASE_URL`, le projet bascule sur une base **SQLite** locale (`db/nba.sqlite`) — pratique
> pour tester hors-ligne.

## Mise en route

```powershell
# 3. Construire l'index vectoriel (OCR des PDF + embeddings)
uv run python scripts/build_index.py
# variante PDF-only pour enriched_v2 (Excel exclu du retrieval texte) :
uv run python scripts/build_index.py --pdf-only

# 4. Charger les stats Excel dans la base SQL
uv run python scripts/load_excel_to_db.py

# (optionnel) vérifier que le rôle applicatif est bien en lecture seule
uv run python scripts/check_db_readonly.py
```

## Utilisation

```powershell
# Interface de chat (recommandé)
uv run streamlit run streamlit_app.py

# …ou en ligne de commande, pour une question ponctuelle
uv run python scripts/ask.py "Quelles équipes ont impressionné en playoffs ?"
```

## Évaluation RAGAS

```powershell
# Prototype d'origine (RAG texte seul, Excel aplati dans l'index)
uv run python eval/evaluate_ragas.py --system baseline

# Agent RAG + SQL sur l'index complet
uv run python eval/evaluate_ragas.py --system enriched

# Agent RAG + SQL sur l'index PDF-only + garde-fou de périmètre (run de référence)
# → nécessite l'index PDF-only : build_index.py --pdf-only (cf. Mise en route)
uv run python eval/evaluate_ragas.py --system enriched_v2

# Ajouter --limit 2 pour un test rapide (2 questions)
```

Chaque run écrit un rapport JSON + tableau Markdown dans [`eval/reports/`](eval/reports/). Le comparatif
avant/après lisible est le HTML : [`comparatif_baseline_v2.html`](eval/reports/comparatif_baseline_v2.html).

## Tests

```powershell
uv run pytest
```

Approche pragmatique : le déterministe (schéma SQL, garde-fous, parsing) est couvert par des tests
unitaires ; les appels LLM sont mockés.

---

## Structure du dépôt

```
src/sportsee_rag/
  ingestion/     chargement & parsing du corpus (PDF/OCR, Excel, docx)
  retrieval/     index FAISS (cosinus) + recherche
  rag/           pipeline RAG (retrieve → prompt → réponse)
  sql/           schéma, chargement Excel→SQL, tool SQL (SELECT-only)
  agent/         agent Pydantic AI (routing RAG / SQL)
  llm/           client Mistral
  config.py      configuration typée (pydantic-settings)
  observability  Logfire
scripts/         entrypoints : build_index, load_excel_to_db, ask, check_db_readonly
eval/            harness RAGAS, questions.yaml, rapports & comparatifs
data/            corpus source (4 PDF de match + classeur NBA)
```

## Limites connues

- **PDF de match = scans OCRisés** (threads Reddit) : texte bruité par nature → plafonne certaines
  métriques de contexte même quand la réponse est correcte.
- **Juge RAGAS = `mistral-small`** (gratuit), pas un modèle frontière : les scores sont *indicatifs* ;
  le signal fiable est le **delta** baseline → enrichi.
- **Périmètre = une seule saison** : le modèle SQL n'a pas de dimension temporelle ; les questions
  hors-schéma (par match, par adversaire, salaires…) sont censées être **refusées** par l'agent.

## Stack technique

Python 3.12 · UV · Pydantic AI · Mistral · FAISS · LangChain (`SQLDatabase`) · SQLAlchemy ·
PostgreSQL/Supabase (repli SQLite) · RAGAS · Pydantic Logfire · Streamlit · EasyOCR/PyTorch.
