# Multi-Agent Research Assistant

A context engineering pipeline built with CrewAI that gathers research from four parallel sources — documents (RAG), memory, web, and arXiv — then filters and synthesizes a comprehensive response.

**Powered by:** Tensorlake · Zep · Firecrawl · ChromaDB · CrewAI

---

![Multi-Agent Research Assistant](./imgs/image%20copy.png)

## Quick Start

### 1. Install Python 3.11 (Recommended)

```bash
winget install Python.Python.3.11
```

Then create a new virtual environment with Python 3.11:

```bash
python3.11 -m venv crewai_env
source crewai_env/Scripts/activate
python -m pip install crewai
```

### 2. Install UV (dependency manager)

```bash
pip install uv
```

### 3. Clone and navigate to the project

```bash
cd research_assistant
```

### 4. Install all dependencies

```bash
uv sync
```

Or install individually:

```bash
uv add crewai[tools] fastapi uvicorn[standard] streamlit firecrawl-py \
       zep-cloud zep-crewai pymilvus tensorlake sentence-transformers \
       httpx python-multipart requests
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

| Variable | Required | Where to get it |
|---|---|---|
| `HF_TOKEN` | Yes | https://huggingface.co/settings/tokens |
| `ZEP_API_KEY` | Yes | https://app.getzep.com |
| `FIRECRAWL_API_KEY` | Yes | https://firecrawl.dev |
| `TENSORLAKE_API_KEY` | Yes (for PDF upload) | https://tensorlake.ai |
| `CHROMA_DB_PATH` | No | Default: `./chroma_db` (local directory) |

> **Note:** ArXiv search requires no API key — it uses the free public API.

---

## Running the Project

### Option A: Run the full flow via CLI

```bash
crewai run
```

This kicks off the `ContextEngineeringFlow` with a default query and prints the final response.

### Option B: Run as a FastAPI + Streamlit app

**Terminal 1 — Start the API server:**
```bash
uv run start_api
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Terminal 2 — Start the Streamlit UI:**
```bash
uv run streamlit run app/streamlit_app.py
# UI available at http://localhost:8501
```

Then open http://localhost:8501 in your browser.

### Option C: Use the API directly

```bash
# Submit a research query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is context engineering?", "user_id": "alice"}'
# Returns: {"session_id": "abc-123", "status": "pending"}

# Poll for the result
curl http://localhost:8000/api/query/abc-123

# Upload a PDF for RAG indexing
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@/path/to/paper.pdf"

# Health check
curl http://localhost:8000/api/health
```

---

## Architecture

```
User Query
    │
    ▼
[Step 1] process_query        → Save query to Zep memory, assign session_id
    │
    ▼
[Step 2] gather_context       → ContextCrew (4 agents run in parallel)
    │                             ┌──────────────────────────────────────┐
    │                             │ rag_agent     → Milvus vector search  │
    │                             │ memory_agent  → Zep conversation hist │
    │                             │ web_search_agent → Firecrawl          │
    │                             │ arxiv_agent   → arXiv API             │
    │                             └──────────────────────────────────────┘
    ▼
[Step 3] evaluate_context     → EvaluationCrew (1 agent)
    │                             Scores each source 0.0–1.0
    │                             Filters sources below 0.4
    │                             Returns ContextEvaluationOutput (Pydantic)
    ▼
[Step 4] synthesize_response  → SynthesisCrew (1 agent)
    │                             Generates markdown research response
    │                             Saves response to Zep memory
    ▼
Final Response
```

---

## Project Structure

```
research_assistant/
├── app/
│   └── streamlit_app.py          # Streamlit chat UI
├── src/research_assistant/
│   ├── flow.py                   # ContextEngineeringFlow (main orchestrator)
│   ├── crew.py                   # ContextCrew, EvaluationCrew, SynthesisCrew
│   ├── schemas.py                # FlowState, ContextEvaluationOutput
│   ├── main.py                   # CLI entrypoints
│   ├── config/
│   │   ├── agents.yaml           # All 6 agent configurations
│   │   └── tasks.yaml            # All 6 task configurations
│   ├── tools/
│   │   ├── arxiv_tool.py         # ArXiv API search
│   │   ├── firecrawl_tool.py     # Firecrawl web search
│   │   ├── milvus_rag_tool.py    # Milvus vector retrieval
│   │   └── tensorlake_tool.py    # PDF parsing + indexing
│   ├── memory/
│   │   └── zep_memory.py         # Zep memory helpers
│   └── api/
│       ├── router.py             # FastAPI route handlers
│       └── schemas.py            # API request/response models
├── knowledge/
│   └── user_preference.txt       # User context (editable)
├── .env.example                  # API key template
└── pyproject.toml
```

---

## Streamlit APP

<video width="100%" controls>
  <source src="./imgs/Screen%20Recording%202026-04-29%20051948.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## Customization

- **Change the LLM**: Edit `MODEL` in `.env`. For better structured output reliability, use `gpt-4o-mini` or `claude-sonnet-4-5` instead of HuggingFace.
- **Add more documents**: Upload PDFs via the Streamlit sidebar or `POST /api/documents/upload`.
- **Change agents**: Edit `src/research_assistant/config/agents.yaml`.
- **Change tasks**: Edit `src/research_assistant/config/tasks.yaml`.
- **Add tools**: Create a new `BaseTool` subclass in `src/research_assistant/tools/` and assign it to an agent in `crew.py`.

---

## Support

- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewai)
- [CrewAI Discord](https://discord.com/invite/X4JWnZnxPb)
