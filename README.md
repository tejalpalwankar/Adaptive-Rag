# Adaptive RAG Chatbot

A question-answering chatbot that decides, for each question, the best way to answer it: search your uploaded documents, answer from the model's own knowledge, or look it up on the web.

Built with FastAPI, LangGraph, FAISS, and MongoDB, with a Streamlit chat UI.

## What it does

You upload a document (PDF or TXT) and ask questions about it. For every question the system first classifies it, then routes it to one of three paths:

- **Index** - the answer is in your uploaded documents, so it retrieves from them.
- **General** - it's common knowledge or small talk, so the model answers directly.
- **Search** - it needs current or outside information, so it searches the web (Tavily).

On the document path it also checks its own work: it grades whether the retrieved text is relevant, rewrites the question and retries if not, and verifies the final answer is supported by the source before returning it.

## How it works

```
                    ┌─────────────────┐
   your question →  │  classify query │
                    └────────┬────────┘
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        ┌─────────┐    ┌───────────┐    ┌────────────┐
        │ retrieve│    │ general   │    │ web search │
        │  docs   │    │ LLM answer│    │  (Tavily)  │
        └────┬────┘    └─────┬─────┘    └──────┬─────┘
             ▼               │                 │
        ┌─────────┐          │                 │
        │  grade  │          │                 │
        └────┬────┘          │                 │
     relevant│ not relevant  │                 │
             │   └─ rewrite ─┘ (retry, capped) │
             ▼                                  ▼
        ┌──────────┐                     ┌──────────┐
        │ generate │◄────────────────────┘          │
        └────┬─────┘                                 │
             ▼                                        │
        ┌──────────┐  not faithful → regenerate       │
        │  verify  │  (capped)                        │
        └────┬─────┘                                  │
             ▼  faithful                              │
          answer ◄────────────────────────────────────
```

Each document you upload is stored in its own vector index, keyed by session, so different users' documents stay separate. Indexes are saved to disk, so they survive a server restart.

## Project structure

```
src/
├── main.py                     # FastAPI entry point
├── api/routes.py               # /rag/query and /rag/documents/upload endpoints
├── core/
│   ├── config.py               # Environment settings
│   ├── security.py             # API-key authentication
│   └── logger.py               # Logging setup
├── db/mongo_client.py          # MongoDB connection
├── llms/openai.py              # OpenAI (gpt-4o) setup
├── memory/
│   ├── chat_history_mongo.py   # Chat history in MongoDB (windowed)
│   └── chathistory_in_memory.py# In-memory fallback
├── models/                     # Pydantic schemas (state, grade, route, etc.)
├── rag/
│   ├── graph_builder.py        # The LangGraph workflow (all nodes)
│   ├── retriever_setup.py      # Per-session FAISS vector stores + persistence
│   └── document_upload.py      # File parsing, chunking, embedding
├── tools/
│   ├── common_tools.py         # Description helper
│   └── graph_tools.py          # Routing and verification logic
└── config/
    ├── settings.py             # Prompt loader
    └── prompts.yaml            # All LLM prompts

streamlit_app/
├── home.py                     # Login page
├── pages/chat.py               # Chat + document upload
└── utils/api_client.py         # Calls the FastAPI backend
```

## Requirements

- Python 3.9 or higher
- MongoDB running locally at `mongodb://localhost:27017`
- An OpenAI API key
- A Tavily API key (for web search)

## Setup

**1. Install dependencies**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Create a `.env` file in the project root**

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
RAG_API_KEY=pick_any_shared_secret     # required, see note below
FAISS_INDEX_PATH=faiss_index           # optional, where indexes are saved
```

`RAG_API_KEY` protects the API. The server rejects every request that does not send this key, so you must set it. Use the same value in the frontend.

**3. Start MongoDB**

Make sure MongoDB is running and reachable at `mongodb://localhost:27017`. Chat history is stored there.

## Running it

**Start the backend:**

```bash
python -m uvicorn src.main:app --reload --port 8000
```

Interactive API docs are then at http://localhost:8000/docs.

**Start the frontend (optional):**

```bash
export RAG_API_KEY=pick_any_shared_secret   # same value as the backend
streamlit run streamlit_app/home.py
```

The UI opens at http://localhost:8501.

> **Note:** The Streamlit login page expects a separate authentication service at `localhost:8080` that is not part of this repository, so login through the UI will not work on its own. To try the system without it, call the backend directly (see below) or replace the login flow with your own. The backend itself runs fine standalone.

## Try it (no frontend needed)

Send the API key on every request, and pick any string as `session_id`. Documents and chat history are tied to that `session_id`.

**Upload a document:**

```bash
curl -X POST http://localhost:8000/rag/documents/upload \
  -H "X-API-KEY: pick_any_shared_secret" \
  -H "X-Session-Id: user123" \
  -H "X-Description: my resume" \
  -F "file=@/path/to/resume.pdf"
```

**Ask a question:**

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "X-API-KEY: pick_any_shared_secret" \
  -H "Content-Type: application/json" \
  -d '{"query": "what was my most recent role?", "session_id": "user123"}'
```

## API reference

### POST /rag/query

Ask a question. Returns the generated answer.

Headers: `X-API-KEY` (required)

Body:

```json
{ "query": "your question", "session_id": "user123" }
```

Response:

```json
{ "result": { "type": "ai", "content": "the answer..." } }
```

### POST /rag/documents/upload

Upload a PDF or TXT file to make it searchable for a session.

Headers:
- `X-API-KEY` (required)
- `X-Session-Id` (required) - which user/session the document belongs to
- `X-Description` (required) - a short description of the document

Form data: `file` - the PDF or TXT file.

Response:

```json
{ "status": true }
```

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | yes | LLM and embeddings |
| `TAVILY_API_KEY` | yes | Web search |
| `RAG_API_KEY` | yes | Shared secret for the API (`X-API-KEY`) |
| `FAISS_INDEX_PATH` | no | Folder for saved vector indexes (default `faiss_index`) |

Prompts live in `src/config/prompts.yaml`: `classify_prompt`, `grading_prompt`, `rewrite_prompt`, `generate_prompt`, and `verify_prompt`.

## Tech stack

| Part | Tool |
|------|------|
| Web API | FastAPI + Uvicorn |
| Workflow | LangGraph / LangChain |
| Vector store | FAISS (saved to disk) |
| Embeddings + LLM | OpenAI (gpt-4o) |
| Web search | Tavily |
| Chat history | MongoDB (via Motor) |
| UI | Streamlit |

## Known limitations

- The Streamlit login depends on an external auth service (`localhost:8080`) not included here.
- The MongoDB connection string is currently hardcoded in `src/db/mongo_client.py`.
- Qdrant support exists in the code but is disabled; FAISS is the active backend.

See `PROJECT_EVALUATION.md` for a fuller list of open items and design notes.

## Contributing

1. Fork the repo and create a branch.
2. Make your changes with clear commit messages.
3. Open a pull request.

Coding conventions are in `CODE_STYLE_GUIDE.md`.

## License

MIT.

## Author

Tejal Palwankar
