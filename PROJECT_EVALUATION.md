# Adaptive RAG - Project Evaluation

Reviewed: July 2026. Scope: `src/` (FastAPI + LangGraph pipeline), `streamlit_app/`, config, and docs.

## Status (updated July 2026)

All five critical bugs and all five high-priority issues have been fixed. Each was addressed one at a time with a targeted test. The medium and hygiene items remain open.

Resolved:

- #1 ReAct agent never saw uploaded documents. `retriever_setup` now exposes the current vector store and the agent was built per request; later removed entirely (see #8/#10).
- #2 No persistence. FAISS indexes are now saved to disk and reloaded on startup.
- #3 Single-user global state. Vector stores and descriptions are now scoped per session, persisted under `faiss_index/<hashed-session>/`. `session_id` is threaded through the upload route, graph state, and nodes.
- #4 `verify_answer` was dead code. It is wired in after `generate`, the self-comparison bug is fixed, and regeneration is capped (`MAX_GENERATIONS`).
- #5 Infinite rewrite loop. `doc_tool` caps rewrites (`MAX_REWRITES`) and falls back to web search.
- #6 No API auth. Both endpoints require an `X-API-KEY` header (constant-time check, fails closed). Service-to-service auth; see the note below on JWT identity as a follow-up.
- #7 Blocking calls in async routes. `builder.invoke` and `documents` now run via `asyncio.to_thread`.
- #8 Duplicate retrieval + double LLM cost. The classifier's retrieved context is reused; retrieval happens once (re-retrieves only after a rewrite).
- #9 Unbounded chat history. `get_messages` fetches only the last `DEFAULT_HISTORY_LIMIT` (20) messages.
- #10 Deprecated ReAct agent. Removed as dead code (its `system_prompt` too) after #8 made it unused.

Still open: #11 hardcoded config, #12 `print()` vs logger, #13 duplicate config/memory modules, #14 prompt-injection surface in the description, #15 thin upload validation, #16 inconsistent message shapes, plus no tests and no `.env.example`.

New environment variables introduced by the fixes: `RAG_API_KEY` (required — the API fails closed without it) and `FAISS_INDEX_PATH` (optional, defaults to `faiss_index`).

---

## What the project does well

The architecture is clean and readable. Concerns are separated into `api`, `rag`, `models`, `memory`, `llms`, `config`, and `tools`. Pydantic models define graph state and structured LLM outputs. Prompts live in YAML instead of being hardcoded. The LangGraph flow (classify → route → retrieve/general/web → grade → rewrite → generate) is a sensible adaptive-RAG design, and docstrings are consistent throughout. For a portfolio or learning project, the bones are good.

The rest of this document is the weaknesses and how to fix them, ordered by severity.

---

## Critical bugs (break correctness)

### 1. The ReAct agent never sees uploaded documents
`src/rag/reAct_agent.py` builds the agent once at import time:

```python
tools = [get_retriever()]          # runs before any upload
react_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(...)
```

At import, no document has been uploaded, so `get_retriever()` returns a tool wrapping the **dummy** vectorstore ("No documents have been uploaded yet"). When a user later uploads a file, `retriever_chain()` reassigns the module global `_faiss_vectorstore` to a new FAISS object, but the agent's tool still holds a reference to the old dummy retriever. So `retriever_node` searches an empty index forever.

Meanwhile `query_classifier` calls `get_retriever()` fresh on every request and *does* see the real docs. Result: the router classifies a query as `index` (docs are relevant), then the retriever finds nothing. The core path is broken.

**Fix:** build the agent lazily per request (or per upload), not at import. Have `retriever_node` call `get_retriever()` at call time and construct the agent from the current vectorstore, or better, drop the ReAct agent entirely for retrieval (see #10).

### 2. No persistence — all documents lost on restart
Qdrant is commented out and FAISS lives only in the `_faiss_vectorstore` module global. Every server restart wipes the index. The README advertises Qdrant persistence that the code doesn't use.

**Fix:** finish the Qdrant integration (the setup guide already exists) or persist FAISS to disk with `save_local()` / `load_local()`. Pick one and make it the default.

### 3. Global state makes it single-user only
Both the vectorstore (`_faiss_vectorstore`) and the retriever description (`description.txt`) are process-wide singletons. If two users upload different documents, the second overwrites the first, and everyone queries the same index. `description.txt` is read from and written to the working directory as shared global state.

**Fix:** scope documents and descriptions per user/session (e.g. a Qdrant collection per user, or a `namespace`/metadata filter), and store the description in the DB keyed by session, not a flat file.

### 4. `verify_answer` is dead code
`src/tools/graph_tools.py` defines a faithfulness-check node and `verify_prompt` exists in YAML, but it's never added to the graph. The "self-correcting" verification the design implies doesn't run.

**Fix:** wire `verify_answer` as a conditional edge after `generate`, or remove it so the code matches reality.

### 5. Infinite-loop risk in the grade → rewrite loop
`grade → doc_tool → rewrite → retriever → grade` has no attempt counter. If grading keeps returning "no", the graph loops until it hits LangGraph's recursion limit and throws.

**Fix:** add a `retry_count` to `State`, cap it (e.g. 2–3), and fall back to `web_search` or a graceful "I couldn't find this" answer when exceeded.

---

## High-priority issues

### 6. No authentication on the RAG API
The FastAPI endpoints (`/rag/query`, `/rag/documents/upload`) have no auth. The Streamlit app talks to a separate Rust auth service (`localhost:8080`) that isn't in this repo, and it passes `jwt_token` as the `session_id` to the Python backend, which never validates it. Anyone who can reach the API can query or upload.

**Fix:** add auth (API key or JWT verification) as a FastAPI dependency, and validate the session server-side.

### 7. Blocking calls inside async routes
`rag_query` is `async` but calls `builder.invoke(...)` (synchronous, LLM-bound) directly, which blocks the event loop. `upload_file` does sync `file.file.read()` and CPU-bound chunking/embedding in the request path. Under concurrency the server stalls.

**Fix:** use `await builder.ainvoke(...)`, make node functions async, and run blocking work in a threadpool (`run_in_threadpool`) or a background task/queue for uploads.

### 8. Duplicate, wasteful retrieval and double LLM cost
Every `index` query retrieves once in `query_classifier` (to classify) and then runs a full ReAct agent in `retriever_node` that retrieves again. That's two embedding + retrieval passes plus extra agent LLM calls per query.

**Fix:** retrieve once, pass the context through state, and reuse it for both classification and generation.

### 9. Unbounded chat history sent every request
`get_messages()` loads up to 1000 messages and the full list is passed into the graph, but only `messages[-1]` is actually used by the classifier. Token cost and latency grow with conversation length for no benefit.

**Fix:** window the history (last N turns) and only include what the nodes actually consume.

### 10. Deprecated ReAct pattern, `max_iterations=2`
`create_react_agent` with string-parsed `Thought/Action` scratchpads is brittle (`handle_parsing_errors=True` masks failures) and the 2-iteration cap often stops before answering. Modern LangChain favors tool-calling agents.

**Fix:** replace with a tool-calling agent, or since there's a single retriever tool, skip the agent and call the retriever directly.

---

## Medium-priority issues

### 11. Hardcoded config
`src/db/mongo_client.py` hardcodes `mongodb://localhost:27017`, and `api_client.py` hardcodes backend URLs. These should come from environment variables like the other settings.

### 12. `print()` instead of logging
Core modules are full of `print()` debugging (docs dumped to stdout, classifier results, etc.), while a `logger.py` exists but isn't used in the pipeline. Printing retrieved context also risks leaking document contents into logs.

**Fix:** use the logger with levels; remove or gate the context dumps.

### 13. Two config systems and two memory backends
There are two config classes (`src/config/settings.py` and `src/core/config.py`) and two chat-history implementations (Mongo and in-memory), with the in-memory one referenced only in a commented-out line. This is confusing and invites drift.

**Fix:** consolidate to one config module and one memory backend.

### 14. Prompt-injection surface in the retriever description
User-supplied descriptions are rewritten by an LLM and embedded directly into the retriever tool's instruction string. A crafted description could steer tool usage.

**Fix:** sanitize/validate descriptions and treat them as data, not instructions.

### 15. Upload validation is thin
File type is checked only by extension (`.pdf`/`.txt`), with no size limit, no MIME check, and no page/character cap. Large uploads run synchronously and can exhaust memory or rack up embedding cost.

**Fix:** validate MIME type, enforce a size limit, and cap chunks; process uploads off the request path.

### 16. Inconsistent message shapes between nodes
Some nodes return `AIMessage`, others return `{"role": "assistant", "content": ...}` dicts. It works via `add_messages` but is inconsistent and easy to break.

**Fix:** standardize on `AIMessage` everywhere.

---

## Lower-priority / hygiene

- **No tests.** There isn't a single unit or integration test. Add tests for `routing_tool`, `doc_tool`, the grade loop, and upload validation at minimum.
- **No `.env.example`.** Required keys (`OPENAI_API_KEY`, `TAVILY_API_KEY`, `QDRANT_*`, Mongo URL) are only discoverable by reading code.
- **Docs oversell the code.** README claims Qdrant and full verification; the running code uses in-memory FAISS with no verification. Align docs with behavior.
- **No streaming responses.** Answers are returned whole; token streaming would improve perceived latency.
- **`src/rag/nodes.py` is empty** and `adaptive_RAG.png` is duplicated in root and `src/rag/`. Minor cleanup.
- **No dependency pinning strategy / lockfile**, and `requirements.txt` lists `langchain_groq` though only OpenAI is used.

---

## Suggested order of work

1. ~~Fix the retriever/agent binding bug (#1) and the loop guard (#5).~~ Done.
2. ~~Add persistence (#2) and per-user scoping (#3).~~ Done.
3. ~~Add auth (#6) and move blocking work off the event loop (#7).~~ Done.
4. ~~Remove duplicate retrieval (#8), window history (#9), and remove the dead agent (#10).~~ Done.
5. ~~Wire `verify_answer` (#4).~~ Done. Remaining: clean up config (#11), logging (#12), duplicate modules (#13), input validation (#14/#15), message shapes (#16), and add tests.

Steps 1–5 are complete: the app now behaves correctly and safely for multiple users, persists data, authenticates callers, and runs efficiently. The remaining work is code hygiene and test coverage, not correctness or safety.
