"""
Retriever setup and vector store configuration.

Vector stores are scoped per session so that different users' uploaded
documents stay isolated. Each session's FAISS index and its description are
persisted under FAISS_INDEX_PATH/<hashed-session-id>/ so they survive restarts.
"""

import hashlib
import os

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import FAISS

from src.core.config import settings

embeddings = OpenAIEmbeddings()

# Root directory under which per-session FAISS indexes are persisted, so
# uploaded documents survive server restarts. Configurable via FAISS_INDEX_PATH.
FAISS_INDEX_ROOT = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# Per-session in-process vector stores: {session_id: FAISS}
# Keeping one store per session prevents one user's upload from overwriting
# another's index.
_vectorstores: dict[str, FAISS] = {}


def _session_dir(session_id: str) -> str:
    """
    Return the on-disk directory for a session's index.

    The session id is hashed to a fixed-length, filesystem-safe name, which
    also prevents path traversal from arbitrary session id values.

    Args:
        session_id: The session identifier.

    Returns:
        Absolute-or-relative path to this session's index directory.
    """
    safe = hashlib.sha256((session_id or "default").encode("utf-8")).hexdigest()[:32]
    return os.path.join(FAISS_INDEX_ROOT, safe)


def _description_path(session_id: str) -> str:
    """Return the path to the persisted description for a session."""
    return os.path.join(_session_dir(session_id), "description.txt")


def get_vectorstore(session_id: str):
    """
    Return the current in-process vector store for a session (or None).

    Args:
        session_id: The session identifier.

    Returns:
        The session's FAISS vector store, or None if not loaded yet.
    """
    return _vectorstores.get(session_id)


def _load_persisted_vectorstore(session_id: str):
    """
    Load a session's persisted FAISS index from disk, if one exists.

    Args:
        session_id: The session identifier.

    Returns:
        The loaded FAISS vector store, or None if none is found or load fails.
    """
    path = _session_dir(session_id)
    if not os.path.isdir(path):
        return None
    try:
        # allow_dangerous_deserialization is required to load a pickled FAISS
        # index. Safe here because we only load files we wrote ourselves.
        vectorstore = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Loaded persisted FAISS index for session from '{path}'")
        return vectorstore
    except Exception as e:
        print(f"Could not load persisted FAISS index: {e}")
        return None


def retriever_chain(chunks: list[Document], session_id: str, description: str = None):
    """
    Initialize and store documents in a session-scoped FAISS vector database.

    Args:
        chunks: List of document chunks to store.
        session_id: The session the documents belong to.
        description: Optional enhanced description for the retriever tool.

    Returns:
        Boolean indicating success of the operation.
    """
    global _vectorstores

    try:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Store per session so get_retriever(session_id) can access it.
        _vectorstores[session_id] = vectorstore

        # Persist to this session's directory so it survives a restart.
        path = _session_dir(session_id)
        vectorstore.save_local(path)

        # Persist the description alongside the index for this session.
        if description is not None:
            with open(_description_path(session_id), "w", encoding="utf-8") as f:
                f.write(description)

        print("FAISS vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")
        print(f"FAISS index persisted to '{path}'")
        return True
    except Exception as e:
        print(f"Error storing documents in FAISS: {e}")
        return False


def get_retriever(session_id: str):
    """
    Get a retriever tool connected to a session's FAISS vector store.

    Restores the session's persisted index if it isn't loaded in this process
    yet. If the session has no documents, falls back to a dummy store.

    Args:
        session_id: The session identifier.

    Returns:
        A LangChain retriever tool configured for the session's vector store.

    Raises:
        Exception: If vector store initialization fails.
    """
    global _vectorstores

    try:
        # Restore a persisted index for this session if not already in memory.
        if _vectorstores.get(session_id) is None:
            loaded = _load_persisted_vectorstore(session_id)
            if loaded is not None:
                _vectorstores[session_id] = loaded

        vectorstore = _vectorstores.get(session_id)
        if vectorstore is not None:
            retriever = vectorstore.as_retriever()
            print("Using existing FAISS vectorstore with uploaded documents")
        else:
            # No documents uploaded yet for this session; create a dummy store.
            print("No documents uploaded yet, creating dummy vectorstore")
            dummy_doc = Document(
                page_content="No documents have been uploaded yet. Please upload a document first.",
                metadata={"source": "initialization"}
            )
            vectorstore = FAISS.from_documents(
                documents=[dummy_doc],
                embedding=embeddings
            )
            _vectorstores[session_id] = vectorstore
            retriever = vectorstore.as_retriever()

        # Load this session's document description.
        description_path = _description_path(session_id)
        if os.path.exists(description_path):
            with open(description_path, "r", encoding="utf-8") as f:
                description = f.read()
        else:
            description = None

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool **only** to answer questions about: {description}\n"
            "Don't use this tool to answer anything else."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)
