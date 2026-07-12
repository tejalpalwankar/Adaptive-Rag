"""
State model for the graph-based RAG system.
"""

from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict):
    """State schema for the RAG graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    binary_score: Optional[str]
    route: Optional[str]
    latest_query: Optional[str]
    # Number of times the query has been rewritten and re-retrieved. Used to
    # cap the grade -> rewrite -> retrieve loop so it cannot run forever.
    retry_count: Optional[int]
    # Identifies whose documents to retrieve, so users' indexes stay isolated.
    session_id: Optional[str]
    # Number of times the answer has been (re)generated. Used to cap the
    # generate -> verify -> generate self-correction loop.
    verify_count: Optional[int]
    # The retrieved context the answer is generated from. Stored so that
    # regeneration and faithfulness verification reuse the same context even
    # after the generated answer becomes the last message.
    context: Optional[str]