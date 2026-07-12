"""
API routes for RAG operations.
"""

import asyncio

from fastapi import APIRouter, UploadFile, File, Header, Depends
from langchain_core.messages import HumanMessage, AIMessage

from src.core.security import verify_api_key
from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


@router.post("/rag/query", dependencies=[Depends(verify_api_key)])
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.

    Args:
        req: The query request containing query text and session_id.

    Returns:
        The generated response from the RAG pipeline.
    """
    #chat_history=ChatInMemoryHistory.get_session_history(req.token)
    chat_history = ChatHistory.get_session_history(req.session_id)
    await chat_history.add_message(HumanMessage(content=req.query))

    # Fetch full history
    messages = await chat_history.get_messages()
    # The graph makes blocking LLM/retrieval calls; run it off the event loop
    # so it doesn't stall other requests.
    result = await asyncio.to_thread(builder.invoke, {
        "messages": messages,
        "session_id": req.session_id
    })
    output_text = result["messages"][-1].content

    # Save assistant message
    await chat_history.add_message(AIMessage(content=output_text))

    return {"result": result["messages"][-1]}


@router.post("/rag/documents/upload", dependencies=[Depends(verify_api_key)])
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
    session_id: str = Header(..., alias="X-Session-Id")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.
        session_id: Session identifier provided via header, used to scope the
            uploaded documents to the uploading user.

    Returns:
        Upload status.
    """
    # Reading the file and building embeddings is blocking and CPU-bound; run
    # it off the event loop so uploads don't stall other requests.
    status_upload = await asyncio.to_thread(
        documents, description, file, session_id
    )
    return {"status": status_upload}

