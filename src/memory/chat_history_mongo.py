"""
Chat history storage using MongoDB backend.
"""

from datetime import datetime
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from src.db.mongo_client import db

collection = db["chat_history"]

# Default number of most-recent messages to load per session. Bounds the tokens,
# memory, and DB fetch per request instead of sending the whole conversation.
DEFAULT_HISTORY_LIMIT = 20


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat history backed by MongoDB."""

    def __init__(self, session_id: str):
        """
        Initialize chat history for a session.

        Args:
            session_id: Unique session identifier.
        """
        self.session_id = session_id

    async def add_message(self, message: BaseMessage) -> None:
        """
        Save a message to MongoDB.

        Args:
            message: The message to save.
        """
        await collection.insert_one({
            "session_id": self.session_id,
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "timestamp": datetime.utcnow(),
        })

    async def get_messages(self, limit: int = DEFAULT_HISTORY_LIMIT) -> List[BaseMessage]:
        """
        Load the most recent messages for a session from MongoDB.

        Only the last ``limit`` messages are fetched (newest first, then
        reversed to chronological order) so token, memory, and DB cost stay
        bounded as conversations grow.

        Args:
            limit: Maximum number of most-recent messages to return.

        Returns:
            List of the most recent messages in chronological order.
        """
        from langchain_core.messages import messages_from_dict

        # Fetch the newest `limit` messages, then restore chronological order.
        cursor = collection.find({"session_id": self.session_id}).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        docs.reverse()

        # Convert to BaseMessage objects
        return messages_from_dict([
            {
                "type": d["type"],
                "data": {
                    "content": d["content"],
                    "additional_kwargs": d.get("additional_kwargs", {}),
                }
            }
            for d in docs
        ])

    async def clear(self) -> None:
        """Delete all messages for a session."""
        await collection.delete_many({"session_id": self.session_id})


class ChatHistory:
    """Factory for MongoDB-backed chat history."""

    @classmethod
    def get_session_history(
        cls,
        session_id: str,
        config: dict = None
    ) -> MongoDBChatMessageHistory:
        """
        Get or create chat history for a session.

        Args:
            session_id: Unique session identifier.
            config: Optional configuration dictionary.

        Returns:
            MongoDBChatMessageHistory instance for the session.
        """
        return MongoDBChatMessageHistory(session_id)
