"""
API authentication for the RAG service.

Requests must present a shared secret in the ``X-API-KEY`` header. The key is
compared in constant time, and the check fails closed: if no key is configured
on the server, all requests are rejected rather than allowed through.
"""

import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.core.config import settings

API_KEY_NAME = "X-API-KEY"

# auto_error=False so we can return a consistent 401 for both missing and
# invalid keys instead of FastAPI's default 403 for a missing header.
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def is_valid_api_key(provided: str) -> bool:
    """
    Check a provided API key against the configured one, in constant time.

    Args:
        provided: The key supplied by the caller (may be None).

    Returns:
        True only if a key is configured and the provided key matches it.
    """
    expected = settings.RAG_API_KEY
    if not expected or not provided:
        # Fail closed when unconfigured or when no key is supplied.
        return False
    return hmac.compare_digest(provided, expected)


def verify_api_key(api_key: str = Security(api_key_header)) -> None:
    """
    FastAPI dependency that rejects requests without a valid API key.

    Raises:
        HTTPException: 401 if the key is missing or invalid.
    """
    if not is_valid_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
