"""Per-request auth + BYOK context.

Mirrors the algrow-mcp/algrow_mcp/client_proxy.py pattern: every HTTP
request to /mcp carries `Authorization: Bearer algrow_<key>`. The auth
middleware in server.py extracts the key and parks it in a ContextVar
so tool handlers can read it (for upstream algora calls + per-user
BYOK lookups) without having to thread an `api_key=` arg through every
function in the codebase.

Public surface:
    current_api_key : ContextVar[str | None]
        Set by AuthMiddleware before the request body runs. Read inside
        tool handlers.

    get_current_api_key() -> str | None
        Convenience wrapper that returns None when nothing's set (instead
        of raising LookupError).

    set_current_api_key(key) -> Token
        Used by tests + the middleware. Returns a Token you can pass to
        current_api_key.reset() if you need to.
"""
from __future__ import annotations

import contextvars

current_api_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "thumbnails_mcp_api_key", default=None
)


def get_current_api_key() -> str | None:
    """Return the per-request algrow API key, or None when not set."""
    try:
        return current_api_key.get()
    except LookupError:
        return None


def set_current_api_key(key: str | None) -> contextvars.Token:
    """Set the per-request API key. Returns a reset token."""
    return current_api_key.set(key)
