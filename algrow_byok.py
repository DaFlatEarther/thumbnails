"""BYOK lookup against algora's /api/byok/keys endpoint.

Mirrors phase-1's storage design: users add their Gemini + Kie keys ONCE
at https://algora.online/settings (the "Model Keys" tab). Both the web
tool and this MCP server resolve the per-user keys via a single endpoint,
so the user never has to paste them into Claude Desktop's connector config.

How it composes with the rest of the request:

  1. AuthMiddleware (server.py) extracts the user's algora API key from
     Bearer header → stashes it in auth_ctx.current_api_key.
  2. Any tool that needs upstream credentials calls `get_keys()` here.
  3. get_keys() reads the current API key, calls `GET /api/byok/keys`
     with it, returns {"gemini": "...", "kie": "..."} (or {} when the
     user hasn't configured any).
  4. Result is cached per algora-API-key for `_TTL_SECONDS` so we don't
     re-hit algora on every tool call within a conversation.

Public surface:
    get_keys() -> dict[str, str]
        {"gemini": "...", "kie": "..."}-shaped dict. Missing providers
        absent from the dict (NOT mapped to None). Caller treats absence
        as "fall back to env".

    get_key(provider) -> str | None
        Convenience for one specific provider.

    invalidate_cache(api_key=None)
        Drop cached lookup for a specific key (or all keys when None).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from auth_ctx import get_current_api_key

logger = logging.getLogger("thumbnails-mcp.byok")

_ALGROW_API_BASE = (os.environ.get("ALGROW_API_BASE_URL") or "https://api.algrow.online").rstrip("/")
_TTL_SECONDS = 300   # 5 min — covers a typical conversation without going stale
_TIMEOUT_S = 8       # short timeout; missing keys is fine, hanging the tool isn't

# Cache: api_key → (expires_at_monotonic, {"gemini": "...", "kie": "..."})
_cache: dict[str, tuple[float, dict[str, str]]] = {}


def _fetch(api_key: str) -> dict[str, str]:
    """One call to algora. Returns {} on any failure (network, auth, parse)
    so the caller transparently falls through to env keys."""
    try:
        resp = requests.get(
            f"{_ALGROW_API_BASE}/api/byok/keys",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT_S,
        )
    except requests.exceptions.RequestException as e:
        logger.warning(f"byok lookup network error: {str(e)[:140]}")
        return {}
    if resp.status_code != 200:
        # 401 means the algora key itself is bad — caller will hit the same
        # 401 on its first real algora request, so just log + return empty.
        logger.warning(f"byok lookup HTTP {resp.status_code}: {resp.text[:160]}")
        return {}
    try:
        data = resp.json()
    except Exception as e:
        logger.warning(f"byok lookup non-JSON response: {e}")
        return {}
    if not data.get("success"):
        return {}
    keys = data.get("keys") or {}
    # Defensive: only keep entries that look like non-empty strings.
    return {k: v for k, v in keys.items() if isinstance(v, str) and v.strip()}


def get_keys() -> dict[str, str]:
    """Return the current user's BYOK keys (Gemini + Kie when configured).

    Reads the per-request algora API key from auth_ctx. Returns an empty
    dict when no key is present (which shouldn't happen on /mcp paths —
    AuthMiddleware 401s those — but is safe for any other caller)."""
    api_key = get_current_api_key()
    if not api_key:
        return {}
    now = time.monotonic()
    cached = _cache.get(api_key)
    if cached and cached[0] > now:
        return cached[1]
    fresh = _fetch(api_key)
    _cache[api_key] = (now + _TTL_SECONDS, fresh)
    return fresh


def get_key(provider: str) -> str | None:
    """Convenience: return one provider's key, or None when not configured."""
    return get_keys().get(provider) or None


def invalidate_cache(api_key: str | None = None) -> None:
    """Drop a specific api_key (or all) from the cache. Used by tests + by
    handlers that just saw a 401 from algora and want a fresh lookup on
    retry."""
    if api_key is None:
        _cache.clear()
    else:
        _cache.pop(api_key, None)
