"""MCP server for AI thumbnail generation via Nano Banana Pro.

Exposes one tool — `generate_thumbnail` — and one UI resource. Tool calls
trigger image generation; the widget renders the result inline and lets the
user iterate without going back through the chat.

Run locally (stdio, for `mcp dev` or Claude Desktop):
    python server.py

Run as remote streamable-HTTP MCP (recommended for claude.ai connectors):
    uvicorn server:app --host 0.0.0.0 --port 8003
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
from collections.abc import AsyncIterator
from typing import Annotated

import requests
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import widgets
from nano_banana_pro import create_task, query_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("thumbnails-mcp")

# Optional auth: if THUMBNAILS_MCP_TOKEN is set, require Bearer <token> on
# /mcp. If unset, the server is open (fine for local dev / self-hosting; not
# recommended for any deployment that pays for Kie credits).
_REQUIRED_TOKEN = os.environ.get("THUMBNAILS_MCP_TOKEN", "").strip() or None

# Optional algrow integration — when set, the find_outlier_references tool
# is enabled and the widget shows a "Find outliers" button. The tool calls
# algrow's public viral-videos search endpoint to surface high-outlier-score
# thumbnails for any topic, which users can pick as references for
# generation.
_ALGROW_API_KEY = os.environ.get("ALGROW_API_KEY", "").strip() or None
_ALGROW_API_BASE = (os.environ.get("ALGROW_API_BASE_URL") or "https://api.algrow.online").rstrip("/")


# Pattern → friendly rewrite. Kie surfaces upstream Gemini errors verbatim,
# and most users don't know what "Generative AI Prohibited Use policy" means
# in practice. Map the ones we see in the wild to actionable hints; fall
# through to the raw message for anything we haven't classified yet.
_FRIENDLY_ERROR_PATTERNS = (
    (
        ("prohibited use policy", "filtered out", "violated google"),
        "Gemini's safety filter blocked this prompt — usually because of a named "
        "character, celebrity, brand, or franchise. Try a generic descriptor "
        "(e.g. \"web-slinging hero\" instead of \"Spider-Man\") and regenerate.",
    ),
    (
        ("rate limit", "too many requests", "429"),
        "Kie is rate-limiting us — too many generations in flight. Wait ~30s and try again.",
    ),
    (
        ("insufficient credits", "not enough credits", "credit balance"),
        "Out of Kie credits. Top up at https://kie.ai before generating more thumbnails.",
    ),
    (
        ("service unavailable", "503", "under maintenance"),
        "Kie / Gemini is temporarily unavailable. Try again in a minute.",
    ),
)


# Reference URLs accepted by generate_thumbnail. The widget surface is a
# free-text input where users paste whatever they have — YouTube watch
# links, shorts URLs, raw video IDs, or direct image URLs. We normalize
# server-side so the agent and the widget don't both need URL-parsing
# logic. Plays well with algrow MCP: Claude can call
# search_viral_videos / find_outlier_faceless_channels on the algrow side
# and pass any `thumbnail_url` (or the watch URL the user copies from a
# browser) straight through here.
_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _resolve_reference_url(url: str) -> str | None:
    """Normalize one reference input.

      - YouTube watch / shorts / embed / live / youtu.be URL → hqdefault.jpg
      - Bare 11-char video ID → hqdefault.jpg
      - Already an i.ytimg.com URL → passed through
      - Any other http(s):// URL → passed through (assumed image)
      - Anything else → None (caller drops it)

    hqdefault.jpg is always present for any public video and is a fine
    visual reference for Gemini even though maxresdefault exists for some
    — picking maxresdefault would 404 unpredictably for older / less
    popular videos and break the whole submit.
    """
    s = (url or "").strip()
    if not s:
        return None
    if "i.ytimg.com" in s:
        return s
    m = _YT_ID_RE.search(s)
    if m:
        return f"https://i.ytimg.com/vi/{m.group(1)}/hqdefault.jpg"
    if _BARE_ID_RE.match(s):
        return f"https://i.ytimg.com/vi/{s}/hqdefault.jpg"
    if s.startswith(("http://", "https://")):
        return s
    return None


def _resolve_reference_urls(urls: list[str] | None) -> list[str]:
    """Resolve and dedupe a list of reference inputs. Caps at 8 (Kie's limit)."""
    if not urls:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        # The widget sends one string per line; also accept comma-separated.
        for piece in re.split(r"[\n,]+", raw or ""):
            r = _resolve_reference_url(piece)
            if r and r not in seen:
                seen.add(r)
                out.append(r)
                if len(out) >= 8:
                    return out
    return out


def _friendly_error(raw: str | None) -> str:
    """Rewrite a known upstream error string into something a user can act on.
    Returns the raw string unchanged if no pattern matches."""
    if not raw:
        return "Generation failed (no detail provided by upstream)."
    low = raw.lower()
    for needles, friendly in _FRIENDLY_ERROR_PATTERNS:
        if any(n in low for n in needles):
            return friendly
    return raw


def _build_mcp() -> FastMCP:
    mcp = FastMCP(
        "thumbnails",
        instructions=(
            "AI thumbnail generation via Nano Banana Pro. Call generate_thumbnail "
            "with a descriptive prompt and the host will render an interactive "
            "preview widget inline; users can refine the prompt and regenerate "
            "from the widget without further conversation turns."
        ),
        host="0.0.0.0",
        port=8003,
        stateless_http=True,
    )
    mcp.settings.transport_security = None

    widgets.register(mcp)

    # The image-generation flow is split in two so neither MCP call sits
    # blocking long enough to trip the host's ~2-minute tool-call timeout:
    #
    #   1. generate_thumbnail — submits to Kie and returns the task_id in
    #      <1s. Widget receives {state:"pending", task_id} and starts polling.
    #   2. check_thumbnail_status — looks up a task_id, returns current state.
    #      Widget calls this every few seconds via app.callServerTool until
    #      state is "success" or "fail".
    #
    # Both tools point at the same widget resource so the host renders the
    # same iframe either way.

    @mcp.tool(
        name="generate_thumbnail",
        title="Generate Thumbnail",
        description=(
            "Submit a thumbnail generation request to Nano Banana Pro (Google's "
            "Gemini 2.5 Flash Image). Returns immediately with a task_id; the "
            "inline widget polls for the result automatically. Default aspect "
            "ratio is 16:9 (YouTube). Costs 3 Kie credits per generation; usually "
            "finishes in 30–90 seconds.\n\n"
            "Use this for any request to make/design/draft a thumbnail, cover "
            "image, or hero image — anything the user wants to see and iterate "
            "on visually.\n\n"
            "Reference images work powerfully — pass up to 8 URLs via "
            "`reference_urls` (YouTube watch/shorts/embed/live URLs, raw 11-char "
            "video IDs, i.ytimg.com URLs, and direct image URLs are all accepted; "
            "the server normalizes them). Composes well with the algrow MCP "
            "(https://mcp.algrow.online): call algrow's search_viral_videos or "
            "find_outlier_faceless_channels first, then pass the resulting "
            "`thumbnail_url`s here as references so Gemini mimics what's already "
            "winning in that niche."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def generate_thumbnail_tool(
        prompt: Annotated[str, "Description of the thumbnail to generate. Be specific about subject, style, mood, and composition — Nano Banana Pro renders detail well."],
        aspect_ratio: Annotated[str, "16:9 / 9:16 / 1:1 / 4:5 / 4:3 / 3:2 / 21:9 / auto. Default 16:9 (YouTube thumbnail)."] = "16:9",
        resolution: Annotated[str, "1K / 2K / 4K. Default 2K — plenty for thumbnails and ~4× faster than 4K."] = "2K",
        reference_urls: Annotated[list[str] | None, "Up to 8 reference inputs. Each can be a YouTube URL (watch / shorts / youtu.be / embed / live), a bare 11-char video ID, an i.ytimg.com URL, or any direct image URL. YouTube URLs are auto-resolved to the video's hqdefault thumbnail server-side."] = None,
        reference_images: Annotated[list[str] | None, "Alias for `reference_urls`, accepted for back-compat. Prefer reference_urls in new code."] = None,
    ) -> str:
        import json

        # Merge & normalize whatever the caller provided.
        combined = list(reference_urls or []) + list(reference_images or [])
        resolved_refs = _resolve_reference_urls(combined)

        submit = create_task(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_input=resolved_refs or None,
        )

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "reference_urls": resolved_refs,
        }
        if submit.get("success"):
            payload.update({
                "state": "pending",
                "task_id": submit["task_id"],
                "model": submit.get("model"),
            })
        else:
            payload.update({
                "state": "fail",
                "error": _friendly_error(submit.get("error") or "Failed to submit task"),
                "raw_error": submit.get("error"),
            })
        return json.dumps(payload, default=str)

    @mcp.tool(
        name="check_thumbnail_status",
        title="Check Thumbnail Status",
        description=(
            "Look up the status of a thumbnail generation task by its task_id. "
            "The widget calls this on a loop while a generation is in flight; "
            "Claude shouldn't normally need to call it directly."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def check_thumbnail_status_tool(
        task_id: Annotated[str, "The task_id returned by generate_thumbnail."],
        prompt: Annotated[str, "Original prompt (echoed back to the widget so its form state is preserved across polls)."] = "",
        aspect_ratio: Annotated[str, "Echoed back to the widget."] = "16:9",
        resolution: Annotated[str, "Echoed back to the widget."] = "2K",
        reference_urls: Annotated[list[str] | None, "Echoed back to the widget so the reference thumbnails stay visible during polling."] = None,
    ) -> str:
        import json

        status = query_task(task_id)
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "task_id": task_id,
            "reference_urls": reference_urls or [],
        }
        if not status.get("success"):
            # Transient query error — keep widget in pending so it retries
            payload["state"] = "pending"
            payload["transient_error"] = status.get("error")
            return json.dumps(payload, default=str)

        state = status.get("state")
        if state == "success":
            payload["state"] = "success"
            payload["images"] = status.get("images", [])
            cost_ms = status.get("cost_time")
            if cost_ms is not None:
                payload["cost_time_s"] = round(cost_ms / 1000, 1)
        elif state == "fail":
            payload["state"] = "fail"
            payload["error"] = _friendly_error(status.get("error") or "Generation failed")
            payload["raw_error"] = status.get("error")
            payload["fail_code"] = status.get("fail_code")
        else:
            payload["state"] = "pending"
            payload["upstream_state"] = state
        return json.dumps(payload, default=str)

    # ----- Optional algrow-powered outlier picker --------------------------
    # Only registered when ALGROW_API_KEY is configured. Lets the widget (and
    # Claude, when called from chat) pull high-outlier-score thumbnails for a
    # topic and offer them as references — same UX a competitor product
    # ships, but powered by algrow's 50k+ channel dataset.
    if _ALGROW_API_KEY:
        @mcp.tool(
            name="find_outlier_references",
            title="Find Outlier Thumbnails on a Topic",
            description=(
                "Search algrow's database (50k+ YouTube channels) for high-"
                "outlier-score videos on a topic and return their thumbnails as "
                "candidate references for thumbnail generation. The widget "
                "renders the results as a clickable grid — users tap one or "
                "more to add them to the generation's reference_urls list.\n\n"
                "Outlier score = video.view_count / channel.avg_views_per_video. "
                "2.5× means the video got 2.5× its channel's typical views — a "
                "strong signal that the thumbnail/title combo is working. Returns "
                "sorted by outlier_score descending. Default content_type is "
                "longform; use shorts for shortform-style references."
            ),
            meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
        )
        async def find_outlier_references_tool(
            topic: Annotated[str, "Search topic — the video idea you want references for. Algrow does semantic search so phrases work better than single keywords (e.g. 'Amazon Prime downfall' is better than 'amazon')."],
            content_type: Annotated[str, "longform or shorts. Default longform."] = "longform",
            limit: Annotated[int, "Max thumbnails to return. Default 12, capped at 24."] = 12,
            min_outlier_score: Annotated[float | None, "Optional floor. Only include videos that outperformed their channel average by at least this factor (e.g. 1.5). Leave None to take whatever algrow returns sorted by outlier_score."] = None,
        ) -> str:
            import json

            limit = max(1, min(int(limit or 12), 24))
            if content_type not in ("longform", "shorts"):
                content_type = "longform"

            try:
                resp = requests.post(
                    f"{_ALGROW_API_BASE}/api/viral-videos/search",
                    headers={
                        "Authorization": f"Bearer {_ALGROW_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "q": topic,
                        "content_type": content_type,
                        "sort_by": "outlier_score",
                        "per_page": limit,
                        "min_outlier_score": min_outlier_score,
                    },
                    timeout=15,
                )
                data = resp.json() if resp.content else {}
            except Exception as e:
                logger.warning(f"algrow API call failed: {e}")
                return json.dumps({
                    "view": "outlier_picker",
                    "topic": topic,
                    "outliers": [],
                    "error": _friendly_error(str(e)),
                })

            if resp.status_code != 200 or not data.get("success"):
                err = data.get("error") or f"algrow returned HTTP {resp.status_code}"
                return json.dumps({
                    "view": "outlier_picker",
                    "topic": topic,
                    "outliers": [],
                    "error": _friendly_error(err),
                })

            # Slim each video down to what the widget actually renders. The
            # grid uses thumbnail_url + outlier_score + title; url is for
            # the "open on YouTube" click-through.
            outliers = []
            for v in (data.get("videos") or [])[:limit]:
                outliers.append({
                    "video_id": v.get("video_id"),
                    "title": v.get("title") or "",
                    "thumbnail_url": v.get("thumbnail_url"),
                    "outlier_score": v.get("outlier_score"),
                    "channel_name": v.get("channel_name") or "",
                    "view_count": v.get("view_count"),
                    "url": v.get("url") or (
                        f"https://www.youtube.com/watch?v={v.get('video_id')}"
                        if v.get("video_id") else None
                    ),
                })

            return json.dumps({
                "view": "outlier_picker",
                "topic": topic,
                "content_type": content_type,
                "outliers": outliers,
            }, default=str)

    return mcp


# ---------------------------------------------------------------------------
# Auth middleware — optional bearer
# ---------------------------------------------------------------------------


CORS_ALLOWED_ORIGINS = {
    "https://chatgpt.com",
    "https://chat.openai.com",
    "https://claude.ai",
}


class CorsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode()
        if origin not in CORS_ALLOWED_ORIGINS:
            await self.app(scope, receive, send)
            return
        if scope.get("method", "") == "OPTIONS":
            await Response(
                status_code=204,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                    "Access-Control-Max-Age": "86400",
                },
            )(scope, receive, send)
            return

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers", []))
                hdrs.append((b"access-control-allow-origin", origin.encode()))
                message = {**message, "headers": hdrs}
            await send(message)

        await self.app(scope, receive, send_with_cors)


class AuthMiddleware:
    """If THUMBNAILS_MCP_TOKEN is set, require it on /mcp via EITHER:

      - HTTP header:  Authorization: Bearer <token>
      - Query string: ?key=<token>

    The query-string form exists because claude.ai's "Add custom connector"
    dialog only accepts a URL + OAuth (no Authorization header field). Pasting
    `https://host/mcp?key=<token>` lets the connector authenticate without
    standing up a full OAuth flow. Header form is preferred for everything
    else (curl, Claude Desktop via mcp-remote --header, etc.) because query
    strings get logged.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _REQUIRED_TOKEN:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith("/mcp"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        header_token = auth[7:].strip() if auth.startswith("Bearer ") else None

        # Parse ?key=… from the raw query string (ASGI exposes it as bytes).
        query = scope.get("query_string", b"").decode()
        from urllib.parse import parse_qs
        query_token = (parse_qs(query).get("key") or [None])[0]

        supplied = header_token or query_token
        if supplied != _REQUIRED_TOKEN:
            await JSONResponse(
                {"error": "unauthorized"}, status_code=401
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# App composition
# ---------------------------------------------------------------------------

_mcp = _build_mcp()
_http_app = _mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with _mcp.session_manager.run():
        yield


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok", "service": "thumbnails-mcp"})


app = Starlette(
    routes=[Route("/health", health)],
    middleware=[Middleware(CorsMiddleware), Middleware(AuthMiddleware)],
    lifespan=lifespan,
)
app.mount("/", _http_app)


def main():
    # stdio entry — for Claude Desktop config or `mcp dev server.py`
    _mcp.run()


if __name__ == "__main__":
    main()
