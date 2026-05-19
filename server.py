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

# Optional Gemini vision — when set, references get auto-analyzed into a
# structured JSON breakdown (composition, palette, lighting, text style,
# etc.) that gets folded into the generation prompt. Massive quality win:
# Gemini's image-gen alone treats reference_urls as loose visual hints;
# spelling out the design rules explicitly forces it to keep the layout
# and only swap the subject content.
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or None
_GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash").strip()
_GEMINI_VISION_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{_GEMINI_VISION_MODEL}:generateContent"
)


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


# ---------------------------------------------------------------------------
# Vision-to-JSON: reference image → structured design breakdown
# ---------------------------------------------------------------------------

_VISION_TO_JSON_PROMPT = """Analyze this YouTube thumbnail and return a structured JSON breakdown describing what makes it work visually. Be specific and concrete — every field should help someone recreate a thumbnail in the same style with different content.

Output ONLY a JSON object matching this schema (no markdown wrapping, no commentary):

{
  "subject": {
    "type": "person | object | scene | text-only | mixed",
    "description": "1 sentence: what's the main subject",
    "position": "left | center | right | full-frame",
    "size_proportion": "small | medium | large | dominant",
    "expression_or_state": "for person: emotion/expression; for object: condition/treatment"
  },
  "composition": {
    "layout": "1 sentence describing element placement (e.g. 'person left occupying 35%, text flanking right, prop bottom right')",
    "depth_style": "flat | layered_cutout | photographic_depth | composite",
    "balance": "symmetric | asymmetric_left | asymmetric_right | dynamic"
  },
  "text_overlay": {
    "present": true,
    "content": "exact visible text",
    "font_style": "bold_sans | gothic_serif | condensed_display | handwritten | tech_mono | other",
    "treatment": "shadow | stroke | glow | gradient | none",
    "position": "top | bottom | left | right | flanking_subject | center",
    "color": "primary text color"
  },
  "color_palette": {
    "background_color": "dominant bg color",
    "primary_accent": "main accent color",
    "secondary_accent": "secondary accent or null",
    "mood": "warm | cool | dark | bright | high_contrast | muted | neon"
  },
  "lighting": {
    "style": "cinematic | studio | natural | spotlight | flat | dramatic",
    "direction": "top | front | back | side | rim | ambient",
    "contrast": "low | medium | high | extreme"
  },
  "background": {
    "type": "solid | gradient | scene | abstract | composite",
    "description": "1 sentence",
    "complexity": "minimal | moderate | busy"
  },
  "style_genre": "mr_beast | unlayered_creator | editorial_explainer | gaming | luxury_brand | horror | educational | brutalist | tech_review | challenge_series | other",
  "emotional_cue": "shock | joy | fear | curiosity | intensity | calm | mystery | urgency | conflict",
  "visual_devices": ["arrow", "circle", "strikethrough", "before_after", "speech_bubble", "money_burst"],
  "key_props": ["list of significant visible objects"],
  "what_makes_it_work": "1-2 sentences on the design technique that makes this thumbnail effective"
}"""


def _analyze_image_via_gemini(image_url: str) -> tuple[dict | None, str | None]:
    """Fetch image bytes, call Gemini vision with the schema prompt, return
    structured dict. Returns (analysis, error) — analysis is None on error.

    Used by the standalone analyze_thumbnail tool and folded into
    generate_thumbnail when analyze_references=True.
    """
    if not _GEMINI_API_KEY:
        return None, "Vision analysis disabled (GEMINI_API_KEY not set)."

    try:
        img_resp = requests.get(image_url, timeout=20)
        img_resp.raise_for_status()
        img_bytes = img_resp.content
        mime_type = (img_resp.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
    except Exception as e:
        return None, f"Couldn't fetch reference image: {str(e)[:140]}"

    import base64 as _b64
    import json as _json

    try:
        resp = requests.post(
            f"{_GEMINI_VISION_URL}?key={_GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"text": _VISION_TO_JSON_PROMPT},
                        {"inline_data": {"mime_type": mime_type, "data": _b64.b64encode(img_bytes).decode("ascii")}},
                    ],
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.2,
                },
            },
            timeout=30,
        )
    except Exception as e:
        return None, f"Gemini vision request failed: {str(e)[:140]}"

    if resp.status_code != 200:
        return None, f"Gemini vision HTTP {resp.status_code}: {resp.text[:200]}"

    try:
        body = resp.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        analysis = _json.loads(text)
    except Exception as e:
        return None, f"Couldn't parse Gemini response: {str(e)[:140]}"

    return analysis, None


def _analyze_references_parallel(urls: list[str], max_n: int = 3
                                  ) -> tuple[list[dict], list[str]]:
    """Run vision analysis on up to `max_n` references in parallel.
    Returns (analyses, errors) — order matches input order, errors is a list
    of any failures (skipped on success). Each analysis adds 3–10s on its
    own; parallel keeps total wall time close to slowest single call.
    """
    if not urls or not _GEMINI_API_KEY:
        return [], []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    picks = urls[:max_n]
    results: dict[int, tuple[dict | None, str | None]] = {}
    with ThreadPoolExecutor(max_workers=len(picks)) as pool:
        futures = {pool.submit(_analyze_image_via_gemini, u): i for i, u in enumerate(picks)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = (None, f"analysis crashed: {str(e)[:120]}")
    analyses, errors = [], []
    for i in range(len(picks)):
        a, err = results.get(i, (None, "missing"))
        if a is not None:
            analyses.append(a)
        elif err:
            errors.append(err)
    return analyses, errors


def _build_reference_directives(analyses: list[dict]) -> str:
    """Distill the vision JSON into a few crisp directive bullets that the
    image-gen model will follow. If multiple analyses, take the first (the
    'primary' reference) and note that more were considered — averaging
    palettes / layouts across multiple refs blurs everything to mush, and
    the user picked an explicit primary by selecting it first.
    """
    if not analyses:
        return ""
    a = analyses[0]
    comp = a.get("composition") or {}
    txt = a.get("text_overlay") or {}
    palette = a.get("color_palette") or {}
    light = a.get("lighting") or {}
    bg = a.get("background") or {}

    devices = a.get("visual_devices") or []
    props = a.get("key_props") or []

    bullets = [
        "TEMPLATE TO MATCH (extracted from the user's chosen reference thumbnail — KEEP these design rules, only swap the subject content):",
        f"• Composition: {comp.get('layout', 'n/a')} (depth: {comp.get('depth_style', 'n/a')}, balance: {comp.get('balance', 'n/a')})",
        f"• Color palette: background {palette.get('background_color', 'n/a')}, primary accent {palette.get('primary_accent', 'n/a')}"
        + (f", secondary accent {palette['secondary_accent']}" if palette.get('secondary_accent') else "")
        + f"; mood {palette.get('mood', 'n/a')}",
        f"• Lighting: {light.get('style', 'n/a')}, {light.get('direction', 'n/a')} direction, {light.get('contrast', 'n/a')} contrast",
        f"• Text treatment: {txt.get('font_style', 'n/a')} font, {txt.get('treatment', 'none')} treatment, positioned {txt.get('position', 'n/a')}, color {txt.get('color', 'n/a')}",
        f"• Background: {bg.get('type', 'n/a')} ({bg.get('complexity', 'n/a')} complexity) — {bg.get('description', 'n/a')}",
        f"• Style genre: {a.get('style_genre', 'n/a')}",
        f"• Emotional cue: {a.get('emotional_cue', 'n/a')}",
    ]
    if devices:
        bullets.append(f"• Visual devices to reuse where natural: {', '.join(devices[:5])}")
    insight = a.get("what_makes_it_work")
    if insight:
        bullets.append(f"• Why this template works: {insight}")
    bullets.append(
        "CRITICAL — REFERENCE USAGE: The reference image attached to this "
        "request is a DESIGN GUIDE ONLY. Apply its composition, palette, "
        "lighting style, text treatment, and style genre to the user's new "
        "subject. NEVER embed, frame, or include the reference image itself "
        "as a visual element in the output — no picture-in-picture, no inset "
        "card, no illustration overlay in a corner, no comic-panel-style "
        "sub-frame, no \"before/after\" duplication of the reference. The "
        "generated thumbnail must be ONE single new image of the user's "
        "subject, rendered in the reference's visual style. The reference's "
        "specific subject/props/text content do NOT carry over — only the "
        "design system does."
    )
    if len(analyses) > 1:
        bullets.append(
            f"(Analyzed {len(analyses)} references total; primary template above. Other refs inform palette breadth but the primary's layout/composition wins.)"
        )
    return "\n".join(bullets)


def _fetch_outliers_from_algrow(topic: str, content_type: str = "longform",
                                 limit: int = 12, min_outlier_score: float = 2.0
                                 ) -> tuple[list[dict], str | None]:
    """Server-internal call to algrow's viral-videos search.

    Returns (outliers, error). On error, outliers is [] and error has the
    user-facing message. Same params + same shape as find_outlier_references
    so the two paths produce identical widget state.
    """
    if not _ALGROW_API_KEY:
        return [], "Algrow integration not configured (set ALGROW_API_KEY)."
    try:
        resp = requests.post(
            f"{_ALGROW_API_BASE}/api/viral-videos/search",
            headers={
                "Authorization": f"Bearer {_ALGROW_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "q": topic,
                "content_type": content_type if content_type in ("longform", "shorts") else "longform",
                "sort_by": "similarity",
                "per_page": max(1, min(int(limit), 24)),
                "min_outlier_score": min_outlier_score,
            },
            timeout=90,
        )
        data = resp.json() if resp.content else {}
    except requests.exceptions.Timeout:
        return [], "Algrow timed out (>90s). Try a more specific topic."
    except Exception as e:
        logger.warning(f"algrow API call failed: {e}")
        return [], _friendly_error(str(e))

    if resp.status_code != 200 or not data.get("success"):
        return [], _friendly_error(data.get("error") or f"algrow returned HTTP {resp.status_code}")

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
    return outliers, None


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


# Composition presets — prepended to the user's prompt before submit so the
# output follows YouTube-thumbnail design conventions even when the prompt
# itself is bare (e.g. "make me a thumbnail about Amazon Prime downfall").
# Tuned from hand-eval against high-CTR Unlayered / Pitagoras / MrBeast-style
# thumbnails. Faceless preset will be added once we hand-eval that family.
_PERSON_FOCAL_STYLE = (
    "COMPOSITION (apply strictly — this is a YouTube thumbnail for a video "
    "featuring a real on-camera creator):\n"
    "• Subject: ONE main person on the LEFT 30–40% of the frame. Face is "
    "  large, well-lit, expressive — close-up to medium shot, eye contact "
    "  with viewer or a strong emotional cue (joy, shock, intensity, fear). "
    "  Never a neutral expression. Use a cutout/photoshop-cut feel with a "
    "  subtle drop shadow so the person reads as layered over the scene.\n"
    "• Background: supports the topic — relevant setting, object, or scene "
    "  — fills the RIGHT 60–70% of the frame. Clearly contrasted from the "
    "  subject (different colors / depth / lighting).\n"
    "• Text overlay: bold heavy sans-serif, MAX 2–3 short words, very high "
    "  contrast (white with hard shadow or stroke, or saturated yellow on "
    "  dark). Positioned to NOT cover the face. Often paired with a chunky "
    "  arrow pointing at the secondary subject.\n"
    "• Props: small in-frame object (phone, gadget, food, weapon) held by "
    "  the subject reinforces the narrative — include one when natural.\n"
    "• Lighting: cinematic, dramatic, clear subject-background separation. "
    "  Rim light on the subject is great.\n"
    "• Color: high saturation, one dominant accent color tied to the topic "
    "  (e.g. red for danger, blue for tech, green for money/nature).\n"
    "• Format: 16:9, sharp focus, no motion blur on the face.\n"
    "AVOID: small text, neutral expressions, busy backgrounds, low contrast, "
    "muted/desaturated palettes, generic stock-photo poses, multiple "
    "competing focal points, watermarks, logos."
)

_FACELESS_STYLE = (
    "COMPOSITION (apply strictly — this is a YouTube thumbnail for a "
    "FACELESS video, no on-camera creator):\n"
    "• Subject: ONE dominant hero object, scene, or prop, usually CENTERED "
    "  (not off to the side). The object IS the story — choose something "
    "  symbolically loaded or visually striking (e.g. skeleton hands holding "
    "  a ring box for a 'cost of love' video; a single highlighted product "
    "  on a pile of competitors; a transformation/before-after visual).\n"
    "• Narrative via metaphor or juxtaposition — pair the hero with a "
    "  supporting visual that creates the curiosity gap (e.g. one bright "
    "  object atop a pile of defeated rivals; a small creature swarmed by "
    "  many; a clean futuristic product against a contrasting map / route / "
    "  context). With no face to carry emotion, the SETUP carries it.\n"
    "• Text overlay: LARGER and more designed than in person-focal — bold "
    "  heavy sans-serif OR decorative gothic / condensed display font. Can "
    "  flank or frame the hero (e.g. 'LUCKIN [cup] COFFEE', 'the price [box] "
    "  you pay'), or use numbers / strikethroughs as the visual device "
    "  ('30 HOURS' crossed out → '6 HOURS'). High contrast — white with "
    "  hard shadow, glowing edges, or single bright color against dark.\n"
    "• Lighting: spotlight / movie-poster / product-shot feel. Often a "
    "  single dramatic light beam on the hero from above, dark surrounding "
    "  void. OR clean editorial / explainer lighting for data-driven topics.\n"
    "• Color: disciplined palette — dark / black background + ONE saturated "
    "  accent color (deep red, neon blue, hot pink, glowing white). For "
    "  editorial-style thumbnails, use a clean limited palette (1 hero "
    "  color + 1 neutral + 1 contrast). Avoid muddy mid-tones.\n"
    "• Composition: tight, often symmetric. No wasted space. Every element "
    "  earns its place. Split layouts work when one side is the physical "
    "  hero and the other is UI / map / data overlay.\n"
    "• Texture / grain: subtle film grain or noise on dark backgrounds adds "
    "  premium feel — never on the bright hero subject itself.\n"
    "• Format: 16:9, sharp focus on the hero subject.\n"
    "AVOID: people / faces (this preset is explicitly faceless), low-contrast "
    "or muted palettes, small text, multiple competing focal points, generic "
    "stock backgrounds, off-center subjects unless paired with a deliberate "
    "left/right split."
)

_STYLE_PRESETS = {
    "person_focal": _PERSON_FOCAL_STYLE,
    "faceless": _FACELESS_STYLE,
    "none": "",
}


def _compose_prompt(user_prompt: str, preset: str, reference_directives: str = "") -> str:
    """Glue the composition preset + (optional) reference-template directives
    onto the user's prompt.

    Final shape:
      SUBJECT / SCENE: <user prompt>
      [TEMPLATE TO MATCH: …]   ← only when refs were analyzed
      [COMPOSITION (preset): …] ← only when preset != "none"

    User intent stays first so it dominates. Reference template comes second
    because it's the most concrete styling signal (extracted from a real
    image). Generic preset rules trail last as fallback constraints.
    """
    parts = [f"SUBJECT / SCENE:\n{user_prompt.strip()}"]
    if reference_directives:
        parts.append(reference_directives)
    style = _STYLE_PRESETS.get(preset or "person_focal", _PERSON_FOCAL_STYLE)
    if style:
        parts.append(style)
    return "\n\n".join(parts)


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
        prompt: Annotated[str, "Describe the SUBJECT and SCENE — who's on camera (or what the visual is about), what's happening, key props, and any text overlay the user wants. You don't need to specify composition / layout / color rules — the server applies them via the style_preset. Focus the prompt on content; let the preset handle composition."],
        aspect_ratio: Annotated[str, "16:9 / 9:16 / 1:1 / 4:5 / 4:3 / 3:2 / 21:9 / auto. Default 16:9 (YouTube thumbnail)."] = "16:9",
        resolution: Annotated[str, "1K / 2K / 4K. Default 2K — plenty for thumbnails and ~4× faster than 4K."] = "2K",
        reference_urls: Annotated[list[str] | None, "Up to 8 reference inputs. Each can be a YouTube URL (watch / shorts / youtu.be / embed / live), a bare 11-char video ID, an i.ytimg.com URL, or any direct image URL. YouTube URLs are auto-resolved to the video's hqdefault thumbnail server-side."] = None,
        reference_images: Annotated[list[str] | None, "Alias for `reference_urls`, accepted for back-compat. Prefer reference_urls in new code."] = None,
        style_preset: Annotated[str, "Composition preset prepended to the prompt. Pick based on whether the video has a face on camera:\n• 'person_focal' (DEFAULT) — for videos featuring a real on-camera creator. Person on the left, face large/expressive, big text right, cutout depth, cinematic lighting (Unlayered / Pitagoras / MrBeast style).\n• 'faceless' — for videos with NO on-camera person (challenge series, business case studies, explainers, ASMR/cooking close-ups, tier-list style). Dominant centered hero object/scene tells the story via metaphor or juxtaposition; large decorative text; spotlight or editorial lighting; dark bg + single accent color.\n• 'none' — pass prompt verbatim with no composition guidance. Use ONLY when the user has very specific creative direction that would conflict with a preset.\nPick faceless if the video idea doesn't naturally include a person on camera, even if the user didn't say 'faceless' explicitly."] = "person_focal",
        find_outliers_first: Annotated[bool, "Set True to auto-fetch viral references from algrow and use the top 3 as reference_urls, in the SAME call. This produces ONE widget with the result AND the outlier grid (user can pick alternates without re-prompting). Prefer this over calling find_outlier_references separately when the user wants 'a thumbnail with viral references for X' — separate calls mount two widgets and the user can't pick from the grid after the result lands. Best used with `outlier_topic` to keep the algrow search query short (algrow's semantic search is loose on long phrases)."] = False,
        outlier_topic: Annotated[str | None, "Topic to search algrow for when find_outliers_first=True. Defaults to `prompt` if unset, but keeping this short (2-3 words: 'Vietnam rail', 'Amazon Prime', 'Minecraft 100 days') gives much better outlier matches than a full prompt."] = None,
        analyze_references: Annotated[bool, "When True (default) AND reference_urls are provided, the server first runs Gemini vision on the references to extract a structured design breakdown (composition, palette, lighting, text style, etc.), then folds those template rules into the prompt. Result: Gemini's image-gen keeps the reference's design system but swaps the subject for the user's. Adds 5–15s of latency. Set False to skip and pass references as loose visual hints only."] = True,
    ) -> str:
        import json

        # Merge & normalize whatever the caller provided.
        combined = list(reference_urls or []) + list(reference_images or [])
        resolved_refs = _resolve_reference_urls(combined)

        # Optional one-call outlier discovery — fetch from algrow, take top
        # 3, fold them into reference_urls so the rest of the flow doesn't
        # care where the refs came from. Keep the full list around for the
        # widget to render.
        outliers_list: list[dict] = []
        outlier_error: str | None = None
        if find_outliers_first and _ALGROW_API_KEY:
            outliers_list, outlier_error = _fetch_outliers_from_algrow(
                topic=(outlier_topic or prompt).strip(),
            )
            for o in outliers_list[:3]:
                tu = o.get("thumbnail_url")
                if tu and tu not in resolved_refs:
                    resolved_refs.append(tu)
            resolved_refs = resolved_refs[:8]

        # Optional reference analysis — vision-to-JSON on each ref, fold the
        # extracted design rules into the prompt as explicit directives.
        # Much higher fidelity than letting Gemini's image-gen guess what's
        # transferable from a raw reference image.
        ref_directives = ""
        ref_analyses: list[dict] = []
        ref_analysis_errors: list[str] = []
        if analyze_references and resolved_refs and _GEMINI_API_KEY:
            ref_analyses, ref_analysis_errors = _analyze_references_parallel(
                resolved_refs, max_n=3,
            )
            ref_directives = _build_reference_directives(ref_analyses)

        composed_prompt = _compose_prompt(prompt, style_preset, ref_directives)

        submit = create_task(
            prompt=composed_prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_input=resolved_refs or None,
        )

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "reference_urls": resolved_refs,
            "style_preset": style_preset,
            "outliers": outliers_list,
        }
        # Optional fields — only include when populated, so Claude doesn't
        # misread null keys as failure signals.
        topic_for_outliers = outlier_topic or (prompt if find_outliers_first else None)
        if topic_for_outliers:
            payload["outlier_topic"] = topic_for_outliers
        if outlier_error:
            payload["outlier_error"] = outlier_error
        if ref_analyses:
            payload["reference_analyses"] = ref_analyses
        if ref_analysis_errors:
            payload["reference_analysis_errors"] = ref_analysis_errors
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
        style_preset: Annotated[str, "Echoed back to the widget so the preset dropdown stays in sync across polls (otherwise the second poll wipes the value Claude originally chose)."] = "person_focal",
        outliers: Annotated[list[dict] | None, "Echoed back to the widget so the outlier grid persists across polling rounds. Widget sends this when the original generate_thumbnail call had find_outliers_first=True."] = None,
        outlier_topic: Annotated[str | None, "Echoed back; lets the widget keep the outlier-section header label."] = None,
    ) -> str:
        import json

        status = query_task(task_id)
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "task_id": task_id,
            "reference_urls": reference_urls or [],
            "style_preset": style_preset,
            "outliers": outliers or [],
            "outlier_topic": outlier_topic,
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

    # ----- Optional standalone vision-to-JSON tool --------------------------
    # Useful as a building block — Claude can call it directly to break down
    # any thumbnail without going through the generation flow. Also exposed
    # for external callers who want the structured analysis without the
    # generate step.
    if _GEMINI_API_KEY:
        @mcp.tool(
            name="analyze_thumbnail",
            title="Analyze Thumbnail (Vision to JSON)",
            description=(
                "Run Gemini vision on a thumbnail image URL and return a "
                "structured JSON breakdown: subject, composition, color "
                "palette, typography, lighting, style genre, emotional cue, "
                "and what makes the design work. Useful for: understanding "
                "why a high-CTR thumbnail performs; extracting reusable "
                "design rules; teaching the user thumbnail design language. "
                "Internally this is the same analysis generate_thumbnail "
                "runs when analyze_references=true."
            ),
        )
        async def analyze_thumbnail_tool(
            image_url: Annotated[str, "Public HTTPS image URL to analyze. Accepts any image; YouTube hqdefault URLs work great."],
        ) -> str:
            import json
            analysis, error = _analyze_image_via_gemini(image_url)
            if analysis is None:
                return json.dumps({"success": False, "error": error or "unknown error"})
            return json.dumps({"success": True, "image_url": image_url, "analysis": analysis}, default=str)

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
                "Search algrow's database (50k+ YouTube channels) for "
                "topically-similar videos with proven outlier performance "
                "and return their thumbnails as candidate references for "
                "thumbnail generation. The widget renders the results as a "
                "clickable grid — users tap one or more to add them to the "
                "generation's reference_urls list.\n\n"
                "Ranking: vector-similarity to the topic, filtered to videos "
                "with outlier_score >= 2.0 (i.e. the video got at least 2× "
                "its channel's typical views — a strong signal the thumbnail/"
                "title combo is working). Outlier score = video.view_count / "
                "channel.avg_views_per_video.\n\n"
                "This combination (similarity sort + 2× outlier floor) is "
                "what produces useful references — sorting by raw outlier "
                "score surfaces topically-off-target winners (e.g. a "
                "99× 'Hidden Villages of the Amazon' video for an 'Amazon "
                "Prime' query), while dropping the floor lets in low-signal "
                "noise. Default content_type is longform; use shorts for "
                "shortform-style references."
            ),
            meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
        )
        async def find_outlier_references_tool(
            topic: Annotated[str, "Search topic — the video idea you want references for. Algrow does semantic search so phrases work better than single keywords (e.g. 'Amazon Prime downfall' is better than 'amazon')."],
            content_type: Annotated[str, "longform or shorts. Default longform."] = "longform",
            limit: Annotated[int, "Max thumbnails to return. Default 12, capped at 24."] = 12,
            min_outlier_score: Annotated[float, "Floor on outlier multiplier — only include videos that outperformed their channel average by at least this factor. Default 2.0 (validated against hand-eval: lower lets in too much noise, higher misses too many strong references)."] = 2.0,
        ) -> str:
            """STRONGLY PREFER generate_thumbnail(find_outliers_first=True)
            over this tool. This tool exists ONLY for the rare 'user wants
            to BROWSE outliers in isolation, no generation yet' case (e.g.
            'show me what's working in the AI niche right now'). For
            anything that ends in 'generate a thumbnail', call
            generate_thumbnail directly with find_outliers_first=True —
            otherwise the chat mounts two separate widgets, the outlier
            picker becomes useless the moment the result lands, and the
            user can't swap reference picks without re-prompting.
            """
            import json

            outliers, error = _fetch_outliers_from_algrow(
                topic=topic,
                content_type=content_type,
                limit=limit,
                min_outlier_score=min_outlier_score,
            )
            # Only include the error key on actual error — Claude tends to
            # read the *presence* of an "error" field as a failure signal
            # even when the value is null. Empty outliers + no error key
            # means "algrow had nothing for this topic", not a tool fault.
            payload = {
                "view": "outlier_picker",
                "topic": topic,
                "content_type": content_type,
                "outliers": outliers,
                "count": len(outliers),
            }
            if error:
                payload["error"] = error
            return json.dumps(payload, default=str)

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
