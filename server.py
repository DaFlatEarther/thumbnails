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

import asyncio
import contextlib
import json as _json_top
import logging
import os
import re
import threading
import time
import uuid as _uuid_top
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import requests
from dotenv import load_dotenv
from pydantic import Field
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ---------------------------------------------------------------------------
# Widget state store — cross-device sync for the in-widget WIP state.
# claude.ai doesn't persist widget-internal state across chat closes, so we
# stash it server-side keyed by lowercased video title. JSON file backing
# makes it survive server restarts; threading.Lock keeps the file safe under
# concurrent saves from PC + phone hitting the same tunnel.
# ---------------------------------------------------------------------------

# Where generated images get persisted on disk and served from. Each call
# to generate_thumbnail saves the Gemini-returned image bytes here under a
# UUID filename so the widget can <img src=...> via the public /generated
# route.
_GENERATED_DIR = Path(__file__).resolve().parent / "generated_images"
_GENERATED_DIR.mkdir(exist_ok=True)

# Where user-uploaded reference images land. Lives under generated_images
# so the existing /generated static mount serves them too — no extra
# Starlette route needed. Sam mirrors this directory to R2 out-of-band.
_REFS_DIR = _GENERATED_DIR / "refs"
_REFS_DIR.mkdir(exist_ok=True)

# In-flight Gemini-direct generations. Keyed by task_id (prefixed with
# "gemini:") so check_thumbnail_status can find them. Stored in process
# memory — a uvicorn restart drops the dict, in which case status checks
# for unknown task_ids report fail with a "server restart" message so the
# user knows to retry rather than poll forever.
#
# We went async on Gemini direct because the synchronous call blocked
# the MCP request for the full Gemini wall-clock (20-90s), which exceeds
# the Claude mobile app's MCP request timeout. With the background task,
# generate_thumbnail returns state=pending+task_id in under a second and
# the widget polls via check_thumbnail_status — same pattern as the
# algrow-backed models.
_GEMINI_TASK_PREFIX = "gemini:"
_GEMINI_TASKS: dict[str, dict] = {}
# Soft cap so a long-running process doesn't grow this dict unbounded.
_GEMINI_TASKS_MAX = 200

# Same idea for algrow-backed models (gpt-image-2, seedream, nano-banana-2).
# The algrow submit endpoint normally returns a job_id in <2s, but on Claude
# mobile the MCP request timeout (<27s) sometimes fires before the submit
# round-trips at all — the widget then sees "no response received" and the
# user is stuck. Fire submit in a background task and return state=pending
# with an `algrow_pending:<uuid>` placeholder; check_thumbnail_status reads
# this dict, and once submit completes it transparently swaps in the real
# algrow job_id for upstream polling.
_ALGROW_SUBMIT_PREFIX = "algrow_pending:"
_ALGROW_SUBMITS: dict[str, dict] = {}
_ALGROW_SUBMITS_MAX = 200


_STATE_FILE = Path(__file__).resolve().parent / "widget_state.json"
_state_lock = threading.Lock()
_MAX_STATE_ENTRIES = 100  # cap so the JSON file doesn't grow unbounded


def _load_state_bucket() -> dict:
    with _state_lock:
        if not _STATE_FILE.exists():
            return {}
        try:
            return _json_top.loads(_STATE_FILE.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}


def _save_state_bucket(bucket: dict) -> None:
    with _state_lock:
        tmp = _STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(_json_top.dumps(bucket), encoding="utf-8")
        tmp.replace(_STATE_FILE)


def _state_key(s: str | None) -> str:
    return (s or "").strip().lower()[:200]
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import widgets
from gemini_image import generate_image as _gemini_generate_image
from nano_banana_pro import create_task, query_task  # legacy Kie path; no longer used in generate_thumbnail
from youtube_extract import extract_video_id, extract_video_info

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
    (
        ("fal.ai timed out", "fal-ai timed out", "upstream timed out", "model timed out"),
        "Seedream's upstream model server (fal.ai) timed out before finishing "
        "this generation — common on cold queues for the edit model. Try "
        "Generate again; algrow refunds the credits for the failed attempt. "
        "If it keeps timing out, switch to Seedream 5.0 Lite (faster) or "
        "back to Nano Banana Pro.",
    ),
)


# ---------------------------------------------------------------------------
# Vision-to-JSON: reference image → structured design breakdown
# ---------------------------------------------------------------------------

_VISION_TO_JSON_PROMPT = """ROLE & OBJECTIVE
You are VisionStruct, an advanced Computer Vision & Data Serialization Engine. Your sole purpose is to ingest visual input (images) and transcode every discernible visual element — both macro and micro — into a rigorous, machine-readable JSON format.

CORE DIRECTIVE
Do not summarize. Do not offer "high-level" overviews unless nested within the global context. You must capture 100% of the visual data available in the image. If a detail exists in pixels, it must exist in your JSON output. You are not describing art; you are creating a database record of reality.

ANALYSIS PROTOCOL
Before generating the final JSON, perform a silent "Visual Sweep" (do not output this):
  • Macro Sweep: scene type, global lighting, atmosphere, primary subjects.
  • Micro Sweep: textures, imperfections, background clutter, reflections, shadow gradients, OCR text.
  • Relationship Sweep: spatial + semantic connections between objects (holding, obscuring, next to, supporting, casting shadow on, visually similar to).

OUTPUT FORMAT (STRICT)
Return ONLY a single valid JSON object. No markdown fencing (no ```json), no preamble, no commentary. Use this schema, expanding arrays as needed to cover every detail:

{
  "meta": {
    "image_quality": "Low | Medium | High",
    "image_type": "Photo | Illustration | Diagram | Screenshot | Composite | Other",
    "resolution_estimation": "approximate dimensions if discernible, else null"
  },
  "global_context": {
    "scene_description": "comprehensive, objective paragraph describing the entire scene",
    "time_of_day": "specific time or lighting condition, or null",
    "weather_atmosphere": "Foggy | Clear | Rainy | Chaotic | Serene | Studio | Other",
    "lighting": {
      "source": "Sunlight | Artificial | Mixed | Ambient",
      "direction": "Top-down | Backlit | Side-lit | Rim-lit | Front | Diffused | Other",
      "quality": "Hard | Soft | Diffused",
      "color_temp": "Warm | Cool | Neutral"
    }
  },
  "color_palette": {
    "dominant_hex_estimates": ["#RRGGBB", "#RRGGBB"],
    "accent_colors": ["color name", "color name"],
    "contrast_level": "High | Medium | Low"
  },
  "composition": {
    "camera_angle": "Eye-level | High-angle | Low-angle | Macro | Aerial | Dutch",
    "framing": "Close-up | Medium-shot | Wide-shot | Extreme close-up",
    "depth_of_field": "Shallow | Deep | Tilt-shift",
    "focal_point": "the primary element drawing the eye"
  },
  "objects": [
    {
      "id": "obj_001",
      "label": "primary object name",
      "category": "Person | Vehicle | Furniture | Animal | Text | Symbol | Other",
      "location": "Center | Top-Left | Top-Right | Bottom-Left | Bottom-Right | Mid-Left | Mid-Right",
      "prominence": "Foreground | Midground | Background",
      "visual_attributes": {
        "color": "detailed color description",
        "texture": "Rough | Smooth | Metallic | Fabric-* | Skin | Other",
        "material": "Wood | Plastic | Metal | Skin | Paper | Digital | Other",
        "state": "Damaged | New | Wet | Dirty | Pristine | Worn",
        "dimensions_relative": "tiny | small | medium | large | dominant relative to frame"
      },
      "micro_details": [
        "specific small details only visible on close inspection"
      ],
      "pose_or_orientation": "Standing | Tilted | Facing-camera | Facing-away | Other",
      "text_content": null
    }
  ],
  "text_ocr": {
    "present": true,
    "content": [
      {
        "text": "exact text content",
        "location": "where it appears (sign, overlay, t-shirt, etc.)",
        "font_style": "Serif | Sans-serif | Display | Handwritten | Bold | Italic | Condensed",
        "legibility": "Clear | Partially obscured | Stylized"
      }
    ]
  },
  "semantic_relationships": [
    "Object A holding Object B",
    "Object C casting shadow on Object A"
  ]
}

CRITICAL CONSTRAINTS
  • Granularity: never write "a crowd of people". Instead, list the crowd as one group object, then list distinct visible individuals as sub-objects or via detailed attributes (clothing color, action).
  • Micro-details: scratches, dust, weather wear, fabric folds, lighting gradients — all noted.
  • Null values: if a field is not applicable, set it to null. Don't omit fields. Schema stability matters."""


# Algrow's CDN serves tiny preview thumbnails (~168×94) — fine for the
# widget's clickable grid, BAD for vision analysis. Same goes for YouTube's
# hqdefault (480×360). When we have a YouTube video_id (either from the
# algrow CDN filename or from a youtube URL), upgrade to maxresdefault
# (1280×720) — ~57× more pixels than the algrow preview, and that's the
# difference between "Gemini can read the panel text" and "Gemini guesses
# from a blurry mosaic."
_YT_ID_FROM_ALGROW_RE = re.compile(
    r"audio\.algrow\.online/thumbnails/(?:longform|shorts)/([A-Za-z0-9_-]{11})\."
)
_YT_ID_FROM_YTIMG_RE = re.compile(r"i\.ytimg\.com/vi/([A-Za-z0-9_-]{11})/")


def _high_res_url_candidates(image_url: str) -> list[str]:
    """Return a fallback chain of URLs to try for vision analysis, highest
    resolution first. The original URL is always last so we never make the
    quality WORSE than what the caller passed.
    """
    candidates: list[str] = []
    video_id: str | None = None
    m = _YT_ID_FROM_ALGROW_RE.search(image_url)
    if m:
        video_id = m.group(1)
    else:
        m = _YT_ID_FROM_YTIMG_RE.search(image_url)
        if m:
            video_id = m.group(1)

    if video_id:
        # maxresdefault is 1280×720 but 404s for some less-popular videos.
        # sddefault (640×480) is much rarer to 404. hqdefault (480×360) is
        # always present for any public video.
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg")
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg")
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
    if image_url not in candidates:
        candidates.append(image_url)
    return candidates


def _fetch_image_bytes_with_fallback(image_url: str) -> tuple[bytes | None, str | None, str | None]:
    """Walk the high-res candidate chain and return the first one that
    fetches OK. Returns (bytes, mime_type, fetched_url) — bytes is None on
    total failure.
    """
    last_err: str | None = None
    for candidate in _high_res_url_candidates(image_url):
        try:
            resp = requests.get(candidate, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            mt = (resp.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
            if not mt.startswith("image/"):
                mt = "image/jpeg"
            return resp.content, mt, candidate
        except Exception as e:
            last_err = f"{candidate}: {str(e)[:100]}"
            continue
    return None, None, last_err


def _analyze_image_via_gemini(image_url: str) -> tuple[dict | None, str | None]:
    """Fetch image bytes (upgrading to the highest available resolution
    via the YouTube CDN where possible), call Gemini vision with the
    schema prompt, return structured dict. Returns (analysis, error) —
    analysis is None on error.

    Used by the standalone analyze_thumbnail tool and folded into
    generate_thumbnail when analyze_references=True.
    """
    if not _GEMINI_API_KEY:
        return None, "Vision analysis disabled (GEMINI_API_KEY not set)."

    img_bytes, mime_type, fetched = _fetch_image_bytes_with_fallback(image_url)
    if img_bytes is None:
        return None, f"Couldn't fetch reference image: {fetched or 'all candidates failed'}"
    logger.info(f"vision analysis fetched {len(img_bytes)} bytes from {fetched}")

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
            timeout=90,
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


# ---------------------------------------------------------------------------
# Map reference DNA → new title: takes (title, analysis JSON, style hint) and
# produces ONE polished natural-language image-gen prompt that swaps the
# reference's subject for the user's title while preserving its design
# system (composition, palette, lighting, text treatment, genre). This is
# the smart-mapping step — without it we'd just dump the analysis verbatim
# and force Nano Banana to figure out the mapping itself (which it does
# badly, especially when the reference's subject and the user's title
# differ in kind, e.g. educational grid → on-camera face).
# ---------------------------------------------------------------------------

_REASONED_MAP_PROMPT = """You are an expert YouTube thumbnail designer. You will reason through WHY a reference thumbnail works for its original title, then transfer that same design logic to a NEW title — with the user's NEW SUBJECT swapped in for the reference's old subject.

You have these inputs:

  1. REFERENCE ANALYSIS — a complete structured JSON breakdown of the reference thumbnail produced by a prior vision pass. It is below. Treat it as ground truth — every visual fact about the reference is in this JSON.
  2. REFERENCE TITLE (the original video this thumbnail was made for): "{reference_title}"
  3. USER'S NEW TITLE (what we are designing for now): "{title}"
  4. Style hint about the user's video format: {style_hint}

REFERENCE ANALYSIS JSON:
```json
{analysis_json}
```

Walk the JSON top to bottom before mapping — every field is load-bearing. Pay particular attention to: `global_context.lighting` (source / direction / quality / color_temp), `color_palette.dominant_hex_estimates` (use these as concrete hex codes in your output prompt), `composition` (camera angle / framing / DOF / focal point — these define the structural slots), every `objects[*]` entry (location, prominence, visual_attributes, micro_details — these are the slot fillings you must REPLACE), every `text_ocr.content[*]` entry (font style / location / legibility — text treatment transfers verbatim, text content gets rewritten for the new title), and `semantic_relationships` (visual devices like arrows, strings, callouts — these patterns transfer).

═══════════════════════════════════════════════════════════════════════════
HARD RULES — these override everything else, including the style hint:
═══════════════════════════════════════════════════════════════════════════

CORE PRINCIPLE — TRANSFER THE DESIGN GRAMMAR, NOT THE SURFACE FORM.
The reference thumbnail is one specific instance of a design grammar that worked for THE REFERENCE'S TITLE. The reference's designer made a SEQUENCE of choices: who is in frame, what they're doing, where they are, what props are around them, what the text says, what the visual device points at. Each of those choices was made because it fit the REFERENCE'S subject.

Your job is NOT to copy those specific choices. Your job is to identify the DESIGN GRAMMAR (the slots, the relationships, the visual language) and then make a NEW set of choices — each one driven by the USER'S TITLE — that fills the same grammar.

If the reference shows a Japanese founder mid-gesture in a 1980s tech-office setting, that grammar is:
  [hero portrait slot] = "the iconic figure most associated with the title's subject"
  [pose slot]          = "a gesture that visually narrates the title's main verb / claim"
  [setting slot]       = "an instantly-readable environment that signals the title's domain"
  [text slot]          = "a punchy label that gives the visual its narrative hook"
  [visual device slot] = "an arrow / callout that directs attention to the hero"

For a McDonald's title, the SAME grammar fills differently:
  hero      → an iconic American business-pitch figure (mid-century salesman archetype)
  pose      → a confident sell / presentation gesture
  setting   → a vintage American fast-food environment
  text      → a hook word/phrase fitting McDonald's story
  device    → the same arrow / callout pattern

The grammar transfers verbatim. The fillings DO NOT.

RULE 1 — THE REFERENCE'S COMPOSITIONAL SLOTS TRANSFER. THE FILLINGS DON'T.
Identify the slots: where is the hero in frame, what's behind them, where's the text, where's the visual device. THOSE structural positions transfer. The specific CONTENTS of each slot (which person, which pose, which background props, which words) get re-chosen from the user's new title.

You do not copy the reference's pose at the limb / finger level. You do not copy the reference's person at the ethnicity / hairstyle level. You CHOOSE new limb positions and new physical descriptors that fit the user's title.

RULE 2 — IF THE REFERENCE HAS NO PERSON, THE OUTPUT HAS NO PERSON.
Check `objects` in the analysis. If none have `category: "Person"`, do NOT include a person, creator, model, or any human figure in the output, even if the style hint says "person_focal". The style hint is about the USER'S VIDEO format, not a license to inject characters that aren't in the reference's design pattern. If the reference is faceless and the user's video happens to have a creator on camera, the user is choosing to use a faceless thumbnail style — that's a deliberate choice signaled by their reference pick.

RULE 3 — IF THE REFERENCE HAS A PERSON, KEEP A PERSON — BUT CHOOSE WHO + HOW BASED ON THE USER'S TITLE.
The reference has a person → the output has a person. But:
  • The PERSON'S identity (ethnicity, age range, hair, clothing, period dress) is derived from the USER'S TITLE, not transferred from the reference.
    - Reference is about a Japanese company → reference's person is Japanese. User's title is about an American company → output person is American-looking.
    - Reference is about a 1980s figure → reference's clothing fits 1980s. User's title is about a 1950s figure → output clothing fits 1950s.
  • The POSE is derived from the USER'S TITLE'S meaning, not transferred from the reference.
    - Reference shows "explaining" because the reference title is about teaching/revelation. User's title is about salesmanship → pose is a confident sell / pitch gesture, not the explaining pose.
    - Reference shows "shock" because the reference title is about a twist. User's title is about quiet competence → pose is composed, not shocked.
  • The EXPRESSION follows the same rule — chosen to match the user's title's tone.

The framing position (e.g. "centered, chest-up close-up", "left-third, full-body") is structural and transfers. WHAT they look like and WHAT they're doing is re-chosen.

RULE 4 — STRUCTURAL ELEMENTS THE REFERENCE USES MUST BE PRESERVED — WITH NEW FILLINGS:
  • Text overlays — same number, same approximate positions, same font style/weight, same colors. The TEXT CONTENT itself is rewritten to fit the user's new title (same hook style, e.g. "ONLY EXPERTS KNOW!" with a parenthetical subhead → adapt the same hook + subhead format for the new title).
  • Visual devices — arrows, circles, strikethroughs, price tags, badges. If the reference has a red curved arrow, the output has a red curved arrow. The device's TARGET adapts (it points at the new hero, not the same point in space).
  • Background TYPE adapts to the user's title's domain. Reference has a "luxury boutique blurred background" because the reference is about luxury → keep blurred + similar lighting/palette/depth-of-field, but change the SETTING (e.g. for a fast-food title → a blurred 1960s diner interior with similar lighting feel and similar warmth).
  • Lighting STYLE transfers (warm vs cold, hard vs diffused, high-key vs low-key). The lighting's COLOR / temperature may shift slightly to suit the new setting.

RULE 5 — NEVER NAME REAL PEOPLE OR REAL BRANDS. DESCRIBE THEM PHYSICALLY INSTEAD.
The downstream image model's safety filter blocks generations that name real public figures or brand names (it returns BlockedReason.OTHER and the user gets nothing). To produce reliable renders, you must NEVER write the name of a real person or a real brand in the output prompt — even when the user's title is explicitly about that person or brand.

  • PEOPLE: Do not write names like "Ray Kroc", "Tom Cruise", "Beyoncé", "Elon Musk", "Pope Francis", "Yutaka Urakami". Instead, describe the person via concrete physical attributes the image generator can render: approximate age, ethnicity / nationality cues, build, hair color and style, complexion, facial features, signature clothing or accessories, characteristic expression or pose.
    Example replacements:
      ✗ "Ray Kroc standing in a McDonald's kitchen"
      ✓ "an older white American man in his 60s, short slicked-back gray hair, wearing a dark business suit with a thin tie, standing in a vintage 1960s diner kitchen"
      ✗ "Tom Cruise with an expression of wide-eyed shock"
      ✓ "a 60-something American man with dark slicked-back hair, athletic build, a sharp jawline, wide-eyed shock expression"

  • BRANDS: Do not write brand names like "McDonald's", "Rolex", "Tesla", "Apple", "Nike", "Ryobi", "Hilti". Describe what the brand LOOKS LIKE — distinctive colors, logo shape (without naming the brand), product silhouette, signature design language.
    Example replacements:
      ✗ "a McDonald's restaurant interior with the golden arches"
      ✓ "a vintage 1960s American fast-food restaurant interior with warm red-and-yellow signage, formica counters, and curved booth seating"
      ✗ "a Rolex Submariner watch on a wrist"
      ✓ "a luxury dive-style wristwatch with a black bezel, oyster-link steel bracelet, and a black sunburst dial"
      ✗ "a lime-green Ryobi power drill"
      ✓ "a modern cordless power drill with a vivid lime-green body and matte-black grip"

  • If the user's title CONTAINS a real name or brand, that's fine in the TEXT OVERLAY portion of the prompt (the visible thumbnail text is allowed to say what the title says) — but the SCENE DESCRIPTION must still use physical / visual descriptors, not names. The text overlay is rendered as letterforms, not interpreted as a directive.

  • The level of specificity should still match the title. If the title implies a specific category, describe a specific-LOOKING instance of that category in detail. NEVER write vague placeholders like "a celebrity-looking person" or "a famous-looking face" — those tell the image model nothing.

RULE 6 — RE-FILL EACH SLOT FROM THE USER'S TITLE. NEVER TRANSFER THE REFERENCE'S FILLINGS.
This is the most common failure mode. The reference's specific person / pose / props / setting were chosen because they fit the REFERENCE'S title. For the user's title you have to choose new fillings — each one driven by what the user's title is about — that occupy the same slots in the same grammar.

Examples (notice every slot's filling is re-derived from the user's title; only the slot structure transfers):

  • Reference: thumbnail for a story about a Japanese tool-company founder. Shows an elderly Japanese man in 1980s business attire, mid-pointing-gesture, against a blurred tech-workshop backdrop, with a red curved arrow + "THE GENIUS" text overlay.
    User's title: "The Man Who Saved McDonald's".
    → Output: an older white American man in his late 60s, slicked-back gray hair, dark mid-century business suit with a thin tie, a confident sell / pitch gesture (one hand forward, palm up, as if presenting), against a blurred vintage 1960s American diner interior with warm red-and-yellow ambient color. Text overlay reads something like "THE SALESMAN" in the same font weight/position. Same red curved arrow pattern pointing at the new hero. The PERSON is American because the title is American. The POSE is a "pitch / sell" gesture because the title is about salesmanship. The SETTING is a 1960s American diner because the title is about a fast-food chain in that era. The GRAMMAR (centered hero + text + arrow + warm blurred backdrop) is the only thing that transferred unchanged.

  • Reference: thumbnail for a pop-star video showing a specific identifiable woman centered, sequins and stage lighting, mid-song expression.
    User's title is about a different pop star from a different era / nationality.
    → Output: a woman matching that DIFFERENT pop star's era and look — a 1970s American woman in glam-rock attire if the title is about a 70s star; a Korean woman in current K-pop styling if the title is about a K-pop star. Stage-lighting STYLE transfers, but stage decor adapts to the new genre.

  • Reference: a Pope thumbnail showing the current Pope mid-blessing in St Peter's Square.
    User's title is about a 1500s Pope.
    → Output: an older European man in Renaissance-era papal vestments (tall white mitre, heavily-embroidered gold-trimmed robes, distinct from the modern simplified vestments), inside a candle-lit stone-cathedral interior, painted in a Renaissance-portrait style. Same close-up framing, same gravitas. Pose chosen for a 1500s-pope feel (e.g. seated, holding a scroll), not the modern blessing pose.

  • Reference: a luxury-watch tier list showing a specific iconic dive watch in the top slot.
    User's title is about a different watchmaker.
    → Output: the visual hallmarks of THAT watchmaker's signature pieces (case shape, dial layout, bracelet style, distinguishing complications). Same tier-list grid structure, same labels' position and font.

THE GRAMMAR TRANSFERS (slot positions, lighting style, palette family, text-overlay structure, visual device kit). EVERY FILLING RE-DERIVES from the user's title.

When you describe the new fillings, BE EXPLICIT in concrete prose AND make the descriptors point in a clearly DIFFERENT direction from the reference's fillings on at least two axes (e.g. different ethnicity AND different clothing era; different setting AND different pose). Vague phrases like "an older man" let the reference image's identity bleed through; specific contrastive descriptors push the generator away from copying. The descriptors should be specific enough that swapping to a named person/brand would only add one word — but you do not add that word.

═══════════════════════════════════════════════════════════════════════════

Do these steps internally — do NOT show your reasoning in the output:

STEP 1 — UNDERSTAND THE REFERENCE'S CHOICES.
Walk the structured analysis element by element (every object, the composition, palette, lighting, every text_ocr entry, every semantic relationship). For each one, ask: "WHY did the designer pick THIS specific filling for THE REFERENCE'S TITLE?" The answer is always semantic, e.g. "they picked a Japanese man because the company is Japanese", "they picked a confident-pointing pose because the title is about teaching/revelation", "they picked a tech-workshop backdrop because the company makes tech". Identify the GRAMMAR (the slots + relationships) separately from the FILLINGS (the specific choices that filled the slots for the reference's title).

STEP 2 — RE-DERIVE EACH FILLING FROM THE USER'S TITLE.
For every slot you identified, ask: "What's the right filling FOR THE USER'S TITLE, given the same grammar?"
  • PERSON slot → ethnicity, age, era, attire derived from the user's title's subject (American business figure → American-looking; 1500s Pope → Renaissance-era vestments; modern K-pop → current K-pop styling).
  • POSE slot → the gesture that visually narrates the user's title's main verb / claim (title about salesmanship → a pitch / sell gesture; title about discovery → an "aha" reach; title about secrecy → a finger-to-lips). Do NOT echo the reference's specific limb position — choose the pose that fits the new title.
  • SETTING / BACKGROUND slot → the environment that signals the user's title's domain at a glance (Japanese tech-co → a tech workshop; American fast-food chain → a vintage diner; medieval cathedral → candle-lit stone interior).
  • TEXT slot → a punchy word/phrase derived from the user's title's narrative hook (same format as the reference's text — single label, headline + subhead, etc).
  • VISUAL DEVICE slot → same kind of device (arrow / circle / price tag), pointing at / labeling the new hero.

The GRAMMAR (slot positions, lighting style, palette family, font weight, visual device shape) transfers unchanged. EVERY FILLING is re-derived.

STEP 3 — WRITE THE OUTPUT.
Output ONE polished image-generation prompt: a single natural-language paragraph, ~300–450 words.

CRITICAL: the prompt will be sent to the image-gen model TEXT-ONLY. No reference image is attached. EVERY visual fact must be made explicit IN THE PROMPT TEXT, pulled from the JSON breakdown above. Do NOT write "as in the reference image", "matching the reference's palette", "in the same style as the reference" — there is no reference at generation time, those phrases tell the model nothing.

Specifically you MUST explicitly describe in the prompt:

  • LIGHTING — name the source ("warm overhead tungsten"), direction ("top-left key with soft fill from below-right"), hardness ("hard / soft / diffused"), color temperature ("warm 3200K feel" or "neutral daylight"). Pull these straight from `global_context.lighting`.

  • PALETTE — list the actual dominant hex codes from `color_palette.dominant_hex_estimates` ("dominated by #8B4513 warm corkboard brown, #E63946 vivid alarm red, #2B2B2B near-black"). Name the accent colors. Set the contrast level.

  • COMPOSITION — camera angle, framing, depth of field, focal point. Where each slot lives in the frame (center / top-left / mid-right etc.) using the `objects[*].location` values.

  • TEXT OVERLAYS — for every entry in `text_ocr.content`: the EXACT font style (serif / sans-serif / display / bold weight / italic / condensed), color, background treatment if any, position. The TEXT CONTENT itself gets REWRITTEN to fit the user's new title — but the typographic treatment stays.

  • VISUAL DEVICES — arrows, strings, callouts, badges, price tags, glow effects. Drawn from `semantic_relationships` and from any `objects` entries categorized as Symbol / device. Describe the shape, color, anchor points, and what they connect.

  • SUBJECT — every Person / focal object: physical attributes (age, build, hair, complexion, clothing), pose, expression. These get FRESHLY DESCRIBED based on the user's new title, NOT copied from the reference's `objects[*]` content. The reference's specific persons / weapons / badges / documents are NEVER mentioned — only their structural ROLE (centered portrait / bottom-left prop / etc.) transfers.

  • TEXTURE / MATERIAL — surfaces from `objects[*].visual_attributes.texture` and `.material` ("rough corkboard texture with visible fiber grain", "weathered paper with creases").

Use natural prose, not bullets. Do NOT use the words "template", "inspiration", "YouTube", "reference image", or "reference". Output ONLY the final prompt — no preamble, no reasoning labels, no commentary."""


def _map_reference_to_title_via_gemini(
    title: str,
    reference_url: str,
    reference_title: str | None,
    style_preset: str = "person_focal",
    custom_instructions: str | None = None,
) -> tuple[str | None, str | None]:
    """Two-step compose pipeline:

      1. Vision-to-JSON — _analyze_image_via_gemini runs VisionStruct over
         the reference and returns a rich structured breakdown (palette
         hex codes, lighting specs, every object's location/material/
         micro-details, exact text-overlay typography, semantic
         relationships between elements).

      2. Text-only synthesis — Gemini reads the JSON + user's title and
         writes a self-contained image-gen prompt that bakes every
         structural detail from the JSON into explicit prose, while
         re-deriving every filling (person, props, text content) from
         the new title.

    Why two calls now? The downstream image generator runs PROMPT-ONLY —
    no reference image is passed to it. That removes the img2img "copy
    the reference" temptation that was making gpt-image-2 / Nano Banana
    return near-identical asset swaps. To compensate for the lost visual
    channel, the prompt has to carry every visual fact explicitly, and
    the richest way to extract those facts is through the VisionStruct
    pass first.

    Mobile MCP timeout note: the original single-call version was 5-15s.
    Two-step is 8-25s — still safely inside the 27s mobile timeout
    because both Gemini calls are short (vision-JSON is heavy compute
    on Gemini's side but returns fast; synthesis is text-only so very
    quick). If we ever start blowing the budget we'd push synthesis to
    a background task like the generate path.

    Returns (mapped_prompt, error) — prompt is None on error.
    """
    if not _GEMINI_API_KEY:
        return None, "Gemini mapping disabled (GEMINI_API_KEY not set)."

    # Step 1: rich vision-to-JSON extraction.
    analysis, vision_err = _analyze_image_via_gemini(reference_url)
    if analysis is None:
        return None, f"Reference vision analysis failed: {vision_err or 'unknown'}"

    import json as _json_mod

    style_hint_text = {
        "person_focal": "the user's video format involves a real on-camera creator (informational only — does NOT override Rule 2. If the reference is faceless, the output stays faceless.)",
        "faceless": "the user's video format has no on-camera person (informational only — does NOT override Rule 3. If the reference features a person, the output keeps a person.)",
        "none": "no hint about the user's video format; rely entirely on the reference's structural pattern.",
    }.get(style_preset or "none", "no hint; rely entirely on the reference's structural pattern")

    reference_title_str = reference_title.strip() if reference_title else "(unknown — reason from the JSON alone)"

    analysis_json_str = _json_mod.dumps(analysis, indent=2, ensure_ascii=False, default=str)

    base_prompt = _REASONED_MAP_PROMPT.format(
        title=title.strip(),
        reference_title=reference_title_str,
        style_hint=style_hint_text,
        analysis_json=analysis_json_str,
    )
    # Custom user-supplied directives go at the very END of the prompt so
    # they have the highest salience and override the default rules where
    # they conflict. Example use: "always show 3 people instead of 1",
    # "stick to a dark/moody palette regardless of reference", "include
    # the word EXPOSED in big red text in the corner". The instructions
    # are user-authored, untrusted free text — wrapped in clear delimiters
    # so they can't impersonate the system rules above.
    if custom_instructions and custom_instructions.strip():
        base_prompt += (
            "\n\n═══════════════════════════════════════════════════════════════════════════\n"
            "USER'S CUSTOM INSTRUCTIONS — apply these on top of every rule above.\n"
            "Where these conflict with the default rules, follow these.\n"
            "═══════════════════════════════════════════════════════════════════════════\n"
            + custom_instructions.strip()
            + "\n═══════════════════════════════════════════════════════════════════════════"
        )

    # Step 2: text-only synthesis call. No inline_data — the JSON is in
    # the prompt text and the model has no image to copy from.
    body = {
        "contents": [{"parts": [{"text": base_prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
        },
    }
    try:
        resp = requests.post(
            f"{_GEMINI_VISION_URL}?key={_GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=90,
        )
    except Exception as e:
        return None, f"Gemini synthesis request failed: {str(e)[:140]}"

    if resp.status_code != 200:
        return None, f"Gemini synthesis HTTP {resp.status_code}: {resp.text[:200]}"

    try:
        out = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return None, f"Couldn't parse Gemini synthesis response: {str(e)[:140]}"

    return out, None


def _build_reference_directives(analyses: list[dict]) -> str:
    """Distill the VisionStruct JSON into directive bullets for the legacy
    auto-pick path (generate_thumbnail with find_outliers_first=True). The
    new manual single-pick compose flow uses _map_reference_to_title_via_gemini
    instead and gets a much richer reasoned-mapping result — this is only
    for the auto-pick fallback where no user-picked reference + title pair
    exists.
    """
    if not analyses:
        return ""
    a = analyses[0]
    comp = a.get("composition") or {}
    palette = a.get("color_palette") or {}
    global_ctx = a.get("global_context") or {}
    light = global_ctx.get("lighting") or {}
    objects = a.get("objects") or []
    text_ocr = a.get("text_ocr") or {}
    text_items = text_ocr.get("content") or []
    rels = a.get("semantic_relationships") or []

    primary_obj = objects[0] if objects else {}
    primary_text = text_items[0] if text_items else {}
    accents = palette.get("accent_colors") or []
    hexes = palette.get("dominant_hex_estimates") or []

    bullets = [
        "TEMPLATE TO MATCH (extracted from the user's chosen reference thumbnail — KEEP these design rules, only swap the subject content):",
        f"• Scene: {global_ctx.get('scene_description', 'n/a')}",
        f"• Composition: {comp.get('framing', 'n/a')} framing, {comp.get('camera_angle', 'n/a')} angle, {comp.get('depth_of_field', 'n/a')} depth-of-field; focal point: {comp.get('focal_point', 'n/a')}",
        f"• Primary subject: {primary_obj.get('label', 'n/a')} ({primary_obj.get('location', 'n/a')}, {primary_obj.get('prominence', 'n/a')})",
        f"• Color palette: dominant hex {', '.join(hexes[:3]) or 'n/a'}; accent colors {', '.join(accents[:3]) or 'n/a'}; contrast {palette.get('contrast_level', 'n/a')}",
        f"• Lighting: {light.get('source', 'n/a')} source, {light.get('direction', 'n/a')} direction, {light.get('quality', 'n/a')} quality, {light.get('color_temp', 'n/a')} temperature",
    ]
    if primary_text:
        bullets.append(
            f"• Text treatment: \"{primary_text.get('text', '')}\" — {primary_text.get('font_style', 'n/a')} font, {primary_text.get('location', 'n/a')}, {primary_text.get('legibility', 'n/a')} legibility"
        )
    atmo = global_ctx.get("weather_atmosphere")
    if atmo:
        bullets.append(f"• Atmosphere / mood: {atmo}")
    if rels:
        bullets.append(f"• Key relationships: {'; '.join(rels[:3])}")
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


# Function/question words to drop when broadening a search query. Algrow's
# semantic search returns 0 results for long YouTube-style titles even when
# the underlying subject HAS coverage in their DB ("How Does Time Become
# Space Inside a Black Hole" → 0; "black hole" → 12). Shortening keeps the
# subject and drops the framing.
_TOPIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "how", "what", "why", "when", "where", "which", "who", "whom", "whose",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into",
    "inside", "outside", "about", "as", "and", "or", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "you", "your", "we", "our", "i", "me", "my",
    "really", "actually", "just", "even", "ever", "always", "never",
    "can", "could", "should", "would", "may", "might", "must", "shall", "will",
}


def _shorten_topic(topic: str, keep: int = 3) -> str:
    """Drop common stop / question words, return the last N meaningful words.

    Used as a fallback when algrow returns 0 for a verbatim title — YouTube
    titles bury the subject under question framing, and the semantic search
    handles bare subject nouns much better than full sentences.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", topic or "")
    significant = [w for w in words if w.lower() not in _TOPIC_STOPWORDS]
    if not significant:
        return ""
    return " ".join(significant[-keep:])


def _algrow_search_once(topic: str, content_type: str, limit: int,
                        min_outlier_score: float, page: int = 1
                        ) -> tuple[list[dict], str | None, int, bool]:
    """One round-trip to algrow. Returns (videos, error, count, has_more).
    `videos` is the raw video dicts (not yet normalized to our outlier shape).
    `has_more` mirrors algrow's response field — True when more pages exist.
    """
    if not _ALGROW_API_KEY:
        return [], "Algrow integration not configured (set ALGROW_API_KEY).", 0, False
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
                "page": max(1, int(page)),
            },
            timeout=90,
        )
        data = resp.json() if resp.content else {}
    except requests.exceptions.Timeout:
        return [], "Algrow timed out (>90s). Try a more specific topic.", 0, False
    except Exception as e:
        logger.warning(f"algrow API call failed: {e}")
        return [], _friendly_error(str(e)), 0, False

    if resp.status_code != 200 or not data.get("success"):
        return [], _friendly_error(data.get("error") or f"algrow returned HTTP {resp.status_code}"), 0, False
    videos = data.get("videos") or []
    return videos, None, int(data.get("count") or len(videos)), bool(data.get("has_more"))


def _fetch_outliers_from_algrow(topic: str, content_type: str = "longform",
                                 limit: int = 12, min_outlier_score: float = 2.0,
                                 page: int = 1
                                 ) -> tuple[list[dict], str | None, str, bool]:
    """Server-internal call to algrow's viral-videos search.

    Returns (outliers, error, effective_topic, has_more). On error,
    outliers is [] and error has the user-facing message; effective_topic
    is the original topic and has_more is False. Same first two slots as
    before so existing tuple-unpacking callers see the same shape via
    indexing.

    Falls back to shorter topic phrasings if the verbatim search returns
    zero — algrow's semantic search struggles with long YouTube-style
    question titles (see _shorten_topic comment). Fallback ONLY runs on
    page=1; for page>1 the caller is asking for the next page of a
    SPECIFIC previously-effective topic, so we use it verbatim to keep
    the result set stable across pages.
    """
    if not _ALGROW_API_KEY:
        return [], "Algrow integration not configured (set ALGROW_API_KEY).", topic, False

    tried: list[str] = []
    videos: list[dict] = []
    last_err: str | None = None
    effective_topic = topic.strip()
    has_more = False

    if page > 1:
        # Pagination request — use the topic verbatim, no fallback.
        tried.append(topic.strip())
        vids, err, count, hm = _algrow_search_once(topic.strip(), content_type, limit, min_outlier_score, page)
        if err:
            return [], err, topic.strip(), False
        videos = vids
        has_more = hm
        logger.info(f"algrow: page {page} returned {count} for q='{topic}'")
    else:
        # First page — try verbatim, then shortened fallbacks.
        candidates = [topic.strip()]
        for n in (3, 2):
            short = _shorten_topic(topic, keep=n)
            if short and short.lower() not in {c.lower() for c in candidates}:
                candidates.append(short)
        for q in candidates:
            if not q:
                continue
            tried.append(q)
            vids, err, count, hm = _algrow_search_once(q, content_type, limit, min_outlier_score, page)
            if err:
                last_err = err
                continue
            if count > 0:
                videos = vids
                has_more = hm
                effective_topic = q
                logger.info(f"algrow: returned {count} for q='{q}' (tried {tried}, has_more={hm})")
                break

    if not videos:
        if last_err:
            return [], last_err, effective_topic, False
        return [], (
            f"No viral references found for any of: {', '.join(repr(q) for q in tried)}. "
            "Try a broader 2-3 word topic (e.g. 'black hole' instead of a full question)."
        ), effective_topic, False

    # Normalize to our outlier shape. The 200/success guard already ran in
    # _algrow_search_once, so we don't need to re-check it here.
    outliers = []
    for v in videos[:limit]:
        outliers.append({
            "video_id": v.get("video_id"),
            "title": v.get("title") or "",
            "thumbnail_url": v.get("thumbnail_url"),
            "outlier_score": v.get("outlier_score"),
            "channel_name": v.get("channel_name") or "",
            "channel_thumbnail": v.get("channel_thumbnail"),
            "view_count": v.get("view_count"),
            "url": v.get("url") or (
                f"https://www.youtube.com/watch?v={v.get('video_id')}"
                if v.get("video_id") else None
            ),
        })
    return outliers, None, effective_topic, has_more


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
    """Resolve and dedupe a list of reference inputs. Caps at 14 (Gemini 3
    Pro Image Preview's per-call reference limit)."""
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
                if len(out) >= 14:
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
            "WHEN TO CALL THIS:\n"
            "  • The user pasted their own reference URLs and just wants you to "
            "    generate from them — call directly.\n"
            "  • The user gave a prompt with no need for outlier references — "
            "    call directly.\n"
            "  • The user explicitly said 'just pick a good reference for me' / "
            "    'surprise me' — call with find_outliers_first=True.\n\n"
            "WHEN NOT TO CALL THIS:\n"
            "  • You just called find_outlier_references in this turn. STOP after "
            "    that call — do NOT also call generate_thumbnail. The widget the "
            "    user sees has its own buttons (pick reference → Create Prompt → "
            "    Generate) and the user drives those clicks themselves. Calling "
            "    generate_thumbnail here mounts a SECOND widget and wastes 3 Kie "
            "    credits on a thumbnail the user didn't get to configure.\n"
            "  • The user asked for help making a thumbnail and a niche/topic is "
            "    available — prefer find_outlier_references first so the user can "
            "    see and pick from proven references.\n\n"
            "Reference images work powerfully — pass up to 8 URLs via "
            "`reference_urls` (YouTube watch/shorts/embed/live URLs, raw 11-char "
            "video IDs, i.ytimg.com URLs, and direct image URLs are all accepted; "
            "the server normalizes them)."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def generate_thumbnail_tool(
        prompt: Annotated[str, Field(description="Describe the SUBJECT and SCENE — who's on camera (or what the visual is about), what's happening, key props, and any text overlay the user wants. You don't need to specify composition / layout / color rules — the server applies them via the style_preset. Focus the prompt on content; let the preset handle composition.")],
        aspect_ratio: Annotated[str, Field(description="16:9 / 9:16 / 1:1 / 4:5 / 4:3 / 3:2 / 21:9 / auto. Default 16:9 (YouTube thumbnail).")] = "16:9",
        resolution: Annotated[str, Field(description="1K / 2K / 4K. Default 2K — plenty for thumbnails and ~4× faster than 4K.")] = "2K",
        reference_urls: Annotated[list[str] | None, Field(description="Up to 8 reference inputs. Each can be a YouTube URL (watch / shorts / youtu.be / embed / live), a bare 11-char video ID, an i.ytimg.com URL, or any direct image URL. YouTube URLs are auto-resolved to the video's hqdefault thumbnail server-side.")] = None,
        reference_images: Annotated[list[str] | None, Field(description="Alias for `reference_urls`, accepted for back-compat. Prefer reference_urls in new code.")] = None,
        style_preset: Annotated[str, Field(description="Composition preset prepended to the prompt. Pick based on whether the video has a face on camera:\n• 'person_focal' (DEFAULT) — for videos featuring a real on-camera creator. Person on the left, face large/expressive, big text right, cutout depth, cinematic lighting (Unlayered / Pitagoras / MrBeast style).\n• 'faceless' — for videos with NO on-camera person (challenge series, business case studies, explainers, ASMR/cooking close-ups, tier-list style). Dominant centered hero object/scene tells the story via metaphor or juxtaposition; large decorative text; spotlight or editorial lighting; dark bg + single accent color.\n• 'none' — pass prompt verbatim with no composition guidance. Use ONLY when the user has very specific creative direction that would conflict with a preset.\nPick faceless if the video idea doesn't naturally include a person on camera, even if the user didn't say 'faceless' explicitly.")] = "person_focal",
        find_outliers_first: Annotated[bool, Field(description="Auto-pick path: fetch viral references from algrow and use the top 3 as reference_urls in the SAME call. ONLY set True when the user has explicitly opted into auto-pick (e.g. 'just pick a good reference for me', 'surprise me', 'don't make me choose'). Default UX is the two-step flow where the user picks ONE reference themselves — call find_outlier_references first for that, then call generate_thumbnail with the user's chosen reference_url. Use `outlier_topic` to keep the algrow search query short.")] = False,
        outlier_topic: Annotated[str | None, Field(description="Topic to search algrow for when find_outliers_first=True. Defaults to `prompt` if unset, but keeping this short (2-3 words: 'Vietnam rail', 'Amazon Prime', 'Minecraft 100 days') gives much better outlier matches than a full prompt.")] = None,
        analyze_references: Annotated[bool, Field(description="When True (default) AND reference_urls are provided, the server first runs Gemini vision on the references to extract a structured design breakdown (composition, palette, lighting, text style, etc.), then folds those template rules into the prompt. Result: Gemini's image-gen keeps the reference's design system but swaps the subject for the user's. Adds 5–15s of latency. Set False to skip and pass references as loose visual hints only.")] = True,
        model: Annotated[str, Field(description="Image-gen backend. CRITICAL: model determines how the reference is used.\n• 'nano-banana-pro' (DEFAULT) — Gemini 3 Pro Image direct. REGENERATES a fresh image from the prompt; reference is a style hint only. Highest quality. 3 credits per gen. Safety filter blocks named real people / recognizable brands / references containing public figures (BlockedReason.OTHER).\n• 'gpt-image-2' — Algrow proxy → OpenAI GPT Image 2. REGENERATES from prompt (text-to-image, or image-to-image when a reference is supplied). CHEAPEST option at 1 credit per gen. Solid quality. Pick this for fast iteration where Nano Banana Pro's quality isn't worth the 3x credit cost, OR when its safety filter is blocking.\n• 'seedream-5.0-lite' — Algrow proxy. REGENERATES from prompt with optional reference as style hint. 2 credits. More permissive filter — pick this when Nano Banana Pro keeps blocking AND the user wants a DIFFERENT subject from the reference.\n• 'seedream-4.5-edit' — Algrow proxy, IMAGE-TO-IMAGE. EDITS the reference (preserves identity, layout, pose). 2 credits. ONLY appropriate when the user wants to keep the reference's subject and just tweak details. Returning the same person doing the same thing is the EXPECTED behavior of this model, not a bug.\nDefault to seedream-5.0-lite (not -4.5-edit) when Nano Banana Pro blocks on a biography / public-figure / brand piece.")] = "nano-banana-pro",
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
            outliers_list, outlier_error, _eff_topic, _has_more = _fetch_outliers_from_algrow(
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

        # If the caller has already composed the prompt (typical signal:
        # widget's "Create Prompt" path sets style_preset="none" and
        # analyze_references=False), don't re-wrap it in our
        # SUBJECT / SCENE / preset scaffolding — that produces a double
        # prefix and pollutes the prompt with rules the user already
        # incorporated. Pass through verbatim.
        if style_preset == "none" and not ref_directives:
            composed_prompt = prompt
        else:
            composed_prompt = _compose_prompt(prompt, style_preset, ref_directives)

        import uuid as _uuid

        # Route to the right backend. Default is Gemini direct (highest
        # quality, but a hard safety filter on named people/brands).
        # Seedream variants go through algrow's proxy — separate filter,
        # tolerates real-figure refs that Gemini blocks.
        #
        # IMPORTANT: algrow is async. We submit and return state=pending
        # with the algrow job_id (prefixed `algrow:`); the widget polls
        # check_thumbnail_status which routes the poll back to algrow.
        # Doing a synchronous poll loop here blew claude.ai's ~60s MCP
        # timeout for slower models (Seedream Edit can take 30-90s).
        algrow_models = {"seedream-4.5-edit", "seedream-5.0-lite", "nano-banana-2", "gpt-image-2"}
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "reference_urls": resolved_refs,
            "style_preset": style_preset,
            "outliers": outliers_list,
            "model": model,
        }
        if model in algrow_models:
            # Background-submit pattern (mirrors the Gemini direct path):
            # the algrow submit POST is a blocking ~1-5s round-trip that has
            # been timing out the Claude mobile MCP client mid-flight. Return
            # state=pending with a placeholder task_id immediately and fire
            # the submit in a background task; check_thumbnail_status reads
            # _ALGROW_SUBMITS and transparently routes onto the real algrow
            # job_id once submit completes.
            placeholder = f"{_ALGROW_SUBMIT_PREFIX}{_uuid_top.uuid4().hex}"
            _ALGROW_SUBMITS[placeholder] = {
                "state": "submitting",
                "started_at": time.time(),
                "model": model,
            }
            if len(_ALGROW_SUBMITS) > _ALGROW_SUBMITS_MAX:
                oldest = sorted(_ALGROW_SUBMITS.items(), key=lambda kv: kv[1].get("started_at", 0))
                for k, _v in oldest[: len(_ALGROW_SUBMITS) - _ALGROW_SUBMITS_MAX]:
                    _ALGROW_SUBMITS.pop(k, None)

            # Image-gen reference policy: when a reference is present we
            # were sending it to algrow as `reference_image_url`, which
            # puts edit-capable models (gpt-image-2 especially) into
            # img2img mode and they end up copying the reference's
            # subject/props verbatim and only swapping a couple of
            # elements. The compose step now bakes the full VisionStruct
            # breakdown of the reference into the prompt text, so the
            # downstream model has every structural detail it needs
            # without the visual copy temptation. Only seedream-4.5-edit
            # actually REQUIRES a reference (it's an edit model) — keep
            # passing it there. Everything else generates text-only.
            ref_for_submit = (
                resolved_refs[0]
                if (resolved_refs and model == "seedream-4.5-edit")
                else None
            )
            async def _run_algrow_submit():
                try:
                    from algrow_image import submit_image as _algrow_submit
                    sub = await asyncio.to_thread(
                        _algrow_submit,
                        prompt=composed_prompt,
                        model=model,
                        aspect_ratio=aspect_ratio,
                        reference_url=ref_for_submit,
                    )
                    if sub.get("success"):
                        _ALGROW_SUBMITS[placeholder] = {
                            "state": "submitted",
                            "started_at": _ALGROW_SUBMITS.get(placeholder, {}).get("started_at", time.time()),
                            "algrow_task_id": sub["task_id"],
                            "credits_used": sub.get("credits_used"),
                            "model": model,
                        }
                    else:
                        _ALGROW_SUBMITS[placeholder] = {
                            "state": "fail",
                            "started_at": _ALGROW_SUBMITS.get(placeholder, {}).get("started_at", time.time()),
                            "error": sub.get("error") or "Algrow submit failed.",
                            "model": model,
                        }
                except Exception as e:
                    _ALGROW_SUBMITS[placeholder] = {
                        "state": "fail",
                        "started_at": _ALGROW_SUBMITS.get(placeholder, {}).get("started_at", time.time()),
                        "error": f"Algrow submit crashed: {e!r}"[:300],
                        "model": model,
                    }
            asyncio.create_task(_run_algrow_submit())

            topic_for_outliers = outlier_topic or (prompt if find_outliers_first else None)
            if topic_for_outliers:
                payload["outlier_topic"] = topic_for_outliers
            if outlier_error:
                payload["outlier_error"] = outlier_error
            payload.update({
                "state": "pending",
                "task_id": placeholder,
                "backend": f"algrow:{model}",
            })
            return json.dumps(payload, default=str)

        # Gemini direct — kick off the actual call in a background task so
        # the MCP request returns immediately with state=pending. The full
        # Gemini wall-clock (20-90s) would otherwise blow the Claude
        # mobile app's MCP request timeout. The widget polls via
        # check_thumbnail_status which reads from _GEMINI_TASKS.
        backend_label = "gemini-3-pro-image-preview"
        topic_for_outliers = outlier_topic or (prompt if find_outliers_first else None)
        if topic_for_outliers:
            payload["outlier_topic"] = topic_for_outliers
        if outlier_error:
            payload["outlier_error"] = outlier_error
        if ref_analyses:
            payload["reference_analyses"] = ref_analyses
        if ref_analysis_errors:
            payload["reference_analysis_errors"] = ref_analysis_errors

        task_id = f"{_GEMINI_TASK_PREFIX}{_uuid_top.uuid4().hex}"
        _GEMINI_TASKS[task_id] = {"state": "pending", "started_at": time.time()}
        # Bound the dict size — drop the oldest entries when over cap.
        if len(_GEMINI_TASKS) > _GEMINI_TASKS_MAX:
            oldest = sorted(_GEMINI_TASKS.items(), key=lambda kv: kv[1].get("started_at", 0))
            for k, _v in oldest[: len(_GEMINI_TASKS) - _GEMINI_TASKS_MAX]:
                _GEMINI_TASKS.pop(k, None)

        async def _run_gemini():
            try:
                # _gemini_generate_image is synchronous (uses requests); run
                # it in a worker thread so the event loop stays free.
                gem = await asyncio.to_thread(
                    _gemini_generate_image,
                    prompt=composed_prompt,
                    reference_urls=resolved_refs or None,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                )
                if gem.get("success"):
                    if gem.get("image_bytes"):
                        mt = gem.get("mime_type") or "image/png"
                        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mt, ".png")
                        fname = f"{_uuid.uuid4().hex}{ext}"
                        (_GENERATED_DIR / fname).write_bytes(gem["image_bytes"])
                        base = (os.environ.get("PUBLIC_BASE_URL") or "").rstrip("/")
                        image_url = f"{base}/generated/{fname}" if base else f"/generated/{fname}"
                    else:
                        image_url = gem.get("image_url")
                    _GEMINI_TASKS[task_id] = {
                        "state": "success",
                        "image_url": image_url,
                        "cost_time_s": gem.get("cost_time_s"),
                        "finished_at": time.time(),
                    }
                else:
                    _GEMINI_TASKS[task_id] = {
                        "state": "fail",
                        "error": _friendly_error(gem.get("error") or f"{backend_label} image generation failed"),
                        "raw_error": gem.get("error"),
                        "finished_at": time.time(),
                    }
            except Exception as e:
                _GEMINI_TASKS[task_id] = {
                    "state": "fail",
                    "error": f"Gemini call threw: {str(e)[:200]}",
                    "finished_at": time.time(),
                }

        asyncio.create_task(_run_gemini())

        payload.update({
            "state": "pending",
            "task_id": task_id,
            "backend": backend_label,
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
        task_id: Annotated[str, Field(description="The task_id returned by generate_thumbnail.")],
        prompt: Annotated[str, Field(description="Original prompt (echoed back to the widget so its form state is preserved across polls).")] = "",
        aspect_ratio: Annotated[str, Field(description="Echoed back to the widget.")] = "16:9",
        resolution: Annotated[str, Field(description="Echoed back to the widget.")] = "2K",
        reference_urls: Annotated[list[str] | None, Field(description="Echoed back to the widget so the reference thumbnails stay visible during polling.")] = None,
        style_preset: Annotated[str, Field(description="Echoed back to the widget so the preset dropdown stays in sync across polls (otherwise the second poll wipes the value Claude originally chose).")] = "person_focal",
        outliers: Annotated[list[dict] | None, Field(description="Echoed back to the widget so the outlier grid persists across polling rounds. Widget sends this when the original generate_thumbnail call had find_outliers_first=True.")] = None,
        outlier_topic: Annotated[str | None, Field(description="Echoed back; lets the widget keep the outlier-section header label.")] = None,
        model: Annotated[str, Field(description="Echoed back so the model dropdown stays in sync across polls.")] = "nano-banana-pro",
    ) -> str:
        import json

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "task_id": task_id,
            "reference_urls": reference_urls or [],
            "style_preset": style_preset,
            "outliers": outliers or [],
            "outlier_topic": outlier_topic,
            "model": model,
        }

        # Algrow tasks have a prefixed task_id; everything else is Kie.
        # Gemini-direct background tasks live in an in-process dict.
        if task_id.startswith(_GEMINI_TASK_PREFIX):
            entry = _GEMINI_TASKS.get(task_id)
            if not entry:
                # Unknown task — usually means uvicorn restarted and lost
                # the dict. Tell the user clearly so they retry instead of
                # polling against a dead reference forever.
                payload.update({
                    "state": "fail",
                    "error": "Server restarted while this generation was in flight — click Generate to retry.",
                    "backend": "gemini-3-pro-image-preview",
                })
                return json.dumps(payload, default=str)
            state = entry.get("state")
            if state == "success":
                payload.update({
                    "state": "success",
                    "images": [entry["image_url"]] if entry.get("image_url") else [],
                    "backend": "gemini-3-pro-image-preview",
                    "cost_time_s": entry.get("cost_time_s"),
                })
            elif state == "fail":
                payload.update({
                    "state": "fail",
                    "error": entry.get("error") or "Gemini generation failed",
                    "raw_error": entry.get("raw_error"),
                    "backend": "gemini-3-pro-image-preview",
                })
            else:
                payload["state"] = "pending"
            return json.dumps(payload, default=str)

        from algrow_image import is_algrow_task_id, check_image_status as _algrow_check

        # algrow_pending:<uuid> placeholder — set when generate_thumbnail
        # backgrounded the algrow submit. Until that submit finishes we have
        # nothing to poll upstream with; once it does, transparently route
        # the poll onto the real algrow job_id so the widget never has to
        # see the placeholder swap.
        if task_id.startswith(_ALGROW_SUBMIT_PREFIX):
            entry = _ALGROW_SUBMITS.get(task_id)
            if not entry:
                payload.update({
                    "state": "fail",
                    "error": "Server restarted while this generation was in flight — click Generate to retry.",
                    "backend": f"algrow:{model}",
                })
                return json.dumps(payload, default=str)
            state = entry.get("state")
            if state == "submitting":
                payload["state"] = "pending"
                return json.dumps(payload, default=str)
            if state == "fail":
                payload.update({
                    "state": "fail",
                    "error": _friendly_error(entry.get("error") or "Algrow submit failed."),
                    "raw_error": entry.get("error"),
                    "backend": f"algrow:{entry.get('model') or model}",
                })
                return json.dumps(payload, default=str)
            # state == "submitted" → fall through to algrow polling using the
            # real job_id stashed by the background submit task.
            algrow_task_id = entry["algrow_task_id"]
            ar = _algrow_check(algrow_task_id)
            ar_state = ar.get("state")
            backend_label = f"algrow:{entry.get('model') or model}"
            if ar_state == "success":
                payload.update({
                    "state": "success",
                    "images": [ar["image_url"]] if ar.get("image_url") else [],
                    "backend": backend_label,
                })
                if entry.get("credits_used") is not None:
                    payload["credits_used"] = entry["credits_used"]
            elif ar_state == "fail":
                payload.update({
                    "state": "fail",
                    "error": _friendly_error(ar.get("error") or "Algrow generation failed"),
                    "raw_error": ar.get("error"),
                    "backend": backend_label,
                })
            else:
                payload["state"] = "pending"
                if ar.get("transient_error"):
                    payload["transient_error"] = ar["transient_error"]
                if ar.get("upstream_state"):
                    payload["upstream_state"] = ar["upstream_state"]
            return json.dumps(payload, default=str)

        if is_algrow_task_id(task_id):
            ar = _algrow_check(task_id)
            state = ar.get("state")
            if state == "success":
                payload.update({
                    "state": "success",
                    "images": [ar["image_url"]] if ar.get("image_url") else [],
                    "backend": f"algrow:{model}",
                })
            elif state == "fail":
                payload.update({
                    "state": "fail",
                    "error": _friendly_error(ar.get("error") or "Algrow generation failed"),
                    "raw_error": ar.get("error"),
                    "backend": f"algrow:{model}",
                })
            else:
                payload["state"] = "pending"
                if ar.get("transient_error"):
                    payload["transient_error"] = ar["transient_error"]
                if ar.get("upstream_state"):
                    payload["upstream_state"] = ar["upstream_state"]
            return json.dumps(payload, default=str)

        # Legacy Kie path.
        status = query_task(task_id)
        if not status.get("success"):
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
            image_url: Annotated[str, Field(description="Public HTTPS image URL to analyze. Accepts any image; YouTube hqdefault URLs work great.")],
        ) -> str:
            import json
            analysis, error = _analyze_image_via_gemini(image_url)
            if analysis is None:
                return json.dumps({"success": False, "error": error or "unknown error"})
            return json.dumps({"success": True, "image_url": image_url, "analysis": analysis}, default=str)

        # Compose-prompt tool: title + picked reference → engineered prompt.
        # The widget calls this via callServerTool after the user clicks
        # "Create Prompt", then fills the prompt textarea with the result so
        # the user can review/edit before generating. Bundled with the
        # vision-key gate because it depends on _analyze_image_via_gemini.
        @mcp.tool(
            name="compose_thumbnail_prompt",
            title="Compose Thumbnail Prompt",
            description=(
                "Given a video title + a chosen reference thumbnail, run "
                "vision analysis on the reference and engineer a full "
                "image-gen prompt that targets the title while applying the "
                "reference's design system (composition, palette, lighting, "
                "text treatment). Returns the prompt as a string. Used by "
                "the widget's 'Create Prompt' button so the user can "
                "preview/edit the engineered prompt before burning a "
                "generation. Do NOT call this from chat — it's a "
                "widget-side helper and produces a fully-composed prompt "
                "that generate_thumbnail expects to receive verbatim "
                "(with style_preset='none' and analyze_references=False)."
            ),
        )
        async def compose_thumbnail_prompt_tool(
            title: Annotated[str, Field(description="What the thumbnail is about — usually the user's video title.")],
            reference_url: Annotated[str, Field(description="The reference thumbnail URL the user picked. YouTube URLs / IDs / image URLs all accepted.")],
            reference_title: Annotated[str | None, Field(description="The ORIGINAL video title that the reference thumbnail was made for. PASS THIS whenever you have it (it comes back as `title` on each outlier from find_outlier_references). Without it, the mapping is forced to photocopy visuals; with it, Gemini can reason about WHY the reference's design choices fit ITS title before adapting that logic to the user's NEW title.")] = None,
            style_preset: Annotated[str, Field(description="person_focal | faceless | none. Default person_focal.")] = "person_focal",
            custom_instructions: Annotated[str | None, Field(description="Optional free-text directives from the user that get appended to the compose prompt with highest salience (they override the default rules where they conflict). Use for sticky design preferences like 'always 3 people not 1', 'dark moody palette regardless of reference', 'include the word EXPOSED in the corner'. Driven by the widget's Custom Instructions textarea.")] = None,
        ) -> str:
            """Single-call reasoned compose:
              1. One multimodal Gemini call sees the reference image + its
                 original title + the user's new title.
              2. Gemini internally walks a comprehensive visual checklist
                 (meta, composition, objects, palette, lighting, text OCR,
                 semantic relationships, visual devices) as chain-of-thought.
              3. Reasons about WHY each element fits the reference title.
              4. Maps that design logic onto the user's new title.
              5. Outputs one polished natural-language image-gen prompt.

            Earlier two-call version (separate vision-to-JSON + mapping)
            blew claude.ai's ~60s MCP timeout. Single-call keeps the same
            reasoning rigor (the checklist is embedded in the prompt) at
            roughly half the latency.
            """
            import json
            resolved = _resolve_reference_url(reference_url)
            if not resolved:
                return json.dumps({"success": False, "error": "Invalid reference URL."})

            # If the caller didn't pass the reference's original title but
            # the URL is a YouTube video, fetch the title via youtubei.js
            # so the reasoned mapping has the semantic anchor it needs.
            # Falls back silently if extraction fails (mapping still works
            # from image alone, just less effectively).
            if not reference_title:
                vid = extract_video_id(reference_url) or extract_video_id(resolved)
                if vid:
                    info = extract_video_info(vid)
                    if info.get("success") and info.get("title"):
                        reference_title = info["title"]
                        logger.info(f"auto-fetched reference title via youtubei.js: {reference_title!r}")

            mapped, map_err = _map_reference_to_title_via_gemini(
                title=title,
                reference_url=resolved,
                reference_title=reference_title,
                style_preset=style_preset,
                custom_instructions=custom_instructions,
            )
            if not mapped:
                return json.dumps({
                    "success": False,
                    "error": f"Reasoned mapping failed: {map_err or 'unknown'}",
                })

            payload = {
                "success": True,
                "title": title,
                "reference_url": resolved,
                "reference_title": reference_title,
                "style_preset": style_preset,
                "prompt": mapped,
                "mode": "reasoned",
            }
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
                "DEFAULT entry point when the user wants a thumbnail and a "
                "title/topic is available. Searches algrow's database (50k+ "
                "YouTube channels) for topically-similar videos with proven "
                "outlier performance and renders them as a clickable grid in "
                "an inline widget.\n\n"
                "AFTER CALLING THIS, STOP. The widget is fully interactive — "
                "the user drives three button clicks themselves:\n"
                "  1. Pick ONE reference card (single-select, replaces prior).\n"
                "  2. Click 'Create Prompt' — widget calls compose_thumbnail_"
                "prompt server-side, fills the prompt textarea with an "
                "engineered prompt.\n"
                "  3. Click 'Generate' — widget calls generate_thumbnail with "
                "the engineered prompt verbatim.\n"
                "You do NOT call compose_thumbnail_prompt or generate_thumbnail "
                "yourself. Calling generate_thumbnail right after this mounts a "
                "SECOND widget, wastes Kie credits, and skips the user's "
                "reference choice. The correct behavior after calling this tool "
                "is to wait — say a short sentence like 'Pick one and I'll be "
                "here if you need anything else' and then stop. The next user "
                "message will tell you what they want next.\n\n"
                "Pass `title` whenever you have it — the widget pre-fills its "
                "title field so the user doesn't retype.\n\n"
                "Ranking: vector-similarity to the topic, filtered to "
                "outlier_score >= 2.0 (video got >= 2× its channel's typical "
                "views). Default content_type is longform; use shorts for "
                "shortform references."
            ),
            meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
        )
        async def find_outlier_references_tool(
            topic: Annotated[str, Field(description="Algrow semantic search query. When `title` is being passed, set `topic` to the EXACT SAME STRING as `title` (verbatim, no elaboration, no keyword expansion, no rewording). Algrow's semantic search handles full titles fine, and your elaborated versions actually retrieve worse — they over-generalize the query and surface off-topic outliers. ONLY produce a different `topic` value when no `title` is available (e.g. the user just said 'show me viral elephant thumbnails' with no specific video idea) — in that case keep it short, 2–4 words.")],
            title: Annotated[str | None, Field(description="The user's video title — what the thumbnail is for. e.g. 'Why Amazon Prime is Failing', 'Every Elephant Explained in 11 Minutes'. The widget pre-fills its TITLE field with this AND the server uses it as the basis for the engineered prompt after the user picks a reference. ALWAYS pass this when you have the title. When you pass it, `topic` should be set to this same string verbatim — see the `topic` field doc.")] = None,
            content_type: Annotated[str, Field(description="longform or shorts. Default longform.")] = "longform",
            limit: Annotated[int, Field(description="Max thumbnails to return per page. Default 12, capped at 24.")] = 12,
            min_outlier_score: Annotated[float, Field(description="Floor on outlier multiplier — only include videos that outperformed their channel average by at least this factor. Default 2.0 (validated against hand-eval: lower lets in too much noise, higher misses too many strong references).")] = 2.0,
            page: Annotated[int, Field(description="1-indexed algrow page. Default 1. Used by the widget's 'load more' pagination — Claude shouldn't normally pass this directly.")] = 1,
        ) -> str:
            """DEFAULT entry point when the user wants a thumbnail and a
            title is available. Drives a three-step in-widget flow that the
            user completes themselves — you call this ONCE and then wait.

              1. You call THIS tool with `topic` (algrow search phrase) +
                 `title` (the user's video title). Widget renders the
                 outlier grid and pre-fills its title field.
              2. User clicks ONE reference card. Widget enables the
                 "Create Prompt" button. User clicks it. Widget calls
                 compose_thumbnail_prompt server-side, fills the prompt
                 textarea with the engineered result.
              3. User reviews/edits the prompt and clicks Generate.

            All three steps happen inside the widget — you do NOT call
            compose_thumbnail_prompt or generate_thumbnail yourself in
            this flow. Just call find_outlier_references and stop.

            Do NOT call generate_thumbnail with find_outliers_first=True
            unless the user explicitly says "just pick a good one for me"
            or "surprise me" — auto-pick removes the user's choice and
            burns a generation on a reference they may not have wanted.
            """
            import json

            # Belt-and-suspenders: if the caller provided a title, use it as
            # the algrow query regardless of what `topic` was set to. The
            # tool description tells Claude to pass title verbatim as topic,
            # but Claude tends to elaborate ("hidden gems budget fashion
            # finds") which over-generalizes the semantic search and
            # surfaces off-topic outliers. Force the verbatim title here.
            search_query = (title or topic or "").strip()
            # Fresh task invocation — clear any saved WIP for this title so
            # the widget starts clean instead of restoring stale state from
            # a prior chat session under the same title.
            if title:
                bucket = _load_state_bucket()
                if bucket.pop(_state_key(title), None) is not None:
                    _save_state_bucket(bucket)
            outliers, error, effective_topic, has_more = _fetch_outliers_from_algrow(
                topic=search_query,
                content_type=content_type,
                limit=limit,
                min_outlier_score=min_outlier_score,
                page=page,
            )
            # Only include the error key on actual error — Claude tends to
            # read the *presence* of an "error" field as a failure signal
            # even when the value is null. Empty outliers + no error key
            # means "algrow had nothing for this topic", not a tool fault.
            payload = {
                "view": "outlier_picker",
                "topic": effective_topic or search_query,
                "content_type": content_type,
                "outliers": outliers,
                "count": len(outliers),
                "page": page,
                "has_more": has_more,
            }
            if title:
                payload["title"] = title
            if error:
                payload["error"] = error
            return json.dumps(payload, default=str)

    # ----- Back-compat alias: generate_thumbnail_from_video ---------------
    # The clean tool is `extract_reference_from_video(url, user_title=X)`
    # — passing user_title triggers the auto-pipeline. But cached connector
    # tool lists in claude.ai may still reference the old name. This alias
    # forwards to the new behavior so stale caches don't break with
    # "Unknown tool" errors. Safe to remove once all connectors are re-synced.
    @mcp.tool(
        name="generate_thumbnail_from_video",
        title="(Alias) Generate Thumbnail From Reference Video",
        description=(
            "DEPRECATED ALIAS — prefer extract_reference_from_video with "
            "user_title set. Forwards to the same auto-pipeline (extract → "
            "compose → generate, mounted in the widget with stage-by-stage "
            "loading indicators). Kept for back-compat with cached "
            "connector tool lists."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def generate_thumbnail_from_video_alias(
        user_title: Annotated[str, Field(description="The user's NEW video title — what THEIR thumbnail is for.")],
        reference_video_url: Annotated[str, Field(description="YouTube URL of the reference video the user wants to base the design on.")],
        style_preset: Annotated[str, Field(description="person_focal | faceless | none. Default 'none'.")] = "none",
        aspect_ratio: Annotated[str, Field(description="16:9 / 9:16 / etc. Default 16:9.")] = "16:9",
        resolution: Annotated[str, Field(description="1K / 2K / 4K. Default 2K.")] = "2K",
        model: Annotated[str, Field(description="nano-banana-pro | seedream-4.5-edit | seedream-5.0-lite. See canonical tool for behavior.")] = "nano-banana-pro",
    ) -> str:
        # Forward directly to the canonical tool's behavior.
        return await extract_reference_from_video_tool(
            url_or_id=reference_video_url,
            user_title=user_title,
            model=model,
        )

    # ----- Reference by direct YouTube URL --------------------------------
    # Lets the user (via chat or widget paste) say "make a thumbnail like
    # THIS video" by handing over a YouTube URL/ID. Server pulls the
    # title + canonical thumbnail via youtubei.js (no API key needed),
    # giving the reasoned-mapping pipeline the same inputs it'd get from
    # an outlier pick.
    @mcp.tool(
        name="extract_reference_from_video",
        title="Use a Specific YouTube Video as the Reference",
        description=(
            "Use this when the user wants to base a thumbnail on a SPECIFIC "
            "YouTube video they've linked. Pulls the video's title + best-"
            "quality thumbnail via youtubei.js and mounts the widget.\n\n"
            "BEHAVIOR DEPENDS ON `user_title`:\n"
            "  • If you pass `user_title` (you have the user's video title): "
            "the widget auto-runs the FULL pipeline — compose the engineered "
            "prompt, then generate the image. The user sees stage-by-stage "
            "progress and the finished thumbnail appears in ~45-65s.\n"
            "  • If you don't pass `user_title` (user only shared a video, "
            "no specific title yet): the widget mounts with the reference "
            "preselected. The user clicks Create Prompt → Generate manually.\n\n"
            "ALWAYS pass `user_title` when you have it. After calling, STOP."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def extract_reference_from_video_tool(
        url_or_id: Annotated[str, Field(description="YouTube URL (watch / shorts / youtu.be / embed / live) or bare 11-character video ID.")],
        user_title: Annotated[str | None, Field(description="The user's NEW video title — what THEIR thumbnail is for (different from the reference video's own title). Widget pre-fills its title field with this. Always pass when you have it.")] = None,
        model: Annotated[str, Field(description="Image-gen backend the widget should preselect. The model dictates HOW the reference is used:\n• 'nano-banana-pro' (DEFAULT) — Gemini 3 Pro Image direct. Highest quality, REGENERATES a fresh image from the prompt + reference as a style hint. 3 credits per gen. Safety filter blocks named real people / brands / references containing public figures (returns BlockedReason.OTHER).\n• 'gpt-image-2' — Algrow proxy → OpenAI GPT Image 2. REGENERATES (text-to-image, or image-to-image when a reference is supplied). CHEAPEST option at 1 credit per gen. Good for fast iteration when Nano Banana Pro's quality isn't worth 3x the cost.\n• 'seedream-5.0-lite' — Algrow proxy, REGENERATES from prompt with optional reference as style hint. 2 credits. More permissive safety filter than Gemini direct — use this when the user's title is a biography / brand-history / public-figure piece and Nano Banana Pro keeps blocking. Right seedream variant for 'thumbnail inspired by reference but with a different subject', because it actually generates a new subject.\n• 'seedream-4.5-edit' — Algrow proxy, IMAGE-TO-IMAGE. Edits the reference rather than regenerating; preserves identity, layout, and pose of the subject. 2 credits. Use this ONLY when the user wants to keep most of the reference and just tweak details. NOT appropriate when the user wants the design pattern applied to a different subject — it'll return the same person doing the same thing.\n\nDecision tree: blocked by Gemini AND wants a different subject from the reference → seedream-5.0-lite. Wants the SAME subject with minor edits → seedream-4.5-edit. Wants cheapest reasonable quality → gpt-image-2. Otherwise → nano-banana-pro.")] = "nano-banana-pro",
    ) -> str:
        import json
        # Fresh task invocation — clear any saved WIP for this title so the
        # widget starts clean. Cross-device sync within one conversation
        # still works because chat reopens replay cached tool results
        # rather than re-firing this tool.
        if user_title:
            bucket = _load_state_bucket()
            if bucket.pop(_state_key(user_title), None) is not None:
                _save_state_bucket(bucket)
        info = extract_video_info(url_or_id)
        if not info.get("success"):
            return json.dumps({
                "view": "outlier_picker",
                "topic": user_title or "",
                "outliers": [],
                "count": 0,
                "error": info.get("error") or "Failed to extract video info.",
                **({"title": user_title} if user_title else {}),
            })

        outlier = {
            "video_id": info.get("video_id"),
            "title": info.get("title") or "",
            "thumbnail_url": info.get("thumbnail_url"),
            "channel_name": info.get("channel_name") or "",
            "channel_thumbnail": info.get("channel_thumbnail"),
            "view_count": info.get("view_count"),
            "outlier_score": None,
            "url": f"https://www.youtube.com/watch?v={info.get('video_id')}" if info.get("video_id") else None,
        }
        payload = {
            "view": "outlier_picker",
            "topic": user_title or info.get("title") or "",
            "outliers": [outlier],
            "count": 1,
            "content_type": "longform",
            "single_reference": True,
            "model": model,
        }
        if user_title:
            payload["title"] = user_title
            # User provided BOTH a title and a reference video → fire the
            # full pipeline automatically. Widget will chain compose +
            # generate via callServerTool with stage-by-stage loading
            # indicators (no single 60s blank wait).
            payload["auto_pipeline"] = True
        return json.dumps(payload, default=str)

    # ----- Claude-as-editor: refine the existing engineered prompt --------
    # The original compose path (find_outlier_references / extract_reference_
    # from_video → user picks → Create Prompt → Generate) stays untouched.
    # These two tools are ONLY for FOLLOW-UP refinements after the user asks
    # Claude to change something. Pattern:
    #   1. get_widget_prompt(title) — Claude reads the current prompt.
    #   2. Claude writes the edited version in chat (user sees the diff).
    #   3. set_widget_prompt(title, new_prompt) — Claude pushes it back.
    #      Widget re-mounts with the new prompt pre-filled; user clicks
    #      Generate. The original reference card stays selected.
    @mcp.tool(
        name="get_widget_prompt",
        title="Read the Current Thumbnail Prompt",
        description=(
            "Read the engineered prompt currently saved in the thumbnail "
            "studio widget for a given user_title. Call this when the user "
            "asks for a refinement to the current thumbnail ('make watches "
            "more realistic', 'darken the background', 'change the text') "
            "— you'll need to see the existing prompt before editing it.\n\n"
            "Returns: {success, title, prompt, reference_thumbnail_url, "
            "reference_title}. After reading, write the edited prompt in "
            "chat (so the user can see your changes), then call "
            "set_widget_prompt to push it to the widget.\n\n"
            "Prerequisites: the user must have already run the original "
            "compose flow for this title (via find_outlier_references or "
            "extract_reference_from_video → Create Prompt in widget). "
            "Returns success:false if no prompt is saved yet — in that "
            "case, don't call set_widget_prompt; instead route the user "
            "back through the original compose flow."
        ),
    )
    async def get_widget_prompt_tool(
        user_title: Annotated[str, Field(description="The user's video title — same value used when the original compose ran. State is keyed by this.")],
    ) -> str:
        import json
        bucket = _load_state_bucket()
        state = bucket.get(_state_key(user_title))
        if not state or not state.get("prompt"):
            return json.dumps({
                "success": False,
                "error": "No engineered prompt found for this title. The user needs to run the original compose flow first (find_outlier_references or extract_reference_from_video → Create Prompt in widget).",
            })
        sel = state.get("selectedOutlier") or {}
        return json.dumps({
            "success": True,
            "title": user_title,
            "prompt": state["prompt"],
            "reference_thumbnail_url": sel.get("thumbnail_url"),
            "reference_title": sel.get("title"),
            "reference_channel": sel.get("channel_name"),
        })

    @mcp.tool(
        name="set_widget_prompt",
        title="Push Edited Prompt to Widget",
        description=(
            "Save an edited engineered prompt to the thumbnail studio "
            "widget for a given user_title. Mounts the widget with the new "
            "prompt pre-filled and the previously-selected reference still "
            "highlighted — the user just clicks Generate.\n\n"
            "Use this AFTER reading the current prompt via get_widget_prompt "
            "AND writing the edited version in chat. Do NOT call this with "
            "a from-scratch prompt that you wrote without seeing the "
            "existing one — refinements should preserve everything the "
            "user didn't explicitly ask to change.\n\n"
            "After calling this, STOP — the widget handles the rest."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def set_widget_prompt_tool(
        user_title: Annotated[str, Field(description="The user's video title — same value used when the original compose ran. State is keyed by this.")],
        new_prompt: Annotated[str, Field(description="The edited engineered prompt to save. Goes verbatim into the widget's prompt textarea and is sent verbatim to the image-gen model when the user clicks Generate. Preserve every unchanged detail from the original prompt — refinements should be surgical, not wholesale rewrites.")],
    ) -> str:
        import json
        bucket = _load_state_bucket()
        state = bucket.get(_state_key(user_title)) or {"title": user_title}
        state["prompt"] = new_prompt
        state["promptIsComposed"] = True
        # Multi-turn editing: BEFORE clearing the cached generation result,
        # stash its image URL into priorImageForReference. That way the
        # next Generate will use the previously-generated image as the
        # reference (so Gemini edits the prior output) instead of falling
        # back to the original picked reference. Without this, the widget
        # mounts fresh with no in-memory lastResultPayload, the async
        # state restore finds a cleared lastResultPayload, and the prior
        # image is lost entirely — making the refinement a fresh
        # generation from the source-video thumbnail.
        prior = state.get("lastResultPayload") or {}
        prior_images = prior.get("images") or []
        if prior_images:
            state["priorImageForReference"] = prior_images[0]
        # Clear the cached generation result — the user is starting a new
        # run from the refined prompt, the OLD image is no longer current.
        # Without this, the widget's async state restore would re-render
        # the stale image into the preview pane on the next mount.
        state["lastResultPayload"] = None
        state["ts"] = int(time.time() * 1000)
        bucket[_state_key(user_title)] = state
        _save_state_bucket(bucket)

        sel = state.get("selectedOutlier")
        payload = {
            "view": "outlier_picker",
            "topic": user_title,
            "title": user_title,
            "outliers": [sel] if sel else [],
            "count": 1 if sel else 0,
            "single_reference": bool(sel),
            "refined_prompt": new_prompt,
        }
        return json.dumps(payload, default=str)

    # ----- Widget state persistence (cross-device WIP sync) ---------------
    # The widget calls these via app.callServerTool to save/restore its
    # in-progress state across chat closes AND devices. They're internal
    # helpers — Claude in chat should never invoke them directly.
    @mcp.tool(
        name="save_widget_state",
        title="Save Widget State (internal)",
        description=(
            "INTERNAL widget helper. The thumbnail-studio widget calls this "
            "to persist its in-progress state (selected reference, composed "
            "prompt, last generation result, etc.) keyed by the video title. "
            "Do NOT call this from chat — only the widget calls it via "
            "callServerTool. Returns {success: true}."
        ),
    )
    async def save_widget_state_tool(
        key: Annotated[str, Field(description="Lowercased, trimmed video title used as the bucket key.")],
        state: Annotated[dict, Field(description="Opaque state blob the widget wants to persist. Stored as-is.")],
    ) -> str:
        import json
        k = _state_key(key)
        if not k:
            return json.dumps({"success": False, "error": "Empty key."})
        bucket = _load_state_bucket()
        bucket[k] = {**(state or {}), "ts": int(time.time() * 1000)}
        # Cap to N most-recent entries so the JSON file stays bounded.
        if len(bucket) > _MAX_STATE_ENTRIES:
            kept = sorted(bucket.items(), key=lambda x: x[1].get("ts", 0), reverse=True)[:_MAX_STATE_ENTRIES]
            bucket = dict(kept)
        _save_state_bucket(bucket)
        return json.dumps({"success": True})

    @mcp.tool(
        name="load_widget_state",
        title="Load Widget State (internal)",
        description=(
            "INTERNAL widget helper. The thumbnail-studio widget calls this "
            "on mount to restore its in-progress state. Do NOT call this "
            "from chat — only the widget calls it via callServerTool. "
            "Returns {success: true, state: <blob or null>}."
        ),
    )
    async def load_widget_state_tool(
        key: Annotated[str, Field(description="Lowercased, trimmed video title used as the bucket key.")],
    ) -> str:
        import json
        k = _state_key(key)
        if not k:
            return json.dumps({"success": True, "state": None})
        bucket = _load_state_bucket()
        return json.dumps({"success": True, "state": bucket.get(k)})

    @mcp.tool(
        name="upload_reference_image",
        title="Upload Reference Image (internal)",
        description=(
            "INTERNAL widget helper. The thumbnail-studio widget calls this "
            "when the user drops or picks a local image file in the "
            "References section. Saves the bytes under /generated/refs/ "
            "with a UUID filename and returns a public URL the widget can "
            "drop into the reference URLs list. Do NOT call from chat."
        ),
    )
    async def upload_reference_image_tool(
        image_b64: Annotated[str, Field(description="Base64-encoded image bytes (no data: prefix). Capped at ~12 MB decoded.")],
        content_type: Annotated[str, Field(description="MIME type — image/png, image/jpeg, image/webp, image/gif.")] = "image/png",
        filename_hint: Annotated[str | None, Field(description="Optional original filename for the suffix only — sanitized server-side. Default: derive extension from content_type.")] = None,
    ) -> str:
        import base64 as _b64
        import json
        import uuid as _uuid_loc

        ALLOWED = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
                   "image/webp": ".webp", "image/gif": ".gif"}
        ct = (content_type or "").lower().strip()
        if ct not in ALLOWED:
            return json.dumps({"success": False, "error": f"Unsupported content_type: {ct or '(none)'}. Allowed: png, jpeg, webp, gif."})
        # Strip optional data-URL prefix that browsers might leak through.
        b64 = (image_b64 or "").strip()
        if b64.startswith("data:"):
            comma = b64.find(",")
            if comma > 0:
                b64 = b64[comma + 1:]
        try:
            raw = _b64.b64decode(b64, validate=True)
        except Exception as e:
            return json.dumps({"success": False, "error": f"Bad base64: {str(e)[:120]}"})
        # 12 MB cap — Gemini Vision rejects anything bigger anyway.
        if len(raw) > 12 * 1024 * 1024:
            return json.dumps({"success": False, "error": f"Image too large ({len(raw):,} bytes). Max 12 MB."})
        if len(raw) < 256:
            return json.dumps({"success": False, "error": "Image too small / empty."})

        ext = ALLOWED[ct]
        fname = f"{_uuid_loc.uuid4().hex}{ext}"
        try:
            (_REFS_DIR / fname).write_bytes(raw)
        except Exception as e:
            logger.warning(f"upload_reference_image write failed: {e}")
            return json.dumps({"success": False, "error": "Couldn't save upload."})
        base = (os.environ.get("PUBLIC_BASE_URL") or "").rstrip("/")
        url = f"{base}/generated/refs/{fname}" if base else f"/generated/refs/{fname}"
        logger.info(f"upload_reference_image saved {len(raw):,}B → {url}")
        _ = filename_hint  # accepted for telemetry but not used for safety
        return json.dumps({"success": True, "url": url, "bytes": len(raw), "content_type": ct})

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


async def generated_image(request: Request) -> Response:
    """Serve a generated image by filename. Files live in _GENERATED_DIR.
    Filenames are server-generated UUIDs so path traversal isn't a worry,
    but we still sanitize to be safe (reject any '/' or '..').
    """
    fname = request.path_params.get("filename", "")
    if "/" in fname or ".." in fname or not fname:
        return Response("Not found", status_code=404)
    path = _GENERATED_DIR / fname
    if not path.is_file():
        return Response("Not found", status_code=404)
    ext = path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        ext.lstrip("."), "application/octet-stream"
    )
    return Response(
        content=path.read_bytes(),
        media_type=mime,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/generated/{filename}", generated_image),
    ],
    middleware=[Middleware(CorsMiddleware), Middleware(AuthMiddleware)],
    lifespan=lifespan,
)
app.mount("/", _http_app)


def main():
    # stdio entry — for Claude Desktop config or `mcp dev server.py`
    _mcp.run()


if __name__ == "__main__":
    main()
