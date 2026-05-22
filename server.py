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

def _tolerant_json_loads(text: str) -> dict:
    """Parse model output as JSON, tolerating leading/trailing prose AND
    markdown code fences. Uses JSONDecoder.raw_decode to consume the first
    valid JSON object and ignore the rest."""
    s = (text or "").strip()
    # Strip ```json ... ``` fences if present
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Find first { or [ — model may prepend explanatory text
    for i, ch in enumerate(s):
        if ch in "{[":
            s = s[i:]
            break
    try:
        return _json_top.loads(s)
    except _json_top.JSONDecodeError:
        # Trailing junk — use raw_decode to consume first valid object only
        decoder = _json_top.JSONDecoder()
        obj, _idx = decoder.raw_decode(s)
        return obj

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
from mcp.server.fastmcp import FastMCP, Context

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

# Compose background-submit cache. compose_thumbnail_prompt_tool returns
# pending+task_id immediately and spawns the real work as an asyncio task.
# claude.ai's MCP client has a hard ~60s request timeout that does NOT
# respect progress notifications, but compose with thinking=high on both
# vision and synth passes routinely runs 60-90s. The cache decouples
# wall-clock latency from per-request deadlines: each individual call
# returns in <1s. The widget polls check_compose_status until the entry
# flips to success/fail.
_COMPOSE_PREFIX = "compose_pending:"
_COMPOSE_PENDING: dict[str, dict] = {}
_COMPOSE_PENDING_MAX = 200

# =============================================================================
# Channel-style DNA pipeline
# =============================================================================
# Build a "channel design DNA" from a creator's top-by-views thumbnails: one
# vision-pass per thumbnail (parallel), then a synthesis step that distills
# the recurring style across all of them and emits a text-only image prompt
# that — applied to a NEW title — recreates the channel's look without any
# reference image at gen time. Cached in Postgres for 30 days because (a) the
# pipeline costs ~$0.30 in API calls and (b) a channel's style barely drifts
# faster than that.

_CHANNEL_STYLE_THUMB_COUNT = 10            # how many top thumbnails to mine
_CHANNEL_STYLE_FETCH_BREADTH = 100         # over-fetch then sort by view_count
_CHANNEL_DNA_TTL_DAYS = 30
_CHANNEL_COMPOSE_PREFIX = "channel_compose_pending:"   # reuses _COMPOSE_PENDING


def _resolve_channel_via_algrow(handle_or_id: str) -> tuple[dict | None, str | None]:
    """Call algrow's /api/channels/resolve. Returns (info, error).
    info contains at minimum: channel_id, channel_name, channel_thumbnail (avatar).
    Reads the per-request algora API key from auth_ctx — caller must be
    inside an AuthMiddleware-wrapped request."""
    from auth_ctx import get_current_api_key
    api_key = (get_current_api_key() or "").strip()
    if not api_key:
        return None, "Algrow integration requires an algora API key on this request."
    s = (handle_or_id or "").strip()
    if not s:
        return None, "Empty channel identifier."
    # Algrow's resolver wants the handle form (@xxx). If the user already
    # gave a UCxxx ID, skip the resolve and just synthesize a stub — the
    # videos endpoint takes channel_id directly.
    if re.match(r"^UC[A-Za-z0-9_-]{22}$", s):
        return {"channel_id": s, "channel_name": s, "channel_thumbnail": None, "handle": None}, None
    payload_input = s if s.startswith("@") else f"@{s}"
    try:
        resp = requests.post(
            f"{_ALGROW_API_BASE}/api/channels/resolve",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"input": payload_input},
            timeout=20,
        )
    except Exception as e:
        return None, f"Algrow resolve failed: {str(e)[:120]}"
    if resp.status_code == 404:
        return None, f"Algrow couldn't resolve {payload_input!r}. Channel may not be indexed."
    if resp.status_code != 200:
        return None, f"Algrow resolve HTTP {resp.status_code}: {resp.text[:160]}"
    try:
        data = resp.json()
    except Exception:
        return None, "Algrow resolve returned non-JSON."
    if not data.get("success") and not data.get("channel_id"):
        return None, data.get("error") or "Algrow resolve had no channel_id in payload."
    info = {
        "channel_id": data.get("channel_id"),
        "channel_name": data.get("channel_name") or data.get("channel_title") or data.get("title") or payload_input,
        "channel_thumbnail": data.get("channel_thumbnail") or data.get("avatar_url"),
        "handle": data.get("handle") or payload_input,
    }
    if not info["channel_id"]:
        return None, "Algrow resolve returned no channel_id."
    return info, None


def _fetch_channel_videos_via_innertube(channel_id: str, limit: int = _CHANNEL_STYLE_FETCH_BREADTH
                                          ) -> tuple[list[dict], dict | None, str | None]:
    """Subprocess the node bridge to scrape a channel's videos + meta via
    Innertube. Returns (videos, channel_meta, error). channel_meta is
    {name, avatar_url, banner_url, subscriber_count_text} or None on failure."""
    import subprocess
    try:
        p = subprocess.run(
            ["node", "/root/apps/thumbnails/extract_channel_videos.mjs",
             channel_id, str(int(limit))],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return [], None, "Innertube scrape timed out (>60s)."
    except Exception as e:
        return [], None, f"Innertube scrape subprocess failed: {str(e)[:120]}"
    if p.returncode != 0 and not p.stdout:
        return [], None, f"Innertube scrape exited {p.returncode}: {p.stderr[:160]}"
    try:
        data = _json_top.loads(p.stdout.strip().split("\n")[-1])
    except Exception:
        return [], None, f"Innertube scrape returned non-JSON: {p.stdout[:160]}"
    if not data.get("success"):
        return [], None, data.get("error") or "Innertube scrape failed."
    raw = data.get("videos") or []
    out = []
    for v in raw:
        vid = v.get("video_id")
        if not vid:
            continue
        out.append({
            "video_id": vid,
            "title": v.get("title") or "",
            "thumbnail_url": v.get("thumbnail_url") or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            "view_count": int(v.get("view_count") or 0),
        })
    meta = data.get("channel") or None
    return out, meta, None


def _fetch_channel_top_videos(channel_id: str, n: int = _CHANNEL_STYLE_THUMB_COUNT,
                              *, api_key: str | None = None
                              ) -> tuple[list[dict], str | None]:
    """Fetch up to _CHANNEL_STYLE_FETCH_BREADTH recent videos from algrow,
    sort by view_count desc, return top-N as [{video_id, title, thumbnail_url,
    view_count}].

    Pass `api_key` explicitly when called from a worker thread (asyncio
    to_thread). Inside a request handler it can be omitted and we'll pull
    it from the ContextVar — but ContextVars don't propagate into the
    thread pool, so explicit-key is the safer default."""
    from auth_ctx import get_current_api_key
    api_key = (api_key or get_current_api_key() or "").strip()
    if not api_key:
        return [], "Algrow integration requires an algora API key on this request."
    try:
        resp = requests.get(
            f"{_ALGROW_API_BASE}/api/channels/{channel_id}/videos",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": str(_CHANNEL_STYLE_FETCH_BREADTH)},
            timeout=45,
        )
    except Exception as e:
        return [], f"Algrow channel-videos fetch failed: {str(e)[:140]}"
    if resp.status_code != 200:
        return [], f"Algrow channel-videos HTTP {resp.status_code}: {resp.text[:160]}"
    try:
        data = resp.json()
    except Exception:
        return [], "Algrow channel-videos returned non-JSON."
    videos = data.get("videos") or []
    if not videos:
        # Algrow doesn't have this channel indexed — scrape via Innertube.
        # Returns raw videos newest-first; we'll sort by view_count below
        # using the same code path so behaviour matches Algrow-served data.
        logger.info(f"channel {channel_id} not in Algrow index — falling back to Innertube scrape")
        scraped, _scr_meta, scr_err = _fetch_channel_videos_via_innertube(channel_id, _CHANNEL_STYLE_FETCH_BREADTH)
        if not scraped:
            return [], scr_err or "Channel has no videos in Algrow OR via Innertube."
        videos = scraped
    # Sort by view_count desc — channel's actual top performers
    videos_sorted = sorted(
        videos,
        key=lambda v: int(v.get("view_count") or 0),
        reverse=True,
    )
    out = []
    for v in videos_sorted[:n]:
        vid = v.get("video_id")
        if not vid:
            continue
        out.append({
            "video_id": vid,
            "title": v.get("title") or "",
            "thumbnail_url": v.get("thumbnail_url") or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            "view_count": int(v.get("view_count") or 0),
        })
    return out, None


def _channel_dna_load(channel_id: str) -> dict | None:
    """Return cached DNA payload or None on miss/expired."""
    try:
        with _db().cursor(cursor_factory=_pg_extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT dna_json, source_thumb_urls, source_titles, source_analyses, "
                "channel_name, channel_avatar, handle, computed_at "
                "FROM channel_style_dna WHERE channel_id = %s AND expires_at > NOW()",
                (channel_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            row = dict(row)
            # Force re-vision-pass if the cache was written under the old
            # schema and has no source_analyses — matching needs those.
            if not row.get("source_analyses"):
                return None
            return row
    except Exception as e:
        logger.error(f"channel_dna_load DB error: {e}")
        return None


def _channel_dna_save(channel_id: str, handle: str | None, channel_name: str,
                      channel_avatar: str | None, dna_json: dict | None,
                      source_thumb_urls: list[str], source_titles: list[str],
                      source_analyses: list[dict]) -> None:
    try:
        with _db().cursor() as cur:
            cur.execute(
                """INSERT INTO channel_style_dna
                       (channel_id, handle, channel_name, channel_avatar, dna_json,
                        source_thumb_urls, source_titles, source_analyses,
                        computed_at, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(),
                           NOW() + INTERVAL '%s days')
                   ON CONFLICT (channel_id) DO UPDATE SET
                       handle = EXCLUDED.handle,
                       channel_name = EXCLUDED.channel_name,
                       channel_avatar = EXCLUDED.channel_avatar,
                       dna_json = EXCLUDED.dna_json,
                       source_thumb_urls = EXCLUDED.source_thumb_urls,
                       source_titles = EXCLUDED.source_titles,
                       source_analyses = EXCLUDED.source_analyses,
                       computed_at = NOW(),
                       expires_at = NOW() + INTERVAL '%s days'""",
                (channel_id[:64], (handle or "")[:128], channel_name[:256],
                 (channel_avatar or "")[:512] or None,
                 _json_top.dumps(dna_json) if dna_json else None,
                 _json_top.dumps(source_thumb_urls),
                 _json_top.dumps(source_titles),
                 _json_top.dumps(source_analyses),
                 _CHANNEL_DNA_TTL_DAYS, _CHANNEL_DNA_TTL_DAYS),
            )
    except Exception as e:
        logger.error(f"channel_dna_save DB error: {e}")


_CHANNEL_DNA_SYNTHESIS_PROMPT = """ROLE & OBJECTIVE
You are ChannelDesignMatcher. You receive (a) a user's NEW video title and (b) JSON vision analyses of N of a single creator's top-performing YouTube thumbnails. The creator does NOT have a single uniform style — different videos use different design treatments depending on topic. Your job is two-step:

STEP 1 — MATCH
From the N analyzed thumbnails, pick the SINGLE one whose design treatment would best fit the new title.

ARCHETYPE GATE — do this FIRST, before any other reasoning:
  1. In one phrase, name the user's title's natural archetype. Choose from: single_hero_object | single_hero_face | comparison_split | character_action | infographic_diagram | reaction_face | scene_panoramic | text_focal. A "single hidden discovery" title like "NASA Just Discovered a Dark Planet No One Was Supposed to See" is single_hero_object. A "X vs Y" title is comparison_split. A "How [person] did [thing]" title is character_action. A "What it's like inside [place]" title is scene_panoramic.
  2. For each of the N candidate thumbnails, read its analysis's `composition.primary_archetype` field (or infer it from the analysis if missing).
  3. RULE: pick ONLY from candidates whose archetype composes with the user's title's archetype. Specifically:
       • A comparison_split candidate is INCOMPATIBLE with a single_hero_object title (and vice-versa). Comparison thumbnails imply "look at these two known things side by side" — they're wrong for a "newly discovered hidden thing" title.
       • A reaction_face candidate is incompatible with an infographic title.
       • A character_action candidate is incompatible with a scene_panoramic title.
       • In general: candidates whose archetype is the same as, or a strict superset of, the user's title's archetype are compatible. Everything else is rejected.
  4. If MULTIPLE candidates are archetype-compatible, fall through to PRIMARY SIGNAL + TIE-BREAKER below to pick among them.
  5. If NO candidate is archetype-compatible, pick the closest archetype (e.g. for single_hero_object pick another single-hero variant rather than a comparison split). Never pick an incompatible candidate just because its topic is close — topic-close + archetype-wrong produces direct-copy outputs that violate HOW-vs-WHAT.

PRIMARY SIGNAL — design treatment fit (apply ONLY to archetype-compatible candidates):
  • Composition archetype alignment (object_hero vs face_close_up vs character_action vs split_screen / comparison vs infographic etc — match the title's natural subject class).
  • Emotional tone alignment (the user's title's intended hook — discovery, shock, awe, curiosity, exposé — vs each thumbnail's tone).
  • Subject framing (does the title call for a single dominant subject? a comparison of two? a scene? pick a thumbnail that frames the analogous structural subject the same way).

TIE-BREAKER — topic similarity:
When two or more candidate thumbnails have equally applicable design treatments, prefer the one whose ORIGINAL TITLE is structurally closest to the user's new title — similar phrasing pattern ("How X Did Y" vs "Why X Happened"), similar narrative hook ("just discovered", "no one knew"), similar subject class (single hidden object vs comparison of two known objects vs character-driven event). For SINGLE-GENRE channels (e.g. a channel that only does space discoveries) topic similarity is a strong positive signal because the creator uses the same design family for the same topic class on purpose.

ANTI-OVERFIT GUARD — when topic similarity is wrong:
Do NOT pick a thumbnail just because the topics are close if its design treatment is clearly wrong for the new title. For MULTI-GENRE channels (a channel that mixes politics, engineering, heists, etc.) the wrong topic match can drag you into the wrong design family. Examples of what to avoid:
  • Picking a "How Iran's Leader Was Killed" looming-villain face composition for a "World's Deadliest Weapon" title — both are geopolitical, but the looming-face archetype doesn't fit a weapon hero shot. The right pick is a thumbnail whose archetype is object_hero / character_action, not face_close_up villain.
  • Picking a "vs" comparison thumbnail for a single-discovery title — same niche, wrong archetype.

Read the candidate thumbnails' archetypes from their analyses before deciding. If the closest-topic candidate has the right archetype, pick it. If the closest-topic candidate has a clearly wrong archetype, pick the next-best topic candidate whose archetype fits.

STEP 2 — APPLY
Take the matched thumbnail's design — its palette, typography, composition, lighting, effects vocabulary, production polish, emotional register — and write a single exhaustive image-gen prompt that recreates that exact design treatment for the new title. The image generator will receive your prompt and NO reference image; encode every visual choice in text.

HARD CONSTRAINT — HOW vs WHAT
You are transferring the matched thumbnail's design (HOW it looks), not its content (WHAT it shows). The matched thumbnail's specific subject — a named person, a named object, a recurring scene — DOES NOT carry over to the new image. The new image's subject comes from the new title, full stop. The matched thumbnail's design tells you how to render that new subject.

Test for every visual element you put in the final prompt: "Would I write this same instruction even if the matched thumbnail were of a totally different topic, as long as the design treatment was identical?" If no, you're smuggling subject content where you should be carrying design choice.

FORBIDDEN CONTENT TRANSFERS — hard rules:
  • Specific NOUNS from the matched reference's analysis do NOT appear in the final prompt unless the new title independently calls for them. If the matched reference's analysis contains an Earth + a Planet, and the new title is "NASA Discovered a Dark Planet", the final prompt must NOT write "Earth", "the Earth", "a familiar Earth", or describe Earth at all — Earth was the reference's content choice for ITS comparison title, not a structural element. Same for any named-object content: a specific weapon, a specific vehicle, a specific landmark, a specific celestial body, a specific person, a specific room, a specific prop.
  • Reference's TEXT-OVERLAY CONTENT (the literal words / labels) NEVER carry over. The text overlay's STRUCTURE transfers — count of labels, font category, weight, color, position, drop-shadow softness, letter-spacing. The actual WORDS are derived fresh from the user's new title. If the matched reference's labels were "EARTH" and "PLANET Y", do NOT write "EARTH" or "PLANET Y" as label content for the new image. Derive labels from the user's title (e.g. for "NASA Discovered a Dark Planet" → a single label like "DARK PLANET" or "DISCOVERED" or whatever the title naturally implies, NOT two labels echoing the comparison structure of the reference if the title isn't a comparison).
  • The matched reference's SLOT COUNT (number of major subjects in the frame) does not automatically transfer. A two-object comparison reference does not force the new image to have two objects if the new title is a single-hero title. The matcher should have rejected it on archetype grounds, but as a backstop: when the new title doesn't justify the slot count, drop the extra slots and let the hero subject claim the whole frame.

Concrete worked example for clarity:
  Matched reference is "EARTH | PLANET Y" — a comparison thumbnail with Earth on the left (label "EARTH") and Planet Y on the right (label "PLANET Y"), pure black void, identical typography on both labels.
  New title: "NASA Just Discovered a Dark Planet No One Was Supposed to See" — single_hero_object archetype.
  CORRECT final prompt: a single dark planet as the sole hero subject claiming most of the frame, on the same pure black void, with the same dramatic key-light + rim-light treatment as the matched reference, with ONE bold all-caps white sans-serif label derived from the new title (e.g. "DARK PLANET" centered, or "HIDDEN" / "DISCOVERED" in a complementary position). Earth does NOT appear. The word "EARTH" does NOT appear. Two-label comparison structure does NOT appear.
  WRONG final prompt (this is what we got last time and we are NOT doing it again): "place a familiar Earth as the scale anchor on the left… the label EARTH in the top-left… cool blues of Earth contrasting against…" — every Earth reference is content smuggling from the matched reference's title.

REFERENCE-AWARE PROMPT
Downstream gen receives your prompt AND the N reference images attached as visual input. Reference image 1 is the MATCHED THUMBNAIL — the one you picked in STEP 1. References 2..N are the other top-by-views thumbnails from the same channel, included as additional style anchors.

Phrase the final prompt in directive language that names the references explicitly so the gen model knows what role each one plays. Examples of the right shape:
  • "Match reference image 1's composition exactly — the looming foreground subject and the smaller hero object below."
  • "Lift the lighting setup from reference image 1: dramatic top-down rim light with deep underexposed shadows."
  • "Use the exact palette and typography treatment shown in reference image 1 (red `#FF0000` accent on near-black `#121212`, bold all-caps sans-serif white text in a solid red bar)."
  • "References 2-N show the channel's broader style range — confirm the palette and effect vocabulary from them, but DO NOT copy any of their subjects."

You can still include concrete hex codes, font specs, bbox coords, and lighting params alongside the directive language — the gen model benefits from both. But the prompt's load-bearing instructions should ANCHOR ON the reference images, not pretend they don't exist.

CRITICAL: even though the references carry strong visual information, the HOW-vs-WHAT rule still applies. The subject in your new image comes 100% from the new title. The references provide HOW to render that subject; they do NOT provide WHAT to render. The matched reference's specific person / brand / object DOES NOT carry over.

OUTPUT FORMAT — STRICT
Return ONLY a single JSON object. No markdown fencing, no preamble. Schema:
{
  "matched_index": <integer 0..N-1>,
  "matched_title": "<exact text of the matched thumbnail's original title>",
  "match_reason": "<one sentence explaining WHY this thumbnail's design treatment fits the new title — focus on design rationale, not topic similarity>",
  "channel_constants_lift": "<short list of channel-wide design constants you observed across all N thumbnails (palette, typography, production polish) that should override the matched thumbnail's value if they differ — only include things that ARE truly consistent across all N>",
  "final_prompt": "<ONE continuous prose paragraph, 500-1200 words, no markdown headings or bullet lists, that the image generator will receive verbatim ALONGSIDE the reference images. Phrase it as reference-aware directives: 'match reference image 1's composition', 'lift the lighting from reference image 1', 'use the palette and typography shown in reference image 1'. References 2-N (the other 9 channel thumbnails) are style anchors — confirm palette/effects but do not copy their subjects. The image's subject comes 100% from the new title; include the exact new-title text inside quotes for the typography spec. Embed concrete values (hex codes, font specs, bbox coords) alongside the reference-anchored directives — both layers help the gen model.>"
}

Be decisive on the match — pick one index. If two thumbnails feel equally applicable, pick the one with stronger design treatment (more distinctive lighting / clearer subject framing / more memorable color choice). Never blend two; that's averaging, which is what we're explicitly avoiding.
"""


def _build_matched_prompt(analyses: list[dict], titles: list[str],
                          thumb_urls: list[str], new_title: str,
                          channel_name: str, channel_handle: str | None,
                          *, gemini_key: str | None = None
                          ) -> tuple[dict | None, str | None]:
    """Single Gemini call that picks the best-matching analyzed thumbnail
    for the new title and applies its design to that title. Returns
    (result_dict, error). On success result_dict contains:
        matched_index, matched_title, matched_thumb_url, match_reason,
        channel_constants_lift, final_prompt

    BYOK-aware: pass `gemini_key` from the request context when calling
    via asyncio.to_thread; falls back to the per-user BYOK key or the
    shared env var when omitted (matches _analyze_image_via_gemini)."""
    if not gemini_key:
        try:
            from algrow_byok import get_key as _byok_get
            gemini_key = _byok_get("gemini") or _GEMINI_API_KEY
        except Exception:
            gemini_key = _GEMINI_API_KEY
    if not gemini_key:
        return None, "Gemini key not configured — set one at https://algora.online/settings/byok"
    if not analyses:
        return None, "No analyses to match against."

    # Build the input. Each analysis is paired with its title and index so
    # the model can refer to them unambiguously.
    indexed = []
    for i, (a, t) in enumerate(zip(analyses, titles)):
        indexed.append({"index": i, "original_title": t, "analysis": a})

    user_text = (
        f"CHANNEL: {channel_name}"
        + (f" ({channel_handle})" if channel_handle else "")
        + f"\nNEW TITLE: {new_title}\n"
        + f"N = {len(indexed)} thumbnails to choose from. Pick exactly one by index.\n"
        + f"ANALYZED THUMBNAILS:\n"
        + _json_top.dumps(indexed, default=str)
    )

    flash_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={gemini_key}"
    try:
        resp = requests.post(
            flash_url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"text": _CHANNEL_DNA_SYNTHESIS_PROMPT},
                        {"text": user_text},
                    ],
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.3,
                    "thinkingConfig": {"thinkingLevel": "high"},
                },
            },
            timeout=240,
        )
    except Exception as e:
        return None, f"Gemini match+apply request failed: {str(e)[:160]}"
    if resp.status_code != 200:
        return None, f"Gemini match+apply HTTP {resp.status_code}: {resp.text[:200]}"
    try:
        body = resp.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        parsed = _tolerant_json_loads(text)
    except Exception as e:
        return None, f"Couldn't parse match+apply response: {str(e)[:160]}"

    idx = parsed.get("matched_index")
    final_prompt = parsed.get("final_prompt")
    if idx is None or not isinstance(idx, int) or not (0 <= idx < len(analyses)):
        return None, f"Match step returned invalid index {idx!r}."
    if not final_prompt:
        return None, "Match step returned no final_prompt."

    return {
        "matched_index": idx,
        "matched_title": parsed.get("matched_title") or titles[idx],
        "matched_thumb_url": thumb_urls[idx],
        "match_reason": parsed.get("match_reason") or "",
        "channel_constants_lift": parsed.get("channel_constants_lift") or "",
        "final_prompt": final_prompt,
    }, None



# Widget state used to live in widget_state.json next to this file. That
# broke when a user generated multiple thumbnails for the same title in
# one chat (each save overwrote the prior under the same title key), and
# on reopen older widget mounts showed blank instead of their own image.
# Storage is now a Postgres table keyed by instance_id (the JSON-RPC id
# of the tool call that mounted the widget), which is globally unique
# per widget mount. The title_key is still indexed so the chat-side
# "use my prior reference for this title" lookup keeps working — it
# resolves to the most recently updated row for that title.
import psycopg2 as _pg
import psycopg2.extras as _pg_extras

_DB_URL = os.environ.get("DATABASE_URL")
_db_conn_lock = threading.Lock()
_db_conn = None  # lazy singleton; reconnected on broken-pipe

def _db():
    """Lazily return a live psycopg2 connection. Reconnect on errors so the
    long-lived MCP process survives transient DB blips."""
    global _db_conn
    if not _DB_URL:
        raise RuntimeError("DATABASE_URL not configured for widget state.")
    with _db_conn_lock:
        if _db_conn is not None:
            try:
                _db_conn.cursor().execute("SELECT 1")
                return _db_conn
            except Exception:
                try: _db_conn.close()
                except Exception: pass
                _db_conn = None
        _db_conn = _pg.connect(_DB_URL)
        _db_conn.autocommit = True
        return _db_conn


def _state_key(s: str | None) -> str:
    return (s or "").strip().lower()[:200]


def _state_load_by_instance(instance_id: str) -> dict | None:
    with _db().cursor(cursor_factory=_pg_extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT state FROM widget_thumbnail_states WHERE instance_id = %s",
            (instance_id,),
        )
        row = cur.fetchone()
        return row["state"] if row else None


def _state_load_latest_by_title(title_key: str) -> dict | None:
    with _db().cursor(cursor_factory=_pg_extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT state FROM widget_thumbnail_states
               WHERE title_key = %s
               ORDER BY updated_at DESC LIMIT 1""",
            (title_key,),
        )
        row = cur.fetchone()
        return row["state"] if row else None


def _state_save(instance_id: str, title_key: str, state: dict) -> None:
    with _db().cursor() as cur:
        cur.execute(
            """INSERT INTO widget_thumbnail_states (instance_id, title_key, state, updated_at)
               VALUES (%s, %s, %s, NOW())
               ON CONFLICT (instance_id) DO UPDATE
                   SET state = EXCLUDED.state,
                       title_key = EXCLUDED.title_key,
                       updated_at = NOW()""",
            (instance_id[:128], title_key[:256], _json_top.dumps(state)),
        )

# Legacy file-backed helpers kept as read-only fallback for the one release
# where some clients might still call without instance_id. After ~2 weeks,
# delete _load_state_bucket and the widget_state.json file.
_STATE_FILE = Path(__file__).resolve().parent / "widget_state.json"
_MAX_STATE_ENTRIES = 100


def _load_state_bucket() -> dict:
    """DEPRECATED: read-only fallback. Returns the legacy JSON file contents
    so any code path still using the old per-title bucket sees something."""
    if not _STATE_FILE.exists():
        return {}
    try:
        return _json_top.loads(_STATE_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
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

# Auth model: per-request algora API key, mirroring algrow-mcp. Every /mcp
# call must carry `Authorization: Bearer algrow_<key>` (or `?key=algrow_…`
# for connector UIs that can't set headers). AuthMiddleware extracts it
# into auth_ctx.current_api_key; downstream tools read it from there.
#
# The previous shared-token model (THUMBNAILS_MCP_TOKEN) is gone — this
# server is now exclusively for algora subscribers, no self-host mode.
_ALGROW_API_BASE = (os.environ.get("ALGROW_API_BASE_URL") or "https://api.algrow.online").rstrip("/")

# Optional Gemini vision — when set, references get auto-analyzed into a
# structured JSON breakdown (composition, palette, lighting, text style,
# etc.) that gets folded into the generation prompt. Massive quality win:
# Gemini's image-gen alone treats reference_urls as loose visual hints;
# spelling out the design rules explicitly forces it to keep the layout
# and only swap the subject content.
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or None
_GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-3-flash-preview").strip()
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

_VISION_TO_JSON_PROMPT = open("/root/apps/thumbnails/vision_prompt_v2.txt").read()


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

    Resolves Gemini key in priority order:
      1. Per-user BYOK key from algrow_byok.get_key('gemini') — set in the
         user's algora.online/settings/byok page. When present, this call
         bills against the user's quota (free tier covers 15 RPM / 1500 RPD).
      2. Shared GEMINI_API_KEY env var as fallback for backward compat
         during the auth migration. Will be removed once every active user
         has BYOK configured.
    """
    from algrow_byok import get_key as _byok_get
    user_gemini = _byok_get("gemini")
    effective_key = user_gemini or _GEMINI_API_KEY
    if not effective_key:
        return None, "Vision analysis disabled — configure a Gemini key at https://algora.online/settings/byok"

    img_bytes, mime_type, fetched = _fetch_image_bytes_with_fallback(image_url)
    if img_bytes is None:
        return None, f"Couldn't fetch reference image: {fetched or 'all candidates failed'}"
    logger.info(f"vision analysis fetched {len(img_bytes)} bytes from {fetched} (byok=%s)", bool(user_gemini))

    import base64 as _b64
    import json as _json

    try:
        resp = requests.post(
            f"{_GEMINI_VISION_URL}?key={effective_key}",
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
                    "thinkingConfig": {"thinkingLevel": "high"},
                },
            },
            timeout=180,
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

CRITICAL: the prompt is sent to the image-gen model ALONGSIDE the reference image attached as visual input. The reference image is the load-bearing source for HOW the new image should look — palette, lighting, composition, typography treatment, effects. Phrase your prompt to ANCHOR ON the reference where it carries information better than words can:

  • DO write: "Match the lighting setup of the reference image exactly — same top-down warm key with deep shadow density and the same rim-light separation on the subject."
  • DO write: "Use the same palette and contrast level as the reference (dominated by `#8B4513` warm brown, `#E63946` alarm red, `#2B2B2B` near-black)."
  • DO write: "Apply the reference's typography treatment to the new headline — same bold display sans-serif, same all-caps, same drop-shadow softness."
  • DO write: "Follow the reference's compositional slot positions: centered hero portrait, text overlay top-third, visual device anchored bottom-right."

DO NOT pretend the reference doesn't exist by encoding every fact in self-contained prose alone. The reference image is the strongest signal the gen model has; your prompt should LEVERAGE it.

You still need to be EXPLICIT about everything the reference doesn't unambiguously convey — most importantly the NEW SUBJECT (the title's filling that replaces the reference's specific person/brand/object). The HARD RULES above still apply: the reference's compositional slots transfer, the FILLINGS get re-derived from the user's title. Describe the new subject in concrete physical descriptors, never names.

Specifically you MUST explicitly describe in the prompt:

  • LIGHTING — name the source ("warm overhead tungsten"), direction ("top-left key with soft fill from below-right"), hardness ("hard / soft / diffused"), color temperature ("warm 3200K feel" or "neutral daylight"). Pull these straight from `global_context.lighting`.

  • PALETTE — list the actual dominant hex codes from `color_palette.dominant_hex_estimates` ("dominated by #8B4513 warm corkboard brown, #E63946 vivid alarm red, #2B2B2B near-black"). Name the accent colors. Set the contrast level.

  • COMPOSITION — camera angle, framing, depth of field, focal point. Where each slot lives in the frame (center / top-left / mid-right etc.) using the `objects[*].location` values.

  • TEXT OVERLAYS — for every entry in `text_ocr.content`: the EXACT font style (serif / sans-serif / display / bold weight / italic / condensed), color, background treatment if any, position. The TEXT CONTENT itself gets REWRITTEN to fit the user's new title — but the typographic treatment stays.

  • VISUAL DEVICES — arrows, strings, callouts, badges, price tags, glow effects. Drawn from `semantic_relationships` and from any `objects` entries categorized as Symbol / device. Describe the shape, color, anchor points, and what they connect.

  • SUBJECT — every Person / focal object: physical attributes (age, build, hair, complexion, clothing), pose, expression. These get FRESHLY DESCRIBED based on the user's new title, NOT copied from the reference's `objects[*]` content. The reference's specific persons / weapons / badges / documents are NEVER mentioned — only their structural ROLE (centered portrait / bottom-left prop / etc.) transfers.

  • TEXTURE / MATERIAL — surfaces from `objects[*].visual_attributes.texture` and `.material` ("rough corkboard texture with visible fiber grain", "weathered paper with creases").

Use natural prose, not bullets. Avoid the words "template", "inspiration", "YouTube". You MAY (and SHOULD) use the word "reference" / "reference image" — it tells the gen model how to read the attached visual. Output ONLY the final prompt — no preamble, no reasoning labels, no commentary.

=======================================================================
TWO-STAGE OUTPUT FORMAT — STRICT
=======================================================================

Your response must follow this exact two-section structure. The downstream
parser will discard everything outside the IMAGE PROMPT section, but the
TRANSLATION TABLE is required because it forces you to commit to a
per-element justification BEFORE you write the final prompt. Without it,
on hard cases you skip the reasoning and fall back to surface-level copying.

STEP 1 - Output a TRANSLATION TABLE. One entry per object in the reference
JSON (use the obj_id). For each one, fill THREE lines:

  obj_001 (short label from the JSON):
    REFERENCE ROLE: why this element was in service of the REFERENCE's
      original title - what concept of that title it visualized. Read
      it from the JSON's semantic_role_in_reference_title field; if that
      field is missing, infer it from design_function + the reference
      title context. Be concrete: NOT the focal subject - that is
      structural, not semantic. SEMANTIC means what idea of the title
      does it embody.
    NEW MAPPING: WHAT element of the USER'S new title needs to fill the
      same semantic role. State it as a concrete description (who/what/
      where, period, profession, props), derived ONLY from the user's
      title - never copied from the reference's content.

STEP 2 - Then output the FINAL IMAGE PROMPT, using the mappings you just
committed to. Wrap it between the markers below EXACTLY (the parser
extracts content between these markers):

=== IMAGE PROMPT ===
<the full natural-language image-gen prompt, written under all the
HARD RULES above. Use the NEW MAPPINGS from STEP 1, not the reference's
original fillings. No preamble, no explanation, no labels - just the
prompt the image model will render.>
=== END IMAGE PROMPT ===

Example shape (truncated):

  STEP 1 - TRANSLATION TABLE
  obj_001 (woman in red hoodie, center frame):
    REFERENCE ROLE: the relatable everyday viewer figure who validates the
      surprise - what the title's hidden discovery feels like through a
      normal person's reaction
    NEW MAPPING: an older funeral director in his 60s, somber black suit,
      composed expression - the trusted insider figure who knows the
      hidden economics of his trade

  obj_002 (red circle highlight on phone screen):
    REFERENCE ROLE: visual punctuation on the surprising element being
      revealed - directs the eye to where the title's payoff lives
    NEW MAPPING: red circle highlight on a tombstone in the background with
      a dollar sign overlay - directs the eye to the unexpected profit
      source the title teases

  ... (one entry per reference object)

  === IMAGE PROMPT ===
  A high-quality 16:9 composite thumbnail showing an older funeral
  director ... (full prompt continues here, never naming real people
  or brands per Rule 5, applying every HARD RULE above, using hex
  colors and concrete physical descriptors throughout).
  === END IMAGE PROMPT ===
"""


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

    Why two calls? The vision pass extracts a rich structured breakdown
    of the reference (palette / lighting / composition / per-object
    placements / text-overlay typography) that the synthesizer reasons
    over to identify the design GRAMMAR vs the topic-specific FILLINGS.
    The image generator then receives the engineered prompt ALONGSIDE
    the reference image as visual input — so the prompt is phrased as
    reference-aware directives ("match the reference's lighting", "lift
    the typography treatment") rather than self-contained prose. The
    reference carries HOW; the prompt re-derives WHAT from the title.

    Mobile MCP timeout note: the original single-call version was 5-15s.
    Two-step is 8-25s — still safely inside the 27s mobile timeout
    because both Gemini calls are short (vision-JSON is heavy compute
    on Gemini's side but returns fast; synthesis is text-only so very
    quick). If we ever start blowing the budget we'd push synthesis to
    a background task like the generate path.

    Returns (mapped_prompt, error) — prompt is None on error.

    Both the vision pass AND the synthesis call below resolve the Gemini
    key with the same BYOK-first priority — see _analyze_image_via_gemini
    for the rule. When the user has a Gemini key configured at
    algora.online/settings/byok, the entire compose pipeline runs on
    their quota and algora's shared key never sees the traffic.
    """
    from algrow_byok import get_key as _byok_get
    user_gemini = _byok_get("gemini")
    effective_key = user_gemini or _GEMINI_API_KEY
    if not effective_key:
        return None, "Gemini mapping disabled — configure a Gemini key at https://algora.online/settings/byok"

    # Step 1: rich vision-to-JSON extraction (uses the same key — see helper).
    import time as _t1
    _vt0 = _t1.monotonic()
    analysis, vision_err = _analyze_image_via_gemini(reference_url)
    logger.info(f"compose: vision pass took {_t1.monotonic()-_vt0:.1f}s")
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
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    }
    logger.info("compose: starting synth pass")
    _st0 = _t1.monotonic()
    try:
        resp = requests.post(
            f"{_GEMINI_VISION_URL}?key={effective_key}",
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=180,
        )
    except Exception as e:
        return None, f"Gemini synthesis request failed: {str(e)[:140]}"

    if resp.status_code != 200:
        return None, f"Gemini synthesis HTTP {resp.status_code}: {resp.text[:200]}"

    try:
        out = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Extract only the IMAGE PROMPT section. Two-stage output puts the
        # per-element TRANSLATION TABLE first and the final prompt between
        # === IMAGE PROMPT === and === END IMAGE PROMPT === markers. Fall
        # back to the full text if markers aren't present (graceful with
        # older synthesis prompts during the deploy window).
        import re as _re
        m = _re.search(r"=== IMAGE PROMPT ===\s*(.*?)\s*=== END IMAGE PROMPT ===", out, _re.DOTALL)
        if m:
            out = m.group(1).strip()
        elif "=== IMAGE PROMPT ===" in out:
            out = out.split("=== IMAGE PROMPT ===", 1)[1].strip()
    except Exception as e:
        return None, f"Couldn't parse Gemini synthesis response: {str(e)[:140]}"

    logger.info(f"compose: synth pass took {_t1.monotonic()-_st0:.1f}s output_chars={len(out)}")
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
                        min_outlier_score: float, page: int = 1,
                        *, api_key: str | None = None
                        ) -> tuple[list[dict], str | None, int, bool]:
    """One round-trip to algrow. Returns (videos, error, count, has_more).
    `videos` is the raw video dicts (not yet normalized to our outlier shape).
    `has_more` mirrors algrow's response field — True when more pages exist.
    """
    # Per-request algora key — pulled from the auth middleware's ContextVar
    # when the caller didn't pass one explicitly. /mcp paths always have it
    # set (AuthMiddleware 401s otherwise); the explicit-arg form is for
    # tests + scripts that drive these helpers outside a request.
    from auth_ctx import get_current_api_key
    api_key = (api_key or get_current_api_key() or "").strip()
    if not api_key:
        return [], "Algrow integration not configured (no API key on request).", 0, False
    try:
        resp = requests.post(
            f"{_ALGROW_API_BASE}/api/viral-videos/search",
            headers={
                "Authorization": f"Bearer {api_key}",
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
            timeout=180,
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
                                 page: int = 1,
                                 *, api_key: str | None = None,
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
    # Same auth resolution as _algrow_search_once. We don't 401 here —
    # surface the missing-key as a user-facing "not configured" error so
    # the tool returns a clean grid-empty state instead of a stack trace.
    from auth_ctx import get_current_api_key
    api_key = (api_key or get_current_api_key() or "").strip()
    if not api_key:
        return [], "Algrow integration not configured (no API key on request).", topic, False

    tried: list[str] = []
    videos: list[dict] = []
    last_err: str | None = None
    effective_topic = topic.strip()
    has_more = False

    if page > 1:
        # Pagination request — use the topic verbatim, no fallback.
        tried.append(topic.strip())
        vids, err, count, hm = _algrow_search_once(topic.strip(), content_type, limit, min_outlier_score, page, api_key=api_key)
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
            vids, err, count, hm = _algrow_search_once(q, content_type, limit, min_outlier_score, page, api_key=api_key)
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
            "    available AND there's no saved widget state for the title — "
            "    prefer find_outlier_references first so the user can see and "
            "    pick from proven references.\n\n"
            "ITERATING ON A PRIOR THUMBNAIL (same title, different model / "
            "tweak / refinement): CALL `load_widget_state` FIRST with "
            "key=<lowercased title>. If it returns a state with "
            "`selectedOutlier.thumbnail_url`, use THAT as your reference "
            "and (optionally) the saved `prompt` as your starting prompt. "
            "Then call this tool directly with the new model / tweak. Do "
            "NOT re-run find_outlier_references — the user already picked "
            "a reference and doesn't want to re-pick. Phrasings that "
            "trigger this: 'use seedream again', 'try with gpt-image-2', "
            "'redo it but [X]', 'regenerate with [model]'.\n\n"
            "REFERENCE-URL RULE (CRITICAL): if the user supplies ANY reference URL "
            "(YouTube link, audio.algrow.online mirror, raw .jpg/.png/.webp, "
            "i.ytimg.com URL, anything visual) DO NOT call generate_thumbnail "
            "directly with a hand-written prompt. Route through the widget's "
            "auto-compose pipeline instead:\n"
            "  • YouTube URL / 11-char ID → call `extract_reference_from_video` "
            "with `user_title`. The widget will compose + generate automatically.\n"
            "  • Direct image URL (R2 mirror, raw .jpg/.png, any non-YouTube image) "
            "→ call `extract_reference_from_image` with `user_title`. Same auto-pipeline.\n"
            "Calling generate_thumbnail directly with a reference URL skips the "
            "vision-analysis step that translates the reference's design DNA into "
            "a tailored prompt, producing a generic free-text result.\n\n"
            "Reference images work powerfully — pass up to 8 URLs via "
            "`reference_urls` (YouTube watch/shorts/embed/live URLs, raw 11-char "
            "video IDs, i.ytimg.com URLs, and direct image URLs are all accepted; "
            "the server normalizes them). This still applies when the user has ALREADY "
            "been through extract_reference_* and you're iterating on the prompt."
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
        if find_outliers_first:
            # No env-key gate — the per-request algora key (set by
            # AuthMiddleware) is what authenticates the algrow call.
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

            # Image-gen reference policy: pass the reference through to
            # algrow if the caller provided one — EXCEPT for gpt-image-2,
            # which has a strong img2img copy tendency (we've seen it
            # near-pixel-clone the reference and only swap a couple of
            # asset slots, ignoring the prompt's content overrides). For
            # gpt-image-2 we rely on the VisionStruct-rich prompt alone.
            # Seedream and nano-banana-2 are more interpretive — they
            # use the reference as a style/palette anchor rather than a
            # base image to edit, so the reference helps them transfer
            # the corkboard texture / lighting / "WANTED-banner" feel
            # without copying the subjects (the prompt's explicit content
            # description still wins for what goes in each slot).
            if resolved_refs and model != "gpt-image-2":
                ref_for_submit = resolved_refs[0]
            else:
                ref_for_submit = None
            # Capture the per-request algora key BEFORE spawning the bg
            # task — the ContextVar gets reset when AuthMiddleware unwinds,
            # which happens the instant we return the placeholder to the
            # client. Without this snapshot the bg task would see None.
            from auth_ctx import get_current_api_key as _get_key_snapshot
            algrow_api_key_for_bg = _get_key_snapshot()

            async def _run_algrow_submit():
                try:
                    from algrow_image import submit_image as _algrow_submit
                    sub = await asyncio.to_thread(
                        _algrow_submit,
                        api_key=algrow_api_key_for_bg,
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

        # Snapshot BYOK Gemini key (if any) BEFORE spawning the bg task —
        # algrow_byok.get_key() reads from the request-scoped ContextVar
        # which AuthMiddleware resets the moment we return placeholder
        # state to the client. Without this snapshot the bg task would
        # always see the env-fallback key.
        from algrow_byok import get_key as _byok_get
        _gemini_byok_key_for_bg = _byok_get("gemini")

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
                    api_key=_gemini_byok_key_for_bg,
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
        from auth_ctx import get_current_api_key as _get_key
        _api_key_for_check = _get_key()

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
            ar = _algrow_check(algrow_task_id, api_key=_api_key_for_check)
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
            ar = _algrow_check(task_id, api_key=_api_key_for_check)
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
            title: Annotated[str, Field(description="What the thumbnail is about - usually the user's video title.")],
            reference_url: Annotated[str, Field(description="The reference thumbnail URL the user picked. YouTube URLs / IDs / image URLs all accepted.")],
            reference_title: Annotated[str | None, Field(description="The ORIGINAL video title that the reference thumbnail was made for. Pass when known so the mapper can reason about WHY the reference design choices fit ITS title before adapting that logic to the user new title.")] = None,
            style_preset: Annotated[str, Field(description="person_focal | faceless | none. Default person_focal.")] = "person_focal",
            custom_instructions: Annotated[str | None, Field(description="Optional free-text directives that get appended with highest salience.")] = None,
        ) -> str:
            """Background-submit compose. Returns {state: pending, task_id}
            immediately and runs the vision+synth chain on an asyncio task.
            The widget polls check_compose_status(task_id) until the entry
            flips to success/fail. Decouples wall-clock latency from
            claude.ai's hard 60s per-request timeout."""
            import json, uuid as _uuid
            import time as _time

            resolved = _resolve_reference_url(reference_url)
            if not resolved:
                return json.dumps({"success": False, "state": "fail", "error": "Invalid reference URL."})

            if not reference_title:
                vid = extract_video_id(reference_url) or extract_video_id(resolved)
                if vid:
                    info = extract_video_info(vid)
                    if info.get("success") and info.get("title"):
                        reference_title = info["title"]
                        logger.info(f"auto-fetched reference title via youtubei.js: {reference_title!r}")

            task_id = f"{_COMPOSE_PREFIX}{_uuid.uuid4().hex}"
            _COMPOSE_PENDING[task_id] = {
                "state": "pending",
                "started_at": _time.time(),
                "title": title,
            }
            if len(_COMPOSE_PENDING) > _COMPOSE_PENDING_MAX:
                oldest = sorted(_COMPOSE_PENDING.items(), key=lambda kv: kv[1].get("started_at", 0))
                for k, _v in oldest[: len(_COMPOSE_PENDING) - _COMPOSE_PENDING_MAX]:
                    _COMPOSE_PENDING.pop(k, None)

            async def _run_compose():
                t0 = _time.monotonic()
                logger.info(f"compose bg-start task_id={task_id[-12:]} title={title!r}")
                try:
                    mapped, map_err = await asyncio.to_thread(
                        _map_reference_to_title_via_gemini,
                        title=title,
                        reference_url=resolved,
                        reference_title=reference_title,
                        style_preset=style_preset,
                        custom_instructions=custom_instructions,
                    )
                except Exception as e:
                    logger.error(f"compose bg-crashed task_id={task_id[-12:]}: {e!r}")
                    _COMPOSE_PENDING[task_id] = {
                        "state": "fail",
                        "error": f"Compose crashed: {type(e).__name__}: {e}"[:300],
                        "title": title,
                    }
                    return
                dt = _time.monotonic() - t0
                if not mapped:
                    logger.warning(f"compose bg-fail task_id={task_id[-12:]} dt={dt:.1f}s err={map_err!r}")
                    _COMPOSE_PENDING[task_id] = {
                        "state": "fail",
                        "error": f"Reasoned mapping failed: {map_err or 'unknown'}",
                        "title": title,
                    }
                else:
                    logger.info(f"compose bg-done task_id={task_id[-12:]} dt={dt:.1f}s chars={len(mapped)}")
                    _COMPOSE_PENDING[task_id] = {
                        "state": "success",
                        "success": True,
                        "title": title,
                        "reference_url": resolved,
                        "reference_title": reference_title,
                        "style_preset": style_preset,
                        "prompt": mapped,
                        "mode": "reasoned",
                        "completed_at": _time.time(),
                    }

            asyncio.create_task(_run_compose())
            return json.dumps({
                "state": "pending",
                "task_id": task_id,
                "title": title,
            })

    # ----- Algrow-powered outlier picker -----------------------------------
    # Always registered — authenticates per-request via the user's algora
    # API key (AuthMiddleware → auth_ctx.current_api_key). Lets the widget
    # (and Claude, when called from chat) pull high-outlier-score thumbnails
    # for a topic and offer them as references.
    if True:

        @mcp.tool(
            name="check_compose_status",
            description=(
                "Poll a background compose job by task_id (returned from "
                "compose_thumbnail_prompt with state=pending). Returns the "
                "same payload shape compose used to return synchronously "
                "once complete: {success: true, prompt, ...}. While running, "
                "returns {state: pending}. Widget should poll every 2s."
            ),
        )
        async def check_compose_status_tool(
            task_id: Annotated[str, Field(description="task_id returned from compose_thumbnail_prompt")],
        ) -> str:
            import json
            info = _COMPOSE_PENDING.get(task_id)
            if not info:
                return json.dumps({"state": "fail", "success": False, "error": "Unknown task_id (expired or invalid)"})
            return json.dumps(info, default=str)

        @mcp.tool(
            name="find_outlier_references",
            title="Find Outlier Thumbnails on a Topic",
            description=(
                "DEFAULT entry point WHEN STARTING A NEW THUMBNAIL on a "
                "title with no prior widget work. Searches algrow's database "
                "(50k+ YouTube channels) for topically-similar videos with "
                "proven outlier performance and renders them as a clickable "
                "grid in an inline widget.\n\n"
                "DO NOT CALL THIS WHEN THE USER IS ITERATING on a thumbnail "
                "they already started — phrasings like 'use seedream "
                "again', 'try with X model', 'redo with [tweak]', "
                "'regenerate with…'. In those cases call `load_widget_state` "
                "with key=<lowercased title> FIRST. If the returned state "
                "has a `selectedOutlier.thumbnail_url`, the user has "
                "already picked a reference — pass that straight to "
                "`generate_thumbnail` (or `compose_thumbnail_prompt` if "
                "they want a fresh prompt) with the new model. Re-running "
                "this tool wipes the user's prior pick and forces them to "
                "repick from a fresh grid for no reason.\n\n"
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
            # Fresh task invocation. With per-instance state rows (keyed by
            # the widget mount's toolInfo.id) each new mount already gets a
            # fresh row, so we no longer need to clear bucket entries to
            # avoid stale-state restore.
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
        # Fresh task invocation. Same rationale as in find_outlier_references:
        # per-instance state rows make clearing-by-title obsolete.
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

    # ----- Reference by direct image URL ---------------------------------
    # Same shape as extract_reference_from_video but for non-YouTube images:
    # R2 mirrors (audio.algrow.online/thumbnails/...), raw .jpg/.png URLs the
    # user uploaded or pasted, anything that's already a visual reference and
    # doesn't need video-info lookup. Returns the same auto_pipeline payload
    # so the widget runs the identical compose-then-generate chain.
    @mcp.tool(
        name="extract_reference_from_image",
        title="Use a Direct Image URL as the Reference",
        description=(
            "Use this when the user wants to base a thumbnail on a SPECIFIC "
            "image they've linked that ISN'T a YouTube video — e.g. "
            "audio.algrow.online R2 mirror URLs, raw .jpg/.png/.webp URLs, "
            "screenshots they uploaded, or any direct image reference. "
            "Mounts the widget with the reference preselected and — if "
            "you pass `user_title` — auto-runs the full compose + "
            "generate pipeline (~45-65s with stage-by-stage progress).\n\n"
            "BEHAVIOR DEPENDS ON `user_title`:\n"
            "  • With `user_title`: widget chains compose → generate "
            "automatically.\n"
            "  • Without `user_title`: widget mounts with the reference "
            "preselected; user clicks Create Prompt → Generate.\n\n"
            "ALWAYS pass `user_title` when you have it. After calling, STOP.\n\n"
            "If the reference is a YouTube link, use extract_reference_from_video "
            "instead so the server can also fetch the original video's title "
            "(which makes the compose step\'s reasoning measurably better)."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def extract_reference_from_image_tool(
        image_url: Annotated[str, Field(description="Direct image URL. Must be http(s). R2 mirrors / raw .jpg / .png / .webp / .gif are all fine; the server passes the URL through unchanged.")],
        user_title: Annotated[str | None, Field(description="The user\'s NEW video title — what THEIR thumbnail is for. Widget pre-fills its title field with this and triggers the auto-pipeline. Always pass when you have it.")] = None,
        reference_title: Annotated[str | None, Field(description="Optional. If the user mentioned what the reference image is OF / what it was made for, pass that here so the compose step can reason about WHY the reference\'s design works before adapting it. Skip if unknown.")] = None,
        model: Annotated[str, Field(description="Image-gen backend the widget preselects. Same options as extract_reference_from_video: 'nano-banana-pro' (default, 3 cr), 'gpt-image-2' (1 cr), 'seedream-5.0-lite' (2 cr, more permissive safety), 'seedream-4.5-edit' (2 cr, image-to-image edit). Default 'nano-banana-pro'.")] = "nano-banana-pro",
    ) -> str:
        import json
        resolved = _resolve_reference_url(image_url)
        if not resolved:
            return json.dumps({
                "view": "outlier_picker",
                "topic": user_title or "",
                "outliers": [],
                "count": 0,
                "error": "Invalid image URL — must be http(s) and resolvable.",
                **({"title": user_title} if user_title else {}),
            })

        # Fresh task invocation — clear any saved WIP for this title so the
        # widget starts clean. (Per-instance state still works because each
        # widget mount has its own toolInfo.id row.)
        if user_title:
            try:
                _state_save(f"legacy:{_state_key(user_title)}", _state_key(user_title), {})
            except Exception:
                pass  # DB blip — widget will just overwrite on first save

        outlier = {
            "video_id": None,
            "title": reference_title or "",
            "thumbnail_url": resolved,
            "channel_name": "",
            "channel_thumbnail": None,
            "view_count": None,
            "outlier_score": None,
            "url": resolved,
        }
        payload = {
            "view": "outlier_picker",
            "topic": user_title or reference_title or "",
            "outliers": [outlier],
            "count": 1,
            "content_type": "longform",
            "single_reference": True,
            "model": model,
        }
        if user_title:
            payload["title"] = user_title
            payload["auto_pipeline"] = True
        return json.dumps(payload, default=str)


    # ----- Channel-style DNA: chat entrypoint ----------------------------
    # When the user wants "a thumbnail in @creator's style" we don't pick a
    # single reference — we fingerprint the channel's recurring visual grammar
    # across the top-10 by-views thumbnails and apply that grammar to the
    # user's new title. Heavy lifting is in compose_channel_style_prompt
    # below; this tool just resolves the channel and mounts the widget with
    # a payload that flips the auto-pipeline into channel-style mode.
    @mcp.tool(
        name="extract_reference_from_channel",
        title="Use a Whole Channel's Style as the Reference",
        description=(
            "Use this when the user wants a thumbnail in the STYLE of a "
            "whole creator/channel — NOT a single reference video. "
            "Examples: 'make a thumbnail in MrBeast's style', "
            "'thumbnail like @fern-tv would make', 'use the Veritasium "
            "design system'. \n\n"
            "WHAT IT DOES (server-side, ~90-120s first time, ~5s on cache hit):\n"
            "  1. Resolves the @handle / UC id via Algrow.\n"
            "  2. Fetches the channel's TOP 10 BY VIEW COUNT thumbnails.\n"
            "  3. Vision-analyzes all 10 in parallel.\n"
            "  4. Synthesizes a channel DNA (palette, typography, "
            "composition, lighting, effects) + an exhaustive text prompt "
            "applying that DNA to the user's title. Cached 30 days.\n"
            "  5. Generates the thumbnail WITHOUT any reference image — "
            "the prompt carries the entire style.\n\n"
            "ALWAYS pass `user_title` when you have it (the new video the "
            "user is making a thumbnail for). After calling, STOP — the "
            "widget runs the rest."
        ),
        meta={"ui": {"resourceUri": widgets.THUMBNAIL_STUDIO_URI}},
    )
    async def extract_reference_from_channel_tool(
        channel: Annotated[str, Field(description="YouTube @handle (e.g. @fern-tv) OR channel ID (UCxxxxxxxxxxxxxxxxxxxxxx, 24 chars). The leading @ is optional for handles.")],
        user_title: Annotated[str | None, Field(description="The user's NEW video title. Channel DNA gets applied to this title to produce the final thumbnail. Always pass when you have it.")] = None,
        model: Annotated[str, Field(description="Image-gen backend the widget preselects. Same options as extract_reference_from_video. Default 'nano-banana-pro'. Note: channel-style mode generates WITHOUT a reference image, so models that rely on img2img (seedream-4.5-edit) are a poor fit here — pick a regenerative model.")] = "nano-banana-pro",
    ) -> str:
        import json
        info, err = _resolve_channel_via_algrow(channel)
        if err or not info:
            return json.dumps({
                "view": "outlier_picker",
                "topic": user_title or channel,
                "outliers": [],
                "count": 0,
                "error": err or "Could not resolve channel.",
                **({"title": user_title} if user_title else {}),
            })

        # Fetch the channel's actual top-by-views thumbnails synchronously
        # so the widget mounts with 10 real cards instead of a single
        # placeholder. Adds ~4-6s to the chat-side tool latency but kills
        # the "ghost placeholder while it analyzes" UX gap.
        channel_id = info["channel_id"]
        top_videos, top_err = _fetch_channel_top_videos(channel_id)
        # Innertube meta (display name + avatar) — call the bridge again
        # cheaply if the first fetch went through Algrow (which doesn't
        # return meta). Skip if top_videos already came from Innertube; in
        # that case we already paid the cost.
        avatar_url = info.get("channel_thumbnail")
        display_name = info["channel_name"]
        try:
            _v, meta, _e = _fetch_channel_videos_via_innertube(channel_id, 1)
            if meta:
                avatar_url = meta.get("avatar_url") or avatar_url
                display_name = meta.get("name") or display_name
        except Exception:
            pass  # avatar/name enrichment is best-effort

        if top_err or not top_videos:
            # Resolution worked but we couldn't pull videos. Fall back to
            # the single-placeholder card so the user still sees the
            # channel mounted (and can retry).
            top_videos = []

        outliers = []
        if top_videos:
            for v in top_videos:
                outliers.append({
                    "video_id": v["video_id"],
                    "title": v.get("title") or "",
                    "thumbnail_url": v["thumbnail_url"],
                    "channel_name": display_name,
                    "channel_thumbnail": avatar_url,
                    "view_count": v.get("view_count"),
                    "outlier_score": None,
                    "url": f"https://www.youtube.com/watch?v={v['video_id']}",
                })
        else:
            outliers.append({
                "video_id": None,
                "title": f"{display_name} channel style",
                "thumbnail_url": avatar_url or "https://www.youtube.com/img/desktop/yt_1200.png",
                "channel_name": display_name,
                "channel_thumbnail": avatar_url,
                "view_count": None,
                "outlier_score": None,
                "url": f"https://www.youtube.com/channel/{channel_id}",
            })

        payload = {
            "view": "outlier_picker",
            "topic": user_title or display_name,
            "outliers": outliers,
            "count": len(outliers),
            "content_type": "longform",
            "single_reference": True,
            "model": model,
            "channel_style_mode": True,
            "channel_id": channel_id,
            "channel_handle": info.get("handle"),
            "channel_name": display_name,
            "channel_avatar": avatar_url,
        }
        if user_title:
            payload["title"] = user_title
            payload["auto_pipeline"] = True
        return json.dumps(payload, default=str)


    # ----- Channel-style DNA: widget poll-driven compose -----------------
    # Background-submit, same shape as compose_thumbnail_prompt. Returns
    # {state: pending, task_id} immediately; widget polls check_compose_status
    # until success/fail. Reuses _COMPOSE_PENDING so the existing poll loop
    # in the widget needs zero changes — the helper that reads task_id
    # doesn't care which compose tool wrote into the dict.
    @mcp.tool(
        name="compose_channel_style_prompt",
        title="Compose Thumbnail Prompt from Channel DNA",
        description=(
            "INTERNAL widget helper. The thumbnail-studio widget calls this "
            "when channel_style_mode is true. Builds (or reads from 30-day "
            "cache) the channel's design DNA across its top-10 by-views "
            "thumbnails, then emits an exhaustive text prompt that applies "
            "that DNA to the user's title. Returns {state: pending, task_id} "
            "immediately; poll check_compose_status. Do NOT call from chat."
        ),
    )
    async def compose_channel_style_prompt_tool(
        channel_id: Annotated[str, Field(description="YouTube channel ID (UCxxxxxxxxxxxxxxxxxxxxxx) — already resolved by extract_reference_from_channel.")],
        title: Annotated[str, Field(description="The user's new video title to apply the channel DNA to.")],
        channel_name: Annotated[str, Field(description="Channel name for display + synthesis prompt context.")] = "",
        channel_handle: Annotated[str | None, Field(description="@handle if known. Optional, used for synthesis prompt context.")] = None,
    ) -> str:
        import json, uuid as _uuid
        import time as _time

        task_id = f"{_CHANNEL_COMPOSE_PREFIX}{_uuid.uuid4().hex}"
        _COMPOSE_PENDING[task_id] = {
            "state": "pending",
            "started_at": _time.time(),
            "title": title,
            "channel_id": channel_id,
            "channel_name": channel_name,
        }
        if len(_COMPOSE_PENDING) > _COMPOSE_PENDING_MAX:
            oldest = sorted(_COMPOSE_PENDING.items(), key=lambda kv: kv[1].get("started_at", 0))
            for k, _v in oldest[: len(_COMPOSE_PENDING) - _COMPOSE_PENDING_MAX]:
                _COMPOSE_PENDING.pop(k, None)

        # Snapshot the request-bound keys NOW — these ContextVars don't
        # propagate to worker threads (asyncio.to_thread), so we must capture
        # them while still inside the request handler.
        from auth_ctx import get_current_api_key as _get_algora_key
        _request_algora_key = (_get_algora_key() or "").strip() or None
        try:
            from algrow_byok import get_key as _byok_get
            _request_gemini_key = _byok_get("gemini") or _GEMINI_API_KEY
        except Exception:
            _request_gemini_key = _GEMINI_API_KEY

        async def _run_channel_compose():
            t0 = _time.monotonic()
            logger.info(f"channel-style bg-start task={task_id[-12:]} channel={channel_id} title={title!r}")
            try:
                cached = await asyncio.to_thread(_channel_dna_load, channel_id)
                if cached:
                    logger.info(f"channel-style cache HIT channel={channel_id}; matching new title")
                    cached_analyses = cached.get("source_analyses") or []
                    if isinstance(cached_analyses, str):
                        try:
                            cached_analyses = _json_top.loads(cached_analyses)
                        except Exception:
                            cached_analyses = []
                    cached_titles = cached.get("source_titles") or []
                    if isinstance(cached_titles, str):
                        try:
                            cached_titles = _json_top.loads(cached_titles)
                        except Exception:
                            cached_titles = []
                    cached_thumbs = cached.get("source_thumb_urls") or []
                    if isinstance(cached_thumbs, str):
                        try:
                            cached_thumbs = _json_top.loads(cached_thumbs)
                        except Exception:
                            cached_thumbs = []

                    match, match_err = await asyncio.to_thread(
                        _build_matched_prompt,
                        cached_analyses, cached_titles, cached_thumbs, title,
                        cached.get("channel_name") or channel_name,
                        cached.get("handle") or channel_handle,
                        gemini_key=_request_gemini_key,
                    )
                    if match_err or not match:
                        _COMPOSE_PENDING[task_id] = {
                            "state": "fail",
                            "error": f"Channel-style match (cached) failed: {match_err or 'no output'}",
                            "title": title,
                        }
                        return

                    _COMPOSE_PENDING[task_id] = {
                        "state": "success",
                        "success": True,
                        "title": title,
                        "channel_id": channel_id,
                        "channel_name": cached.get("channel_name") or channel_name,
                        "prompt": match["final_prompt"],
                        "matched_index": match["matched_index"],
                        "matched_title": match["matched_title"],
                        "matched_thumb_url": match["matched_thumb_url"],
                        "match_reason": match["match_reason"],
                        "channel_constants_lift": match["channel_constants_lift"],
                        "source_thumb_urls": cached_thumbs,
                        "source_titles": cached_titles,
                        "mode": "channel_style_cached",
                        "completed_at": _time.time(),
                    }
                    logger.info(
                        f"channel-style bg-done CACHED task={task_id[-12:]} "
                        f"dt={_time.monotonic()-t0:.1f}s matched_idx={match['matched_index']} "
                        f"chars={len(match['final_prompt'])}"
                    )
                    return

                # ---- cache miss: full pipeline ----
                videos, fetch_err = await asyncio.to_thread(
                    _fetch_channel_top_videos, channel_id,
                    api_key=_request_algora_key,
                )
                if not videos:
                    _COMPOSE_PENDING[task_id] = {
                        "state": "fail",
                        "error": fetch_err or "Channel has no analyzable videos.",
                        "title": title,
                    }
                    return

                thumb_urls = [v["thumbnail_url"] for v in videos]
                titles = [v["title"] for v in videos]
                logger.info(f"channel-style fetched {len(thumb_urls)} thumbs; firing parallel vision passes")

                # Parallel vision pass — _analyze_image_via_gemini is sync,
                # use to_thread per call inside gather.
                async def _one(u):
                    return await asyncio.to_thread(_analyze_image_via_gemini, u)
                results = await asyncio.gather(*[_one(u) for u in thumb_urls], return_exceptions=True)
                analyses: list[dict] = []
                vision_errors: list[str] = []
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        vision_errors.append(f"thumb {i}: {r!r}")
                        continue
                    analysis, err = r
                    if analysis:
                        analyses.append(analysis)
                    else:
                        vision_errors.append(f"thumb {i}: {err}")

                logger.info(f"channel-style vision: {len(analyses)} succeeded, {len(vision_errors)} failed")
                if len(analyses) < 3:
                    _COMPOSE_PENDING[task_id] = {
                        "state": "fail",
                        "error": f"Too few thumbnails analyzed successfully ({len(analyses)}/{len(thumb_urls)}). Errors: " + " | ".join(vision_errors[:3]),
                        "title": title,
                    }
                    return

                # Fresh build: cache the analyses FIRST so a crash during
                # the match step doesn't waste the ~$0.30 vision spend.
                try:
                    await asyncio.to_thread(
                        _channel_dna_save, channel_id, channel_handle, channel_name,
                        None, None, thumb_urls, titles, analyses,
                    )
                except Exception as e:
                    logger.warning(f"channel-style cache save (analyses-only) failed: {e}")

                match, match_err = await asyncio.to_thread(
                    _build_matched_prompt,
                    analyses, titles, thumb_urls, title,
                    channel_name, channel_handle,
                    gemini_key=_request_gemini_key,
                )
                if match_err or not match:
                    _COMPOSE_PENDING[task_id] = {
                        "state": "fail",
                        "error": f"Channel-style match (fresh) failed: {match_err or 'no output'}",
                        "title": title,
                    }
                    return

                _COMPOSE_PENDING[task_id] = {
                    "state": "success",
                    "success": True,
                    "title": title,
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "prompt": match["final_prompt"],
                    "matched_index": match["matched_index"],
                    "matched_title": match["matched_title"],
                    "matched_thumb_url": match["matched_thumb_url"],
                    "match_reason": match["match_reason"],
                    "channel_constants_lift": match["channel_constants_lift"],
                    "source_thumb_urls": thumb_urls,
                    "source_titles": titles,
                    "mode": "channel_style",
                    "completed_at": _time.time(),
                }
                logger.info(
                    f"channel-style bg-done FRESH task={task_id[-12:]} "
                    f"dt={_time.monotonic()-t0:.1f}s matched_idx={match['matched_index']} "
                    f"chars={len(match['final_prompt'])}"
                )
            except Exception as e:
                logger.error(f"channel-style crashed task={task_id[-12:]}: {e!r}")
                _COMPOSE_PENDING[task_id] = {
                    "state": "fail",
                    "error": f"Channel DNA pipeline crashed: {type(e).__name__}: {e}"[:300],
                    "title": title,
                }

        asyncio.create_task(_run_channel_compose())
        return json.dumps({
            "state": "pending",
            "task_id": task_id,
            "title": title,
        })


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
        # Lookup by title returns the most recently updated widget mount for
        # that title — which is what the user means by "the prompt for this
        # title" when iterating in chat.
        try:
            state = _state_load_latest_by_title(_state_key(user_title))
        except Exception as e:
            logger.error(f"get_widget_prompt DB error: {e}")
            state = None
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
        title_key = _state_key(user_title)
        try:
            state = _state_load_latest_by_title(title_key) or {"title": user_title}
        except Exception as e:
            logger.error(f"set_widget_prompt DB load error: {e}")
            state = {"title": user_title}
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
        # Write under a stable legacy id so any fresh widget mount (which
        # gets a brand-new toolInfo.id) falls back to this row via the
        # most-recent-by-title lookup in load_widget_state.
        try:
            _state_save(f"legacy:{title_key}", title_key, state)
        except Exception as e:
            logger.error(f"set_widget_prompt DB save error: {e}")
            return json.dumps({"success": False, "error": f"Could not save prompt: {e}"})

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
        key: Annotated[str, Field(description="Lowercased, trimmed video title used as the bucket key (title_key).")],
        state: Annotated[dict, Field(description="Opaque state blob the widget wants to persist. Stored as-is.")],
        instance_id: Annotated[str | None, Field(description="Per-widget-mount identifier (hostContext.toolInfo.id). When provided, each widget mount keeps its own row instead of overwriting prior mounts for the same title. The widget always passes this; chat-side callers may omit it (legacy path falls back to a synthetic id derived from the title).")] = None,
    ) -> str:
        import json
        k = _state_key(key)
        if not k:
            return json.dumps({"success": False, "error": "Empty key."})
        # If the caller didn't pass an instance_id, derive a per-title synthetic
        # one so chat-originated saves still land in the table (preserving the
        # "use my prior reference for this title" flow). Each per-title chat
        # save overwrites the same synthetic row, matching old JSON behaviour.
        iid = (instance_id or "").strip() or f"legacy:{k}"
        try:
            _state_save(iid[:128], k, {**(state or {}), "ts": int(time.time() * 1000)})
            return json.dumps({"success": True})
        except Exception as e:
            logger.error(f"save_widget_state DB error: {e}")
            return json.dumps({"success": False, "error": f"DB write failed: {e}"})

    @mcp.tool(
        name="load_widget_state",
        title="Load Widget State",
        description=(
            "Fetches the saved widget state for a title. The state blob "
            "contains: `selectedOutlier.thumbnail_url` (the reference the "
            "user picked last time), `prompt` (the last composed/edited "
            "prompt), `lastResultPayload` (the last generation result + "
            "its model), `stylePreset`, `model`, `aspectRatio`, "
            "`resolution`, `customInstructions`.\n\n"
            "CALL THIS FIRST WHENEVER THE USER REFERENCES PRIOR WORK ON A "
            "TITLE. Phrasings that mean 'use the prior widget state':\n"
            "  • 'use seedream again' / 'try with gpt-image-2'\n"
            "  • 'redo it but [tweak]'\n"
            "  • 'regenerate with [model]'\n"
            "  • 'same reference, different prompt'\n"
            "  • any follow-up that implies a previous reference/prompt\n\n"
            "Pass `key` as the lowercased + trimmed title. If the returned "
            "state has a `selectedOutlier.thumbnail_url`, USE THAT as the "
            "reference for compose/generate — do NOT call find_outlier_"
            "references again, and do NOT ask the user which reference. "
            "If state is null, fall back to the normal flow.\n\n"
            "Also used by the widget itself on mount to restore in-progress "
            "state; that call path is internal and doesn't affect chat use."
        ),
    )
    async def load_widget_state_tool(
        key: Annotated[str, Field(description="Lowercased, trimmed video title used as the bucket key (title_key).")],
        instance_id: Annotated[str | None, Field(description="Optional per-mount identifier. When provided, returns ONLY that widget mount's state. When omitted, returns the most recently updated state for the title (the legacy chat-side behaviour).")] = None,
    ) -> str:
        import json
        k = _state_key(key)
        if not k:
            return json.dumps({"success": True, "state": None})
        try:
            if instance_id and instance_id.strip():
                # Strict per-instance lookup. NO fallback to title-keyed row
                # here — letting that fallback fire on a fresh widget mount
                # was bleeding prior failed runs into brand-new chats that
                # happened to use the same title. If this instance has no
                # row, the widget gets null and starts clean.
                st = _state_load_by_instance(instance_id.strip()[:128])
            else:
                # Chat-side path (LLM tool call with no instance_id) — the
                # "use my prior reference for this title" flow still wants
                # the most-recent-by-title row, that's its whole purpose.
                st = _state_load_latest_by_title(k)
            return json.dumps({"success": True, "state": st})
        except Exception as e:
            logger.error(f"load_widget_state DB error: {e}")
            # Last-ditch: fall back to the legacy file so a transient DB
            # outage doesn't blank everyone's widgets.
            try:
                bucket = _load_state_bucket()
                return json.dumps({"success": True, "state": bucket.get(k), "fallback": "file"})
            except Exception:
                return json.dumps({"success": True, "state": None, "error": str(e)})

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
    """Per-request algora API-key auth, mirroring algrow-mcp/remote.py.

    Every /mcp request must carry an algora-issued API key via EITHER:

      - HTTP header:  Authorization: Bearer algrow_<key>
      - Query string: ?key=algrow_<key>

    The query-string form exists because claude.ai's "Add custom connector"
    dialog only accepts a URL + OAuth (no Authorization header field).
    Pasting `https://host/mcp?key=algrow_…` lets that connector authenticate
    without standing up a full OAuth flow. Header form is preferred
    everywhere else (curl, Claude Desktop via mcp-remote --header, etc.)
    because query strings get logged.

    We don't validate the key shape here beyond non-emptiness — the first
    real algora call (find_outlier_references, image submit, BYOK lookup)
    will fail with a clean 401 if the key is bogus, and the user sees that
    immediately. Keeping middleware dumb avoids a per-request algora round
    trip just to validate auth.

    The extracted key is stored in the `current_api_key` ContextVar so
    every tool handler downstream can read it without threading it through
    function signatures."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        # Auth only enforced on /mcp; /health and /generated/* stay open.
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

        supplied = (header_token or query_token or "").strip()
        if not supplied:
            await JSONResponse(
                {"error": "unauthorized", "detail": "Authorization: Bearer <algora_api_key> required. Get yours at https://algora.online/settings"},
                status_code=401,
            )(scope, receive, send)
            return

        # Stash for tool handlers — see auth_ctx.py.
        from auth_ctx import set_current_api_key
        token = set_current_api_key(supplied)
        logger.info(f"[Auth] path={path} key={supplied[:12]}...")
        try:
            await self.app(scope, receive, send)
        finally:
            from auth_ctx import current_api_key
            current_api_key.reset(token)


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


# ---------------------------------------------------------------------------
# OAuth 2.0 discovery + flow endpoints
# ---------------------------------------------------------------------------
# Ported wholesale from algrow-mcp's remote.py. Lets MCP clients (Claude
# Desktop, claude.ai, ChatGPT) auth via the standard "Connect to thumbnails"
# button instead of forcing the user to mint + paste an algora API key.
# The flow returns the user's existing algora API key as the OAuth
# access_token (see oauth.py header for the full sequence).

from oauth import (
    authorize as _oauth_authorize,
    callback as _oauth_callback,
    approve as _oauth_approve,
    login_email as _oauth_login_email,
    token as _oauth_token,
    register_client as _oauth_register_client,
)


async def oauth_protected_resource(request: Request) -> Response:
    """RFC 9728 resource metadata.

    Served at both /.well-known/oauth-protected-resource (root) and
    /.well-known/oauth-protected-resource/mcp (path-mounted MCP). The /mcp
    variant is required by MCP March 2025 auth spec for path-mounted
    servers; without it, Claude Desktop on macOS fails OAuth discovery
    with "invalid response" (claude.ai web tolerates the 404 and falls
    back, Desktop does not).
    """
    base = os.environ.get("MCP_BASE_URL", "https://thumbnails-mcp.algrow.online")
    path = request.url.path
    if path.rstrip("/").endswith("/mcp"):
        resource_url = f"{base.rstrip('/')}/mcp"
    else:
        resource_url = base
    return JSONResponse({
        "resource": resource_url,
        "authorization_servers": [base],
        "bearer_methods_supported": ["header"],
        "scopes_supported": [],
    })


async def oauth_metadata(request: Request) -> Response:
    """GET /.well-known/oauth-authorization-server — RFC 8414 discovery."""
    base = os.environ.get("MCP_BASE_URL", "https://thumbnails-mcp.algrow.online")
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "registration_endpoint": f"{base}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "none"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": [],
    })


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/generated/{filename}", generated_image),
        # OAuth discovery — RFC 9728 + RFC 8414. Both root + /mcp path
        # variants for Desktop/Web compat (see comments above).
        Route("/.well-known/oauth-protected-resource", oauth_protected_resource),
        Route("/.well-known/oauth-protected-resource/mcp", oauth_protected_resource),
        Route("/.well-known/oauth-authorization-server", oauth_metadata),
        Route("/.well-known/oauth-authorization-server/mcp", oauth_metadata),
        # OAuth 2.0 flow endpoints.
        Route("/oauth/authorize", _oauth_authorize),
        Route("/oauth/callback", _oauth_callback),
        Route("/oauth/approve", _oauth_approve, methods=["POST"]),
        Route("/oauth/login-email", _oauth_login_email, methods=["POST"]),
        Route("/oauth/token", _oauth_token, methods=["POST"]),
        Route("/oauth/register", _oauth_register_client, methods=["POST"]),
        # Root-level aliases — some MCP connectors hit /authorize, /token,
        # /register directly without the /oauth prefix.
        Route("/authorize", _oauth_authorize),
        Route("/token", _oauth_token, methods=["POST"]),
        Route("/register", _oauth_register_client, methods=["POST"]),
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
