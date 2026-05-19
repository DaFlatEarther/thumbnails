"""Gemini 3 Pro Image (Nano Banana Pro) generation wrapper.

Synchronous: one call to Google's official Gemini Image API returns the
generated image as inline bytes. Replaces the previous Kie-based
async submit/poll path.

Uses the official google-genai SDK so we don't have to hand-maintain the
enum string conversions or thought-signature plumbing.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("gemini-image")

_DEFAULT_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
_GENERATED_DIR = Path(__file__).resolve().parent / "generated_images"
_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# YouTube ID extractors (duplicated from server.py to avoid a circular
# import — server.py imports gemini_image at module load).
_YT_ID_FROM_ALGROW_RE = re.compile(
    r"audio\.algrow\.online/thumbnails/(?:longform|shorts)/([A-Za-z0-9_-]{11})\."
)
_YT_ID_FROM_YTIMG_RE = re.compile(r"i\.ytimg\.com/vi(?:_webp)?/([A-Za-z0-9_-]{11})/")


def _high_res_url_candidates(image_url: str) -> list[str]:
    """Same upgrade strategy as the compose-side vision call: when the URL
    is an algrow CDN preview (~168x94) or a YouTube hqdefault (480x360),
    try maxresdefault → sddefault → hqdefault → original. Critical for
    Gemini Image generation too — passing it a 168x94 reference is barely
    informative, while passing 1280x720 gives Gemini real style signal.
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
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg")
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg")
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
    if image_url not in candidates:
        candidates.append(image_url)
    return candidates


def _fetch_image_bytes(url: str) -> tuple[bytes, str] | None:
    """Download a reference image; returns (bytes, mime_type) or None.

    Shortcut A — /generated/: if the URL points to our own /generated/
    endpoint, read the file off disk directly. Avoids an HTTP roundtrip
    through the public tunnel (cloudflared → Cloudflare → back to us)
    on multi-turn editing.

    Shortcut B — high-res upgrade: if the URL is an algrow CDN preview
    or a YouTube hqdefault, walk the upgrade candidate chain (maxresdefault
    first) so Gemini sees a usable resolution instead of a 168x94 blob.
    """
    if "/generated/" in url:
        fname = url.rsplit("/", 1)[-1].split("?", 1)[0]
        local_path = _GENERATED_DIR / fname
        if local_path.is_file():
            mime = _MIME_BY_EXT.get(local_path.suffix.lower(), "image/png")
            try:
                return local_path.read_bytes(), mime
            except Exception as e:
                logger.warning(f"local /generated/ read failed for {fname}: {e}")
                # fall through to HTTP fetch

    last_err: str | None = None
    for candidate in _high_res_url_candidates(url):
        try:
            r = requests.get(candidate, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            mime = (r.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
            if not mime.startswith("image/"):
                mime = "image/jpeg"
            logger.info(f"reference fetched for generation: {len(r.content)} bytes from {candidate}")
            return r.content, mime
        except Exception as e:
            last_err = f"{candidate}: {str(e)[:120]}"
            continue
    logger.warning(f"reference fetch failed for all candidates of {url}: {last_err}")
    return None


def generate_image(
    prompt: str,
    reference_urls: list[str] | None = None,
    aspect_ratio: str = "16:9",
    resolution: str = "2K",
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate one image via Gemini's official Image API.

    Returns:
      {"success": True, "image_bytes": bytes, "mime_type": str, "cost_time_s": float}
      {"success": False, "error": str}
    """
    import time as _time

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY not set"}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"success": False, "error": "google-genai package not installed (pip install google-genai)"}

    client = genai.Client(api_key=api_key)
    model = model or _DEFAULT_MODEL

    # Build the contents list: prompt text first, then up to 14 inline
    # reference images. The SDK accepts raw bytes wrapped in
    # types.Part.from_bytes, which is faster than spinning up PIL just to
    # serialize back to bytes.
    parts: list[Any] = [prompt]
    for url in (reference_urls or [])[:14]:
        fetched = _fetch_image_bytes(url)
        if not fetched:
            continue
        img_bytes, mime = fetched
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=resolution,
            # NOTE: person_generation is NOT settable on the Developer API
            # tier (it errors out — only available on Gemini Enterprise
            # Agent Platform). The Developer API uses its own defaults
            # which already allow most adult public figures, so we rely on
            # those + the prompt-side guidance for naming celebrities
            # directly.
        ),
    )

    started = _time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=config,
        )
    except Exception as e:
        return {"success": False, "error": f"Gemini call failed: {str(e)[:400]}"}

    cost_time = round(_time.time() - started, 2)

    # When Gemini blocks the prompt (safety filter, recitation, etc.) it
    # returns a response with parts=None and a populated prompt_feedback
    # or candidate.finish_reason field instead. Surface that reason so the
    # caller doesn't see a generic TypeError ("NoneType is not iterable").
    response_parts = response.parts if getattr(response, "parts", None) is not None else None
    if not response_parts:
        block_reason = None
        # Check prompt_feedback for a top-level block.
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None:
            block_reason = (
                getattr(feedback, "block_reason_message", None)
                or getattr(feedback, "block_reason", None)
            )
        # Check the first candidate's finish_reason (e.g. SAFETY, RECITATION,
        # IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT, PROHIBITED_CONTENT).
        candidate_finish = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            candidate_finish = getattr(candidates[0], "finish_reason", None)
            if hasattr(candidate_finish, "name"):
                candidate_finish = candidate_finish.name
        reason_bits = [str(x) for x in (block_reason, candidate_finish) if x]
        reason_str = " / ".join(reason_bits) or "no image part and no block reason"
        return {
            "success": False,
            "error": (
                f"Gemini returned no image — {reason_str}. "
                "This usually means a safety filter triggered (e.g. a named "
                "public figure, brand, or sensitive subject). Try editing the "
                "prompt to remove specific names or sensitive details, then "
                "regenerate."
            ),
            "cost_time_s": cost_time,
        }

    # Walk parts; ignore "thought" parts (interim images during the
    # Thinking phase) and return the first real image part.
    try:
        for part in response_parts:
            if getattr(part, "thought", False):
                continue
            inline = getattr(part, "inline_data", None)
            if inline is None or not getattr(inline, "data", None):
                continue
            data = inline.data
            if isinstance(data, str):
                # Some SDK paths still return base64-encoded strings.
                import base64
                data = base64.b64decode(data)
            return {
                "success": True,
                "image_bytes": data,
                "mime_type": inline.mime_type or "image/png",
                "cost_time_s": cost_time,
            }
    except Exception as e:
        return {"success": False, "error": f"Couldn't parse Gemini response: {str(e)[:200]}", "cost_time_s": cost_time}

    return {"success": False, "error": "Gemini response contained parts but no image data.", "cost_time_s": cost_time}
