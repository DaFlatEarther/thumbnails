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


def _fetch_image_bytes(url: str) -> tuple[bytes, str] | None:
    """Download a reference image; returns (bytes, mime_type) or None.

    Shortcut: if the URL points to our own /generated/ endpoint, read the
    file off disk directly. Avoids an HTTP roundtrip through the public
    tunnel (cloudflared → Cloudflare → back to us) when we already have
    the file locally. Important for multi-turn editing where the previous
    generation's image is passed as a reference for the next one.
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
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        mime = (r.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        return r.content, mime
    except Exception as e:
        logger.warning(f"reference fetch failed: {url}: {e}")
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

    # Walk response parts; ignore "thought" parts (interim images during
    # the Thinking phase) and return the first real image part.
    try:
        for part in response.parts:
            if getattr(part, "thought", False):
                continue
            inline = getattr(part, "inline_data", None)
            if inline is None or not inline.data:
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

    return {"success": False, "error": "Gemini response contained no image part", "cost_time_s": cost_time}
