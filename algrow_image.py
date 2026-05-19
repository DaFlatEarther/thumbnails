"""Algrow image-generation API wrapper.

Submits a generation job to algrow's `/api/generate-image`, polls
`/api/job-status/:id`, and returns the final image URL plus timing.
Algrow proxies several model families:

  • nano-banana-pro       — 3 credits, supports up to 8 reference images
  • nano-banana-2         — 2 credits, fast general-purpose
  • seedream-4.5-edit     — 2 credits, image-to-image (REQUIRES reference)
  • seedream-5.0-lite     — 2 credits, lightweight, optional reference

Why this exists alongside the direct Gemini path in gemini_image.py:
the direct path's safety filter blocks named real people and recognizable
brand references, which hits us hard on biography / brand-history
thumbnails — both in the prompt and in the reference image content.
Seedream's filter is more permissive. Letting users switch model is the
escape hatch.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import quote_plus

import requests

logger = logging.getLogger("algrow-image")

_API_BASE = (os.environ.get("ALGROW_API_BASE_URL") or "https://api.algrow.online").rstrip("/")
_API_KEY = (os.environ.get("ALGROW_API_KEY") or "").strip()

# Models the API accepts. Anything not in this set is rejected client-side
# so we don't spam algrow with bad model names.
KNOWN_MODELS = {
    "nano-banana-pro",
    "nano-banana-2",
    "seedream-4.5-edit",
    "seedream-5.0-lite",
}

# seedream-4.5-edit requires a reference image — the API rejects calls
# without one. We surface that as a clean error rather than letting it
# fail in the upstream.
REQUIRES_REFERENCE = {"seedream-4.5-edit"}


def generate_image(
    *,
    prompt: str,
    model: str,
    aspect_ratio: str = "16:9",
    reference_url: str | None = None,
    timeout_s: float = 180.0,
    poll_interval_s: float = 2.0,
) -> dict[str, Any]:
    """Submit + poll. Blocks until the job completes, fails, or times out.

    Returns {"success": True, "image_url", "image_bytes" (None), "mime_type",
              "cost_time_s", "job_id", "credits_used", "model"}
    or       {"success": False, "error", "job_id" (if known), "model"}.

    image_bytes is None here because algrow already uploads the result to
    its own CDN — we just hand the URL back. (Caller persists it locally
    if it wants stable serving from PUBLIC_BASE_URL.)
    """
    if not _API_KEY:
        return {"success": False, "error": "Algrow image generation disabled (ALGROW_API_KEY not set)."}
    if model not in KNOWN_MODELS:
        return {"success": False, "error": f"Unknown algrow model '{model}'. Allowed: {sorted(KNOWN_MODELS)}."}
    if model in REQUIRES_REFERENCE and not reference_url:
        return {"success": False, "error": f"Model {model} requires a reference_image_url."}

    started = time.time()
    submit_body: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "aspect_ratio": aspect_ratio,
    }
    if reference_url:
        submit_body["reference_image_url"] = reference_url

    try:
        resp = requests.post(
            f"{_API_BASE}/api/generate-image",
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            },
            json=submit_body,
            timeout=30,
        )
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Algrow submit timed out (>30s)."}
    except Exception as e:
        return {"success": False, "error": f"Algrow submit failed: {str(e)[:200]}"}

    if resp.status_code != 200:
        return {
            "success": False,
            "error": f"Algrow HTTP {resp.status_code}: {resp.text[:300]}",
        }
    try:
        data = resp.json()
    except Exception:
        return {"success": False, "error": "Algrow returned non-JSON submit response."}
    if not data.get("success"):
        return {"success": False, "error": data.get("error") or "Algrow submit returned success=false."}

    job_id = data.get("job_id")
    credits_used = data.get("credits_used")
    if not job_id:
        return {"success": False, "error": "Algrow didn't return a job_id."}

    # Poll job status. Algrow's pricing is upfront-deduct + refund-on-fail,
    # so we don't need to worry about leaking credits if the poll times
    # out — the job will resolve on their side and refund.
    deadline = started + timeout_s
    while time.time() < deadline:
        try:
            sresp = requests.get(
                f"{_API_BASE}/api/job-status/{quote_plus(job_id)}",
                headers={"Authorization": f"Bearer {_API_KEY}"},
                timeout=15,
            )
        except Exception as e:
            logger.warning(f"algrow poll error: {e}")
            time.sleep(poll_interval_s)
            continue
        if sresp.status_code != 200:
            logger.warning(f"algrow poll HTTP {sresp.status_code}: {sresp.text[:200]}")
            time.sleep(poll_interval_s)
            continue
        try:
            sdata = sresp.json()
        except Exception:
            time.sleep(poll_interval_s)
            continue
        status = sdata.get("status")
        if status == "completed":
            urls = sdata.get("image_urls") or []
            if not urls:
                return {
                    "success": False,
                    "error": "Algrow job completed but image_urls was empty.",
                    "job_id": job_id,
                    "model": model,
                }
            return {
                "success": True,
                "image_url": urls[0],
                "all_image_urls": urls,
                "image_bytes": None,
                "mime_type": _mime_from_url(urls[0]),
                "cost_time_s": round(time.time() - started, 2),
                "job_id": job_id,
                "credits_used": credits_used,
                "model": model,
            }
        if status == "failed":
            return {
                "success": False,
                "error": sdata.get("error") or "Algrow job failed (no error message returned).",
                "job_id": job_id,
                "model": model,
            }
        # status in {"pending", "processing", None} → keep polling
        time.sleep(poll_interval_s)

    return {
        "success": False,
        "error": f"Algrow job timed out after {timeout_s:.0f}s (still polling).",
        "job_id": job_id,
        "model": model,
    }


def _mime_from_url(url: str) -> str:
    u = url.lower().split("?", 1)[0]
    if u.endswith(".png"):  return "image/png"
    if u.endswith(".jpg") or u.endswith(".jpeg"): return "image/jpeg"
    if u.endswith(".webp"): return "image/webp"
    return "image/png"
