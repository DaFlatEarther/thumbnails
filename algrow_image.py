"""Algrow image-generation API wrapper.

Two-step pattern (the synchronous combined wrapper kept blowing
claude.ai's ~60s MCP timeout for slow models):

  • submit_image(...)  → returns {"success", "job_id", "credits_used", ...}
                          almost immediately.
  • check_image_status(job_id) → polls /api/job-status/:id once. Returns
                                   {"state": "pending" | "success" | "fail",
                                    "image_url", "error", ...}.

generate_thumbnail uses submit_image and returns state=pending+task_id;
check_thumbnail_status uses check_image_status on every poll round.

Algrow proxies several model families:
  • nano-banana-pro       — 3 credits, supports up to 8 reference images
  • nano-banana-2         — 2 credits, fast general-purpose
  • seedream-4.5-edit     — 2 credits, image-to-image (REQUIRES reference)
  • seedream-5.0-lite     — 2 credits, lightweight, optional reference

Why this exists alongside the direct Gemini path in gemini_image.py:
the direct path's safety filter blocks named real people and recognizable
brand references, which hits us hard on biography / brand-history
thumbnails — both in the prompt AND in the reference image content.
Seedream's filter is more permissive. The model switch is the escape
hatch when Gemini keeps returning BlockedReason.OTHER.
"""
from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import quote_plus

import requests

logger = logging.getLogger("algrow-image")

_API_BASE = (os.environ.get("ALGROW_API_BASE_URL") or "https://api.algrow.online").rstrip("/")
_API_KEY = (os.environ.get("ALGROW_API_KEY") or "").strip()

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

# task_id prefix that marks a job as algrow-backed. check_thumbnail_status
# uses this to route between Kie (legacy) and algrow polling.
TASK_ID_PREFIX = "algrow:"


def is_algrow_task_id(task_id: str) -> bool:
    return isinstance(task_id, str) and task_id.startswith(TASK_ID_PREFIX)


def submit_image(
    *,
    prompt: str,
    model: str,
    aspect_ratio: str = "16:9",
    reference_url: str | None = None,
) -> dict[str, Any]:
    """POST /api/generate-image — returns job_id quickly (no polling)."""
    if not _API_KEY:
        return {"success": False, "error": "Algrow image generation disabled (ALGROW_API_KEY not set)."}
    if model not in KNOWN_MODELS:
        return {"success": False, "error": f"Unknown algrow model '{model}'. Allowed: {sorted(KNOWN_MODELS)}."}
    if model in REQUIRES_REFERENCE and not reference_url:
        return {"success": False, "error": f"Model {model} requires a reference_image_url."}

    body: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "aspect_ratio": aspect_ratio,
    }
    if reference_url:
        body["reference_image_url"] = reference_url

    try:
        resp = requests.post(
            f"{_API_BASE}/api/generate-image",
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Algrow submit timed out (>30s)."}
    except Exception as e:
        return {"success": False, "error": f"Algrow submit failed: {str(e)[:200]}"}

    if resp.status_code != 200:
        return {"success": False, "error": f"Algrow HTTP {resp.status_code}: {resp.text[:300]}"}
    try:
        data = resp.json()
    except Exception:
        return {"success": False, "error": "Algrow returned non-JSON submit response."}
    if not data.get("success"):
        return {"success": False, "error": data.get("error") or "Algrow submit returned success=false."}

    job_id = data.get("job_id")
    if not job_id:
        return {"success": False, "error": "Algrow didn't return a job_id."}

    return {
        "success": True,
        "job_id": job_id,
        "task_id": f"{TASK_ID_PREFIX}{job_id}",  # prefixed for check_thumbnail_status routing
        "credits_used": data.get("credits_used"),
        "model": model,
    }


def check_image_status(task_id: str) -> dict[str, Any]:
    """GET /api/job-status/:id — one round of polling.

    Returns {"state": "pending"|"success"|"fail", "image_url"?, "error"?}.
    Accepts a prefixed task_id (algrow:<job_id>) or a raw job_id.
    """
    if not _API_KEY:
        return {"state": "fail", "error": "Algrow integration not configured."}
    job_id = task_id[len(TASK_ID_PREFIX):] if task_id.startswith(TASK_ID_PREFIX) else task_id
    try:
        resp = requests.get(
            f"{_API_BASE}/api/job-status/{quote_plus(job_id)}",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            timeout=15,
        )
    except Exception as e:
        # Treat as pending so the widget keeps polling instead of bailing.
        return {"state": "pending", "transient_error": f"Algrow poll error: {str(e)[:150]}"}

    if resp.status_code != 200:
        return {"state": "pending", "transient_error": f"Algrow HTTP {resp.status_code}: {resp.text[:200]}"}
    try:
        data = resp.json()
    except Exception:
        return {"state": "pending", "transient_error": "Algrow returned non-JSON status response."}

    status = data.get("status")
    if status == "completed":
        urls = data.get("image_urls") or []
        if not urls:
            return {"state": "fail", "error": "Algrow job completed but image_urls was empty."}
        return {
            "state": "success",
            "image_url": urls[0],
            "all_image_urls": urls,
            "mime_type": _mime_from_url(urls[0]),
            "completed_at": data.get("completed_at"),
        }
    if status == "failed":
        return {"state": "fail", "error": data.get("error") or "Algrow job failed (no error message)."}
    # pending / processing / unknown
    return {"state": "pending", "upstream_state": status}


def _mime_from_url(url: str) -> str:
    u = url.lower().split("?", 1)[0]
    if u.endswith(".png"):  return "image/png"
    if u.endswith(".jpg") or u.endswith(".jpeg"): return "image/jpeg"
    if u.endswith(".webp"): return "image/webp"
    return "image/png"
