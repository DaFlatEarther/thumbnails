"""Extract YouTube video metadata (title, thumbnail, channel) via youtubei.js.

Calls the Node.js bridge `extract_video.mjs` as a subprocess. The bridge
uses youtubei.js (Innertube) so we don't need a YouTube Data API key.

Used by:
  • compose_thumbnail_prompt — auto-fetches the reference video's title
    when the caller didn't pass one but the URL is a YouTube video.
  • extract_reference_from_video — direct entry point for Claude/widget
    when the user wants to base a thumbnail on a specific video they pasted.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("youtube-extract")

_BRIDGE = (Path(__file__).resolve().parent / "extract_video.mjs")

_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    s = url_or_id.strip()
    if _BARE_ID_RE.match(s):
        return s
    m = _YT_ID_RE.search(s)
    return m.group(1) if m else None


def extract_video_info(url_or_id: str, timeout_s: float = 20.0) -> dict[str, Any]:
    """Return video metadata via the youtubei.js Node bridge.

    Success shape:
      {"success": True, "video_id", "title", "channel_name", "thumbnail_url",
       "duration_s", "view_count"}
    Failure shape:
      {"success": False, "error": "..."}
    """
    if not _BRIDGE.is_file():
        return {"success": False, "error": f"Node bridge missing: {_BRIDGE}"}
    node = shutil.which("node")
    if not node:
        return {"success": False, "error": "Node.js not on PATH"}
    try:
        proc = subprocess.run(
            [node, str(_BRIDGE), url_or_id],
            capture_output=True,
            timeout=timeout_s,
            check=False,
            text=True,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Node bridge timed out after {timeout_s}s"}
    except Exception as e:
        return {"success": False, "error": f"Node bridge exec failed: {str(e)[:200]}"}

    out = (proc.stdout or "").strip()
    if not out:
        err = (proc.stderr or "").strip()[:300] or "empty stdout"
        return {"success": False, "error": f"Node bridge returned no output. stderr: {err}"}
    try:
        data = json.loads(out)
    except Exception as e:
        return {"success": False, "error": f"Couldn't parse bridge output: {str(e)[:150]}; out: {out[:200]}"}
    return data
