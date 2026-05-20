"""MCP widget — inlines the ext-apps bundle into the HTML, registers the
resource on the FastMCP server, and exposes the URI / CSP for the tool to
reference via _meta.ui.

The bundle is shipped verbatim so the iframe doesn't need a CDN fetch
(iframe CSP blocks that). We rewrite the trailing `export{…}` into a
`globalThis.ExtApps={…}` assignment at startup so an inline
`<script type="module">` block can `globalThis.ExtApps.App`.
"""
from __future__ import annotations

import pathlib
import re
from functools import lru_cache

RESOURCE_MIME_TYPE = "text/html;profile=mcp-app"

_WIDGETS_DIR = pathlib.Path(__file__).resolve().parent
_BUNDLE_PATH = _WIDGETS_DIR / "ext_apps_bundle.js"
_PLACEHOLDER = "/*__EXT_APPS_BUNDLE__*/"

_EXPORT_RE = re.compile(r"export\{([^}]+)\};?\s*$")


def _rewrite_exports(bundle: str) -> str:
    match = _EXPORT_RE.search(bundle)
    if not match:
        raise RuntimeError("ext_apps_bundle.js is missing the trailing export{…} statement")
    pairs = []
    for raw in match.group(1).split(","):
        parts = [p.strip() for p in raw.split(" as ")]
        local = parts[0]
        exported = parts[1] if len(parts) > 1 else parts[0]
        pairs.append(f"{exported}:{local}")
    return _EXPORT_RE.sub("globalThis.ExtApps={" + ",".join(pairs) + "};", bundle)


@lru_cache(maxsize=1)
def _bundle() -> str:
    return _rewrite_exports(_BUNDLE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_widget_html(filename: str) -> str:
    """Read a widget HTML file and inline the ExtApps bundle into it."""
    html = (_WIDGETS_DIR / filename).read_text(encoding="utf-8")
    if _PLACEHOLDER not in html:
        raise RuntimeError(f"Widget {filename} is missing {_PLACEHOLDER}")
    return html.replace(_PLACEHOLDER, _bundle())


# ---------------------------------------------------------------------------
# Widget registry — single thumbnail studio for now.
# ---------------------------------------------------------------------------

THUMBNAIL_STUDIO_URI = "ui://widgets/thumbnail-studio-v30.html"

# CSP allowlist for the iframe. We now serve generated images from this
# server itself (under /generated/<uuid>.png), so PUBLIC_BASE_URL is added
# automatically when present. Kie's legacy CDNs are kept on the list for
# back-compat in case anything is still referencing them, but the active
# generation path is the official Gemini image API → local disk → /generated.
import os as _os
_PUBLIC_BASE_URL = (_os.environ.get("PUBLIC_BASE_URL") or "").rstrip("/")
THUMBNAIL_STUDIO_CSP = {
    "connectDomains": [],
    "resourceDomains": [
        # Where the official Gemini-generated image is served from.
        *([_PUBLIC_BASE_URL] if _PUBLIC_BASE_URL else []),
        # Legacy: Kie's image CDNs (no longer used by default).
        "https://tempfile.aiquickdraw.com",
        "https://file.aiquickdraw.com",
        "https://cdn.kie.ai",
        "https://kieai.erweima.ai",
        # YouTube thumbnail CDNs — where reference image previews load from
        # when the user pastes a YouTube URL (resolver normalizes them to
        # i.ytimg.com/vi/<id>/hqdefault.jpg). Without this the reference
        # tiles render as black boxes — generation still works because
        # Gemini fetches the URL server-side, but the widget preview is
        # CSP-blocked.
        "https://i.ytimg.com",
        "https://yt3.ggpht.com",
        "https://yt3.googleusercontent.com",
        # Algora's own R2-backed thumbnail mirror — the outlier picker
        # surfaces algrow's viral_videos rows whose `thumbnail_url` points
        # here (algora downloads + republishes YouTube thumbs to dodge
        # hotlink throttling). Without this on the allowlist the outlier
        # grid renders black boxes.
        "https://audio.algrow.online",
        # Google Fonts — the widget loads DM Sans for the algrow-style card
        # aesthetic. googleapis.com serves the stylesheet, gstatic.com
        # serves the woff2 files. Without these the !important font-family
        # rule still fires but resolves to the system fallback stack.
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
    ],
    "baseUriDomains": [],
}


def register(mcp) -> None:
    """Register widget resources on the FastMCP server."""

    @mcp.resource(
        uri=THUMBNAIL_STUDIO_URI,
        name="Thumbnail Studio",
        description="Prompt-driven AI thumbnail generator (Nano Banana Pro)",
        mime_type=RESOURCE_MIME_TYPE,
        meta={"ui": {"csp": THUMBNAIL_STUDIO_CSP}},
    )
    def thumbnail_studio_resource() -> str:
        return load_widget_html("thumbnail_studio.html")
