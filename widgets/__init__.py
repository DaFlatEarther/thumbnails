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

THUMBNAIL_STUDIO_URI = "ui://widgets/thumbnail-studio.html"

# CSP allowlist for the iframe. Nano Banana Pro returns image URLs hosted on
# Kie's temporary file CDN — declare those origins so the <img src> loads.
THUMBNAIL_STUDIO_CSP = {
    "connectDomains": [],
    "resourceDomains": [
        "https://tempfile.aiquickdraw.com",
        "https://file.aiquickdraw.com",
        "https://cdn.kie.ai",
        "https://kieai.erweima.ai",
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
