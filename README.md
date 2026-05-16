# thumbnails

Open-source AI thumbnail generation. Ships as both a Python library and an
**MCP server with an interactive widget** — drop it into Claude (or any
apps-aware MCP host) and you get a prompt textarea + live preview inline.

Powered by **Nano Banana Pro** (Google's Gemini 2.5 Flash Image) via the
[Kie](https://kie.ai) API.

![demo](./.github/demo.png) <!-- placeholder, add when you have one -->

## What you get

| Layer | File |
|---|---|
| Kie API wrapper (`create_task`, `query_task`) | `nano_banana_pro.py` |
| Sync helper with thumbnail defaults (16:9, 2K) | `thumbnails.py` |
| **MCP server + widget** (the headline feature) | `server.py` + `widgets/` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env, put in your KIE_API_KEY
export $(grep -v '^#' .env | xargs)
```

## Use as a library

```python
from thumbnails import generate_thumbnail

result = generate_thumbnail(
    "Cinematic close-up of a hooded figure in fog, neon highlights",
    reference_images=["https://example.com/style-ref.png"],  # optional
)
# {'success': True, 'task_id': '...', 'images': ['https://...png'], 'cost_time_s': 38.2}
```

CLI smoke test:
```bash
python thumbnails.py "a cat in a spacesuit, octane render"
```

## Use as an MCP server (widget mode)

### Run locally

Streamable-HTTP (recommended — works with claude.ai connectors):
```bash
uvicorn server:app --host 0.0.0.0 --port 8003
# → http://localhost:8003/mcp
```

stdio (for Claude Desktop / `mcp dev`):
```bash
python server.py
```

### Add to Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "thumbnails": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/mcp",
               "--allow-http", "--transport", "http-only"]
    }
  }
}
```

### Add to claude.ai (custom connector)

After deploying behind HTTPS, paste your server URL (e.g.
`https://thumbnails-mcp.example.com/mcp`) into claude.ai's "Add custom
connector" dialog. If `THUMBNAILS_MCP_TOKEN` is set, supply it as the bearer.

### How the widget works

When Claude calls `generate_thumbnail`, the host renders an iframe with a
prompt textarea, aspect-ratio/resolution dropdowns, and a preview pane.
After the first image arrives the user can edit the prompt and hit
"Generate" again — clicks call back via `app.callServerTool` so subsequent
generations bypass the conversation entirely. The widget honours
host theme (light/dark) and is `autoResize`d.

CSP allowlist on the resource permits the Kie CDN
(`tempfile.aiquickdraw.com`, `file.aiquickdraw.com`, `cdn.kie.ai`,
`kieai.erweima.ai`) so the result `<img>` loads without base64-inlining.
If Kie serves you from a different origin, add it to
`widgets/__init__.py::THUMBNAIL_STUDIO_CSP`.

## Configuration

| Env var | Required | Effect |
|---|---|---|
| `KIE_API_KEY` | yes | Kie account API key. |
| `THUMBNAILS_MCP_TOKEN` | no | If set, `/mcp` requires `Authorization: Bearer <token>`. Leave unset for local dev; set on anything publicly reachable. |

## Defaults (override per call)

| Knob | Default | Notes |
|---|---|---|
| `aspect_ratio` | `16:9` | YouTube thumbnail standard. |
| `resolution` | `2K` | ~2048×1152. 4K is ~4× the cost in time and rarely worth it for thumbs. |
| `output_format` | `png` | Use `jpg` for smaller files. |
| `timeout_s` | `300` | Most jobs return in 30–90 s. |
| `poll_interval_s` | `3` | Cheap. |

## Aspect ratios

`1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, `auto`

## Pricing

3 Kie credits per generation regardless of resolution.

## License

MIT.
