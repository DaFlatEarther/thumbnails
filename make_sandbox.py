"""Build a standalone sandbox.html for quick UI tweaking of the widget.

The widget normally runs inside a Claude.ai iframe with globalThis.ExtApps
provided by the host (see widgets/__init__.py — the placeholder
/*__EXT_APPS_BUNDLE__*/ is replaced server-side). For offline UI work
we replace that placeholder with an inline mock and bolt on a floating
dev panel with buttons that inject canned tool-result payloads through
the same ontoolresult hook the widget already registers.

Usage:  python make_sandbox.py
Then open sandbox.html in any browser. No server needed.
"""
from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "widgets" / "thumbnail_studio.html"
DST = ROOT / "sandbox.html"

# --- 1. Mock that satisfies globalThis.ExtApps.App + applyHostStyleVariables.
# All callServerTool routes return canned content; ontoolresult is the
# real driver of state changes via the dev panel below.
EXT_APPS_MOCK = r"""
globalThis.ExtApps = {
  applyHostStyleVariables: function () {},
  App: class {
    constructor(meta, hostCtx, opts) {
      this.meta = meta;
      this.hostCtx = hostCtx || { theme: "dark" };
      this.opts = opts || {};
      this.ontoolresult = null;
      this.onhostcontextchanged = null;
      window.__sandboxApp = this;
    }
    async connect() {
      console.log("[sandbox] App.connect() — wire dev panel via window.__sandboxApp");
      return true;
    }
    getHostContext() { return this.hostCtx; }
    async openLink({ url }) { window.open(url, "_blank"); }
    async callServerTool({ name, arguments: args }) {
      console.log("[sandbox] callServerTool", name, args);
      return SANDBOX_TOOL_HANDLERS[name]
        ? SANDBOX_TOOL_HANDLERS[name](args)
        : { content: [{ text: "{}" }] };
    }
  },
};

// Canned responses for the tools the widget calls internally.
const SANDBOX_TOOL_HANDLERS = {
  load_widget_state: () => ({ content: [{ text: JSON.stringify({ state: null }) }] }),
  save_widget_state: () => ({ content: [{ text: "{}" }] }),
  find_outlier_references: (args) => ({
    content: [{ text: JSON.stringify(sandboxOutlierPickerMulti(args && args.topic)) }],
  }),
  compose_thumbnail_prompt: (args) => ({
    content: [{ text: JSON.stringify({
      success: true,
      title: (args && args.title) || "",
      prompt: "A bold cinematic thumbnail. [compose mock — paste a real engineered prompt here to test rendering]",
    }) }],
  }),
  generate_thumbnail: (args) => ({
    content: [{ text: JSON.stringify(sandboxSuccessPayload(args)) }],
  }),
  check_thumbnail_status: (args) => ({
    content: [{ text: JSON.stringify(sandboxSuccessPayload(args)) }],
  }),
};

// --- Canned payloads the dev panel can fire through ontoolresult. -------

function sandboxOutlierPickerSingle(opts) {
  return {
    view: "outlier_picker",
    topic: (opts && opts.title) || "The Man Who Saved McDonald's",
    title: (opts && opts.title) || "The Man Who Saved McDonald's",
    outliers: [{
      video_id: "fT4zkqM6JSQ",
      title: "The Man Who Accidentally Saved Nintendo",
      thumbnail_url: "https://i.ytimg.com/vi/fT4zkqM6JSQ/maxresdefault.jpg",
      channel_name: "Modern Ideas",
      channel_thumbnail: null,
      view_count: 19559,
      outlier_score: null,
      url: "https://www.youtube.com/watch?v=fT4zkqM6JSQ",
    }],
    count: 1,
    content_type: "longform",
    single_reference: true,
    auto_pipeline: (opts && opts.auto) !== false,
    model: (opts && opts.model) || "nano-banana-pro",
  };
}

function sandboxOutlierPickerMulti(topic) {
  const samples = [
    { vid: "Rm-hSXCv5ko", title: "The 7 Levels of Kpop",  channel: "internetsnathan", views: 312445, score: 3.8 },
    { vid: "7zD0hGhMVqg", title: "The Man Who Turned A Lifelong Failure Into KFC", channel: "Biz Stories", views: 209926, score: 3.78 },
    { vid: "OyKPASXUy48", title: "The Fascinating Story of Ryobi", channel: "Brand Origins", views: 145001, score: 2.9 },
    { vid: "CIJ7PmGvYns", title: "Why This Watch Costs $50,000", channel: "Watch Lab", views: 980132, score: 4.6 },
    { vid: "SlEfdb6wvlE", title: "Kings Who Lost Their Minds", channel: "History Capsule", views: 458321, score: 3.1 },
    { vid: "bIEe29lh6_I", title: "How Time Becomes Space Inside a Black Hole", channel: "Cosmic", views: 712998, score: 5.2 },
  ];
  return {
    view: "outlier_picker",
    topic: topic || "viral thumbnail references",
    title: "",
    content_type: "longform",
    outliers: samples.map((s) => ({
      video_id: s.vid,
      title: s.title,
      thumbnail_url: `https://i.ytimg.com/vi/${s.vid}/maxresdefault.jpg`,
      channel_name: s.channel,
      channel_thumbnail: null,
      view_count: s.views,
      outlier_score: s.score,
      url: `https://www.youtube.com/watch?v=${s.vid}`,
    })),
    count: samples.length,
  };
}

function sandboxOutlierPickerEmpty(topic) {
  return {
    view: "outlier_picker",
    topic: topic || "How Does Time Become Space Inside a Black Hole",
    title: topic || "How Does Time Become Space Inside a Black Hole",
    outliers: [],
    count: 0,
    content_type: "longform",
  };
}

function sandboxPending(taskId) {
  return {
    state: "pending",
    task_id: taskId || "sandbox-pending-task-1",
    prompt: "test prompt",
    aspect_ratio: "16:9",
    resolution: "2K",
    reference_urls: [],
    style_preset: "none",
    model: "nano-banana-pro",
  };
}

function sandboxSuccessPayload(args) {
  return {
    state: "success",
    images: ["https://picsum.photos/seed/sandbox/1280/720"],
    prompt: (args && args.prompt) || "test prompt",
    aspect_ratio: (args && args.aspect_ratio) || "16:9",
    resolution: (args && args.resolution) || "2K",
    reference_urls: (args && args.reference_urls) || [],
    style_preset: (args && args.style_preset) || "none",
    model: (args && args.model) || "nano-banana-pro",
    cost_time_s: 24.3,
  };
}

function sandboxFail() {
  return {
    state: "fail",
    error: "Gemini returned no image — BlockedReason.OTHER. This usually means a safety filter triggered.",
    raw_error: "BlockedReason.OTHER",
    prompt: "test prompt",
    aspect_ratio: "16:9",
    resolution: "2K",
    style_preset: "none",
    model: "nano-banana-pro",
  };
}

// --- Dev panel wiring (runs after the widget script has installed App). --

function sandboxFire(payload) {
  const app = window.__sandboxApp;
  if (!app || typeof app.ontoolresult !== "function") {
    console.warn("[sandbox] app not ready yet");
    return;
  }
  app.ontoolresult({ content: [{ text: JSON.stringify(payload) }] });
}

window.SANDBOX = {
  pickerSingle: (opts) => sandboxFire(sandboxOutlierPickerSingle(opts)),
  pickerSingleNoAuto: () => sandboxFire(sandboxOutlierPickerSingle({ auto: false })),
  pickerMulti: (topic) => sandboxFire(sandboxOutlierPickerMulti(topic)),
  pickerEmpty: (topic) => sandboxFire(sandboxOutlierPickerEmpty(topic)),
  pending: () => sandboxFire(sandboxPending()),
  success: () => sandboxFire(sandboxSuccessPayload({})),
  fail: () => sandboxFire(sandboxFail()),
  reset: () => location.reload(),
};
""".strip()

# --- 2. Floating dev panel CSS + HTML + wiring.
DEV_PANEL = r"""
<style>
  /* Sandbox-only chrome — won't ship to production widget. */
  body { background: #1f1f1f; }
  #sandbox-panel {
    position: fixed; bottom: 12px; right: 12px;
    z-index: 99999;
    background: rgba(20, 20, 20, 0.96);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.55);
    font-family: "DM Sans", system-ui, sans-serif;
    color: #fff;
    max-width: 280px;
  }
  #sandbox-panel h4 {
    margin: 0 0 8px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.55);
    font-weight: 700;
  }
  #sandbox-panel .row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-bottom: 6px;
  }
  #sandbox-panel button {
    background: rgba(77,156,255,0.12);
    border: 1px solid rgba(77,156,255,0.35);
    color: #cfe1ff;
    font-size: 11.5px;
    font-weight: 600;
    padding: 6px 10px;
    border-radius: 8px;
    cursor: pointer;
    font-family: inherit;
  }
  #sandbox-panel button:hover {
    background: rgba(77,156,255,0.22);
  }
  #sandbox-panel button.danger {
    background: rgba(255,107,107,0.12);
    border-color: rgba(255,107,107,0.35);
    color: #ffb8b8;
  }
  #sandbox-panel .toggle {
    position: absolute; top: -10px; right: 10px;
    background: #4d9cff; border: none; color: #fff;
    width: 22px; height: 22px; border-radius: 50%;
    font-size: 14px; font-weight: 700; cursor: pointer;
  }
  #sandbox-panel.collapsed > *:not(.toggle) { display: none; }
  #sandbox-panel.collapsed { padding: 4px 10px 4px 14px; }
  /* Make the widget container roughly the size claude.ai gives it so
     the iframe-relative math (aspect-ratio etc.) looks right. */
  body > .wrap, body > div.wrap { max-width: 720px; margin: 24px auto; padding: 0 16px 16px; }
</style>
<div id="sandbox-panel">
  <button class="toggle" onclick="this.parentElement.classList.toggle('collapsed')">_</button>
  <h4>Sandbox · widget states</h4>
  <div class="row">
    <button onclick="SANDBOX.pickerMulti()">Picker (multi)</button>
    <button onclick="SANDBOX.pickerEmpty()">Picker (empty)</button>
  </div>
  <div class="row">
    <button onclick="SANDBOX.pickerSingle()">Single + auto-pipeline</button>
    <button onclick="SANDBOX.pickerSingleNoAuto()">Single (no auto)</button>
  </div>
  <div class="row">
    <button onclick="SANDBOX.pending()">Loader / game</button>
    <button onclick="SANDBOX.success()">Success result</button>
    <button onclick="SANDBOX.fail()">Fail (Blocked)</button>
  </div>
  <div class="row">
    <button class="danger" onclick="SANDBOX.reset()">Reload</button>
  </div>
</div>
"""

# --- 3. Build sandbox.html. --------------------------------------------

def main() -> int:
    if not SRC.is_file():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1
    src = SRC.read_text(encoding="utf-8")
    if "/*__EXT_APPS_BUNDLE__*/" not in src:
        print("source missing /*__EXT_APPS_BUNDLE__*/ placeholder", file=sys.stderr)
        return 1
    out = src.replace("/*__EXT_APPS_BUNDLE__*/", EXT_APPS_MOCK)
    out = out.replace("</body>", DEV_PANEL + "\n</body>", 1)
    DST.write_text(out, encoding="utf-8")
    print(f"wrote {DST} ({len(out):,} bytes)")
    print("open it in any browser — no server needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
