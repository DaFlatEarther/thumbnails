// Node bridge: extract YouTube video metadata via youtubei.js
//
// Usage: node extract_video.mjs <url_or_id>
// Prints JSON to stdout:
//   { success: true, video_id, title, channel_name, thumbnail_url, duration_s, view_count }
//   { success: false, error: "..." }
//
// Called from Python (server.py) via subprocess to back the
// extract_reference_from_video MCP tool. Uses youtubei.js (Innertube) so
// we don't need a YouTube Data API key.

import { Innertube } from 'youtubei.js';

const YT_ID_RE = /(?:youtube\.com\/(?:watch\?(?:[^#]*&)?v=|shorts\/|embed\/|live\/)|youtu\.be\/)([A-Za-z0-9_-]{11})/;
const BARE_ID_RE = /^[A-Za-z0-9_-]{11}$/;

function extractId(input) {
  if (!input) return null;
  const s = input.trim();
  if (BARE_ID_RE.test(s)) return s;
  const m = s.match(YT_ID_RE);
  return m ? m[1] : null;
}

async function main() {
  const arg = process.argv[2];
  const id = extractId(arg);
  if (!id) {
    console.log(JSON.stringify({ success: false, error: `Couldn't extract video ID from: ${arg}` }));
    process.exit(0);
  }

  try {
    const yt = await Innertube.create({ retrieve_player: false });
    const info = await yt.getBasicInfo(id);
    const basic = info?.basic_info || {};
    // Pick the largest available thumbnail; fall back to maxresdefault URL.
    const thumbs = basic.thumbnail || [];
    let thumbnail_url = thumbs.length
      ? thumbs.reduce((a, b) => ((a?.width || 0) >= (b?.width || 0) ? a : b))?.url
      : null;
    if (!thumbnail_url) {
      thumbnail_url = `https://i.ytimg.com/vi/${id}/maxresdefault.jpg`;
    }

    console.log(JSON.stringify({
      success: true,
      video_id: id,
      title: basic.title || null,
      channel_name: basic.author || null,
      thumbnail_url,
      duration_s: basic.duration ?? null,
      view_count: basic.view_count ?? null,
    }));
  } catch (e) {
    console.log(JSON.stringify({ success: false, error: String(e?.message || e).slice(0, 400) }));
  }
}

main();
