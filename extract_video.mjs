// Node bridge: extract YouTube video metadata.
//
// Usage: node extract_video.mjs <url_or_id>
// Prints JSON to stdout:
//   { success: true, video_id, title, channel_name, channel_thumbnail,
//     thumbnail_url, duration_s, view_count }
//   { success: false, error: "..." }
//
// Called from Python (server.py) via subprocess to back the
// extract_reference_from_video MCP tool.
//
// Two-stage strategy:
//   1. YouTube oEmbed (unauthenticated, HTTP GET) → title + channel name.
//      Works from datacenter IPs without bot challenges. This is the
//      reliable path that production depends on.
//   2. youtubei.js (Innertube) → richer fields (channel thumbnail, view
//      count, duration). YouTube routinely challenges Innertube from
//      datacenter IPs with LOGIN_REQUIRED, in which case basic_info comes
//      back empty. Treated as best-effort; failures are silent and we
//      return whatever oEmbed gave us plus null for the rest.

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

// YouTube's own oEmbed endpoint. Returns:
//   { title, author_name, author_url, thumbnail_url, ... }
// Unauthenticated, returns 404 for private/unlisted videos.
async function fetchOEmbed(id) {
  try {
    const url = `https://www.youtube.com/oembed?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D${id}&format=json`;
    const resp = await fetch(url, {
      headers: { 'user-agent': 'Mozilla/5.0' },
      signal: AbortSignal.timeout(8000),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_) { return null; }
}

async function tryInnertube(id) {
  try {
    const yt = await Innertube.create({ retrieve_player: false });
    const info = await yt.getBasicInfo(id);
    const basic = info?.basic_info || {};

    // basic.thumbnail is the largest variant array; pick the biggest.
    const thumbs = basic.thumbnail || [];
    const thumbnail_url = thumbs.length
      ? thumbs.reduce((a, b) => ((a?.width || 0) >= (b?.width || 0) ? a : b))?.url
      : null;

    // Channel avatar — needs a second fetch.
    let channel_thumbnail = null;
    const channelId = basic?.channel?.id || basic?.channel_id || null;
    if (channelId) {
      try {
        const ch = await yt.getChannel(channelId);
        const avatar = ch?.metadata?.thumbnail
          || ch?.header?.author?.thumbnails
          || [];
        if (Array.isArray(avatar) && avatar.length) {
          channel_thumbnail = avatar.reduce(
            (a, b) => ((a?.width || 0) >= (b?.width || 0) ? a : b)
          )?.url || null;
        }
      } catch (_) { /* nice-to-have */ }
    }

    return {
      title: basic.title || null,
      channel_name: basic.author || null,
      channel_thumbnail,
      thumbnail_url,
      duration_s: basic.duration ?? null,
      view_count: basic.view_count ?? null,
    };
  } catch (_) {
    return null;
  }
}

async function main() {
  const arg = process.argv[2];
  const id = extractId(arg);
  if (!id) {
    console.log(JSON.stringify({ success: false, error: `Couldn't extract video ID from: ${arg}` }));
    process.exit(0);
  }

  // Race oEmbed and Innertube in parallel — oEmbed wins on speed (~150ms)
  // and reliability; Innertube fills in extras when YouTube isn't
  // bot-challenging us.
  const [oembed, inner] = await Promise.all([
    fetchOEmbed(id),
    tryInnertube(id),
  ]);

  // Title / channel preference: oEmbed first (canonical and reliable),
  // then Innertube, then null.
  const title = (oembed?.title ?? inner?.title) || null;
  const channel_name = (oembed?.author_name ?? inner?.channel_name) || null;
  const thumbnail_url =
    inner?.thumbnail_url
    || oembed?.thumbnail_url
    || `https://i.ytimg.com/vi/${id}/maxresdefault.jpg`;

  console.log(JSON.stringify({
    success: true,
    video_id: id,
    title,
    channel_name,
    channel_thumbnail: inner?.channel_thumbnail ?? null,
    thumbnail_url,
    duration_s: inner?.duration_s ?? null,
    view_count: inner?.view_count ?? null,
  }));
}

main();
