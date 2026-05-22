// Node bridge: list a channel's videos via youtubei.js Innertube, AND
// return channel metadata (display name + avatar).
//
// Usage: node extract_channel_videos.mjs <channel_id> [limit]
// Prints JSON to stdout:
//   { success: true,
//     channel: { name, avatar_url, banner_url, subscriber_count_text },
//     videos: [{ video_id, title, thumbnail_url, view_count }] }
//   { success: false, error: "..." }

import { Innertube } from 'youtubei.js';

const channelId = process.argv[2];
const limit = parseInt(process.argv[3] || '60', 10);

if (!channelId) {
  console.log(JSON.stringify({ success: false, error: 'channel_id required' }));
  process.exit(0);
}

function parseViews(s) {
  if (!s) return 0;
  const m = String(s).match(/([\d.]+)\s*([KMB])?/i);
  if (!m) return 0;
  let n = parseFloat(m[1]);
  const suf = (m[2] || '').toUpperCase();
  if (suf === 'K') n *= 1e3;
  else if (suf === 'M') n *= 1e6;
  else if (suf === 'B') n *= 1e9;
  return Math.round(n);
}

function extractVideo(richItem) {
  const inner = richItem?.content;
  if (!inner || inner.content_type !== 'VIDEO') return null;
  const vid = inner.content_id;
  if (!vid) return null;
  const title = inner?.metadata?.title?.text || '';
  let viewsText = '';
  try {
    viewsText = inner?.metadata?.metadata?.metadata_rows?.[0]?.metadata_parts?.[0]?.text?.text || '';
  } catch (_) { viewsText = ''; }
  return {
    video_id: vid,
    title,
    thumbnail_url: `https://i.ytimg.com/vi/${vid}/hqdefault.jpg`,
    view_count: parseViews(viewsText),
  };
}

// Pull channel header metadata. Shape varies across Innertube versions; try
// several known field paths and return the first that gives us a value.
function extractChannelMeta(channel) {
  const meta = { name: null, avatar_url: null, banner_url: null, subscriber_count_text: null };
  // metadata.title is the modern path
  try { meta.name = channel?.metadata?.title || channel?.header?.author?.name || null; } catch (_) {}
  // Avatar — try header.content.image first (new shape), then header.author.thumbnails (legacy)
  try {
    const headerImg = channel?.header?.content?.image?.avatar?.image
                   || channel?.header?.content?.image?.image
                   || channel?.header?.author?.thumbnails;
    if (Array.isArray(headerImg) && headerImg.length > 0) {
      // Pick the largest
      const sorted = [...headerImg].sort((a, b) => (b.width || 0) - (a.width || 0));
      meta.avatar_url = sorted[0]?.url || null;
    } else if (Array.isArray(headerImg?.sources)) {
      const sorted = [...headerImg.sources].sort((a, b) => (b.width || 0) - (a.width || 0));
      meta.avatar_url = sorted[0]?.url || null;
    }
  } catch (_) {}
  // Banner (less critical, but cheap)
  try {
    const banner = channel?.header?.banner?.[0]?.url
                || channel?.header?.content?.banner?.image?.[0]?.url;
    if (banner) meta.banner_url = banner;
  } catch (_) {}
  try {
    meta.subscriber_count_text = channel?.metadata?.subscriber_count?.text
                              || channel?.header?.content?.subscriber_count?.text
                              || null;
  } catch (_) {}
  return meta;
}

(async () => {
  let yt;
  try {
    yt = await Innertube.create({ retrieve_player: false });
  } catch (e) {
    console.log(JSON.stringify({ success: false, error: `Innertube init failed: ${e?.message || e}` }));
    return;
  }

  let channel;
  try {
    channel = await yt.getChannel(channelId);
  } catch (e) {
    console.log(JSON.stringify({ success: false, error: `getChannel failed: ${e?.message || e}` }));
    return;
  }

  const channelMeta = extractChannelMeta(channel);

  let videosTab;
  try {
    videosTab = await channel.getVideos();
  } catch (e) {
    console.log(JSON.stringify({
      success: true,
      channel: channelMeta,
      videos: [],
      warning: `getVideos failed: ${e?.message || e}`,
    }));
    return;
  }

  const videos = [];
  let items = videosTab?.current_tab?.content?.contents || [];
  for (const it of items) {
    const v = extractVideo(it);
    if (v) videos.push(v);
    if (videos.length >= limit) break;
  }

  let page = videosTab?.current_tab?.content;
  let safety = 10;
  while (videos.length < limit && page?.has_continuation && safety-- > 0) {
    try {
      page = await page.getContinuation();
    } catch (e) {
      break;
    }
    const more = page?.contents || [];
    for (const it of more) {
      const v = extractVideo(it);
      if (v) videos.push(v);
      if (videos.length >= limit) break;
    }
  }

  console.log(JSON.stringify({
    success: true,
    channel: channelMeta,
    videos: videos.slice(0, limit),
  }));
})().catch(e => {
  console.log(JSON.stringify({ success: false, error: `top-level: ${e?.message || e}` }));
});
