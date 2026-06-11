// 统一 API 封装。所有后端调用集中于此，便于维护与错误处理。
// SSE 解析逻辑移植自原 Vanilla app.js（fetch + ReadableStream，EventSource 不支持 POST）。

async function jsonFetch(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

export const api = {
  // ---- 对话 ----
  async streamChat({ userId, message, history }, handlers) {
    const resp = await fetch("/agent/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, message, history: (history || []).slice(-10) }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        for (const line of part.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          let event;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }
          handlers.onEvent?.(event);
        }
      }
    }
  },
  chatSync: (userId, message, history) =>
    jsonFetch("/chat", { method: "POST", body: JSON.stringify({ user_id: userId, message, history }) }),

  // ---- 搜索 / 推荐 ----
  search: (userId, query) =>
    jsonFetch("/search", { method: "POST", body: JSON.stringify({ user_id: userId, query, include_external: true, top_k: 12 }) }),
  dailyRecommend: (userId, timeOfDay) =>
    jsonFetch("/recommend/daily", { method: "POST", body: JSON.stringify({ user_id: userId, time_of_day: timeOfDay }) }),

  // ---- 库 ----
  listAssets: () => jsonFetch("/assets"),
  rate: (userId, assetId, score) =>
    jsonFetch("/rate", { method: "POST", body: JSON.stringify({ user_id: userId, asset_id: assetId, score }) }),
  deleteAsset: (assetId, userId) =>
    jsonFetch(`/assets/${assetId}?user_id=${encodeURIComponent(userId)}`, { method: "DELETE" }),
  ingest: (url) =>
    jsonFetch("/assets/ingest_full", { method: "POST", body: JSON.stringify({ url, force_refresh: false }) }),

  // ---- 歌单 ----
  generatePlaylist: (userId, instruction) =>
    jsonFetch("/playlist/generate", { method: "POST", body: JSON.stringify({ user_id: userId, instruction }) }),
  autoPlaylists: (userId) =>
    jsonFetch(`/playlist/auto/${encodeURIComponent(userId)}`, { method: "POST" }),
  listPlaylists: (userId) => jsonFetch(`/playlists/${encodeURIComponent(userId)}`),
  deletePlaylist: (userId, pid) =>
    jsonFetch(`/playlist/${encodeURIComponent(userId)}/${encodeURIComponent(pid)}`, { method: "DELETE" }),

  // ---- 记忆 / 反馈 ----
  updateMemory: (userId, event) =>
    jsonFetch("/memory/update", { method: "POST", body: JSON.stringify({ user_id: userId, event }) }),
  getMemory: (userId) => jsonFetch(`/memory/${encodeURIComponent(userId)}`),
  getTaste: (userId) => jsonFetch(`/taste/${encodeURIComponent(userId)}`),
  getRatings: (userId) => jsonFetch(`/ratings/${encodeURIComponent(userId)}`),
  dislike: (userId, track) =>
    jsonFetch("/feedback/dislike", { method: "POST", body: JSON.stringify({
      user_id: userId, title: track.title, artist: track.artist,
      source: track.source, source_id: track.source_id || track.external_id || "",
    }) }),

  // ---- 排除规则（偏好设置） ----
  listExclusions: (userId) => jsonFetch(`/exclusions/${encodeURIComponent(userId)}`),
  addExclusion: (userId, rule) =>
    jsonFetch(`/exclusions/${encodeURIComponent(userId)}`, { method: "POST", body: JSON.stringify({ rule }) }),
  removeExclusion: (userId, rule) =>
    jsonFetch(`/exclusions/${encodeURIComponent(userId)}/${encodeURIComponent(rule)}`, { method: "DELETE" }),

  // ---- 播放 ----
  playbackAudio: (userId, track) =>
    jsonFetch("/api/playback/audio", { method: "POST", body: JSON.stringify({ track, user_id: userId }) }),
  playbackMv: (userId, track) =>
    jsonFetch("/api/playback/mv", { method: "POST", body: JSON.stringify({ track, user_id: userId }) }),

  // ---- 网易云认证 ----
  neteaseQrKey: () => jsonFetch("/auth/netease/qr/key"),
  neteaseQrStatus: (unikey, userId) =>
    jsonFetch(`/auth/netease/qr/status?unikey=${encodeURIComponent(unikey)}&user_id=${encodeURIComponent(userId)}`),
  neteaseAccount: (userId) => jsonFetch(`/auth/netease/account?user_id=${encodeURIComponent(userId)}`),
  neteaseUnbind: (userId) =>
    jsonFetch("/auth/netease/unbind", { method: "POST", body: JSON.stringify({ user_id: userId }) }),
  importNetease: (userId, playlistRef, limit) =>
    jsonFetch("/playlist/import/netease", { method: "POST", body: JSON.stringify({ user_id: userId, playlist_ref: playlistRef, limit }) }),
  neteasePlaylistList: (userId) => jsonFetch(`/playlist/netease/list?user_id=${encodeURIComponent(userId)}`),
};
