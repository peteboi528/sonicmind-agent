// 统一 API 封装。所有后端调用集中于此，便于维护与错误处理。
// SSE 解析逻辑移植自原 Vanilla app.js（fetch + ReadableStream，EventSource 不支持 POST）。

async function jsonFetch(url, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const resp = await fetch(url, {
    ...rest,
    headers: { "Content-Type": "application/json", ...extraHeaders },
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

async function streamSse(url, body, handlers, signal) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const consume = (text) => {
    for (const line of text.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      try { handlers.onEvent?.(JSON.parse(line.slice(6))); } catch { /* malformed event */ }
    }
  };
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    parts.forEach(consume);
  }
  if (buffer.trim()) consume(buffer);
}

export const api = {
  // ---- 对话 ----
  streamChat: ({ userId, threadId, message, history }, handlers, signal) =>
    streamSse("/agent/stream", {
      user_id: userId, thread_id: threadId, message, history: (history || []).slice(-10),
    }, handlers, signal),
  resumeAgent: ({ userId, threadId, actionId, approved }, handlers, signal) =>
    streamSse("/agent/resume", {
      user_id: userId, thread_id: threadId, action_id: actionId, approved,
    }, handlers, signal),
  chatSync: (userId, message, history) =>
    jsonFetch("/chat", { method: "POST", body: JSON.stringify({ user_id: userId, message, history }) }),

  // ---- 搜索 / 推荐 ----
  search: (userId, query) =>
    jsonFetch("/search", { method: "POST", body: JSON.stringify({ user_id: userId, query, include_external: true, top_k: 12 }) }),
  dailyRecommend: (userId, timeOfDay, noLocal = false) =>
    jsonFetch("/recommend/daily", { method: "POST", body: JSON.stringify({ user_id: userId, time_of_day: timeOfDay, no_local: noLocal }) }),

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
  generateTasteExperiment: (userId, prompt, total = 12) =>
    jsonFetch("/taste/experiment/generate", { method: "POST", body: JSON.stringify({ user_id: userId, prompt, total }) }),
  listTasteExperiments: (userId) =>
    jsonFetch(`/taste/experiments/${encodeURIComponent(userId)}`),
  getTasteExperiment: (userId, experimentId) =>
    jsonFetch(`/taste/experiment/${encodeURIComponent(userId)}/${encodeURIComponent(experimentId)}`),
  tasteExperimentFeedback: (userId, experimentId, trackKey, signal, score = null) =>
    jsonFetch("/taste/experiment/feedback", { method: "POST", body: JSON.stringify({
      user_id: userId, experiment_id: experimentId, track_key: trackKey, signal, score,
    }) }),
  tasteExperimentReport: (userId, experimentId) =>
    jsonFetch("/taste/experiment/report", { method: "POST", body: JSON.stringify({ user_id: userId, experiment_id: experimentId }) }),
  regenerateTasteBucket: (userId, experimentId, bucket) =>
    jsonFetch("/taste/experiment/regenerate", { method: "POST", body: JSON.stringify({ user_id: userId, experiment_id: experimentId, bucket }) }),
  deleteTasteExperiment: (userId, experimentId) =>
    jsonFetch(`/taste/experiment/${encodeURIComponent(userId)}/${encodeURIComponent(experimentId)}`, { method: "DELETE" }),
  getRatings: (userId) => jsonFetch(`/ratings/${encodeURIComponent(userId)}`),

  // ---- 用户画像（可解释品味仪表盘）----
  getProfile: (userId) => jsonFetch(`/profile/${encodeURIComponent(userId)}`),
  profileInsightFeedback: (userId, insightId, action) =>
    jsonFetch(`/profile/insights/${encodeURIComponent(insightId)}/feedback`, {
      method: "POST", body: JSON.stringify({ user_id: userId, action }),
    }),
  deleteProfileInsight: (userId, insightId) =>
    jsonFetch(`/profile/insights/${encodeURIComponent(userId)}/${encodeURIComponent(insightId)}`, { method: "DELETE" }),
  clearProfile: (userId) =>
    jsonFetch(`/profile/${encodeURIComponent(userId)}`, { method: "DELETE" }),
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
  // ---- 收听行为上报（喂行为锚：听完=completed，秒跳=skip）----
  listen: (userId, assetId, duration, completed, context = "player") =>
    jsonFetch("/listen", { method: "POST", body: JSON.stringify({
      user_id: userId, asset_id: assetId, duration, completed, context,
    }) }),

  // ---- 歌词 ----
  getLyrics: (userId, { title, artist, sourceId }) =>
    jsonFetch("/lyrics", { method: "POST", body: JSON.stringify({
      user_id: userId, title, artist, source_id: sourceId || "",
    }) }),

  // ---- 发现 / 歌手 ----
  discoverBrowse: (userId, category, value, limit = 12, seed = 0) =>
    jsonFetch("/discover/browse", { method: "POST", body: JSON.stringify({ user_id: userId, category, value, limit, seed }) }),
  discoverTrending: (userId, limit = 12) =>
    jsonFetch("/discover/trending", { method: "POST", body: JSON.stringify({ user_id: userId, limit }) }),
  discoverClassify: (query) =>
    jsonFetch("/discover/classify", { method: "POST", body: JSON.stringify({ query }) }),
  discoverSearch: (userId, query, topK = 12) =>
    jsonFetch("/discover/search", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, query, include_external: false, top_k: topK }),
    }),
  discoverSearchExternal: (userId, query, topK = 12) =>
    jsonFetch("/discover/search", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, query, external_only: true, top_k: topK }),
    }),
  artistInfo: (artist) =>
    jsonFetch("/artist/info", { method: "POST", body: JSON.stringify({ artist }) }),
  artistAlbumTracks: (artist, album, albumId, limit = 100) =>
    jsonFetch("/artist/album_tracks", { method: "POST", body: JSON.stringify({ artist, album, album_id: albumId || null, limit }) }),

  // ---- 收藏专辑 ----
  saveAlbum: (userId, album) =>
    jsonFetch("/album/save", { method: "POST", body: JSON.stringify({ user_id: userId, ...album }) }),
  unsaveAlbum: (userId, albumId) =>
    jsonFetch(`/album/saved/${encodeURIComponent(userId)}/${encodeURIComponent(albumId)}`, { method: "DELETE" }),
  listSavedAlbums: (userId) => jsonFetch(`/albums/saved/${encodeURIComponent(userId)}`),
  isAlbumSaved: (userId, albumId) => jsonFetch(`/album/saved/${encodeURIComponent(userId)}/${encodeURIComponent(albumId)}`),

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
