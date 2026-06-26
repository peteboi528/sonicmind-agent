<script setup>
import { ref, watch } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const props = defineProps({
  card: { type: Object, required: true },
  showReason: { type: Boolean, default: true },
});
const emit = defineEmits(["toast", "added"]);

const adding = ref(false);
const coverBroken = ref(false);
watch(() => props.card.cover_url, () => { coverBroken.value = false; });

async function play() {
  try {
    const data = await api.playbackAudio(store.userId, props.card);
    if (!data.url) {
      const hints = {
        vip_required: "⚠️ 网易云付费歌曲，扫码登录后可播放。可点 MV 观看。",
        not_found: "⚠️ 该来源没有音频直链，点 MV 观看吧。",
        error: "⚠️ 取流失败，请稍后重试。",
      };
      emit("toast", hints[data.reason] || "⚠️ 暂无试听链接");
      return;
    }
    store.playTrack({
      title: props.card.title, artist: props.card.artist || "",
      cover: props.card.cover_url || "", url: data.url,
      source: props.card.source || "",
      sourceId: props.card.source_id || "",
      assetId: props.card.asset_id || "",
    });
  } catch { emit("toast", "⚠️ 播放失败"); }
}

async function playMv() {
  try {
    const data = await api.playbackMv(store.userId, props.card);
    if (!data.url) { emit("toast", "⚠️ 暂无 MV 链接"); return; }
    store.showMv(data.url);
  } catch { emit("toast", "⚠️ MV 播放失败"); }
}

async function dislike() {
  try {
    await api.dislike(store.userId, props.card);
    emit("toast", "已记录，以后会少推这类。");
  } catch { emit("toast", "操作失败"); }
}

async function addToLibrary() {
  const card = props.card;
  // 入库靠「解析来源页 URL → 识别 → 入库」（与歌单导入同路径），不是喂 CDN 直链。
  // 推荐/每日卡片不预带 playback_url（播放时才取流），故按 source+source_id 构造页 URL。
  let url = "";
  if (card.source === "netease" && card.source_id) {
    url = `https://music.163.com/song?id=${card.source_id}`;
  } else if (card.playback_url) {
    url = card.playback_url;  // bilibili/youtube 等回退到直链
  }
  if (!url) { emit("toast", "⚠️ 无可入库链接"); return; }
  adding.value = true;
  try {
    await api.ingest(url);
    emit("toast", `《${card.title}》已加入曲库 📥`);
    emit("added", card);
  } catch { emit("toast", "⚠️ 入库失败，请稍后重试"); }
  finally { adding.value = false; }
}
</script>

<template>
  <div class="song-card">
    <div class="cover-wrap">
      <img v-if="card.cover_url && !coverBroken" class="cover" :src="card.cover_url" alt="" loading="lazy" @error="coverBroken = true" />
      <div v-else class="cover-ph">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" opacity="0.4"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55C7.79 13 6 14.79 6 17s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
      </div>
    </div>
    <div class="info">
      <div class="title">{{ card.title || "未知曲目" }}</div>
      <div class="artist">
        {{ card.artist || "未知" }}
        <span v-if="card.source" class="src-tag">{{ card.source }}</span>
      </div>
      <div v-if="showReason && card.reason" class="reason">{{ card.reason }}</div>
    </div>
    <div class="actions">
      <button class="icon-btn play-btn" title="听歌" @click="play">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
      </button>
      <button class="icon-btn" title="MV" @click="playMv">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>
      </button>
      <button v-if="card.source !== 'local'" class="icon-btn" title="加入我的库" :disabled="adding" @click="addToLibrary">
        <svg v-if="!adding" width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
        <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="currentColor" class="spin"><path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"/></svg>
      </button>
      <button class="icon-btn" title="不喜欢" @click="dislike">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z"/></svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.song-card {
  display: flex; align-items: center; gap: 14px;
  background: var(--bg-card); border-radius: var(--radius);
  padding: 12px 16px; margin-bottom: 8px;
  border: 1px solid transparent;
  transition: all var(--dur-norm) var(--ease-out);
}
.song-card:hover {
  background: var(--bg-hover);
  border-color: var(--border-light);
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}

.cover-wrap {
  width: 50px; height: 50px; border-radius: var(--radius-sm);
  flex-shrink: 0; overflow: hidden; position: relative;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.cover {
  width: 100%; height: 100%; object-fit: cover;
  transition: transform 0.4s var(--ease-out);
}
.song-card:hover .cover { transform: scale(1.08); }
.cover-ph {
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, rgba(29,185,84,0.12), rgba(100,60,180,0.10));
}

.info { flex: 1; min-width: 0; }
.title {
  font-family: var(--font-display);
  font-weight: 600; font-size: 0.93rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.artist { color: var(--text-sub); font-size: 0.82rem; margin-top: 3px; }
.src-tag {
  display: inline-block; margin-left: 8px; padding: 2px 8px;
  background: var(--accent-dim); color: var(--accent);
  border-radius: var(--radius-pill); font-size: 0.68rem;
  font-weight: 600; letter-spacing: 0.02em;
}
.reason { color: var(--text-muted); font-size: 0.78rem; margin-top: 4px; line-height: 1.4; }

.actions {
  display: flex; gap: 2px; flex-shrink: 0;
  opacity: 0.4;
  transition: opacity var(--dur-norm) var(--ease-out);
}
.song-card:hover .actions { opacity: 1; }

.icon-btn {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-sub);
  transition: all var(--transition);
}
.icon-btn:hover {
  background: var(--bg-elevated); color: var(--text);
  transform: scale(1.1);
}
.icon-btn.play-btn:hover { color: var(--accent); }
.icon-btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
.spin { animation: vinyl-spin 1s linear infinite; }
@keyframes vinyl-spin { to { transform: rotate(360deg); } }
</style>
