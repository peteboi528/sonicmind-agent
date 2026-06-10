<script setup>
import { ref } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const props = defineProps({
  card: { type: Object, required: true },
  showReason: { type: Boolean, default: true },
});
const emit = defineEmits(["toast", "added"]);

const adding = ref(false);

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
  const url = props.card.playback_url;
  if (!url) { emit("toast", "⚠️ 无可入库链接"); return; }
  adding.value = true;
  try {
    await api.ingest(url);
    emit("toast", `《${props.card.title}》已加入曲库 📥`);
    emit("added", props.card);
  } catch { emit("toast", "⚠️ 入库失败，请稍后重试"); }
  finally { adding.value = false; }
}
</script>

<template>
  <div class="song-card">
    <img v-if="card.cover_url" class="cover" :src="card.cover_url" alt="" loading="lazy" />
    <div v-else class="cover-ph">🎵</div>
    <div class="info">
      <div class="title">{{ card.title || "未知曲目" }}</div>
      <div class="artist">
        {{ card.artist || "未知" }}
        <span v-if="card.source" class="src-tag">{{ card.source }}</span>
      </div>
      <div v-if="showReason && card.reason" class="reason">{{ card.reason }}</div>
    </div>
    <div class="actions">
      <button class="icon-btn" title="听歌" @click="play">🎵</button>
      <button class="icon-btn" title="MV" @click="playMv">📺</button>
      <button v-if="card.source !== 'local'" class="icon-btn" title="加入我的库" :disabled="adding" @click="addToLibrary">{{ adding ? '⏳' : '📥' }}</button>
      <button class="icon-btn" title="不喜欢" @click="dislike">👎</button>
    </div>
  </div>
</template>

<style scoped>
.song-card {
  display: flex; align-items: center; gap: 12px;
  background: var(--bg-card); border-radius: var(--radius);
  padding: 10px 14px; margin-bottom: 8px; transition: var(--transition);
}
.song-card:hover { background: var(--bg-hover); }
.cover, .cover-ph {
  width: 48px; height: 48px; border-radius: var(--radius-sm);
  flex-shrink: 0; object-fit: cover;
}
.cover-ph { display: flex; align-items: center; justify-content: center; background: var(--bg-elevated); font-size: 1.3rem; }
.info { flex: 1; min-width: 0; }
.title { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.artist { color: var(--text-sub); font-size: 0.85rem; margin-top: 2px; }
.src-tag {
  display: inline-block; margin-left: 6px; padding: 1px 7px;
  background: var(--accent-dim); color: var(--accent);
  border-radius: var(--radius-pill); font-size: 0.7rem;
}
.reason { color: var(--text-muted); font-size: 0.8rem; margin-top: 3px; }
.actions { display: flex; gap: 4px; flex-shrink: 0; }
.icon-btn {
  width: 36px; height: 36px; border-radius: 50%;
  font-size: 1rem; transition: var(--transition);
}
.icon-btn:hover { background: var(--bg-elevated); }
.icon-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
