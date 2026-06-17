<script setup>
import { ref, watch, computed } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const audio = ref(null);
const spinning = ref(false);
const toastVisible = ref(false);
const toastText = ref("");
let toastTimer = null;

const hasQueue = computed(() => store.queue.length > 1);
const queueProgress = computed(() =>
  store.queueIndex >= 0 ? `${store.queueIndex + 1} / ${store.queue.length}` : "",
);

// ── 收听行为采集（喂行为锚：听完=completed，秒跳=skip）──
// 每首歌一个 session：累计听了多少秒、是否已上报。切歌/播完时上报 /listen。
// key 用 asset_id(本地) 或 source_id(在线)，与后端 rerank 的 _track_id 同命名空间，
// 行为锚才能把"听完/秒跳"映射回候选并改变排序。
const session = { key: "", elapsed: 0, reported: false };
let lastCurrentTime = 0;

function playerKey() {
  return store.player.assetId || store.player.sourceId || "";
}

function flushSession(completed) {
  if (!session.key || session.reported) return;
  // 秒跳类：只在真正听过(≥1s)才报，避免 VIP/取流失败被误记成秒跳（伪负反馈）
  if (!completed && session.elapsed < 1) {
    session.reported = true;
    return;
  }
  session.reported = true;
  api
    .listen(store.userId, session.key, Math.round(session.elapsed), completed)
    .catch(() => {});
}

function onTimeUpdate() {
  const a = audio.value;
  if (!a) return;
  const delta = a.currentTime - lastCurrentTime;
  if (delta > 0 && delta < 5) session.elapsed += delta; // 只累加正常播放增量，忽略拖拽大跳
  lastCurrentTime = a.currentTime;
}

// URL 变化 = 换曲：先结算上一首(没听完被换走=秒跳)，再开新 session 并自动播放
watch(() => store.player.url, (url) => {
  flushSession(false);
  if (url && audio.value) {
    session.key = playerKey();
    session.elapsed = 0;
    session.reported = false;
    lastCurrentTime = 0;
    audio.value.src = url;
    audio.value.play().catch(() => {});
  } else {
    session.key = "";
  }
});

// 监听队列 toast
watch(() => store.toastKey, () => {
  if (store.toastMsg) {
    toastText.value = store.toastMsg;
    toastVisible.value = true;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastVisible.value = false; }, 3000);
  }
});

function onEnded() {
  const a = audio.value;
  if (a) session.elapsed = Math.max(session.elapsed, a.duration || a.currentTime || 0);
  flushSession(true); // 自然播完 = completed
  spinning.value = false;
  store.nextTrack();
}

function close() {
  flushSession(false);
  if (audio.value) { audio.value.pause(); audio.value.src = ""; }
  spinning.value = false;
  store.closePlayer();
}
</script>

<template>
  <div class="player-bar" :class="{ visible: store.player.visible }">
    <!-- 队列 toast -->
    <Transition name="toast">
      <div v-if="toastVisible" class="player-toast">{{ toastText }}</div>
    </Transition>

    <!-- 背景模糊光晕 -->
    <div v-if="store.player.cover" class="player-bg" :style="{ backgroundImage: `url(${store.player.cover})` }"></div>

    <div class="disc-wrap" :class="{ spinning }">
      <div class="disc">
        <div class="disc-ring"></div>
        <div class="disc-hole"></div>
        <img v-if="store.player.cover" :src="store.player.cover" alt="" />
      </div>
    </div>

    <div class="meta">
      <div class="ptitle">{{ store.player.title || "未知" }}</div>
      <div class="partist">{{ store.player.artist }}</div>
      <div v-if="hasQueue" class="queue-info">{{ queueProgress }}</div>
    </div>

    <!-- 队列控制 -->
    <div v-if="hasQueue" class="queue-controls">
      <button class="ctrl-btn" :disabled="store.queueIndex <= 0" @click="store.prevTrack()" title="上一首">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zm3.5 6 8.5 6V6z"/></svg>
      </button>
      <button class="ctrl-btn" :disabled="store.queueIndex >= store.queue.length - 1" @click="store.nextTrack()" title="下一首">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/></svg>
      </button>
    </div>

    <div class="audio-wrap">
      <audio
        ref="audio" controls
        @play="spinning = true" @pause="spinning = false"
        @timeupdate="onTimeUpdate" @ended="onEnded"
      ></audio>
    </div>
    <button class="close" @click="close">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>
</template>

<style scoped>
.player-bar {
  position: fixed; bottom: 0; left: 0; right: 0; height: var(--player-h);
  display: flex; align-items: center; gap: 18px; padding: 0 24px;
  background: var(--bg-glass);
  backdrop-filter: blur(32px) saturate(1.3);
  -webkit-backdrop-filter: blur(32px) saturate(1.3);
  border-top: 1px solid var(--border);
  transform: translateY(100%);
  transition: transform 0.45s var(--ease-out);
  z-index: 200;
  overflow: hidden;
}
.player-bar.visible { transform: translateY(0); }

/* 背景封面模糊光晕 */
.player-bg {
  position: absolute; inset: -40px; z-index: 0;
  background-size: cover; background-position: center;
  filter: blur(60px) saturate(1.5) brightness(0.25);
  opacity: 0.6;
  pointer-events: none;
}
.player-bar > *:not(.player-bg) { position: relative; z-index: 1; }

/* Toast 浮层 */
.player-toast {
  position: absolute; top: -42px; left: 50%; transform: translateX(-50%);
  padding: 8px 18px; background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-pill); font-size: 0.82rem; white-space: nowrap;
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
}
.toast-enter-active, .toast-leave-active { transition: all 0.3s var(--ease-out); }
.toast-enter-from, .toast-leave-to { opacity: 0; transform: translateX(-50%) translateY(8px); }

/* Vinyl Disc */
.disc-wrap {
  width: 52px; height: 52px; flex-shrink: 0;
}
.disc-wrap.spinning .disc {
  animation: vinyl-spin 3s linear infinite;
}
.disc {
  width: 100%; height: 100%; border-radius: 50%;
  background: conic-gradient(from 0deg, #111 0%, #1a1a1a 25%, #0d0d0d 50%, #181818 75%, #111 100%);
  position: relative; overflow: hidden;
  box-shadow: 0 0 0 2px rgba(255,255,255,0.05), inset 0 0 12px rgba(0,0,0,0.6);
}
.disc img {
  position: absolute; inset: 10px; border-radius: 50%;
  object-fit: cover; z-index: 2;
  box-shadow: 0 0 8px rgba(0,0,0,0.4);
}
.disc-ring {
  position: absolute; inset: 0; border-radius: 50%; z-index: 1;
  background: repeating-radial-gradient(circle at center,
    transparent 0px, transparent 3px,
    rgba(255,255,255,0.02) 3px, rgba(255,255,255,0.02) 4px
  );
}
.disc-hole {
  position: absolute; top: 50%; left: 50%; width: 6px; height: 6px;
  transform: translate(-50%, -50%);
  border-radius: 50%; background: #08080c; z-index: 3;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.1);
}
@keyframes vinyl-spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

.meta { min-width: 140px; }
.ptitle {
  font-family: var(--font-display);
  font-weight: 700; font-size: 0.92rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 200px;
}
.partist { color: var(--text-sub); font-size: 0.8rem; margin-top: 2px; }
.queue-info { color: var(--text-muted); font-size: 0.72rem; margin-top: 3px; letter-spacing: 0.04em; }

.queue-controls { display: flex; gap: 4px; }
.ctrl-btn {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-sub);
  transition: all var(--transition);
}
.ctrl-btn:hover:not(:disabled) {
  background: var(--bg-hover); color: var(--accent);
}
.ctrl-btn:disabled { opacity: 0.25; cursor: not-allowed; }

.audio-wrap { flex: 1; max-width: 540px; }
.audio-wrap audio {
  width: 100%; height: 40px;
  border-radius: var(--radius-sm);
}

.close {
  width: 34px; height: 34px; border-radius: 50%;
  color: var(--text-muted);
  display: flex; align-items: center; justify-content: center;
  transition: all var(--transition);
}
.close:hover { background: var(--bg-hover); color: var(--text); }

@media (max-width: 768px) {
  .audio-wrap { max-width: none; }
  .player-bg { display: none; }
  .meta { min-width: 0; flex: 1; }
  .ptitle { max-width: 120px; font-size: 0.82rem; }
  .partist, .queue-info { display: none; }
  .queue-controls { display: flex; }
  .disc-wrap { width: 40px; height: 40px; }
  .disc img { inset: 7px; }
  .player-bar { gap: 12px; padding: 0 16px; }
}
</style>
