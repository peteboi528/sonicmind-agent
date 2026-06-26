<script setup>
import { ref, watch, computed, nextTick, onBeforeUnmount } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const audio = ref(null);
const spinning = ref(false);
const toastVisible = ref(false);
const toastText = ref("");
let toastTimer = null;

// ── 歌词 ──
const showLyrics = ref(false);
const lyrics = ref([]);
const lyricsLoading = ref(false);
const lyricsEmpty = ref(false);
const currentTime = ref(0);
const lyricsBody = ref(null);

// ── 自定义播放控件状态（替代浏览器原生 <audio controls>）──
const duration = ref(0);
const seeking = ref(false);
const seekRatio = ref(0);
const seekEl = ref(null);

async function loadLyrics() {
  const p = store.player;
  if (!p.title && !p.sourceId) { lyricsEmpty.value = true; return; }
  lyricsLoading.value = true;
  lyricsEmpty.value = false;
  lyrics.value = [];
  try {
    const data = await api.getLyrics(store.userId, {
      title: p.title, artist: p.artist, sourceId: p.sourceId,
    });
    lyrics.value = data.lines || [];
    lyricsEmpty.value = !lyrics.value.length;
  } catch {
    lyricsEmpty.value = true;
  } finally {
    lyricsLoading.value = false;
  }
}

function toggleLyrics() {
  showLyrics.value = !showLyrics.value;
  if (showLyrics.value && !lyrics.value.length && !lyricsLoading.value) {
    loadLyrics();
  }
}

// ── 字幕同步：当前行高亮 + 平滑居中滚动 + 点行跳转 ──
const hasTimedLyrics = computed(() => lyrics.value.some(l => l.time != null));
const currentLineIndex = computed(() => {
  if (!hasTimedLyrics.value) return null;
  const t = currentTime.value;
  let idx = null;
  for (let i = 0; i < lyrics.value.length; i++) {
    const lt = lyrics.value[i].time;
    if (lt != null && lt <= t) idx = i;
    else if (lt != null && lt > t) break;
  }
  return idx;
});

function seekTo(line) {
  if (line.time == null || !audio.value) return;
  audio.value.currentTime = line.time;
  currentTime.value = line.time;
}

function scrollToCurrent() {
  const body = lyricsBody.value;
  const idx = currentLineIndex.value;
  if (!body || idx == null) return;
  const el = body.querySelector(`[data-idx="${idx}"]`);
  if (el) {
    const top = el.offsetTop - body.clientHeight / 2 + el.clientHeight / 2;
    body.scrollTo({ top, behavior: "smooth" });
  }
}

watch(currentLineIndex, () => { if (showLyrics.value) scrollToCurrent(); });
watch(showLyrics, (open) => { if (open) nextTick(scrollToCurrent); });
watch(lyrics, () => { if (showLyrics.value) nextTick(scrollToCurrent); });

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
  currentTime.value = a.currentTime; // 字幕同步
  const delta = a.currentTime - lastCurrentTime;
  if (delta > 0 && delta < 5) session.elapsed += delta; // 只累加正常播放增量，忽略拖拽大跳
  lastCurrentTime = a.currentTime;
}

// ── 字幕实时同步：用 requestAnimationFrame（~60fps）驱动，不靠 timeupdate（仅 ~4Hz，
// 行高亮会明显滞后于人声）。播放时跑 rAF 把 audio.currentTime 灌进 reactive currentTime，
// currentLineIndex 随之每帧重算、跨行时立即切高亮并居中——真正"实时跟踪"。
// 暂停/结束/关面板/切歌时停 rAF，高亮停在当前行。
let rafId = null;
function startLyricSync() {
  stopLyricSync();
  const tick = () => {
    if (audio.value) currentTime.value = audio.value.currentTime;
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}
function stopLyricSync() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
}

function onPlay() { spinning.value = true; startLyricSync(); }
function onPause() { spinning.value = false; stopLyricSync(); }

// ── 自定义播放控件：时间格式 / 进度百分比 / 播放暂停 / 拖拽 seek ──
function fmt(t) {
  if (!t || !isFinite(t)) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
const seekPct = computed(() => {
  if (seeking.value) return Math.min(100, Math.max(0, seekRatio.value * 100));
  if (!duration.value) return 0;
  return Math.min(100, Math.max(0, (currentTime.value / duration.value) * 100));
});
function togglePlay() {
  const a = audio.value;
  if (!a) return;
  if (a.paused) a.play().catch(() => {});
  else a.pause();
}
function onLoaded() {
  if (audio.value && isFinite(audio.value.duration)) duration.value = audio.value.duration;
}
function ratioFromX(clientX) {
  const el = seekEl.value;
  if (!el) return 0;
  const rect = el.getBoundingClientRect();
  return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
}
function onSeekDown(e) {
  if (!duration.value) return;
  seeking.value = true;
  seekRatio.value = ratioFromX(e.clientX);
  window.addEventListener("pointermove", onSeekMove);
  window.addEventListener("pointerup", onSeekUp);
}
function onSeekMove(e) {
  if (seeking.value) seekRatio.value = ratioFromX(e.clientX);
}
function onSeekUp() {
  if (seeking.value) {
    if (audio.value && duration.value) {
      audio.value.currentTime = seekRatio.value * duration.value;
      currentTime.value = audio.value.currentTime;
    }
    seeking.value = false;
  }
  window.removeEventListener("pointermove", onSeekMove);
  window.removeEventListener("pointerup", onSeekUp);
}

onBeforeUnmount(() => {
  stopLyricSync();
  window.removeEventListener("pointermove", onSeekMove);
  window.removeEventListener("pointerup", onSeekUp);
});

// URL 变化 = 换曲：先结算上一首(没听完被换走=秒跳)，再开新 session 并自动播放
watch(() => store.player.url, (url) => {
  flushSession(false);
  // 换曲：重置歌词；面板开着就自动重取新曲歌词
  lyrics.value = [];
  lyricsEmpty.value = false;
  currentTime.value = 0;
  duration.value = 0;       // 换曲清时长，避免 metadata 到手前显示上一首的 3:45
  seeking.value = false;
  if (showLyrics.value && url) loadLyrics();
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
  stopLyricSync();
  store.nextTrack();
}

function close() {
  flushSession(false);
  if (audio.value) { audio.value.pause(); audio.value.src = ""; }
  spinning.value = false;
  stopLyricSync();
  showLyrics.value = false;
  lyrics.value = [];
  currentTime.value = 0;
  store.closePlayer();
}
</script>

<template>
  <!-- 歌词面板：独立浮层，从播放器上方滑出 -->
  <Transition name="lyrics-slide">
    <div v-if="showLyrics" class="lyrics-panel">
      <div v-if="store.player.cover" class="lyrics-bg" :style="{ backgroundImage: `url(${store.player.cover})` }"></div>
      <div class="lyrics-head">
        <div class="lyrics-meta">
          <div class="lyrics-title">{{ store.player.title || "未知" }}</div>
          <div class="lyrics-artist">{{ store.player.artist }}</div>
        </div>
        <div class="lyrics-actions">
          <button class="lyrics-pp" @click="togglePlay" :title="spinning ? '暂停' : '播放'">
            <svg v-if="spinning" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>
            <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          </button>
          <button class="lyrics-x" @click="toggleLyrics" title="收起">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
      </div>
      <div ref="lyricsBody" class="lyrics-body">
        <div v-if="lyricsLoading" class="lyrics-hint"><span class="lyrics-spin"></span>加载歌词…</div>
        <div v-else-if="lyricsEmpty" class="lyrics-hint">暂无歌词</div>
        <p v-else v-for="(line, i) in lyrics" :key="i" class="lyrics-line"
           :data-idx="i"
           :class="{ active: i === currentLineIndex, dim: hasTimedLyrics && i !== currentLineIndex }"
           @click="hasTimedLyrics && seekTo(line)">{{ line.text }}</p>
      </div>
    </div>
  </Transition>

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
      <div class="partist">
        <span class="pa-name">{{ store.player.artist }}</span>
        <template v-if="hasQueue"><span class="pa-dot">·</span><span class="pa-qp">{{ queueProgress }}</span></template>
      </div>
    </div>

    <!-- 传输区：时间 + 自定义进度条 + 上一首/播放/下一首（替代原生 audio 控件） -->
    <div class="transport">
      <div class="t-progress">
        <span class="ttime">{{ fmt(currentTime) }}</span>
        <div ref="seekEl" class="seek" @pointerdown="onSeekDown">
          <div class="seek-track"></div>
          <div class="seek-fill" :style="{ width: seekPct + '%' }"></div>
          <div class="seek-thumb" :style="{ left: seekPct + '%' }"></div>
        </div>
        <span class="ttime">{{ fmt(duration) }}</span>
      </div>
      <div class="t-controls">
        <button v-if="hasQueue" class="ctrl-btn" :disabled="store.queueIndex <= 0" @click="store.prevTrack()" title="上一首">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zm3.5 6 8.5 6V6z"/></svg>
        </button>
        <button class="play-btn" @click="togglePlay" :title="spinning ? '暂停' : '播放'">
          <svg v-if="spinning" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>
          <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        </button>
        <button v-if="hasQueue" class="ctrl-btn" :disabled="store.queueIndex >= store.queue.length - 1" @click="store.nextTrack()" title="下一首">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/></svg>
        </button>
      </div>
    </div>

    <button class="lyrics-btn" :class="{ active: showLyrics }" @click="toggleLyrics" title="歌词">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h10"/></svg>
    </button>
    <button class="close" @click="close" title="关闭">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>

    <!-- 隐藏的原生 audio，仅作播放引擎，UI 由上方自定义控件接管 -->
    <audio ref="audio" class="audio-hidden"
      @play="onPlay" @pause="onPause" @ended="onEnded"
      @timeupdate="onTimeUpdate" @loadedmetadata="onLoaded" @durationchange="onLoaded"
    ></audio>
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

.meta { min-width: 130px; max-width: 220px; }
.ptitle {
  font-family: var(--font-display);
  font-weight: 700; font-size: 0.92rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.partist {
  color: var(--text-sub); font-size: 0.8rem; margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pa-dot { margin: 0 5px; opacity: 0.5; }
.pa-qp { color: var(--text-muted); font-variant-numeric: tabular-nums; }

/* ── 传输区：自定义进度条 + 控件（替代原生 <audio controls>）── */
.transport {
  flex: 1; min-width: 0;
  display: flex; align-items: center; justify-content: center; gap: 18px;
}
.t-progress {
  flex: 1; max-width: 380px; min-width: 140px;
  display: flex; align-items: center; gap: 10px;
}
.ttime {
  font-family: var(--font-display); font-size: 0.72rem; font-weight: 600;
  color: var(--text-muted); font-variant-numeric: tabular-nums;
  min-width: 34px; text-align: center;
}
.seek {
  flex: 1; position: relative; height: 16px; cursor: pointer;
  touch-action: none; /* 拖拽 seek 时别触发页面滚动 */
}
.seek-track {
  position: absolute; top: 50%; left: 0; right: 0; height: 4px;
  transform: translateY(-50%); border-radius: 2px;
  background: rgba(255,255,255,0.12);
  transition: height 0.15s var(--ease-out);
}
.seek-fill {
  position: absolute; top: 50%; left: 0; height: 4px;
  transform: translateY(-50%); border-radius: 2px;
  background: var(--accent);
  transition: height 0.15s var(--ease-out); pointer-events: none;
}
.seek-thumb {
  position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 50%;
  background: #fff; transform: translate(-50%, -50%); opacity: 0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.5);
  transition: opacity 0.15s var(--ease-out); pointer-events: none;
}
.seek:hover .seek-track, .seek:active .seek-track,
.seek:hover .seek-fill, .seek:active .seek-fill { height: 6px; }
.seek:hover .seek-thumb, .seek:active .seek-thumb { opacity: 1; }

.t-controls { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
.ctrl-btn {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-sub);
  transition: all var(--transition);
}
.ctrl-btn:hover:not(:disabled) { background: var(--bg-hover); color: var(--accent); }
.ctrl-btn:disabled { opacity: 0.25; cursor: not-allowed; }
.play-btn {
  width: 38px; height: 38px; border-radius: 50%;
  background: var(--text); color: #000;
  display: flex; align-items: center; justify-content: center;
  transition: transform var(--transition), background var(--transition);
}
.play-btn:hover { transform: scale(1.06); background: #fff; }
.play-btn:active { transform: scale(0.94); }

.audio-hidden {
  position: absolute; width: 1px; height: 1px; opacity: 0;
  overflow: hidden; pointer-events: none;
} /* 仅作播放引擎；UI 由自定义控件接管。offscreen 而非 display:none，播放/事件最稳 */

.close {
  width: 34px; height: 34px; border-radius: 50%;
  color: var(--text-muted);
  display: flex; align-items: center; justify-content: center;
  transition: all var(--transition);
}
.close:hover { background: var(--bg-hover); color: var(--text); }

/* ── 歌词 ── */
.lyrics-btn {
  width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-sub); transition: all var(--transition);
}
.lyrics-btn:hover { background: var(--bg-hover); color: var(--text); }
.lyrics-btn.active { color: var(--accent); background: var(--accent-dim); }

.lyrics-panel {
  position: fixed; left: 50%; bottom: var(--player-h);
  transform: translateX(-50%);
  width: min(720px, 94vw); height: min(720px, 78vh);
  display: flex; flex-direction: column;
  background: var(--bg-glass);
  backdrop-filter: blur(32px) saturate(1.3);
  -webkit-backdrop-filter: blur(32px) saturate(1.3);
  border: 1px solid var(--border-light); border-bottom: none;
  border-radius: var(--radius) var(--radius) 0 0;
  box-shadow: var(--shadow-lg); overflow: hidden; z-index: 199;
}
.lyrics-bg {
  position: absolute; inset: 0; z-index: 0;
  background-size: cover; background-position: center;
  filter: blur(50px) saturate(1.5) brightness(0.3);
  opacity: 0.5; pointer-events: none;
}
.lyrics-head {
  position: relative; z-index: 1;
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px 12px; gap: 12px; border-bottom: 1px solid var(--border);
}
.lyrics-title {
  font-family: var(--font-display); font-weight: 700; font-size: 1rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 460px;
}
.lyrics-artist { color: var(--text-sub); font-size: 0.82rem; margin-top: 2px; }
.lyrics-x {
  width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-muted); transition: all var(--transition);
}
.lyrics-x:hover { background: var(--bg-hover); color: var(--text); }
.lyrics-actions { display: flex; align-items: center; gap: 4px; flex-shrink: 0; }
.lyrics-pp {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: rgba(255,255,255,0.06); color: var(--text);
  transition: all var(--transition);
}
.lyrics-pp:hover { background: var(--accent); color: #000; }
.lyrics-body {
  position: relative; z-index: 1; flex: 1; overflow-y: auto;
  padding: 32px 28px 48px;
  display: flex; flex-direction: column; align-items: center;
  /* 上下渐隐，字幕感：mask 不影响布局，offsetTop/offsetHeight 不变，滚动定位不受影响 */
  -webkit-mask-image: linear-gradient(to bottom, transparent, #000 14%, #000 86%, transparent);
  mask-image: linear-gradient(to bottom, transparent, #000 14%, #000 86%, transparent);
}
.lyrics-line {
  margin: 0; text-align: center;
  font-size: 1.02rem; line-height: 2.15; color: var(--text);
  cursor: pointer; padding: 2px 0;
  transition: all 0.35s var(--ease-out);
}
/* 当前行高亮放大，非当前行淡出 —— 焦点突出，字幕感 */
.lyrics-line.active {
  font-size: 1.34rem; font-weight: 700;
  color: var(--accent);
  transform: scale(1.03);
  text-shadow: 0 0 18px var(--accent-glow);
}
.lyrics-line.dim {
  opacity: 0.32;
}
.lyrics-hint {
  margin: auto; color: var(--text-muted); font-size: 0.88rem;
  display: flex; align-items: center; gap: 8px;
}
.lyrics-spin {
  width: 14px; height: 14px; border-radius: 50%;
  border: 2px solid var(--border-light); border-top-color: var(--accent);
  animation: lyrics-spin 0.8s linear infinite;
}
@keyframes lyrics-spin { to { transform: rotate(360deg); } }

.lyrics-slide-enter-active { transition: transform 0.4s var(--ease-out), opacity 0.3s; }
.lyrics-slide-leave-active { transition: transform 0.25s ease, opacity 0.2s; }
.lyrics-slide-enter-from, .lyrics-slide-leave-to {
  opacity: 0; transform: translate(-50%, 40px);
}

@media (max-width: 768px) {
  .player-bg { display: none; }
  .meta { min-width: 0; max-width: 120px; }
  .ptitle { font-size: 0.82rem; }
  .partist { font-size: 0.72rem; }
  .pa-dot, .pa-qp { display: none; }
  .transport { gap: 10px; }
  .ttime { display: none; }          /* 小屏藏时间，给进度条更多空间 */
  .t-progress { min-width: 70px; }
  .t-controls .ctrl-btn { display: none; }  /* 窄屏只留播放键，上下首在桌面端 */
  .disc-wrap { width: 40px; height: 40px; }
  .disc img { inset: 7px; }
  .play-btn { width: 34px; height: 34px; }
  .player-bar { gap: 10px; padding: 0 12px; }
  .lyrics-panel { width: 100vw; height: 80vh; border-radius: 0; }
  .lyrics-line { font-size: 0.94rem; }
  .lyrics-line.active { font-size: 1.14rem; }
}
</style>
