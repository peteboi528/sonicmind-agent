<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const loading = ref(false);
const rec = ref(null);
const msg = ref("");
const timeOfDay = ref("");
const hasRequested = ref(false);

const SLOTS = [
  { v: "", label: "自动" },
  { v: "morning", label: "清晨" },
  { v: "focus", label: "专注" },
  { v: "afternoon", label: "午后" },
  { v: "evening", label: "傍晚" },
  { v: "night", label: "深夜" },
];

// 当日缓存 key：每个 userId 每天独立一份
function cacheKey() {
  const today = new Date().toISOString().slice(0, 10); // "2026-07-01"
  return `daily_rec_${store.userId}_${today}`;
}

function saveCache(data) {
  try { localStorage.setItem(cacheKey(), JSON.stringify(data)); } catch {}
}

function loadCache() {
  try {
    const raw = localStorage.getItem(cacheKey());
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

async function load(forceRefresh = false) {
  if (!forceRefresh) {
    const cached = loadCache();
    if (cached) {
      rec.value = cached;
      hasRequested.value = true;
      return;
    }
  }
  loading.value = true;
  msg.value = "";
  hasRequested.value = true;
  try {
    const result = await api.dailyRecommend(store.userId, timeOfDay.value || undefined);
    rec.value = result;
    saveCache(result);
  } catch { msg.value = "生成推荐失败，请稍后重试。"; }
  finally { loading.value = false; }
}

// 页面挂载时自动恢复当天缓存（不发请求）
onMounted(() => load(false));

function toCard(item) {
  const t = item.asset;
  return {
    title: t.title, artist: t.artist || "", source: t.source || "local",
    source_id: t.source_id || t.external_id || "", cover_url: t.cover_url,
    reason: item.reason, components: item.components,
  };
}
</script>

<template>
  <div>
    <div class="section-title">今日推荐</div>
    <div class="section-sub">结合你的口味、行为与记忆生成的可追溯候选。</div>

    <!-- 初始空状态（今天还没有缓存且未加载） -->
    <div v-if="!hasRequested && !loading" class="empty-state">
      <div class="empty-rings">
        <div class="ring r1"></div>
        <div class="ring r2"></div>
        <div class="ring r3"></div>
        <div class="ring-icon">🎵</div>
      </div>
      <div class="empty-text">点击下方按钮，为你生成今日推荐</div>
      <button class="btn-recommend" @click="load(true)">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/></svg>
        获取今日推荐
      </button>
    </div>

    <template v-if="hasRequested">
    <div class="slot-row">
      <button
        v-for="(s, i) in SLOTS" :key="s.v"
        class="slot" :class="{ active: timeOfDay === s.v }"
        :style="{ animationDelay: `${i * 40}ms` }"
        @click="timeOfDay = s.v; load(true)"
      >{{ s.label }}</button>
      <!-- 手动刷新按钮：强制重新生成，不用缓存 -->
      <button class="btn-ghost refresh-btn" :disabled="loading" @click="load(true)" title="重新生成今日推荐">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
      </button>
    </div>

    <div v-if="loading" class="loading-hint">
      <div class="loading-dots">
        <span></span><span></span><span></span>
      </div>
      正在为你挑选…
    </div>
    <div v-if="msg" class="empty-hint">{{ msg }}</div>

    <template v-if="rec && !loading">
      <div v-if="rec.reason_summary" class="summary">{{ rec.reason_summary }}</div>
      <div v-if="rec.tracks?.length > 1" class="play-all-wrap">
        <button class="play-all-btn" @click="store.playAll(rec.tracks.map(toCard))">
          ▶ 全部播放（{{ rec.tracks.length }}首）
        </button>
      </div>
      <div v-for="(item, i) in rec.tracks" :key="i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
        <SongCard :card="toCard(item)" @toast="(m) => msg = m" />
      </div>
      <div v-if="!rec.tracks?.length" class="empty-hint">还没有足够数据，先去发现页听几首吧。</div>
    </template>
    </template>
  </div>
</template>

<style scoped>
.slot-row {
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; align-items: center;
}
.slot {
  padding: 8px 18px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  color: var(--text-sub); font-family: var(--font-display);
  font-size: 0.84rem; font-weight: 600;
  transition: all var(--dur-norm) var(--ease-out);
  animation: fadeInUp 0.4s var(--ease-out) both;
}
.slot:hover {
  color: var(--text); border-color: var(--border-light);
  transform: translateY(-1px);
}
.slot.active {
  background: var(--accent); color: #000;
  border-color: var(--accent);
  box-shadow: 0 2px 12px var(--accent-glow);
}
.refresh-btn {
  width: 36px; height: 36px; padding: 0;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
}
.summary {
  background: var(--bg-card); padding: 16px 18px;
  border-radius: var(--radius); margin-bottom: 20px;
  line-height: 1.6; border: 1px solid var(--border);
}
.play-all-wrap { margin-bottom: 4px; }

/* ── Empty State ── */
.empty-state {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 80px 20px; gap: 20px;
  animation: fadeInUp 0.6s var(--ease-out);
}
.empty-rings {
  position: relative; width: 100px; height: 100px;
}
.ring {
  position: absolute; border-radius: 50%;
  border: 1px solid var(--accent);
  animation: ring-pulse 3s ease-in-out infinite;
}
.r1 { inset: 0; opacity: 0.15; animation-delay: 0s; }
.r2 { inset: 15px; opacity: 0.25; animation-delay: 0.5s; }
.r3 { inset: 30px; opacity: 0.35; animation-delay: 1s; }
.ring-icon {
  position: absolute; inset: 38px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem;
  animation: float 4s ease-in-out infinite;
}
@keyframes ring-pulse {
  0%, 100% { transform: scale(1); opacity: 0.15; }
  50% { transform: scale(1.08); opacity: 0.3; }
}
.empty-text { color: var(--text-sub); font-size: 0.95rem; }
.btn-recommend {
  padding: 13px 32px;
  background: var(--accent); color: #000;
  border: none; border-radius: var(--radius-pill);
  font-family: var(--font-display);
  font-size: 1rem; font-weight: 700;
  display: flex; align-items: center; gap: 8px;
  transition: all var(--dur-norm) var(--ease-out);
}
.btn-recommend:hover {
  filter: brightness(1.1); transform: translateY(-2px);
  box-shadow: 0 4px 20px var(--accent-glow);
}

/* ── Loading ── */
.loading-hint {
  display: flex; flex-direction: column; align-items: center;
  gap: 14px; color: var(--text-sub); padding: 48px 0;
}
.loading-dots { display: flex; gap: 6px; }
.loading-dots span {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent);
  animation: dot-bounce 1.4s ease-in-out infinite;
}
.loading-dots span:nth-child(2) { animation-delay: 0.16s; }
.loading-dots span:nth-child(3) { animation-delay: 0.32s; }
</style>
