<script setup>
import { ref } from "vue";
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

async function load() {
  loading.value = true;
  msg.value = "";
  hasRequested.value = true;
  try {
    rec.value = await api.dailyRecommend(store.userId, timeOfDay.value || undefined);
  } catch { msg.value = "生成推荐失败，请稍后重试。"; }
  finally { loading.value = false; }
}

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

    <!-- 初始空状态：按钮触发推荐 -->
    <div v-if="!hasRequested && !loading" class="empty-state">
      <div class="empty-icon">🎵</div>
      <div class="empty-text">点击下方按钮，为你生成今日推荐</div>
      <button class="btn-recommend" @click="load">获取今日推荐</button>
    </div>

    <template v-if="hasRequested">
    <div class="slot-row">
      <button
        v-for="s in SLOTS" :key="s.v"
        class="slot" :class="{ active: timeOfDay === s.v }"
        @click="timeOfDay = s.v; load()"
      >{{ s.label }}</button>
      <button class="btn-ghost" :disabled="loading" @click="load">刷新</button>
    </div>

    <div v-if="loading" class="loading-hint">正在为你挑选…</div>
    <div v-if="msg" class="empty-hint">{{ msg }}</div>

    <template v-if="rec && !loading">
      <div v-if="rec.reason_summary" class="summary">{{ rec.reason_summary }}</div>
      <SongCard v-for="(item, i) in rec.tracks" :key="i" :card="toCard(item)" @toast="(m) => msg = m" />
      <div v-if="!rec.tracks?.length" class="empty-hint">还没有足够数据，先去发现页听几首吧。</div>
    </template>
    </template>
  </div>
</template>

<style scoped>
.slot-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 22px; align-items: center; }
.slot { padding: 7px 16px; background: var(--bg-card); border-radius: var(--radius-pill); color: var(--text-sub); font-size: 0.85rem; transition: var(--transition); }
.slot:hover { color: var(--text); }
.slot.active { background: var(--accent); color: #000; font-weight: 700; }
.summary { background: var(--bg-card); padding: 14px 16px; border-radius: var(--radius); margin-bottom: 18px; line-height: 1.5; }
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 20px; gap: 16px; }
.empty-icon { font-size: 3rem; opacity: 0.7; }
.empty-text { color: var(--text-sub); font-size: 0.95rem; }
.btn-recommend { padding: 12px 32px; background: var(--accent); color: #000; border: none; border-radius: var(--radius-pill); font-size: 1rem; font-weight: 700; cursor: pointer; transition: var(--transition); margin-top: 8px; }
.btn-recommend:hover { filter: brightness(1.1); transform: translateY(-1px); }
</style>
