<script setup>
import { ref } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const query = ref("");
const loading = ref(false);
const results = ref(null);
const msg = ref("");

async function search() {
  const q = query.value.trim();
  if (!q) return;
  loading.value = true;
  msg.value = "";
  results.value = null;
  try {
    const data = await api.search(store.userId, q);
    results.value = data;
  } catch { msg.value = "搜索失败，请稍后重试。"; }
  finally { loading.value = false; }
}

function toCard(t) {
  return {
    title: t.title, artist: t.artist || "", source: t.source || "local",
    source_id: t.source_id || t.external_id || "", cover_url: t.cover_url,
    playback_url: t.playback_url,
  };
}
</script>

<template>
  <div>
    <div class="section-title">发现</div>
    <div class="section-sub">搜索本地库 + 全网真实候选。</div>

    <div class="search-row">
      <input v-model="query" class="input" placeholder="歌手、歌名、心情、场景…" @keyup.enter="search" />
      <button class="btn" :disabled="loading || !query.trim()" @click="search">搜索</button>
    </div>

    <div v-if="loading" class="loading-hint">搜索中…</div>
    <div v-if="msg" class="empty-hint">{{ msg }}</div>

    <template v-if="results">
      <div v-if="results.summary" class="summary">{{ results.summary }}</div>

      <div v-if="results.external?.length" class="group">
        <div class="group-title">全网候选</div>
        <SongCard v-for="(t, i) in results.external" :key="'e'+i" :card="toCard(t)" @toast="(m) => msg = m" />
      </div>
      <div v-if="results.local?.length" class="group">
        <div class="group-title">本地库</div>
        <SongCard v-for="(t, i) in results.local" :key="'l'+i" :card="toCard(t)" @toast="(m) => msg = m" />
      </div>
      <div v-if="!results.external?.length && !results.local?.length" class="empty-hint">
        没找到可追溯的结果，换个说法试试。
      </div>
    </template>
  </div>
</template>

<style scoped>
.search-row { display: flex; gap: 10px; margin-bottom: 24px; max-width: 640px; }
.summary { background: var(--bg-card); padding: 14px 16px; border-radius: var(--radius); margin-bottom: 18px; line-height: 1.5; }
.group { margin-bottom: 24px; }
.group-title { font-weight: 700; color: var(--text-sub); margin-bottom: 10px; font-size: 0.9rem; }
</style>
