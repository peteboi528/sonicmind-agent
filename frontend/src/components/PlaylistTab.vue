<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const playlists = ref([]);
const loading = ref(false);
const instruction = ref("");
const generating = ref(false);
const msg = ref("");
const expanded = ref(null);

async function load() {
  loading.value = true;
  try {
    const data = await api.listPlaylists(store.userId);
    playlists.value = data.playlists || [];
  } catch { msg.value = "加载歌单失败。"; }
  finally { loading.value = false; }
}

async function generate() {
  const ins = instruction.value.trim();
  if (!ins) return;
  generating.value = true;
  msg.value = "";
  try {
    await api.generatePlaylist(store.userId, ins);
    instruction.value = "";
    await load();
  } catch { msg.value = "生成失败，请稍后重试。"; }
  finally { generating.value = false; }
}

async function autoClassify() {
  generating.value = true;
  msg.value = "";
  try {
    await api.autoPlaylists(store.userId);
    await load();
  } catch { msg.value = "自动分类失败。"; }
  finally { generating.value = false; }
}

async function remove(pl) {
  if (!confirm(`删除歌单《${pl.name}》？`)) return;
  try {
    await api.deletePlaylist(store.userId, pl.playlist_id);
    playlists.value = playlists.value.filter((p) => p.playlist_id !== pl.playlist_id);
  } catch { msg.value = "删除失败。"; }
}

function toCard(t) {
  return {
    title: t.title, artist: t.artist || "", source: t.source || "local",
    source_id: t.source_id || t.external_id || "", cover_url: t.cover_url,
  };
}

onMounted(load);
</script>

<template>
  <div>
    <div class="section-title">我的歌单</div>
    <div class="section-sub">用一句话生成歌单，或让我按口味自动分类。</div>

    <div class="gen-row">
      <input v-model="instruction" class="input" placeholder="如：做 20 首适合雨天的 city pop" @keyup.enter="generate" />
      <button class="btn" :disabled="generating || !instruction.trim()" @click="generate">生成</button>
      <button class="btn-ghost" :disabled="generating" @click="autoClassify">自动分类</button>
    </div>

    <div v-if="msg" class="toast">{{ msg }}</div>
    <div v-if="loading || generating" class="loading-hint">{{ generating ? "生成中…" : "加载中…" }}</div>

    <div v-if="!loading && !playlists.length" class="empty-hint">还没有歌单，生成一个试试。</div>

    <div v-for="pl in playlists" :key="pl.playlist_id" class="pl-card">
      <div class="pl-head" @click="expanded = expanded === pl.playlist_id ? null : pl.playlist_id">
        <div>
          <div class="pl-name">{{ pl.name }}</div>
          <div class="pl-desc">{{ pl.description || (pl.tracks?.length + " 首") }}</div>
        </div>
        <div class="pl-actions">
          <span class="count">{{ pl.tracks?.length || 0 }} 首</span>
          <button class="del" @click.stop="remove(pl)">🗑</button>
          <span class="chev">{{ expanded === pl.playlist_id ? "▲" : "▼" }}</span>
        </div>
      </div>
      <div v-if="expanded === pl.playlist_id" class="pl-tracks">
        <SongCard v-for="(t, i) in pl.tracks" :key="i" :card="toCard(t)" :show-reason="false" @toast="(m) => msg = m" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.gen-row { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
.gen-row .input { flex: 1; min-width: 240px; }
.toast { background: var(--accent-dim); color: var(--accent); padding: 10px 14px; border-radius: var(--radius-sm); margin-bottom: 14px; }
.pl-card { background: var(--bg-card); border-radius: var(--radius); margin-bottom: 10px; overflow: hidden; }
.pl-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px; cursor: pointer; }
.pl-head:hover { background: var(--bg-hover); }
.pl-name { font-weight: 700; }
.pl-desc { color: var(--text-sub); font-size: 0.83rem; margin-top: 2px; }
.pl-actions { display: flex; align-items: center; gap: 12px; }
.count { color: var(--text-muted); font-size: 0.8rem; }
.del { color: var(--text-sub); }
.del:hover { color: var(--danger); }
.chev { color: var(--text-muted); font-size: 0.7rem; }
.pl-tracks { padding: 4px 12px 12px; }
</style>
