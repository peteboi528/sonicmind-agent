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

    <div v-for="(pl, idx) in playlists" :key="pl.playlist_id" class="pl-card stagger-item" :style="{ animationDelay: `${idx * 60}ms` }">
      <div class="pl-head" @click="expanded = expanded === pl.playlist_id ? null : pl.playlist_id">
        <div class="pl-info">
          <div class="pl-name">{{ pl.name }}</div>
          <div class="pl-desc">{{ pl.description || (pl.tracks?.length + " 首") }}</div>
        </div>
        <div class="pl-actions">
          <span class="count">{{ pl.tracks?.length || 0 }} 首</span>
          <button class="del" @click.stop="remove(pl)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
          </button>
          <span class="chev" :class="{ open: expanded === pl.playlist_id }">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/></svg>
          </span>
        </div>
      </div>
      <Transition name="expand">
        <div v-if="expanded === pl.playlist_id" class="pl-tracks">
          <button v-if="pl.tracks?.length > 1" class="play-all-btn" @click="store.playAll(pl.tracks.map(toCard))">
            ▶ 全部播放（{{ pl.tracks.length }}首）
          </button>
          <SongCard v-for="(t, i) in pl.tracks" :key="i" :card="toCard(t)" :show-reason="false" @toast="(m) => msg = m" />
        </div>
      </Transition>
    </div>
  </div>
</template>

<style scoped>
.gen-row { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
.gen-row .input { flex: 1; min-width: 240px; }
.toast {
  background: var(--accent-dim); color: var(--accent); padding: 10px 14px;
  border-radius: var(--radius-sm); margin-bottom: 16px;
  border: 1px solid rgba(29,185,84,0.12);
}

.pl-card {
  background: var(--bg-card); border-radius: var(--radius);
  margin-bottom: 10px; overflow: hidden;
  border: 1px solid var(--border);
  transition: all var(--dur-norm) var(--ease-out);
}
.pl-card:hover { border-color: var(--border-light); }

.pl-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 18px; cursor: pointer;
  transition: background var(--transition);
}
.pl-head:hover { background: var(--bg-hover); }

.pl-info { min-width: 0; flex: 1; }
.pl-name {
  font-family: var(--font-display);
  font-weight: 700; font-size: 0.95rem;
}
.pl-desc { color: var(--text-sub); font-size: 0.82rem; margin-top: 3px; }
.pl-actions { display: flex; align-items: center; gap: 12px; }
.count {
  color: var(--text-muted); font-size: 0.78rem;
  font-family: var(--font-display); font-weight: 600;
}
.del {
  color: var(--text-muted); padding: 4px;
  border-radius: 50%; transition: all var(--transition);
}
.del:hover { color: var(--danger); background: rgba(231,76,60,0.1); }

.chev {
  color: var(--text-muted);
  transition: transform var(--dur-norm) var(--ease-out);
  display: flex;
}
.chev.open { transform: rotate(180deg); }

.pl-tracks { padding: 4px 14px 14px; }

/* ── Expand Animation ── */
.expand-enter-active {
  animation: expand-in 0.35s var(--ease-out);
}
.expand-leave-active {
  animation: expand-in 0.25s var(--ease-out) reverse;
}
@keyframes expand-in {
  from { opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0; }
  to   { opacity: 1; max-height: 2000px; }
}
</style>
