<script setup>
import { ref, onMounted, watch } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const playlists = ref([]);
const loading = ref(false);
const instruction = ref("");
const generating = ref(false);
const msg = ref("");
const expanded = ref(null);
const draft = ref(null);
const picked = ref([]);
const draftName = ref("");
const saving = ref(false);
const revising = ref(false);
const adjustment = ref("");

// ── 多选删除 ──
const selectMode = ref(false);
const selectedIds = ref(new Set());
const bulkDeleting = ref(false);

function toggleSelectMode() {
  selectMode.value = !selectMode.value;
  if (!selectMode.value) selectedIds.value = new Set();
}

function toggleSelect(pid) {
  const s = new Set(selectedIds.value);
  s.has(pid) ? s.delete(pid) : s.add(pid);
  selectedIds.value = s;
}

function selectAll() {
  if (selectedIds.value.size === playlists.value.length) {
    selectedIds.value = new Set();
  } else {
    selectedIds.value = new Set(playlists.value.map(p => p.playlist_id));
  }
}

async function deleteSelected() {
  const ids = [...selectedIds.value];
  if (!ids.length || bulkDeleting.value) return;
  if (!confirm(`确定删除选中的 ${ids.length} 个歌单？`)) return;
  bulkDeleting.value = true;
  let ok = 0;
  for (const pid of ids) {
    try {
      await api.deletePlaylist(store.userId, pid);
      ok++;
    } catch { /* 单个失败不阻断其余 */ }
  }
  selectedIds.value = new Set();
  selectMode.value = false;
  bulkDeleting.value = false;
  msg.value = ok === ids.length ? `已删除 ${ok} 个歌单。` : `删除 ${ok}/${ids.length} 个，部分失败。`;
  await load();  // 与后端对齐：删成功的消失，失败的仍在
}

async function load() {
  const owner = store.userId;
  loading.value = true;
  try {
    const data = await api.listPlaylists(store.userId);
    if (owner === store.userId) playlists.value = data.playlists || [];
  } catch { msg.value = "加载歌单失败。"; }
  finally { loading.value = false; }
}

async function generate() {
  const ins = instruction.value.trim();
  if (!ins || generating.value || saving.value || revising.value) return;
  const owner = store.userId;
  generating.value = true;
  msg.value = "";
  try {
    const preview = await api.previewPlaylist(owner, ins);
    if (store.userId !== owner) return;
    draft.value = preview.tracks.length ? preview : null;
    picked.value = preview.tracks.map((_, index) => index);
    draftName.value = preview.name.slice(0, 200);
    adjustment.value = "";
    if (!draft.value) msg.value = "没有找到合适的曲目，试试换一个场景或曲风。";
  } catch { msg.value = "生成失败，请稍后重试。"; }
  finally { generating.value = false; }
}

async function revise(keepSelected) {
  if (!draft.value || generating.value || saving.value || revising.value) return;
  const owner = store.userId;
  const kept = keepSelected ? picked.value : [];
  const note = adjustment.value.trim();
  msg.value = "";
  revising.value = true;
  try {
    const result = await api.revisePlaylist(owner, draft.value.playlist_id, kept, note);
    if (owner !== store.userId) return;
    if (!result.added_count) {
      msg.value = "暂时没有找到新的候选，当前草稿已保留。可以试试更具体的曲风。";
      return;
    }
    draft.value = result.draft;
    picked.value = result.draft.tracks.map((_, index) => index);
    adjustment.value = "";
    const target = result.draft.target_count;
    const count = result.draft.tracks.length;
    msg.value = `保留 ${kept.length} 首，找到 ${result.added_count} 首新候选。`
      + (count < target ? `当前共有 ${count} 首，目标 ${target} 首，可继续补歌。` : "");
  } catch (error) {
    if (owner !== store.userId) return;
    msg.value = error.message === "HTTP 409" ? "草稿已变更，请重新预览。" : "换歌失败，可以重试。";
  } finally { revising.value = false; }
}

async function saveSelection() {
  if (!draft.value || !picked.value.length || saving.value || generating.value || revising.value) return;
  const owner = store.userId;
  saving.value = true;
  try {
    const result = await api.confirmPlaylist(owner, draft.value.playlist_id, picked.value, draftName.value);
    if (store.userId !== owner) return;
    draft.value = null;
    expanded.value = result.playlist_id;
    msg.value = `已保存《${result.name}》，共 ${result.tracks.length} 首。`;
    await load();
  } catch (error) {
    if (owner !== store.userId) return;
    msg.value = error.message === "HTTP 409" ? "草稿已变更，请重新预览。" : "保存失败，可以重试。";
  } finally { saving.value = false; }
}

watch(() => store.userId, () => {
  draft.value = null;
  picked.value = [];
  adjustment.value = "";
  selectedIds.value = new Set();
  playlists.value = [];
  instruction.value = "";
  msg.value = "";
  expanded.value = null;
  load();
});

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
    asset_id: t.asset_id, playback_url: t.playback_url,
    source_id: t.source_id || t.external_id || "", cover_url: t.cover_url,
  };
}

// covers for collage: first 3 unique cover_url values
function playlistCovers(pl) {
  const covers = [];
  for (const t of pl.tracks || []) {
    if (t.cover_url && !covers.includes(t.cover_url)) covers.push(t.cover_url);
    if (covers.length >= 4) break;
  }
  return covers;
}

// real-height expand transition
function onEnter(el) {
  el.style.overflow = "hidden";
  el.style.maxHeight = "0";
  el.style.opacity = "0";
  requestAnimationFrame(() => {
    el.style.transition = "max-height 0.4s cubic-bezier(0.16,1,0.3,1), opacity 0.3s ease";
    el.style.maxHeight = el.scrollHeight + "px";
    el.style.opacity = "1";
  });
}
function onAfterEnter(el) {
  el.style.maxHeight = "";
  el.style.overflow = "";
  el.style.transition = "";
  el.style.opacity = "";
}
function onLeave(el) {
  el.style.overflow = "hidden";
  el.style.maxHeight = el.scrollHeight + "px";
  el.style.opacity = "1";
  requestAnimationFrame(() => {
    el.style.transition = "max-height 0.32s cubic-bezier(0.16,1,0.3,1), opacity 0.22s ease";
    el.style.maxHeight = "0";
    el.style.opacity = "0";
  });
}

onMounted(load);
</script>

<template>
  <div class="playlist-view">
    <div class="pl-page-head">
      <div>
        <h1>为此刻挑一份歌单</h1>
        <p class="curation-intro">告诉我你在做什么，先看看候选曲目，再留下想听的。</p>
      </div>
      <div v-if="playlists.length" class="pl-head-actions">
        <template v-if="selectMode">
          <button class="sel-btn" @click="selectAll">
            {{ selectedIds.size === playlists.length ? "取消全选" : "全选" }}
          </button>
          <button class="sel-btn danger" :disabled="!selectedIds.size || bulkDeleting" @click="deleteSelected">
            {{ bulkDeleting ? "删除中…" : `删除选中（${selectedIds.size}）` }}
          </button>
          <button class="sel-btn ghost" @click="toggleSelectMode">取消</button>
        </template>
        <button v-else class="sel-btn" @click="toggleSelectMode">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
          多选
        </button>
      </div>
    </div>

    <!-- generator -->
    <div class="gen-block">
      <div class="gen-input-wrap">
        <input v-model="instruction" aria-label="听歌场景" class="gen-input" placeholder="用一句话描述歌单，如「雨天 city pop 20首」" @keyup.enter="generate" />
        <button class="gen-send" :disabled="generating || saving || revising || !instruction.trim()" @click="generate">
          <svg v-if="!generating" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          <svg v-else class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>
          {{ generating ? "整理中" : "预览歌单" }}
        </button>
      </div>
      <button class="gen-auto" :disabled="generating || saving || revising" @click="autoClassify">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>
        自动分类
      </button>
    </div>

    <section v-if="draft" class="draft-panel" aria-label="歌单草稿">
      <label class="draft-title">草稿名称
        <input v-model="draftName" class="gen-input" maxlength="200" :disabled="saving || generating || revising" />
      </label>
      <span class="draft-count">候选 {{ draft.tracks.length }} 首 · 已选 {{ picked.length }} 首</span>
      <label v-for="(track, i) in draft.tracks" :key="i" class="draft-row">
        <input v-model="picked" type="checkbox" :value="i" :disabled="saving || generating || revising" />
        <span>{{ track.title }} <small>{{ track.artist }}</small></span>
      </label>
      <div class="draft-adjust">
        <label class="draft-title">调整要求 <small>可选</small>
          <input v-model="adjustment" class="gen-input" maxlength="300" placeholder="例如少一点人声，偏爵士" :disabled="saving || generating || revising" />
        </label>
        <div class="draft-actions">
          <button class="sel-btn" :disabled="saving || generating || revising || picked.length >= draft.target_count" @click="revise(true)">
            {{ revising ? "寻找新曲目…" : "保留已选，其余换歌" }}
          </button>
          <button class="sel-btn" :disabled="saving || generating || revising" @click="revise(false)">全部换一批</button>
          <button class="gen-send draft-save" :disabled="!picked.length || saving || generating || revising" @click="saveSelection">
            {{ saving ? "保存中…" : "保存选中的曲目" }}
          </button>
        </div>
      </div>
    </section>
    <div v-if="msg" role="status" class="pl-toast">{{ msg }}</div>
    <div v-if="loading || generating" class="loading-hint">{{ generating ? "生成中…" : "加载中…" }}</div>

    <!-- empty state -->
    <div v-if="!loading && !playlists.length && !draft && !generating" class="pl-empty">
      <svg class="pl-empty-icon" viewBox="0 0 120 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="20" width="60" height="8" rx="4" fill="currentColor" opacity="0.2"/>
        <rect x="10" y="36" width="45" height="8" rx="4" fill="currentColor" opacity="0.15"/>
        <rect x="10" y="52" width="52" height="8" rx="4" fill="currentColor" opacity="0.1"/>
        <circle cx="90" cy="42" r="18" fill="currentColor" opacity="0.06"/>
        <path d="M90 33v12.5M90 33c0-1.1.9-2 2-2h5v5h-5c-1.1 0-2-.9-2-2z" stroke="currentColor" stroke-width="2" stroke-linecap="round" opacity="0.3"/>
        <circle cx="88" cy="47" r="3" stroke="currentColor" stroke-width="2" opacity="0.3"/>
      </svg>
      <p>试试用一个场景开始</p>
      <div class="pl-empty-actions">
        <button class="empty-btn" @click="instruction = '推荐 20 首适合深夜的治愈歌单'">🌙 深夜治愈</button>
        <button class="empty-btn" @click="instruction = '帮我做 20 首跑步歌单，节奏感强'">🏃 跑步节奏</button>
      </div>
    </div>

    <h2 v-if="playlists.length">我的歌单</h2>
    <!-- playlist cards -->
    <div v-for="(pl, idx) in playlists" :key="pl.playlist_id" class="pl-card stagger-item" :class="{ selecting: selectMode, selected: selectedIds.has(pl.playlist_id) }" :style="{ animationDelay: `${idx * 55}ms` }">
      <div class="pl-head">
        <!-- 多选复选框 -->
        <label v-if="selectMode" class="pl-check" @click.stop>
          <input type="checkbox" :checked="selectedIds.has(pl.playlist_id)" @change="toggleSelect(pl.playlist_id)" />
          <span class="pl-check-box"></span>
        </label>

        <!-- cover collage -->
        <div class="pl-collage" :class="{ single: playlistCovers(pl).length === 1, empty: !playlistCovers(pl).length }">
          <template v-if="playlistCovers(pl).length >= 4">
            <img v-for="(src, ci) in playlistCovers(pl).slice(0, 4)" :key="ci" :src="src" alt="" loading="lazy" />
          </template>
          <template v-else-if="playlistCovers(pl).length > 0">
            <img :src="playlistCovers(pl)[0]" alt="" loading="lazy" class="solo-cover" />
          </template>
          <template v-else>
            <div class="pl-cover-ph">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" opacity="0.5"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55C7.79 13 6 14.79 6 17s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
            </div>
          </template>
        </div>

        <!-- info -->
        <div class="pl-info" @click="selectMode ? toggleSelect(pl.playlist_id) : (expanded = expanded === pl.playlist_id ? null : pl.playlist_id)">
          <div class="pl-name">{{ pl.name }}</div>
          <div class="pl-desc">{{ pl.description || `${pl.tracks?.length || 0} 首曲目` }}</div>
        </div>

        <!-- action group -->
        <div v-if="!selectMode" class="pl-actions">
          <button
            v-if="pl.tracks?.length"
            class="pl-play-btn"
            title="播放全部"
            @click.stop="store.playAll(pl.tracks.map(toCard))"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
            播放
          </button>
          <button class="pl-icon-btn" title="删除歌单" @click.stop="remove(pl)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
          </button>
          <button
            class="pl-chev"
            :class="{ open: expanded === pl.playlist_id }"
            @click.stop="expanded = expanded === pl.playlist_id ? null : pl.playlist_id"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/></svg>
          </button>
        </div>
      </div>

      <Transition
        @enter="onEnter"
        @after-enter="onAfterEnter"
        @leave="onLeave"
      >
        <div v-if="expanded === pl.playlist_id" class="pl-tracks">
          <SongCard v-for="(t, i) in pl.tracks" :key="i" :card="toCard(t)" :show-reason="false" @toast="(m) => msg = m" />
        </div>
      </Transition>
    </div>
  </div>
</template>

<style scoped>
.playlist-view {
  padding: 28px clamp(18px, 4vw, 42px) calc(var(--player-h) + 40px);
}

.curation-intro, .draft-count, .draft-row small { color: var(--text-sub); }
.curation-intro { margin: .5rem 0 0; }
.draft-panel {
  display: grid; gap: 12px; padding: 20px; margin-bottom: 24px;
  border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: var(--radius); background: var(--bg-card);
}
.draft-title { display: grid; gap: 6px; font-weight: 700; }
.draft-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); }
.draft-row input { accent-color: var(--accent); }
.draft-row small { margin-left: 6px; }
.draft-adjust { display: grid; gap: 12px; }
.draft-adjust .draft-title { display: block; }
.draft-adjust .draft-title small { margin-left: 4px; color: var(--text-sub); font-weight: 400; }
.draft-adjust .gen-input { display: block; width: 100%; box-sizing: border-box; margin-top: 6px; }
.draft-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.draft-save { margin-left: auto; }
@media (max-width: 600px) {
  .draft-save { margin-left: 0; }
}
.gen-input { min-width: 0; }
.pl-page-head {
  margin-bottom: 24px;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
}
.pl-head-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}
.sel-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 36px;
  padding: 0 14px;
  border-radius: var(--radius-pill);
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-sub);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  transition: all var(--transition);
}
.sel-btn:hover:not(:disabled) { border-color: var(--border-light); color: var(--text); }
.sel-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.sel-btn.danger {
  background: rgba(231,76,60,0.1);
  border-color: rgba(231,76,60,0.3);
  color: #e74c3c;
}
.sel-btn.danger:hover:not(:disabled) { background: #e74c3c; color: #fff; border-color: #e74c3c; }
.sel-btn.ghost { background: transparent; }
.eyebrow {
  margin: 0 0 6px;
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: clamp(1.6rem, 3vw, 2.2rem);
  letter-spacing: -0.02em;
}

/* ── Generator ── */
.gen-block {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 24px;
}
.gen-input-wrap {
  display: flex;
  gap: 10px;
}
.gen-input {
  flex: 1;
  min-height: 46px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  padding: 0 16px;
  font: inherit;
  font-size: 0.92rem;
  transition: border-color var(--transition);
}
.gen-input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-dim);
}
.gen-input::placeholder { color: var(--text-muted); }
.gen-send {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 46px;
  padding: 0 20px;
  border-radius: var(--radius);
  background: var(--accent);
  color: #07120b;
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 0.88rem;
  white-space: nowrap;
  transition: all var(--transition);
}
.gen-send:hover:not(:disabled) { background: var(--accent-hover); transform: translateY(-1px); }
.gen-send:disabled { opacity: 0.5; cursor: not-allowed; }
.gen-auto {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 38px;
  padding: 0 16px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-sub);
  font-size: 0.84rem;
  font-weight: 600;
  align-self: flex-start;
  transition: all var(--transition);
}
.gen-auto:hover:not(:disabled) { border-color: var(--border-light); color: var(--text); }
.gen-auto:disabled { opacity: 0.5; cursor: not-allowed; }
.spin { animation: pl-spin 0.9s linear infinite; }
@keyframes pl-spin { to { transform: rotate(360deg); } }

/* ── Toast ── */
.pl-toast {
  padding: 10px 14px;
  margin-bottom: 16px;
  border-radius: var(--radius-sm);
  background: var(--accent-dim);
  color: var(--accent);
  border: 1px solid rgba(29,185,84,0.12);
  font-size: 0.86rem;
}

/* ── Empty state ── */
.pl-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 56px 24px;
  text-align: center;
  color: var(--text-sub);
}
.pl-empty-icon {
  width: 100px;
  height: 80px;
  color: var(--text-muted);
  margin-bottom: 4px;
}
.pl-empty p { font-size: 0.95rem; margin: 0; }
.pl-empty-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
}
.empty-btn {
  padding: 9px 16px;
  border-radius: var(--radius-pill);
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-sub);
  font-size: 0.84rem;
  font-weight: 600;
  transition: all var(--transition);
}
.empty-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

/* ── Playlist card ── */
.pl-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  margin-bottom: 10px;
  overflow: hidden;
  transition: border-color var(--dur-norm) var(--ease-out);
}
.pl-card:hover { border-color: var(--border-light); }
.pl-card.selecting { cursor: pointer; }
.pl-card.selected {
  border-color: var(--accent);
  background: var(--accent-dim);
}

/* 多选复选框 */
.pl-check {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  cursor: pointer;
}
.pl-check input { position: absolute; opacity: 0; width: 0; height: 0; }
.pl-check-box {
  width: 22px;
  height: 22px;
  border-radius: 6px;
  border: 2px solid var(--border-light);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
}
.pl-check input:checked + .pl-check-box {
  background: var(--accent);
  border-color: var(--accent);
}
.pl-check input:checked + .pl-check-box::after {
  content: "";
  width: 6px;
  height: 11px;
  border: solid #07120b;
  border-width: 0 2.5px 2.5px 0;
  transform: rotate(45deg);
  margin-top: -2px;
}

.pl-head {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 12px 14px;
}

/* Cover collage */
.pl-collage {
  width: 56px;
  height: 56px;
  border-radius: var(--radius-sm);
  flex-shrink: 0;
  overflow: hidden;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  background: var(--bg-elevated);
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.pl-collage img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.4s var(--ease-out);
}
.pl-card:hover .pl-collage img { transform: scale(1.06); }
.pl-collage.single, .pl-collage.empty {
  grid-template-columns: 1fr;
  grid-template-rows: 1fr;
}
.pl-collage .solo-cover { width: 100%; height: 100%; object-fit: cover; }
.pl-cover-ph {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, rgba(29,185,84,0.12), rgba(100,60,180,0.08));
  color: var(--text-muted);
}

/* Info */
.pl-info {
  flex: 1;
  min-width: 0;
  cursor: pointer;
}
.pl-name {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 1.02rem;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pl-desc {
  margin-top: 4px;
  color: var(--text-sub);
  font-size: 0.82rem;
  line-height: 1.45;
  white-space: normal;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Actions */
.pl-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}
.pl-play-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 32px;
  padding: 0 12px;
  border-radius: var(--radius-pill);
  background: var(--accent-dim);
  border: 1px solid rgba(29,185,84,0.2);
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  white-space: nowrap;
  transition: all var(--transition);
}
.pl-play-btn:hover { background: var(--accent); color: #fff; border-color: var(--accent); transform: scale(1.04); }
.pl-icon-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  transition: all var(--transition);
}
.pl-icon-btn:hover { background: rgba(231,76,60,0.1); color: #e74c3c; }
.pl-chev {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  transition: all var(--dur-norm) var(--ease-out);
}
.pl-chev.open { transform: rotate(180deg); color: var(--text-sub); }
.pl-chev:hover { background: var(--bg-hover); color: var(--text); }

/* Track list */
.pl-tracks {
  padding: 4px 14px 14px;
  border-top: 1px solid var(--border);
}
</style>
