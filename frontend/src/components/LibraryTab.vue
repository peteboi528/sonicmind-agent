<script setup>
import { ref, computed, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";
import EnergyBadge from "./EnergyBadge.vue";

const assets = ref([]);
const loading = ref(false);
const msg = ref("");
const ingestUrl = ref("");
const ingesting = ref(false);
const selectedTags = ref(new Set());   // 当前已选筛选标签
const searchQuery = ref("");           // 库内文字搜索（歌名/歌手/专辑/标签）

const savedAlbums = ref([]);
const savedLoading = ref(false);

// ── 多选建歌单 ──
const selectMode = ref(false);
const selectedIds = ref(new Set());
const creating = ref(false);
const showNameDialog = ref(false);
const newPlaylistName = ref("");

function toggleSelectMode() {
  selectMode.value = !selectMode.value;
  if (!selectMode.value) selectedIds.value = new Set();
}

function toggleSelect(assetId) {
  const s = new Set(selectedIds.value);
  s.has(assetId) ? s.delete(assetId) : s.add(assetId);
  selectedIds.value = s;
}

function selectAllFiltered() {
  const ids = filteredAssets.value.map(a => a.asset_id);
  const allSelected = ids.every(id => selectedIds.value.has(id));
  const s = new Set(selectedIds.value);
  if (allSelected) {
    for (const id of ids) s.delete(id);
  } else {
    for (const id of ids) s.add(id);
  }
  selectedIds.value = s;
}

function openNameDialog() {
  if (!selectedIds.value.size) return;
  newPlaylistName.value = "";
  showNameDialog.value = true;
}

async function createPlaylist() {
  const ids = [...selectedIds.value];
  if (!ids.length || creating.value) return;
  creating.value = true;
  try {
    const pl = await api.createPlaylistFromAssets(store.userId, newPlaylistName.value.trim() || "我的歌单", ids);
    msg.value = `已创建歌单《${pl.name}》：${pl.tracks?.length || ids.length} 首。`;
    showNameDialog.value = false;
    selectMode.value = false;
    selectedIds.value = new Set();
  } catch { msg.value = "创建歌单失败。"; }
  finally { creating.value = false; }
}

function notify(m) { msg.value = m; }

// ── 标签筛选 ──────────────────────────────────────────────────────────────

// 所有出现过的标签，按出现频次降序（genre 优先，mood 跟在后面）。
const allTags = computed(() => {
  const freq = {};
  for (const a of assets.value) {
    for (const g of (a.genre || [])) freq[g] = (freq[g] || 0) + 1;
    for (const m of (a.mood  || [])) freq[m] = (freq[m] || 0) + 1;
  }
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .map(([tag]) => tag);
});

function toggleTag(tag) {
  const s = new Set(selectedTags.value);
  s.has(tag) ? s.delete(tag) : s.add(tag);
  selectedTags.value = s;
}

function clearFilter() { selectedTags.value = new Set(); searchQuery.value = ""; }

// 筛选后的列表：标签 AND 语义 + 文字搜索（歌名/歌手/专辑/标签任一命中，不区分大小写）。
const filteredAssets = computed(() => {
  let list = assets.value;
  if (selectedTags.value.size) {
    list = list.filter(a => {
      const tags = new Set([...(a.genre || []), ...(a.mood || [])]);
      for (const t of selectedTags.value) if (!tags.has(t)) return false;
      return true;
    });
  }
  const q = searchQuery.value.trim().toLowerCase();
  if (q) {
    list = list.filter(a => {
      const hay = [
        a.title || "", a.artist || "", a.album || "",
        ...(a.genre || []), ...(a.mood || []), ...(a.tags_fingerprint || []),
      ].join(" ").toLowerCase();
      return hay.includes(q);
    });
  }
  return list;
});

// ── 播放 ──────────────────────────────────────────────────────────────────

function savedToCard(t) {
  return {
    title: t.title, artist: t.artist || "", source: t.source || "netease",
    source_id: t.external_id || t.source_id || "", cover_url: t.cover_url,
    playback_url: t.playback_url,
  };
}

function assetToCard(a) {
  return {
    title: a.title || "未命名", artist: a.artist || "",
    source: a.source || "local", source_id: a.external_id || "",
    source_url: a.source_url || "", playback_url: a.source_url || "",
    cover_url: a.cover_url || "", asset_id: a.asset_id || "",
  };
}

function playFrom(idx) {
  if (!filteredAssets.value.length) return;
  store.queue = filteredAssets.value.map(assetToCard);
  store.playQueueIndex(idx);
  msg.value = `▶ ${filteredAssets.value[idx]?.title || "播放"}`;
}

function playAllAssets() {
  if (!filteredAssets.value.length) return;
  store.playAll(filteredAssets.value.map(assetToCard));
  const filtered = selectedTags.value.size || searchQuery.value;
  msg.value = `播放${filtered ? "筛选结果" : "全部"}：${filteredAssets.value.length} 首`;
}

async function loadSavedAlbums() {
  savedLoading.value = true;
  try {
    const data = await api.listSavedAlbums(store.userId);
    savedAlbums.value = (data.albums || []).map(a => ({ ...a, _expanded: false }));
  } catch { savedAlbums.value = []; }
  finally { savedLoading.value = false; }
}

function playSavedAlbum(al) {
  if (!al.tracks?.length) return;
  store.playAll(al.tracks.map(savedToCard));
  msg.value = `播放《${al.name}》：${al.tracks.length} 首`;
}

async function unsaveAlbum(al) {
  try {
    await api.unsaveAlbum(store.userId, al.album_id);
    savedAlbums.value = savedAlbums.value.filter(a => a.album_id !== al.album_id);
    msg.value = `已取消收藏《${al.name}》。`;
  } catch { msg.value = "取消收藏失败。"; }
}

async function load() {
  loading.value = true;
  try {
    const data = await api.listAssets();
    assets.value = data.assets || [];
    try {
      const ratingsData = await api.getRatings(store.userId);
      const ratingMap = {};
      for (const r of ratingsData.ratings || []) ratingMap[r.asset_id] = r.score;
      for (const a of assets.value) if (ratingMap[a.asset_id] != null) a._rated = ratingMap[a.asset_id];
    } catch { /* 评分加载失败不影响主流程 */ }
  } catch { msg.value = "加载库失败。"; }
  finally { loading.value = false; }
}

async function rate(asset, score) {
  try {
    await api.rate(store.userId, asset.asset_id, score);
    asset._rated = score;
    msg.value = `已为《${asset.title}》评 ${score} 分。`;
  } catch { msg.value = "评分失败。"; }
}

async function remove(asset) {
  if (!confirm(`确定删除《${asset.title}》？`)) return;
  try {
    await api.deleteAsset(asset.asset_id, store.userId);
    assets.value = assets.value.filter((a) => a.asset_id !== asset.asset_id);
  } catch { msg.value = "删除失败。"; }
}

async function ingest() {
  const url = ingestUrl.value.trim();
  if (!url) return;
  ingesting.value = true;
  msg.value = "";
  try {
    const asset = await api.ingest(url);
    msg.value = `已入库：《${asset.title || "新资源"}》`;
    ingestUrl.value = "";
    await load();
  } catch { msg.value = "入库失败，请检查链接。"; }
  finally { ingesting.value = false; }
}

onMounted(() => { load(); loadSavedAlbums(); });
</script>

<template>
  <div>
    <div class="lib-head">
      <div>
        <div class="section-title">我的库</div>
        <div class="section-sub">管理已入库的曲目，评分会实时更新你的口味画像。</div>
      </div>
      <div v-if="assets.length" class="lib-head-actions">
        <template v-if="selectMode">
          <button class="sel-btn" @click="selectAllFiltered">
            {{ filteredAssets.length && filteredAssets.every(a => selectedIds.has(a.asset_id)) ? "取消全选" : "全选" }}
          </button>
          <button class="sel-btn accent" :disabled="!selectedIds.size || creating" @click="openNameDialog">
            建歌单（{{ selectedIds.size }}）
          </button>
          <button class="sel-btn ghost" @click="toggleSelectMode">取消</button>
        </template>
        <button v-else class="sel-btn" @click="toggleSelectMode">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
          多选建歌单
        </button>
      </div>
    </div>

    <!-- ── 收藏的专辑 ── -->
    <section v-if="savedAlbums.length || savedLoading" class="saved-section">
      <div class="saved-head">
        <span class="saved-label">💿 收藏的专辑</span>
        <span v-if="savedAlbums.length" class="saved-count">{{ savedAlbums.length }} 张</span>
      </div>
      <div v-if="savedLoading" class="loading-hint">加载中…</div>
      <div v-for="al in savedAlbums" :key="al.album_id" class="saved-album stagger-item">
        <div class="saved-album-row" role="button" @click="al._expanded = !al._expanded">
          <div class="saved-album-cover-wrap">
            <img v-if="al.image" :src="al.image" class="saved-album-cover" alt="" loading="lazy" />
            <div v-else class="saved-album-cover-ph">💿</div>
          </div>
          <div class="saved-album-info">
            <div class="saved-album-name">{{ al.name }}</div>
            <div class="saved-album-meta">{{ al.artist || '未知' }} · {{ al.tracks?.length || al.track_count || 0 }} 首</div>
          </div>
          <button class="saved-play-btn" @click.stop="playSavedAlbum(al)" title="播放整张专辑">▶ 播放</button>
          <button class="del" @click.stop="unsaveAlbum(al)" title="取消收藏">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
          </button>
        </div>
        <div v-if="al._expanded" class="saved-album-tracks">
          <div v-for="(t, i) in (al.tracks || [])" :key="(t.external_id || t.title) + '_' + i" class="saved-track-row">
            <span class="saved-track-no">{{ i + 1 }}</span>
            <SongCard :card="savedToCard(t)" :show-reason="false" @toast="notify" />
          </div>
          <div v-if="!al.tracks?.length" class="empty-hint">该专辑暂无曲目。</div>
        </div>
      </div>
    </section>

    <div class="ingest-row">
      <input v-model="ingestUrl" class="input" placeholder="粘贴音乐链接入库（网易云/B站/YouTube）" @keyup.enter="ingest" />
      <button class="btn" :disabled="ingesting || !ingestUrl.trim()" @click="ingest">
        {{ ingesting ? "入库中…" : "添加" }}
      </button>
    </div>

    <Transition name="toast-slide">
      <div v-if="msg" class="toast">{{ msg }}</div>
    </Transition>
    <div v-if="loading" class="loading-hint">加载中…</div>

    <div v-if="!loading && !assets.length" class="empty-hint">库还是空的，先添加点音乐吧。</div>

    <!-- ── 搜索 ── -->
    <div v-if="assets.length" class="search-row">
      <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input v-model="searchQuery" class="search-input" placeholder="搜索库内歌曲：歌名 / 歌手 / 专辑 / 标签" />
      <button v-if="searchQuery" class="search-clear" @click="searchQuery = ''" title="清除搜索">×</button>
    </div>

    <!-- ── 标签筛选 ── -->
    <div v-if="assets.length && allTags.length" class="filter-bar">
      <span class="filter-label">筛选</span>
      <button
        v-for="tag in allTags"
        :key="tag"
        class="filter-tag"
        :class="{ on: selectedTags.has(tag) }"
        @click="toggleTag(tag)"
      >{{ tag }}</button>
      <button v-if="selectedTags.size" class="filter-clear" @click="clearFilter">
        清除筛选（{{ filteredAssets.length }}/{{ assets.length }}）
      </button>
    </div>

    <div v-if="!loading && assets.length && !filteredAssets.length" class="empty-hint">
      <template v-if="searchQuery">没有匹配「{{ searchQuery }}」的曲目。<button class="link-btn" @click="searchQuery = ''">清除搜索</button></template>
      <template v-else>没有符合所选标签的曲目。<button class="link-btn" @click="clearFilter">清除筛选</button></template>
    </div>

    <button v-if="filteredAssets.length" class="play-all-btn" @click="playAllAssets">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
      播放{{ (selectedTags.size || searchQuery) ? "筛选结果" : "全部" }}（{{ filteredAssets.length }} 首）
    </button>

    <div v-for="(a, idx) in filteredAssets" :key="a.asset_id" class="lib-row stagger-item" :class="{ selecting: selectMode, selected: selectedIds.has(a.asset_id) }" :style="{ animationDelay: `${idx * 40}ms` }" @click="selectMode && toggleSelect(a.asset_id)">
      <label v-if="selectMode" class="lib-check" @click.stop>
        <input type="checkbox" :checked="selectedIds.has(a.asset_id)" @change="toggleSelect(a.asset_id)" />
        <span class="lib-check-box"></span>
      </label>
      <button v-else class="play-btn" title="播放" @click="playFrom(idx)">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
      </button>
      <div class="info">
        <div class="title">{{ a.title || "未命名" }}</div>
        <div class="artist">
          {{ a.artist || "未知" }}
          <span v-for="g in (a.genre || []).slice(0,2)" :key="g" class="tag">{{ g }}</span>
          <EnergyBadge v-if="a.energy_level != null" :energy="a.energy_level" :source="a.features_source" />
        </div>
      </div>
      <div v-if="!selectMode" class="rate">
        <button v-for="s in [2,4,6,8,10]" :key="s"
          class="star" :class="{ on: a._rated >= s }" @click="rate(a, s)">★</button>
      </div>
      <button v-if="!selectMode" class="del" @click="remove(a)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
      </button>
    </div>

    <!-- 多选建歌单：命名对话框 -->
    <div v-if="showNameDialog" class="dialog-mask" @click.self="showNameDialog = false">
      <div class="dialog">
        <h3>新建歌单</h3>
        <p class="dialog-sub">已选 {{ selectedIds.size }} 首曲目</p>
        <input
          v-model="newPlaylistName"
          class="dialog-input"
          placeholder="给歌单起个名字"
          @keyup.enter="createPlaylist"
        />
        <div class="dialog-actions">
          <button class="sel-btn ghost" @click="showNameDialog = false">取消</button>
          <button class="sel-btn accent" :disabled="creating" @click="createPlaylist">
            {{ creating ? "创建中…" : "创建" }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.lib-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 8px;
}
.lib-head-actions { display: flex; gap: 8px; flex-shrink: 0; }
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
.sel-btn.accent {
  background: var(--accent-dim);
  border-color: rgba(29,185,84,0.3);
  color: var(--accent);
}
.sel-btn.accent:hover:not(:disabled) { background: var(--accent); color: #07120b; border-color: var(--accent); }
.sel-btn.ghost { background: transparent; }

/* 多选复选框 */
.lib-check {
  display: flex; align-items: center; justify-content: center;
  width: 34px; height: 34px; flex-shrink: 0; cursor: pointer;
}
.lib-check input { position: absolute; opacity: 0; width: 0; height: 0; }
.lib-check-box {
  width: 20px; height: 20px; border-radius: 6px;
  border: 2px solid var(--border-light);
  display: flex; align-items: center; justify-content: center;
  transition: all var(--transition);
}
.lib-check input:checked + .lib-check-box {
  background: var(--accent); border-color: var(--accent);
}
.lib-check input:checked + .lib-check-box::after {
  content: ""; width: 5px; height: 10px;
  border: solid #07120b; border-width: 0 2.5px 2.5px 0;
  transform: rotate(45deg); margin-top: -2px;
}

.ingest-row { display: flex; gap: 10px; margin-bottom: 20px; max-width: 640px; }

/* ── 库内搜索 ── */
.search-row {
  display: flex; align-items: center; gap: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 14px;
  margin-bottom: 16px;
  transition: all var(--transition);
}
.search-row:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-dim);
}
.search-icon { color: var(--text-muted); flex-shrink: 0; }
.search-input {
  flex: 1; min-width: 0;
  background: transparent; border: none; outline: none;
  color: var(--text);
  font-size: 0.9rem; font-family: var(--font-display);
}
.search-input::placeholder { color: var(--text-muted); }
.search-clear {
  width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-elevated); border: none;
  color: var(--text-muted); font-size: 1rem; line-height: 1;
  cursor: pointer; transition: all var(--transition);
}
.search-clear:hover { background: rgba(231,76,60,0.15); color: var(--danger); }

/* ── 标签筛选 ── */
.filter-bar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  margin-bottom: 18px;
}
.filter-label {
  color: var(--text-muted); font-size: 0.8rem; font-weight: 600;
  margin-right: 4px;
}
.filter-tag {
  padding: 5px 12px;
  border-radius: var(--radius-pill);
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  color: var(--text-sub);
  font-size: 0.76rem; font-weight: 600;
  font-family: var(--font-display);
  cursor: pointer;
  transition: all var(--transition);
}
.filter-tag:hover {
  border-color: var(--border-light);
  color: var(--text);
  transform: translateY(-1px);
}
.filter-tag.on {
  background: var(--accent-dim);
  border-color: var(--accent);
  color: var(--accent);
}
.filter-clear {
  margin-left: auto;
  padding: 5px 12px;
  border-radius: var(--radius-pill);
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.76rem; font-weight: 600;
  cursor: pointer;
  transition: all var(--transition);
}
.filter-clear:hover {
  color: var(--danger);
  border-color: rgba(231,76,60,0.3);
}
.link-btn {
  background: none; border: none; color: var(--accent);
  font-size: inherit; font-weight: 600; cursor: pointer;
  text-decoration: underline; padding: 0;
}
.link-btn:hover { color: var(--accent-hover); }
.toast {
  background: var(--accent-dim); color: var(--accent); padding: 10px 14px;
  border-radius: var(--radius-sm); margin-bottom: 16px;
  border: 1px solid rgba(29,185,84,0.12);
}
.toast-slide-enter-active { animation: fadeInUp 0.25s var(--ease-out); }
.toast-slide-leave-active { transition: all 0.15s ease; }
.toast-slide-leave-to { opacity: 0; transform: translateY(-6px); }

.lib-row {
  display: flex; align-items: center; gap: 14px;
  background: var(--bg-card); padding: 14px 18px;
  border-radius: var(--radius); margin-bottom: 8px;
  border: 1px solid var(--border);
  transition: all var(--dur-norm) var(--ease-out);
}
.lib-row:hover {
  background: var(--bg-hover);
  border-color: var(--border-light);
  transform: translateY(-1px);
}
.lib-row.selecting { cursor: pointer; }
.lib-row.selected { border-color: var(--accent); background: var(--accent-dim); }

/* 命名对话框 */
.dialog-mask {
  position: fixed; inset: 0; z-index: 50;
  background: rgba(0,0,0,0.55);
  display: flex; align-items: center; justify-content: center;
  padding: 20px;
}
.dialog {
  width: 100%; max-width: 360px;
  background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius);
  padding: 22px;
  box-shadow: 0 18px 50px rgba(0,0,0,0.45);
}
.dialog h3 { margin: 0 0 4px; font-family: var(--font-display); font-size: 1.1rem; }
.dialog-sub { margin: 0 0 16px; color: var(--text-sub); font-size: 0.84rem; }
.dialog-input {
  width: 100%; min-height: 44px;
  background: var(--bg-elevated); border: 1px solid var(--border);
  border-radius: var(--radius-sm); color: var(--text);
  padding: 0 14px; font: inherit; margin-bottom: 18px;
}
.dialog-input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-dim); }
.dialog-actions { display: flex; justify-content: flex-end; gap: 10px; }
.info { flex: 1; min-width: 0; }
.title {
  font-family: var(--font-display);
  font-weight: 600; font-size: 0.93rem;
}
.artist { color: var(--text-sub); font-size: 0.82rem; margin-top: 3px; }

.play-btn {
  width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-dim); color: var(--accent);
  border: 1px solid var(--border-light);
  transition: all var(--transition);
}
.play-btn:hover {
  background: var(--accent); color: #fff;
  border-color: var(--accent); transform: scale(1.08);
}
.tag {
  display: inline-block; margin-left: 6px; padding: 2px 8px;
  background: var(--bg-elevated); border-radius: var(--radius-pill);
  font-size: 0.68rem; color: var(--text-muted);
  font-family: var(--font-display); font-weight: 600;
}

.rate { display: flex; gap: 2px; }
.star {
  font-size: 1.15rem; color: var(--text-muted);
  transition: all 0.2s var(--ease-spring);
}
.star.on {
  color: var(--accent);
  text-shadow: 0 0 8px var(--accent-glow);
}
.star:hover {
  color: var(--accent-hover);
  transform: scale(1.25);
}

.del {
  width: 34px; height: 34px; border-radius: 50%;
  color: var(--text-muted); display: flex; align-items: center; justify-content: center;
  transition: all var(--transition);
}
.del:hover { background: rgba(231,76,60,0.1); color: var(--danger); }

/* ── 收藏的专辑 ── */
.saved-section {
  margin-bottom: 32px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.saved-head {
  display: flex; align-items: baseline; gap: 10px;
  margin-bottom: 14px;
}
.saved-label {
  font-family: var(--font-display);
  font-size: 1.1rem; font-weight: 700;
}
.saved-count {
  color: var(--text-muted); font-size: 0.8rem; font-weight: 600;
}
.saved-album {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px 14px;
  margin-bottom: 10px; transition: all var(--dur-norm) var(--ease-out);
}
.saved-album:hover { border-color: var(--border-light); }
.saved-album-row {
  display: flex; align-items: center; gap: 14px; cursor: pointer;
}
.saved-album-cover-wrap {
  width: 52px; height: 52px; border-radius: var(--radius-sm);
  overflow: hidden; flex-shrink: 0;
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.saved-album-cover { width: 100%; height: 100%; object-fit: cover; }
.saved-album-cover-ph {
  width: 100%; height: 100%; display: flex;
  align-items: center; justify-content: center; font-size: 1.4rem;
  background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
}
.saved-album-info { flex: 1; min-width: 0; }
.saved-album-name {
  font-family: var(--font-display); font-weight: 600; font-size: 0.93rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.saved-album-meta { color: var(--text-sub); font-size: 0.8rem; margin-top: 3px; }
.saved-play-btn {
  padding: 6px 14px; border-radius: var(--radius-pill);
  background: var(--accent-dim); border: 1px solid var(--border-light);
  color: var(--accent); font-size: 0.8rem; font-weight: 600;
  cursor: pointer; transition: all var(--transition); white-space: nowrap;
}
.saved-play-btn:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
.saved-album-tracks {
  margin-top: 12px; padding-top: 12px;
  border-top: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 8px;
}
.saved-track-row {
  display: grid; grid-template-columns: 28px minmax(0, 1fr);
  align-items: center; gap: 8px;
}
.saved-track-no {
  color: var(--text-muted); font-size: 0.78rem;
  text-align: right; font-variant-numeric: tabular-nums;
}
</style>
