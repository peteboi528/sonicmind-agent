<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const assets = ref([]);
const loading = ref(false);
const msg = ref("");
const ingestUrl = ref("");
const ingesting = ref(false);

const savedAlbums = ref([]);       // 收藏的专辑（含完整曲目）
const savedLoading = ref(false);

function notify(m) { msg.value = m; }

// 收藏曲目（ExternalTrack 形）→ 播放卡片
function savedToCard(t) {
  return {
    title: t.title, artist: t.artist || "", source: t.source || "netease",
    source_id: t.external_id || t.source_id || "", cover_url: t.cover_url,
    playback_url: t.playback_url,
  };
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
      for (const r of ratingsData.ratings || []) {
        ratingMap[r.asset_id] = r.score;
      }
      for (const a of assets.value) {
        if (ratingMap[a.asset_id] != null) {
          a._rated = ratingMap[a.asset_id];
        }
      }
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

onMounted(() => {
  load();
  loadSavedAlbums();
});
</script>

<template>
  <div>
    <div class="section-title">我的库</div>
    <div class="section-sub">管理已入库的曲目，评分会实时更新你的口味画像。</div>

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

    <div v-for="(a, idx) in assets" :key="a.asset_id" class="lib-row stagger-item" :style="{ animationDelay: `${idx * 40}ms` }">
      <div class="info">
        <div class="title">{{ a.title || "未命名" }}</div>
        <div class="artist">
          {{ a.artist || "未知" }}
          <span v-for="g in (a.genre || []).slice(0,2)" :key="g" class="tag">{{ g }}</span>
        </div>
      </div>
      <div class="rate">
        <button v-for="s in [2,4,6,8,10]" :key="s"
          class="star" :class="{ on: a._rated >= s }" @click="rate(a, s)">★</button>
      </div>
      <button class="del" @click="remove(a)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.ingest-row { display: flex; gap: 10px; margin-bottom: 20px; max-width: 640px; }
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
.info { flex: 1; min-width: 0; }
.title {
  font-family: var(--font-display);
  font-weight: 600; font-size: 0.93rem;
}
.artist { color: var(--text-sub); font-size: 0.82rem; margin-top: 3px; }
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
