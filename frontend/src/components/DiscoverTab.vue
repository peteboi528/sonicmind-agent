<script setup>
import { ref, onMounted, watch } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

// ── Genre & Mood card definitions ──
const GENRES = [
  { label: "流行", emoji: "🎤", color: "#E91E63" },
  { label: "摇滚", emoji: "🎸", color: "#FF5722" },
  { label: "电子", emoji: "🎛️", color: "#00BCD4" },
  { label: "说唱", emoji: "🎙️", color: "#FF9800" },
  { label: "R&B", emoji: "🎶", color: "#9C27B0" },
  { label: "爵士", emoji: "🎷", color: "#FFC107" },
  { label: "民谣", emoji: "🪕", color: "#8BC34A" },
  { label: "古典", emoji: "🎻", color: "#607D8B" },
  { label: "国风", emoji: "🏮", color: "#F44336" },
  { label: "金属", emoji: "⚡", color: "#455A64" },
];

const MOODS = [
  { label: "放松", emoji: "🌿", color: "#4CAF50" },
  { label: "治愈", emoji: "☀️", color: "#FFEB3B" },
  { label: "运动", emoji: "🏃", color: "#FF5722" },
  { label: "专注", emoji: "🎯", color: "#2196F3" },
  { label: "浪漫", emoji: "💕", color: "#E91E63" },
  { label: "伤感", emoji: "🌧️", color: "#90A4AE" },
  { label: "深夜", emoji: "🌙", color: "#3F51B5" },
  { label: "清晨", emoji: "🌅", color: "#FF9800" },
  { label: "通勤", emoji: "🚇", color: "#009688" },
  { label: "派对", emoji: "🎉", color: "#E040FB" },
];

// ── State ──
const query = ref("");
const loading = ref(false);
const searchResults = ref(null);
const searchMsg = ref("");

const forYou = ref([]);
const forYouLoading = ref(false);

const trending = ref([]);    // [{ name, icon, tracks: [...] }]
const trendingLoading = ref(false);

const browseCategory = ref(null); // { type: "genre"|"mood", value: "摇滚", tracks: [], loading: false }
const browseSeed = ref(0);        // 换一批：递增 seed 让后端轮换关键词/歌单，取不同曲目
const artistInfo = ref(null);     // { name, image, bio, tags, top_albums, top_tracks }
const artistLoading = ref(false);

const toast = ref("");
const toastKey = ref(0);

function showToast(msg) {
  toast.value = msg;
  toastKey.value++;
}

// ── Card mapper ──
function toCard(t) {
  return {
    title: t.title, artist: t.artist || "", source: t.source || "local",
    source_id: t.source_id || t.external_id || "", cover_url: t.cover_url,
    playback_url: t.playback_url,
  };
}

// ── Load "为你推荐" ──
async function loadForYou() {
  forYouLoading.value = true;
  try {
    const data = await api.dailyRecommend(store.userId);
    const tracks = data.tracks || [];
    forYou.value = tracks.slice(0, 8).map(t => {
      const a = t.asset;
      return {
        title: a.title, artist: a.artist || "", source: getattr(a, "source", "local"),
        source_id: getattr(a, "source_id", getattr(a, "external_id", "")),
        cover_url: getattr(a, "cover_url", null),
        playback_url: getattr(a, "playback_url", null),
        reason: t.reason,
      };
    });
  } catch {
    // silently fail — don't block the page
    forYou.value = [];
  } finally {
    forYouLoading.value = false;
  }
}

function getattr(obj, key, fallback = "") {
  return obj[key] !== undefined ? obj[key] : fallback;
}

// ── Load "热门趋势" ──
async function loadTrending() {
  trendingLoading.value = true;
  try {
    const data = await api.discoverTrending(store.userId, 8);
    trending.value = data.charts || [];
  } catch {
    trending.value = [];
  } finally {
    trendingLoading.value = false;
  }
}

// ── Search (enhanced with artist detection) ──
async function search() {
  const q = query.value.trim();
  if (!q) return;
  loading.value = true;
  searchMsg.value = "";
  searchResults.value = null;
  artistInfo.value = null;
  artistLoading.value = true;

  // Fire artist info lookup in parallel with search
  const artistPromise = api.artistInfo(q).catch(() => null);

  try {
    const data = await api.search(store.userId, q);
    searchResults.value = data;
  } catch {
    searchMsg.value = "搜索失败，请稍后重试。";
  } finally {
    loading.value = false;
  }

  // Resolve artist info
  try {
    const info = await artistPromise;
    if (info && info.name && (info.bio || info.top_tracks?.length || info.top_albums?.length)) {
      artistInfo.value = info;
    }
  } catch { /* no artist info, fine */ }
  artistLoading.value = false;
}

// ── Browse genre / mood ──
async function runBrowse(type, value, seed) {
  browseCategory.value = { type, value, tracks: [], loading: true, summary: "" };
  try {
    const data = await api.discoverBrowse(store.userId, type, value, 12, seed);
    browseCategory.value.tracks = (data.tracks || []).map(t => ({
      ...toCard(t),
      reason: t.reason || "",
    }));
    browseCategory.value.summary = data.summary || "";
  } catch {
    browseCategory.value.tracks = [];
    browseCategory.value.summary = "加载失败，点「换一批」重试。";
  } finally {
    browseCategory.value.loading = false;
  }
}

function browse(type, value) {
  browseSeed.value = 0;
  return runBrowse(type, value, 0);
}

function refreshBrowse() {
  if (!browseCategory.value) return;
  browseSeed.value += 1;
  return runBrowse(browseCategory.value.type, browseCategory.value.value, browseSeed.value);
}

function closeBrowse() {
  browseCategory.value = null;
  browseSeed.value = 0;
}

// ── Init ──
onMounted(() => {
  loadForYou();
  loadTrending();
});
</script>

<template>
  <div>
    <!-- ── Header ── -->
    <div class="section-title">发现</div>
    <div class="section-sub">搜索、探索、发现属于你的新音乐。</div>

    <!-- ── Search Bar ── -->
    <div class="search-row">
      <input
        v-model="query"
        class="input search-input"
        placeholder="歌手、歌名、心情、场景…"
        @keyup.enter="search"
      />
      <button class="btn" :disabled="loading || !query.trim()" @click="search">搜索</button>
    </div>

    <!-- ── Search Loading ── -->
    <div v-if="loading" class="loading-hint">
      <div class="loading-dots"><span></span><span></span><span></span></div>
      搜索中…
    </div>
    <div v-if="searchMsg" class="empty-hint">{{ searchMsg }}</div>

    <!-- ── Artist Info Card (on search) ── -->
    <div v-if="artistInfo" class="artist-card stagger-item">
      <div class="artist-header">
        <div class="artist-avatar-wrap">
          <img
            v-if="artistInfo.image"
            class="artist-avatar"
            :src="artistInfo.image"
            alt=""
            loading="lazy"
          />
          <div v-else class="artist-avatar-ph">🎵</div>
        </div>
        <div class="artist-meta">
          <div class="artist-name">{{ artistInfo.name }}</div>
          <div v-if="artistInfo.tags?.length" class="artist-tags">
            <span v-for="tag in artistInfo.tags.slice(0, 5)" :key="tag" class="tag-chip">{{ tag }}</span>
          </div>
          <div v-if="artistInfo.bio" class="artist-bio">{{ artistInfo.bio }}</div>
        </div>
      </div>

      <!-- Top Albums -->
      <div v-if="artistInfo.top_albums?.length" class="artist-albums">
        <div class="artist-section-label">代表专辑</div>
        <div class="album-grid">
          <div v-for="album in artistInfo.top_albums.slice(0, 6)" :key="album.name" class="album-item">
            <div class="album-cover-wrap">
              <img v-if="album.image" class="album-cover" :src="album.image" alt="" loading="lazy" />
              <div v-else class="album-cover-ph">💿</div>
            </div>
            <div class="album-name">{{ album.name }}</div>
          </div>
        </div>
      </div>

      <!-- Top Tracks -->
      <div v-if="artistInfo.top_tracks?.length" class="artist-tracks">
        <div class="artist-section-label">热门歌曲</div>
        <button
          v-if="artistInfo.top_tracks.length > 1"
          class="play-all-btn"
          @click="store.playAll(artistInfo.top_tracks.map(t => toCard(t)))"
        >
          ▶ 全部播放（{{ artistInfo.top_tracks.length }}首）
        </button>
        <div v-for="(t, i) in artistInfo.top_tracks" :key="'at'+i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
          <SongCard :card="toCard(t)" @toast="(m) => showToast(m)" />
        </div>
      </div>
    </div>

    <!-- ── Search Results ── -->
    <template v-if="searchResults">
      <div v-if="searchResults.summary" class="summary">{{ searchResults.summary }}</div>
      <div v-if="searchResults.external?.length" class="group">
        <div class="group-title">全网候选</div>
        <button v-if="searchResults.external.length > 1" class="play-all-btn" @click="store.playAll(searchResults.external.map(toCard))">
          ▶ 全部播放（{{ searchResults.external.length }}首）
        </button>
        <div v-for="(t, i) in searchResults.external" :key="'e'+i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
          <SongCard :card="toCard(t)" @toast="(m) => showToast(m)" />
        </div>
      </div>
      <div v-if="searchResults.local?.length" class="group">
        <div class="group-title">本地库</div>
        <button v-if="searchResults.local.length > 1" class="play-all-btn" @click="store.playAll(searchResults.local.map(toCard))">
          ▶ 全部播放（{{ searchResults.local.length }}首）
        </button>
        <div v-for="(t, i) in searchResults.local" :key="'l'+i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
          <SongCard :card="toCard(t)" @toast="(m) => showToast(m)" />
        </div>
      </div>
      <div v-if="!searchResults.external?.length && !searchResults.local?.length && !artistInfo" class="empty-hint">
        没找到可追溯的结果，换个说法试试。
      </div>
    </template>

    <!-- ══════════════════════════════════════════════ -->
    <!-- Browse Overlay (when genre/mood is selected)   -->
    <!-- ══════════════════════════════════════════════ -->
    <template v-if="browseCategory">
      <div class="browse-header">
        <button class="btn-back" @click="closeBrowse">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
          返回发现
        </button>
        <div class="browse-title">
          {{ browseCategory.type === 'genre' ? '🎸' : '💭' }} {{ browseCategory.value }}
        </div>
        <button class="btn-refresh" :disabled="browseCategory.loading" @click="refreshBrowse" title="换一批不同的">
          🔄 换一批
        </button>
      </div>

      <div v-if="browseCategory.loading" class="loading-hint">
        <div class="loading-dots"><span></span><span></span><span></span></div>
        加载中…
      </div>

      <template v-if="!browseCategory.loading">
        <button
          v-if="browseCategory.tracks.length > 1"
          class="play-all-btn"
          @click="store.playAll(browseCategory.tracks)"
        >
          ▶ 全部播放（{{ browseCategory.tracks.length }}首）
        </button>
        <div v-for="(t, i) in browseCategory.tracks" :key="'b'+i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
          <SongCard :card="t" @toast="(m) => showToast(m)" />
        </div>
        <div v-if="!browseCategory.tracks.length" class="empty-hint">
          {{ browseCategory.summary || '暂无该分类的歌曲，点「换一批」再试试。' }}
        </div>
      </template>
    </template>

    <!-- ══════════════════════════════════════════════ -->
    <!-- Default Home Content (hidden when browsing)    -->
    <!-- ══════════════════════════════════════════════ -->
    <template v-if="!browseCategory && !searchResults && !loading">
      <!-- ── 为你推荐 ── -->
      <section class="discover-section">
        <div class="section-header">
          <span class="section-icon">🎯</span>
          <span class="section-label">为你推荐</span>
          <span class="section-badge">个性化</span>
        </div>

        <div v-if="forYouLoading" class="card-skeleton-grid">
          <div v-for="i in 6" :key="'sk'+i" class="skeleton-card">
            <div class="skeleton-cover shimmer"></div>
            <div class="skeleton-lines">
              <div class="skeleton-line shimmer" style="width:70%"></div>
              <div class="skeleton-line shimmer" style="width:45%"></div>
            </div>
          </div>
        </div>

        <template v-if="!forYouLoading">
          <button v-if="forYou.length > 1" class="play-all-btn" @click="store.playAll(forYou)">
            ▶ 全部播放（{{ forYou.length }}首）
          </button>
          <div v-if="forYou.length" class="song-list">
            <div v-for="(t, i) in forYou" :key="'fy'+i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
              <SongCard :card="t" @toast="(m) => showToast(m)" />
            </div>
          </div>
          <div v-else class="section-empty">暂无推荐，先听几首歌建立口味档案吧。</div>
        </template>
      </section>

      <!-- ── 热门趋势 ── -->
      <section class="discover-section">
        <div class="section-header">
          <span class="section-icon">🔥</span>
          <span class="section-label">热门趋势</span>
        </div>

        <div v-if="trendingLoading" class="card-skeleton-grid">
          <div v-for="i in 4" :key="'tsk'+i" class="skeleton-card">
            <div class="skeleton-cover shimmer"></div>
            <div class="skeleton-lines">
              <div class="skeleton-line shimmer" style="width:65%"></div>
              <div class="skeleton-line shimmer" style="width:40%"></div>
            </div>
          </div>
        </div>

        <template v-if="!trendingLoading">
          <div v-if="trending.length" class="charts-wrap">
            <div v-for="chart in trending" :key="chart.name" class="chart-group">
              <div class="chart-header">
                <span class="chart-icon">{{ chart.icon }}</span>
                <span class="chart-name">{{ chart.name }}</span>
                <button v-if="chart.tracks.length > 1" class="play-all-btn chart-play-all" @click="store.playAll(chart.tracks.map(toCard))">
                  ▶ 播放
                </button>
              </div>
              <div v-for="(t, i) in chart.tracks" :key="chart.name+'_'+i" class="stagger-item" :style="{ animationDelay: `${i * 40}ms` }">
                <SongCard :card="toCard(t)" @toast="(m) => showToast(m)" />
              </div>
            </div>
          </div>
          <div v-else class="section-empty">暂无热门趋势数据。</div>
        </template>
      </section>

      <!-- ── 曲风探索 ── -->
      <section class="discover-section">
        <div class="section-header">
          <span class="section-icon">🎸</span>
          <span class="section-label">曲风探索</span>
        </div>
        <div class="tag-grid">
          <button
            v-for="g in GENRES"
            :key="g.label"
            class="tag-card"
            :style="{ '--tag-color': g.color }"
            @click="browse('genre', g.label)"
          >
            <span class="tag-emoji">{{ g.emoji }}</span>
            <span class="tag-label">{{ g.label }}</span>
          </button>
        </div>
      </section>

      <!-- ── 心情 / 场景 ── -->
      <section class="discover-section">
        <div class="section-header">
          <span class="section-icon">💭</span>
          <span class="section-label">心情 / 场景</span>
        </div>
        <div class="tag-grid">
          <button
            v-for="m in MOODS"
            :key="m.label"
            class="tag-card"
            :style="{ '--tag-color': m.color }"
            @click="browse('mood', m.label)"
          >
            <span class="tag-emoji">{{ m.emoji }}</span>
            <span class="tag-label">{{ m.label }}</span>
          </button>
        </div>
      </section>
    </template>

    <!-- ── Toast ── -->
    <Transition name="toast">
      <div v-if="toast" :key="toastKey" class="discover-toast">{{ toast }}</div>
    </Transition>
  </div>
</template>

<style scoped>
/* ── Search ── */
.search-row {
  display: flex; gap: 10px; margin-bottom: 28px; max-width: 640px;
}
.search-input { flex: 1; }
.summary {
  background: var(--bg-card); padding: 16px 18px;
  border-radius: var(--radius); margin-bottom: 20px;
  line-height: 1.6; border: 1px solid var(--border);
}
.group { margin-bottom: 28px; }
.group-title {
  font-family: var(--font-display);
  font-weight: 700; color: var(--text-sub);
  margin-bottom: 12px; font-size: 0.88rem;
  text-transform: uppercase; letter-spacing: 0.5px;
}
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

/* ── Artist Info Card ── */
.artist-card {
  background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.artist-card::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0;
  height: 120px;
  background: linear-gradient(135deg, rgba(29,185,84,0.08), rgba(100,60,180,0.06));
  pointer-events: none;
}
.artist-header {
  display: flex; gap: 20px; align-items: flex-start;
  position: relative; z-index: 1; margin-bottom: 20px;
}
.artist-avatar-wrap {
  width: 88px; height: 88px; border-radius: 50%;
  flex-shrink: 0; overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  border: 3px solid rgba(255,255,255,0.08);
}
.artist-avatar {
  width: 100%; height: 100%; object-fit: cover;
}
.artist-avatar-ph {
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem;
  background: linear-gradient(135deg, rgba(29,185,84,0.15), rgba(100,60,180,0.12));
}
.artist-meta { flex: 1; min-width: 0; }
.artist-name {
  font-family: var(--font-display);
  font-size: 1.5rem; font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 8px;
}
.artist-tags {
  display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px;
}
.tag-chip {
  display: inline-block; padding: 3px 10px;
  background: var(--accent-dim); color: var(--accent);
  border-radius: var(--radius-pill);
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.02em;
}
.artist-bio {
  color: var(--text-sub); font-size: 0.88rem;
  line-height: 1.6; max-height: 80px;
  overflow: hidden; position: relative;
}
.artist-bio::after {
  content: "";
  position: absolute; bottom: 0; left: 0; right: 0;
  height: 30px;
  background: linear-gradient(transparent, var(--bg-card));
}
.artist-section-label {
  font-family: var(--font-display);
  font-weight: 700; color: var(--text-sub);
  font-size: 0.84rem; text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 12px;
  margin-top: 16px;
}

/* ── Album Grid ── */
.album-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 14px;
}
.album-item { text-align: center; }
.album-cover-wrap {
  width: 100%; aspect-ratio: 1; border-radius: var(--radius-sm);
  overflow: hidden; margin-bottom: 6px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.album-cover {
  width: 100%; height: 100%; object-fit: cover;
  transition: transform 0.3s var(--ease-out);
}
.album-item:hover .album-cover { transform: scale(1.06); }
.album-cover-ph {
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem;
  background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
}
.album-name {
  font-size: 0.78rem; color: var(--text-sub);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Discover Sections ── */
.discover-section {
  margin-bottom: 36px;
}
.section-header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 16px;
}
.section-icon { font-size: 1.2rem; }
.section-label {
  font-family: var(--font-display);
  font-size: 1.15rem; font-weight: 700;
  letter-spacing: -0.01em;
}
.section-badge {
  padding: 2px 10px;
  background: var(--accent-dim); color: var(--accent);
  border-radius: var(--radius-pill);
  font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.02em;
}
.section-empty {
  color: var(--text-muted); font-size: 0.88rem;
  padding: 20px 0; text-align: center;
}

/* ── Tag Card Grid (Genre / Mood) ── */
.tag-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
}
.tag-card {
  display: flex; flex-direction: column;
  align-items: center; gap: 10px;
  padding: 20px 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: all var(--dur-norm) var(--ease-out);
  position: relative; overflow: hidden;
}
.tag-card::before {
  content: "";
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 30%, var(--tag-color), transparent 70%);
  opacity: 0;
  transition: opacity var(--dur-norm) var(--ease-out);
  pointer-events: none;
}
.tag-card:hover {
  border-color: var(--tag-color);
  transform: translateY(-3px);
  box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}
.tag-card:hover::before { opacity: 0.12; }
.tag-card:active { transform: translateY(-1px) scale(0.98); }

.tag-emoji { font-size: 1.8rem; position: relative; z-index: 1; }
.tag-label {
  font-family: var(--font-display);
  font-size: 0.88rem; font-weight: 600;
  position: relative; z-index: 1;
}

/* ── Chart Groups ── */
.charts-wrap {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}
.chart-group {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}
.chart-header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 12px;
}
.chart-icon { font-size: 1.1rem; }
.chart-name {
  font-family: var(--font-display);
  font-size: 0.95rem; font-weight: 700;
  flex: 1;
}
.chart-play-all {
  padding: 5px 14px; font-size: 0.78rem;
}

/* ── Browse Header ── */
.browse-header {
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 20px;
}
.btn-back {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: var(--radius-pill);
  background: var(--bg-card); border: 1px solid var(--border-light);
  color: var(--text-sub); font-size: 0.86rem;
  font-weight: 600; transition: all var(--transition);
}
.btn-back:hover {
  border-color: var(--accent); color: var(--accent);
  background: var(--accent-dim);
}
.btn-refresh {
  margin-left: auto;
  padding: 8px 16px; border-radius: var(--radius-pill);
  background: var(--accent-dim); border: 1px solid var(--border-light);
  color: var(--accent); font-size: 0.84rem;
  font-weight: 600; transition: all var(--transition);
  cursor: pointer;
}
.btn-refresh:hover:not(:disabled) {
  background: var(--accent); color: #fff;
  border-color: var(--accent);
}
.btn-refresh:disabled { opacity: 0.5; cursor: not-allowed; }
.browse-title {
  font-family: var(--font-display);
  font-size: 1.3rem; font-weight: 800;
  letter-spacing: -0.01em;
}

/* ── Skeleton Loading ── */
.card-skeleton-grid {
  display: flex; flex-direction: column; gap: 10px;
}
.skeleton-card {
  display: flex; gap: 14px; align-items: center;
  padding: 12px 16px;
  background: var(--bg-card); border-radius: var(--radius);
}
.skeleton-cover {
  width: 50px; height: 50px; border-radius: var(--radius-sm);
  flex-shrink: 0;
  background: var(--bg-hover);
}
.skeleton-lines { flex: 1; }
.skeleton-line {
  height: 12px; border-radius: 6px;
  background: var(--bg-hover);
  margin-bottom: 8px;
}
.skeleton-line:last-child { margin-bottom: 0; }

/* ── Toast ── */
.discover-toast {
  position: fixed; bottom: 100px; left: 50%;
  transform: translateX(-50%);
  background: var(--bg-elevated);
  color: var(--text); padding: 12px 24px;
  border-radius: var(--radius-pill);
  font-size: 0.88rem; font-weight: 600;
  box-shadow: var(--shadow-lg);
  border: 1px solid var(--border-light);
  z-index: 1000;
  pointer-events: none;
}
.toast-enter-active { transition: all 0.3s var(--ease-spring); }
.toast-leave-active { transition: all 0.2s ease; }
.toast-enter-from { opacity: 0; transform: translateX(-50%) translateY(20px); }
.toast-leave-to { opacity: 0; transform: translateX(-50%) translateY(10px); }

/* ── Responsive ── */
@media (max-width: 768px) {
  .artist-header { flex-direction: column; align-items: center; text-align: center; }
  .artist-tags { justify-content: center; }
  .tag-grid { grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 8px; }
  .tag-card { padding: 14px 8px; }
  .tag-emoji { font-size: 1.4rem; }
  .album-grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 10px; }
}
</style>
