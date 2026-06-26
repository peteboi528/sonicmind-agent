<script setup>
import { computed, ref, onMounted } from "vue";
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
const externalLoading = ref(false); // 在线源后补的独立加载态：本地结果已铺出，在线仍在跑
const searchResults = ref(null);
const searchMsg = ref("");
const searchIntent = ref(null); // { kind: category|artist|track, label, tags, ... }

const forYou = ref([]);
const forYouLoading = ref(false);
const forYouOnlineOnly = ref(false); // 「仅线上」开关：不推本地曲库

const trending = ref([]);    // [{ name, icon, tracks: [...] }]
const trendingLoading = ref(false);

const browseCategory = ref(null); // { type: "genre"|"mood", value: "摇滚", tracks: [], loading: false }
const browseSeed = ref(0);        // 换一批：递增 seed 让后端轮换关键词/歌单，取不同曲目
const artistInfo = ref(null);     // { name, image, bio, tags, top_albums, top_tracks }
const artistLoading = ref(false);
const bioExpanded = ref(false);   // 歌手简介展开/收起（默认收起，CSS max-height 裁剪）
const albumDetail = ref(null);    // { album, tracks, summary, loading, error, key }
const albumLoadingKey = ref("");
const searchSeq = ref(0);         // 搜索代次令牌：每发起一次新搜索自增，旧搜索的异步回调发现代次不符即丢弃，杜绝 A 覆盖 B
const savedAlbumIds = ref(new Set());  // 已收藏专辑 id 集合，驱动 ♡/♥ 切换态

const toast = ref("");
const toastKey = ref(0);

const searchResultTitle = computed(() => ({
  category: "分类电台",
  artist: "相关歌曲",
  track: "歌曲结果",
}[searchIntent.value?.kind] || "搜索结果"));

const searchIntentIcon = computed(() => ({
  category: "◉",
  artist: "◎",
  track: "♫",
}[searchIntent.value?.kind] || "⌕"));

function showToast(msg) {
  toast.value = msg;
  toastKey.value++;
}

function withTimeout(promise, timeoutMs, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = window.setTimeout(() => reject(new Error(`${label}_timeout`)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => window.clearTimeout(timer));
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
    const data = await api.dailyRecommend(store.userId, undefined, forYouOnlineOnly.value);
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

// ── Search: classify first, then launch only the relevant data path ──
async function search() {
  const q = query.value.trim();
  if (!q) return;
  const seq = ++searchSeq.value;  // 本次搜索的代次令牌
  loading.value = true;
  externalLoading.value = false;
  searchMsg.value = "";
  searchResults.value = null;
  searchIntent.value = null;
  artistInfo.value = null;
  albumDetail.value = null;
  albumLoadingKey.value = "";
  bioExpanded.value = false;
  artistLoading.value = false;

  try {
    const classification = await withTimeout(api.discoverClassify(q), 5000, "classify");
    if (seq !== searchSeq.value) return;
    searchIntent.value = classification;

    if (classification.kind === "category") {
      const data = await withTimeout(api.discoverBrowse(
        store.userId,
        classification.browse_category || "mood",
        classification.browse_value || classification.normalized_query,
        12,
        0,
      ), 12000, "browse");
      if (seq !== searchSeq.value) return;
      searchResults.value = {
        external: data.tracks || [],
        local: [],
        summary: data.summary || `已进入${classification.label}`,
      };
      return;
    }

    const resolvedQuery = classification.kind === "artist"
      ? (classification.normalized_query || q)
      : q;
    // 本地秒出：只读曲库，先把结果铺出来。
    const localPromise = withTimeout(api.discoverSearch(store.userId, resolvedQuery), 10000, "search");
    // 在线后补：独立请求，跑到哪补到哪，永不因慢而被丢弃。本地慢也不拖它。
    externalLoading.value = true;
    withTimeout(api.discoverSearchExternal(store.userId, resolvedQuery), 14000, "search-external")
      .then((ext) => {
        if (seq !== searchSeq.value) return;
        const tracks = ext?.external || [];
        if (!searchResults.value) searchResults.value = { local: [], external: [], summary: "" };
        searchResults.value.external = tracks;
        // 本地为空且在线也空时，给一句在线源的实情，别让结果区一片空白。
        if (!searchResults.value.local?.length && !tracks.length && ext?.summary) {
          searchResults.value.summary = ext.summary;
        }
      })
      .catch(() => { /* 在线源可选，失败不影响本地结果 */ })
      .finally(() => {
        if (seq === searchSeq.value) externalLoading.value = false;
      });
    // 歌手卡：artist 分类必查；track 分类也试探性查一次——曲库里没有歌手数据时，
    // 裸打"周杰伦"会被保守判成 track，但 /artist/info 自带反误判（歌名/活动词不会
    // 匹配成歌手），所以试探是安全的，匹配上才显示，恢复"输歌手名即出歌手卡"的体验。
    if (classification.kind === "artist" || classification.kind === "track") {
      if (classification.kind === "artist") artistLoading.value = true;
      withTimeout(api.artistInfo(classification.normalized_query || q), 35000, "artist").then((info) => {
        if (seq !== searchSeq.value) return;
        if (info?.matched && info.name && (info.bio || info.top_tracks?.length || info.top_albums?.length)) {
          artistInfo.value = info;
          // track 分类下试探命中歌手时，把路由条从"歌曲搜索"提升为"歌手档案"，
          // 否则顶部显示「♫ 歌曲搜索」却又出歌手卡，自相矛盾。
          if (searchIntent.value?.kind === "track") {
            searchIntent.value = { ...searchIntent.value, kind: "artist", label: "歌手档案", normalized_query: info.name };
          }
        }
      }).catch(() => { /* artist detail is optional */ }).finally(() => {
        if (seq === searchSeq.value) artistLoading.value = false;
      });
    }

    const data = await localPromise;
    if (seq !== searchSeq.value) return;
    // 在线结果可能已先到（externalLoading 那条路），合并时保住它，别被本地覆盖。
    const alreadyExternal = searchResults.value?.external || [];
    searchResults.value = {
      local: data.local || [],
      external: data.external?.length ? data.external : alreadyExternal,
      summary: data.summary || searchResults.value?.summary || "",
    };
  } catch (error) {
    if (seq !== searchSeq.value) return;
    searchMsg.value = String(error?.message || "").endsWith("_timeout")
      ? "搜索服务响应较慢，请稍后重试。"
      : "搜索失败，请稍后重试。";
  } finally {
    if (seq === searchSeq.value) {
      loading.value = false;
      if (searchIntent.value?.kind !== "artist") artistLoading.value = false;
    }
  }
}

function clearSearch() {
  searchSeq.value += 1;
  query.value = "";
  externalLoading.value = false;
  searchResults.value = null;
  searchIntent.value = null;
  searchMsg.value = "";
  artistInfo.value = null;
  artistLoading.value = false;
  albumDetail.value = null;
  loading.value = false;
}

// ── Play an album: 加载网易云专辑详情，按专辑原始曲序播放并展示曲目清单 ──
function albumKey(album) {
  return album?.id || `${artistInfo.value?.name || ""}|${album?.name || ""}`;
}

async function playAlbum(album) {
  if (!album?.name || !artistInfo.value?.name) return;
  const key = albumKey(album);
  albumLoadingKey.value = key;
  albumDetail.value = {
    album,
    key,
    tracks: [],
    summary: `正在加载《${album.name}》曲目…`,
    loading: true,
    error: "",
  };
  try {
    const data = await api.artistAlbumTracks(artistInfo.value.name, album.name, album.id, 100);
    const cards = (data.tracks || []).map(toCard);
    albumDetail.value = {
      album: data.album || album,
      key,
      tracks: cards,
      summary: data.summary || "",
      loading: false,
      error: "",
    };
    if (cards.length) {
      store.playAll(cards);
      showToast(`播放《${data.album?.name || album.name}》：${cards.length} 首`);
    } else {
      showToast(`没找到《${album.name}》的可播放曲目`);
    }
  } catch {
    albumDetail.value = {
      album,
      key,
      tracks: [],
      summary: "",
      loading: false,
      error: "专辑加载失败，稍后重试",
    };
    showToast("专辑加载失败，稍后重试");
  } finally {
    albumLoadingKey.value = "";
  }
}

// ── 收藏专辑 ──
function isAlbumSaved(id) {
  return !!id && savedAlbumIds.value.has(id);
}

async function loadSavedAlbumIds() {
  try {
    const data = await api.listSavedAlbums(store.userId);
    savedAlbumIds.value = new Set((data.albums || []).map(a => a.album_id));
  } catch { /* 静默：收藏态加载失败不阻塞主流程 */ }
}

async function toggleSaveAlbum() {
  const album = albumDetail.value?.album;
  const id = album?.id;
  if (!id) return;  // 无真实 album_id（Last.fm 兜底专辑）不可收藏
  const tracks = albumDetail.value?.tracks || [];
  const wasSaved = savedAlbumIds.value.has(id);
  albumDetail.value._saving = true;
  try {
    if (wasSaved) {
      await api.unsaveAlbum(store.userId, id);
    } else {
      await api.saveAlbum(store.userId, {
        album_id: id,
        name: album.name,
        artist: album.artist || artistInfo.value?.name || "",
        image: album.image || "",
        track_count: album.track_count ?? tracks.length,
        tags: artistInfo.value?.tags || [],
        tracks: tracks.map(c => ({
          external_id: c.source_id, title: c.title, artist: c.artist,
          source: c.source, cover_url: c.cover_url, playback_url: c.playback_url,
        })),
      });
    }
    const next = new Set(savedAlbumIds.value);
    if (wasSaved) next.delete(id); else next.add(id);
    savedAlbumIds.value = next;  // 重建 Set 触发响应式
    showToast(wasSaved ? `已取消收藏《${album.name}》` : `已收藏《${album.name}》`);
  } catch {
    showToast("操作失败，稍后重试");
  } finally {
    albumDetail.value._saving = false;
  }
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
  loadSavedAlbumIds();
});
</script>

<template>
  <div class="discover-workbench">
    <!-- ── Header ── -->
    <div class="section-title">探索工作台</div>
    <div class="section-sub">先识别你在找歌曲、歌手还是一种状态，再进入对应的探索路径。</div>

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

    <div v-if="searchIntent" class="query-route" :class="`route-${searchIntent.kind}`">
      <span class="route-glyph">{{ searchIntentIcon }}</span>
      <div class="route-copy">
        <span class="route-kicker">识别为</span>
        <strong>{{ searchIntent.label }}</strong>
        <small v-if="searchIntent.normalized_query">{{ searchIntent.normalized_query }}</small>
      </div>
      <div v-if="searchIntent.tags" class="route-tags">
        <span v-for="tag in [...(searchIntent.tags.genre || []), ...(searchIntent.tags.mood || []), ...(searchIntent.tags.scenario || [])]" :key="tag">
          {{ tag }}
        </span>
      </div>
      <button class="route-close" title="清除搜索" @click="clearSearch">✕</button>
    </div>

    <!-- ── Search Loading ── -->
    <div v-if="loading" class="loading-hint">
      <div class="loading-dots"><span></span><span></span><span></span></div>
      搜索中…
    </div>
    <div v-if="searchMsg" class="empty-hint">{{ searchMsg }}</div>

    <!-- ── Artist Info Skeleton (loading) ── -->
    <div v-if="searchIntent?.kind === 'artist' && artistLoading && !artistInfo" class="artist-card artist-skeleton">
      <div class="artist-header">
        <div class="skeleton-cover shimmer" style="width:88px;height:88px;border-radius:50%;flex-shrink:0"></div>
        <div class="skeleton-lines" style="flex:1">
          <div class="skeleton-line shimmer" style="width:40%;height:20px"></div>
          <div class="skeleton-line shimmer" style="width:70%"></div>
          <div class="skeleton-line shimmer" style="width:55%"></div>
        </div>
      </div>
    </div>

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
          <div v-if="artistInfo.bio" class="artist-bio" :class="{ expanded: bioExpanded }">{{ artistInfo.bio }}</div>
          <button v-if="artistInfo.bio" class="bio-toggle" @click="bioExpanded = !bioExpanded">
            {{ bioExpanded ? '收起' : '展开' }}
          </button>
        </div>
      </div>

      <!-- Top Albums -->
      <div v-if="artistInfo.top_albums?.length" class="artist-albums">
        <div class="artist-section-label">代表专辑 · 按专辑页曲序播放</div>
        <div class="album-grid">
          <div
            v-for="album in artistInfo.top_albums.slice(0, 12)"
            :key="albumKey(album)"
            class="album-item"
            :class="{ active: albumDetail?.key === albumKey(album) }"
            role="button"
            tabindex="0"
            @click="playAlbum(album)"
            @keydown.enter="playAlbum(album)"
          >
            <div class="album-cover-wrap">
              <img v-if="album.image" class="album-cover" :src="album.image" alt="" loading="lazy" />
              <div v-else class="album-cover-ph">💿</div>
              <div v-if="albumLoadingKey === albumKey(album)" class="album-loading">加载中</div>
            </div>
            <div class="album-name">{{ album.name }}</div>
          </div>
        </div>
      </div>

      <!-- Album Tracks -->
      <div v-if="albumDetail" class="album-detail">
        <div class="album-detail-head">
          <div class="album-detail-title">
            专辑曲目：{{ albumDetail.album?.name || '未知专辑' }}
            <span v-if="albumDetail.tracks?.length" class="album-count">{{ albumDetail.tracks.length }} 首</span>
          </div>
          <div class="album-detail-actions">
            <button
              v-if="albumDetail.tracks?.length"
              class="play-all-btn album-play-btn"
              @click="store.playAll(albumDetail.tracks)"
            >
              ▶ 播放专辑
            </button>
            <button
              v-if="albumDetail.album?.id && !albumDetail.loading"
              class="album-save-btn"
              :class="{ saved: isAlbumSaved(albumDetail.album?.id) }"
              :disabled="albumDetail._saving"
              @click="toggleSaveAlbum"
            >
              {{ isAlbumSaved(albumDetail.album?.id) ? '♥ 已收藏' : '♡ 收藏' }}
            </button>
            <button class="album-collapse-btn" @click="albumDetail = null" title="收起曲目">✕</button>
          </div>
        </div>
        <div v-if="albumDetail.loading" class="album-status">正在按专辑顺序加载曲目…</div>
        <div v-else-if="albumDetail.error" class="album-status error">{{ albumDetail.error }}</div>
        <div v-else-if="!albumDetail.tracks?.length" class="album-status">
          {{ albumDetail.summary || '暂时没有拿到这张专辑的曲目。' }}
        </div>
        <div v-else class="album-track-list">
          <div
            v-for="(t, i) in albumDetail.tracks"
            :key="`${t.source_id || t.title}_${i}`"
            class="album-track-row"
          >
            <span class="album-track-no">{{ i + 1 }}</span>
            <SongCard :card="t" :show-reason="false" @toast="(m) => showToast(m)" />
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
      <div v-if="searchResults.local?.length" class="group">
        <div class="group-title"><span>我的曲库匹配</span><em>{{ searchResults.local.length }} local</em></div>
        <button v-if="searchResults.local.length > 1" class="play-all-btn" @click="store.playAll(searchResults.local.map(toCard))">
          ▶ 全部播放（{{ searchResults.local.length }}首）
        </button>
        <div v-for="(t, i) in searchResults.local" :key="'l'+i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
          <SongCard :card="toCard(t)" @toast="(m) => showToast(m)" />
        </div>
      </div>
      <div v-if="searchResults.external?.length" class="group">
        <div class="group-title">
          <span>{{ searchResultTitle }}</span>
          <em>{{ searchResults.external.length }} tracks</em>
        </div>
        <button v-if="searchResults.external.length > 1" class="play-all-btn" @click="store.playAll(searchResults.external.map(toCard))">
          ▶ 全部播放（{{ searchResults.external.length }}首）
        </button>
        <div v-for="(t, i) in searchResults.external" :key="'e'+i" class="stagger-item" :style="{ animationDelay: `${i * 50}ms` }">
          <SongCard :card="toCard(t)" @toast="(m) => showToast(m)" />
        </div>
      </div>
      <!-- 在线源后补：本地已铺出，在线仍在跑时显示进度，到了就替换成结果。 -->
      <div v-if="externalLoading && !searchResults.external?.length" class="external-pending">
        <div class="loading-dots"><span></span><span></span><span></span></div>
        <span>在线来源搜索中，结果会自动补上…</span>
      </div>
      <div v-if="!externalLoading && !searchResults.external?.length && !searchResults.local?.length && !artistInfo" class="empty-hint">
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
          <label class="online-toggle" title="开启后只推线上候选，不混入本地曲库">
            <input type="checkbox" v-model="forYouOnlineOnly" @change="loadForYou" />
            <span>仅线上</span>
          </label>
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
                <span class="chart-title-wrap">
                  <span class="chart-name">{{ chart.name }}</span>
                  <span class="chart-meta">
                    {{ chart.source === 'netease' ? '网易云榜单' : 'Last.fm' }}
                    <template v-if="chart.updated_at"> · {{ new Date(chart.updated_at).toLocaleDateString('zh-CN') }}更新</template>
                  </span>
                </span>
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
.discover-workbench {
  position: relative;
  isolation: isolate;
}
.discover-workbench::before {
  content: "";
  position: absolute;
  z-index: -1;
  top: -54px;
  right: -8vw;
  width: 420px;
  height: 240px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(29,185,84,0.07), transparent 70%);
  pointer-events: none;
}
.search-row {
  display: flex; gap: 10px; margin-bottom: 14px; max-width: 760px;
}
.search-input {
  flex: 1;
  min-height: 48px;
  font-size: 0.98rem;
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025));
}
.query-route {
  --route-color: var(--accent);
  display: grid;
  grid-template-columns: 42px minmax(0, auto) minmax(0, 1fr) 32px;
  align-items: center;
  gap: 12px;
  max-width: 760px;
  margin: 0 0 24px;
  padding: 11px 12px;
  border: 1px solid color-mix(in srgb, var(--route-color) 34%, var(--border));
  border-radius: var(--radius);
  background: linear-gradient(110deg, color-mix(in srgb, var(--route-color) 11%, var(--bg-card)), var(--bg-card) 58%);
  animation: fadeInUp 0.32s var(--ease-out) both;
}
.query-route.route-artist { --route-color: #f2b84b; }
.query-route.route-track { --route-color: #62a8ff; }
.query-route.route-category { --route-color: #46d483; }
.route-glyph {
  width: 42px; height: 42px;
  display: grid; place-items: center;
  border-radius: 50%;
  color: var(--route-color);
  border: 1px solid color-mix(in srgb, var(--route-color) 45%, transparent);
  background: color-mix(in srgb, var(--route-color) 12%, transparent);
  font-size: 1.1rem; font-weight: 900;
}
.route-copy { display: flex; flex-direction: column; min-width: 116px; }
.route-kicker {
  color: var(--text-muted); font-size: 0.62rem; font-weight: 800;
  letter-spacing: 0.12em; text-transform: uppercase;
}
.route-copy strong { color: var(--text); font-size: 0.9rem; }
.route-copy small {
  max-width: 220px; color: var(--text-sub); font-size: 0.72rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.route-tags { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; }
.route-tags span {
  padding: 4px 9px; border-radius: var(--radius-pill);
  color: var(--route-color); background: color-mix(in srgb, var(--route-color) 10%, transparent);
  font-size: 0.7rem; font-weight: 700;
}
.route-close {
  width: 30px; height: 30px; border-radius: 50%;
  color: var(--text-muted); background: transparent;
  transition: all var(--transition);
}
.route-close:hover { color: var(--text); background: var(--bg-hover); }
.summary {
  background: var(--bg-card); padding: 16px 18px;
  border-radius: var(--radius); margin-bottom: 20px;
  line-height: 1.6; border: 1px solid var(--border);
}
.group { margin-bottom: 28px; }
.group-title {
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  font-family: var(--font-display);
  font-weight: 700; color: var(--text-sub);
  margin-bottom: 12px; font-size: 0.88rem;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.group-title em {
  color: var(--text-muted); font-family: var(--font-body);
  font-size: 0.68rem; font-style: normal; font-weight: 600;
  letter-spacing: 0.08em;
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

/* 在线源后补的内联提示：本地已铺出，在线还在跑 */
.external-pending {
  display: flex; align-items: center; gap: 12px;
  color: var(--text-sub); font-size: 13px;
  padding: 14px 4px; margin-top: 4px;
}

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
.artist-bio.expanded { max-height: none; }
.artist-bio.expanded::after { display: none; }
.artist-bio::after {
  content: "";
  position: absolute; bottom: 0; left: 0; right: 0;
  height: 30px;
  background: linear-gradient(transparent, var(--bg-card));
}
.bio-toggle {
  margin-top: 6px; padding: 2px 10px;
  background: none; border: none;
  color: var(--accent); font-size: 0.78rem;
  font-weight: 600; cursor: pointer;
}
.bio-toggle:hover { text-decoration: underline; }
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
.album-item { text-align: center; cursor: pointer; }
.album-cover-wrap {
  width: 100%; aspect-ratio: 1; border-radius: var(--radius-sm);
  overflow: hidden; margin-bottom: 6px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  position: relative;
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
.album-loading {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  background: rgba(0,0,0,0.58);
  color: #fff; font-size: 0.72rem; font-weight: 700;
}
.album-detail {
  margin-top: 18px;
  padding: 16px;
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.album-detail-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; margin-bottom: 12px;
}
.album-detail-title {
  font-family: var(--font-display);
  font-weight: 700; font-size: 0.96rem;
}
.album-count {
  margin-left: 8px; color: var(--text-muted);
  font-family: var(--font-body); font-size: 0.78rem; font-weight: 600;
}
.album-play-btn {
  margin: 0; white-space: nowrap;
}
.album-detail-actions {
  display: flex; align-items: center; gap: 8px;
}
.album-collapse-btn {
  width: 30px; height: 30px; border-radius: 50%;
  background: var(--bg-hover); border: 1px solid var(--border);
  color: var(--text-sub); font-size: 0.9rem; line-height: 1;
  cursor: pointer; transition: all var(--transition);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.album-collapse-btn:hover {
  background: var(--accent-dim); color: var(--accent); border-color: var(--accent);
}
.album-item.active .album-cover-wrap {
  box-shadow: 0 0 0 3px var(--accent), 0 4px 18px rgba(0,0,0,0.4);
}
.album-item.active .album-name {
  color: var(--accent); font-weight: 700;
}
.album-save-btn {
  padding: 6px 14px; border-radius: var(--radius-pill);
  background: var(--bg-hover); border: 1px solid var(--border);
  color: var(--text-sub); font-size: 0.8rem; font-weight: 600;
  cursor: pointer; transition: all var(--transition); white-space: nowrap;
}
.album-save-btn:hover:not(:disabled) {
  border-color: var(--accent); color: var(--accent); background: var(--accent-dim);
}
.album-save-btn.saved {
  background: var(--accent-dim); color: var(--accent); border-color: var(--accent);
}
.album-save-btn:disabled { opacity: 0.5; cursor: progress; }
.album-status {
  color: var(--text-sub); font-size: 0.86rem;
  padding: 12px 0;
}
.album-status.error { color: var(--danger, #ff6b6b); }
.album-track-list {
  display: flex; flex-direction: column; gap: 8px;
}
.album-track-row {
  display: grid; grid-template-columns: 28px minmax(0, 1fr);
  align-items: center; gap: 8px;
}
.album-track-no {
  color: var(--text-muted); font-size: 0.78rem;
  text-align: right; font-variant-numeric: tabular-nums;
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
.online-toggle {
  margin-left: auto; display: inline-flex; align-items: center; gap: 5px;
  color: var(--text-muted); font-size: 0.74rem; cursor: pointer;
  user-select: none; transition: color var(--transition);
}
.online-toggle input { accent-color: var(--accent); width: 13px; height: 13px; cursor: pointer; }
.online-toggle:hover { color: var(--text); }
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
.chart-title-wrap {
  display: flex;
  flex: 1;
  min-width: 0;
  flex-direction: column;
  gap: 2px;
}
.chart-name {
  font-family: var(--font-display);
  font-size: 0.95rem; font-weight: 700;
}
.chart-meta {
  color: var(--text-muted);
  font-size: 0.7rem;
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
  .query-route { grid-template-columns: 38px minmax(0, 1fr) 30px; }
  .route-glyph { width: 38px; height: 38px; }
  .route-tags { grid-column: 1 / -1; justify-content: flex-start; padding-left: 50px; }
  .artist-header { flex-direction: column; align-items: center; text-align: center; }
  .artist-tags { justify-content: center; }
  .tag-grid { grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 8px; }
  .tag-card { padding: 14px 8px; }
  .tag-emoji { font-size: 1.4rem; }
  .album-grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 10px; }
}
</style>
