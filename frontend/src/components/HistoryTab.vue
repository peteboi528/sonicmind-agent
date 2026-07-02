<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const section = ref("chats");     // "chats" = 最近对话 | "listening" = 听歌记录
const chats = ref([]);
const listening = ref([]);
const loadingChats = ref(false);
const loadingListening = ref(false);
const msg = ref("");

function notify(m) { msg.value = m; }

// ── 最近对话 ──────────────────────────────────────────────────────────────
function threadTitle(t) {
  if (t.title?.trim()) return t.title.trim();
  const firstUser = (t.messages || []).find(m => m.role === "user");
  return firstUser?.content?.slice(0, 40) || "新对话";
}
function threadPreview(t) {
  const msgs = t.messages || [];
  const last = msgs[msgs.length - 1];
  return last?.content?.replace(/\s+/g, " ").slice(0, 90) || "";
}

async function loadChats() {
  loadingChats.value = true;
  try {
    const data = await api.listChatHistory(store.userId);
    chats.value = data.threads || [];
  } catch { chats.value = []; }
  finally { loadingChats.value = false; }
}

function openThread(t) {
  store.navigateTo("chat", t.thread_id);
}

async function clearChats() {
  if (!chats.value.length) return;
  if (!confirm(`确定清空全部 ${chats.value.length} 条历史对话？`)) return;
  try {
    await api.deleteChatHistory(store.userId);
    chats.value = [];
    msg.value = "已清空历史对话。";
  } catch { msg.value = "清空失败，请重试。"; }
}

// ── 听歌记录 ──────────────────────────────────────────────────────────────
function fmtDur(sec) {
  if (sec == null) return "";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}m${s ? s + "s" : ""}`;
}
function listenLabel(item) {
  if (item.completed) return "完整收听";
  if ((item.duration_listened || 0) < 30) return "秒跳";
  return `听了 ${fmtDur(item.duration_listened)}`;
}
function relTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Date.now() - then;
  const m = Math.floor(diff / 60000);
  if (m < 1) return "刚刚";
  if (m < 60) return `${m} 分钟前`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} 小时前`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d} 天前`;
  return new Date(then).toLocaleDateString("zh-CN");
}

async function loadListening() {
  loadingListening.value = true;
  try {
    const data = await api.listListeningHistory(store.userId);
    listening.value = data.items || [];
  } catch { listening.value = []; }
  finally { loadingListening.value = false; }
}

// 点击听歌记录里的一首 → 入播放器。available=False（在线曲已无元数据/已删）不可播。
function playItem(item) {
  if (!item.available || !item.title) return;
  const online = item.source && item.source !== "local";
  store.playAll([{
    title: item.title,
    artist: item.artist || "",
    cover_url: item.cover_url || "",
    source: online ? item.source : "local",
    source_id: online ? (item.source_id || item.asset_id) : "",
    asset_id: online ? "" : item.asset_id,
  }]);
  msg.value = `▶ ${item.title}`;
}

onMounted(() => { loadChats(); loadListening(); });
</script>

<template>
  <div>
    <div class="section-title">历史</div>
    <div class="section-sub">回看最近的对话与听歌记录。</div>

    <!-- 分段开关 -->
    <div class="seg">
      <button :class="['seg-btn', { on: section === 'chats' }]" @click="section = 'chats'">
        最近对话<span v-if="chats.length" class="seg-count">{{ chats.length }}</span>
      </button>
      <button :class="['seg-btn', { on: section === 'listening' }]" @click="section = 'listening'">
        听歌记录<span v-if="listening.length" class="seg-count">{{ listening.length }}</span>
      </button>
    </div>

    <Transition name="toast-slide">
      <div v-if="msg" class="toast">{{ msg }}</div>
    </Transition>

    <!-- ── 最近对话 ── -->
    <template v-if="section === 'chats'">
      <div v-if="loadingChats" class="loading-hint">加载中…</div>
      <div v-else-if="!chats.length" class="empty-hint">还没有历史对话。</div>
      <button v-else class="clear-btn" @click="clearChats">清空全部</button>

      <div v-for="t in chats" :key="t.thread_id" class="row stagger-item" role="button" @click="openThread(t)">
        <div class="row-icon chats-ic">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        </div>
        <div class="row-info">
          <div class="row-title">{{ threadTitle(t) }}</div>
          <div v-if="threadPreview(t)" class="row-preview">{{ threadPreview(t) }}</div>
          <div class="row-meta">
            <span>{{ (t.messages || []).length }} 条消息</span>
            <span class="dot">·</span>
            <span>{{ relTime(t.updated_at || t.created_at) }}</span>
          </div>
        </div>
        <svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
      </div>
    </template>

    <!-- ── 听歌记录 ── -->
    <template v-if="section === 'listening'">
      <div v-if="loadingListening" class="loading-hint">加载中…</div>
      <div v-else-if="!listening.length" class="empty-hint">还没有听歌记录，播放几首试试。</div>

      <div
        v-for="(item, idx) in listening" :key="(item.asset_id || item.source_id || '') + '_' + idx"
        class="row lstn stagger-item"
        :class="{ unplayable: !item.available || !item.title }"
        :role="item.available && item.title ? 'button' : null"
        @click="playItem(item)"
      >
        <div class="cover-wrap">
          <img v-if="item.cover_url" :src="item.cover_url" class="cover" alt="" loading="lazy" />
          <div v-else class="cover-ph">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
          </div>
        </div>
        <div class="row-info">
          <div class="row-title">{{ item.title || "未知曲目" }}</div>
          <div class="row-meta">
            <span class="badge" :class="{ ok: item.completed, skip: !item.completed && (item.duration_listened || 0) < 30 }">
              {{ listenLabel(item) }}
            </span>
            <span class="dot">·</span>
            <span>{{ item.artist || "未知歌手" }}</span>
            <span class="dot">·</span>
            <span>{{ relTime(item.timestamp) }}</span>
          </div>
        </div>
        <svg v-if="item.available && item.title" class="play-ic" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
      </div>
    </template>
  </div>
</template>

<style scoped>
.seg {
  display: inline-flex; gap: 4px; padding: 4px;
  background: var(--bg-elevated); border: 1px solid var(--border);
  border-radius: var(--radius-pill); margin-bottom: 20px;
}
.seg-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 7px 16px; border-radius: var(--radius-pill);
  background: transparent; border: none; cursor: pointer;
  color: var(--text-sub); font-size: 0.82rem; font-weight: 600;
  font-family: var(--font-display);
  transition: all var(--transition);
}
.seg-btn:hover { color: var(--text); }
.seg-btn.on { background: var(--accent-dim); color: var(--accent); }
.seg-count {
  font-size: 0.68rem; padding: 1px 7px; border-radius: var(--radius-pill);
  background: var(--bg-card); color: var(--text-muted);
}
.seg-btn.on .seg-count { background: var(--accent); color: #fff; }

.clear-btn {
  margin-bottom: 14px; padding: 5px 12px; border-radius: var(--radius-pill);
  background: transparent; border: 1px solid var(--border);
  color: var(--text-muted); font-size: 0.76rem; font-weight: 600; cursor: pointer;
  transition: all var(--transition);
}
.clear-btn:hover { color: var(--danger); border-color: rgba(231,76,60,0.3); }

.toast {
  background: var(--accent-dim); color: var(--accent); padding: 10px 14px;
  border-radius: var(--radius-sm); margin-bottom: 16px;
  border: 1px solid rgba(29,185,84,0.12);
}
.toast-slide-enter-active { animation: fadeInUp 0.25s var(--ease-out); }
.toast-slide-leave-active { transition: all 0.15s ease; }
.toast-slide-leave-to { opacity: 0; transform: translateY(-6px); }

.row {
  display: flex; align-items: center; gap: 14px;
  background: var(--bg-card); padding: 14px 18px;
  border-radius: var(--radius); margin-bottom: 8px;
  border: 1px solid var(--border);
  transition: all var(--dur-norm) var(--ease-out);
}
.row[role="button"] { cursor: pointer; }
.row[role="button"]:hover {
  background: var(--bg-hover); border-color: var(--border-light);
  transform: translateY(-1px);
}
.row.lstn.unplayable { opacity: 0.55; }

.row-icon {
  width: 40px; height: 40px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-dim); color: var(--accent);
  border: 1px solid var(--border-light);
}
.cover-wrap {
  width: 44px; height: 44px; border-radius: var(--radius-sm); flex-shrink: 0;
  overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.cover { width: 100%; height: 100%; object-fit: cover; }
.cover-ph {
  width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
  color: var(--text-muted); background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
}

.row-info { flex: 1; min-width: 0; }
.row-title {
  font-family: var(--font-display); font-weight: 600; font-size: 0.92rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.row-preview {
  color: var(--text-sub); font-size: 0.8rem; margin-top: 3px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.row-meta {
  color: var(--text-muted); font-size: 0.76rem; margin-top: 5px;
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.dot { opacity: 0.5; }

.badge {
  padding: 1px 8px; border-radius: var(--radius-pill);
  background: var(--bg-elevated); color: var(--text-sub);
  font-size: 0.68rem; font-weight: 600;
}
.badge.ok { background: var(--accent-dim); color: var(--accent); }
.badge.skip { background: rgba(231,76,60,0.12); color: var(--danger); }

.chev { color: var(--text-muted); flex-shrink: 0; }
.play-ic { color: var(--accent); flex-shrink: 0; }
</style>
