<script setup>
import { ref, reactive, nextTick, watch, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const STORAGE_KEY = `sonicmind_chat_${store.userId}`;
const newThreadId = () => globalThis.crypto?.randomUUID?.() || `thread-${Date.now()}-${Math.random().toString(16).slice(2)}`;

const messages = ref([]);
const input = ref("");
const isStreaming = ref(false);
const thinking = ref("");
const scroller = ref(null);
const savedAlbumIds = ref(new Set());
const history = [];
let msgId = 0;
let abortController = null;
let threadId = newThreadId();

const QUICK = [
  { text: "🎵 推荐几首适合深夜的歌", prompt: "推荐几首适合深夜的歌" },
  { text: "💿 讲讲 Blonde 这张专辑", prompt: "讲讲 Blonde 这张专辑，乐评怎么说？" },
  { text: "🧭 The Weeknd 音乐路线", prompt: "The Weeknd 的音乐路线是什么？" },
  { text: "🧪 推荐点不一样的", prompt: "推荐点不一样的，做个品味实验" },
  { text: "🏃 帮我做 20 首跑步歌单", prompt: "帮我做 20 首跑步歌单" },
];

function scrollDown() {
  nextTick(() => { if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight; });
}

function toast(text) {
  messages.value.push({ id: ++msgId, role: "bot", text, cards: [], albums: [], traceSummary: null, tasteExperiment: null, dossier: null });
  scrollDown();
}

function trackToCard(track) {
  return {
    title: track.title,
    artist: track.artist || "",
    source: track.source || "netease",
    source_id: track.source_id || track.external_id || "",
    cover_url: track.cover_url,
    playback_url: track.playback_url,
  };
}

function cardToExternalTrack(card) {
  return {
    external_id: card.source_id || card.external_id || "",
    title: card.title,
    artist: card.artist || "",
    source: card.source || "netease",
    cover_url: card.cover_url,
    playback_url: card.playback_url,
  };
}

function normalizeAlbumCard(payload) {
  const raw = payload?.album || payload || {};
  const id = raw.id || raw.album_id || "";
  return {
    id,
    name: raw.name || "",
    artist: raw.artist || payload?.artist || "",
    image: raw.image || raw.cover || "",
    track_count: raw.track_count || raw.size || null,
    tracks: Array.isArray(raw.tracks) ? raw.tracks.map(trackToCard) : [],
    loading: false,
    saving: false,
    saved: !!id && savedAlbumIds.value.has(String(id)),
  };
}

// ── Persistence ──
function saveToStorage() {
  try {
    const data = {
      messages: messages.value.map(m => ({
        id: m.id, role: m.role, text: m.text,
        cards: (m.cards || []).map(c => ({
          title: c.title, artist: c.artist, source: c.source,
          source_id: c.source_id, cover_url: c.cover_url,
          playback_url: c.playback_url, reason: c.reason,
        })),
        albums: (m.albums || []).map(a => ({
          id: a.id, name: a.name, artist: a.artist, image: a.image,
          track_count: a.track_count, tracks: a.tracks || [], saved: !!a.saved,
        })),
        artists: m.artists || [],
        dossier: m.dossier || null,
        sampleDossier: m.sampleDossier || null,
        traceSummary: m.traceSummary || null,
        tasteExperiment: m.tasteExperiment || null,
        pendingActions: m.pendingActions || [],
      })),
      history: history.slice(-20),
      msgId,
      threadId,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch { /* quota exceeded etc */ }
}

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data.messages?.length) {
      messages.value = data.messages.map(m => ({
        ...m,
        cards: m.cards || [],
        albums: (m.albums || []).map(a => ({ ...a, loading: false, saving: false })),
        artists: m.artists || [],
        dossier: m.dossier || null,
        sampleDossier: m.sampleDossier || null,
        traceSummary: m.traceSummary || null,
        tasteExperiment: m.tasteExperiment || null,
      }));
      msgId = data.msgId || data.messages.length;
    }
    if (data.history?.length) {
      history.push(...data.history);
    }
    threadId = data.threadId || threadId;
  } catch { /* corrupt data */ }
}

function clearHistory() {
  messages.value = [];
  history.length = 0;
  msgId = 0;
  threadId = newThreadId();
  localStorage.removeItem(STORAGE_KEY);
}

function refreshAlbumSavedState() {
  for (const message of messages.value) {
    for (const album of message.albums || []) {
      album.saved = !!album.id && savedAlbumIds.value.has(String(album.id));
    }
  }
}

async function loadSavedAlbumIds() {
  try {
    const data = await api.listSavedAlbums(store.userId);
    savedAlbumIds.value = new Set((data.albums || []).map(a => String(a.album_id)));
    refreshAlbumSavedState();
  } catch { /* 收藏态失败不影响聊天主流程 */ }
}

async function ensureAlbumTracks(album) {
  if (album.tracks?.length) return album.tracks;
  if (!album.name) return [];
  album.loading = true;
  try {
    const data = await api.artistAlbumTracks(album.artist, album.name, album.id, 100);
    const detail = data.album || {};
    album.id = detail.id || album.id;
    album.name = detail.name || album.name;
    album.artist = detail.artist || album.artist;
    album.image = detail.image || album.image;
    album.track_count = detail.track_count || data.tracks?.length || album.track_count;
    album.tracks = (data.tracks || []).map(trackToCard);
    return album.tracks;
  } catch {
    toast(`专辑《${album.name}》加载失败，稍后重试`);
    return [];
  } finally {
    album.loading = false;
  }
}

async function playAlbum(album) {
  const tracks = await ensureAlbumTracks(album);
  if (!tracks.length) {
    toast(`没找到《${album.name}》的可播放曲目`);
    return;
  }
  store.playAll(tracks);
  toast(`播放《${album.name}》：${tracks.length} 首`);
}

async function toggleAlbumSave(album) {
  if (album.saving) return;
  album.saving = true;
  try {
    if (album.saved && album.id) {
      await api.unsaveAlbum(store.userId, album.id);
      album.saved = false;
      const next = new Set(savedAlbumIds.value);
      next.delete(String(album.id));
      savedAlbumIds.value = next;
      toast(`已取消收藏《${album.name}》`);
      return;
    }

    const tracks = await ensureAlbumTracks(album);
    if (!album.id) {
      toast("这张专辑缺少真实 album_id，暂时不能收藏");
      return;
    }
    await api.saveAlbum(store.userId, {
      album_id: album.id,
      name: album.name,
      artist: album.artist || "",
      image: album.image || "",
      track_count: album.track_count ?? tracks.length,
      tracks: tracks.map(cardToExternalTrack),
    });
    album.saved = true;
    savedAlbumIds.value = new Set(savedAlbumIds.value).add(String(album.id));
    toast(`已收藏《${album.name}》`);
  } catch {
    toast("操作失败，稍后重试");
  } finally {
    album.saving = false;
  }
}

// Auto-save whenever messages change
watch(messages, saveToStorage, { deep: true });

onMounted(() => {
  loadFromStorage();
  loadSavedAlbumIds();
  scrollDown();
});

// ── Send ──
async function send(text) {
  const msg = (text ?? input.value).trim();
  if (!msg || isStreaming.value) return;
  input.value = "";
  messages.value.push({ id: ++msgId, role: "user", text: msg, cards: [], albums: [], tasteExperiment: null });
  history.push({ role: "user", content: msg });
  isStreaming.value = true;
  thinking.value = "思考中...";
  scrollDown();

  // 必须用 reactive：后续 candidates/song_card/final 事件会持续 push/splice botMsg.cards，
  // 若是普通对象，这些改动绕过响应式代理、Vue 检测不到，导致流式阶段只显示第一个
  // candidates 批次（约 5 张），final 的完整列表不刷新——只能靠刷新页面从 storage 重建。
  const botMsg = reactive({ id: ++msgId, role: "bot", text: "", cards: [], albums: [], artists: [], dossier: null, sampleDossier: null, traceSummary: null, tasteExperiment: null, pendingActions: [] });
  let finalText = "";
  abortController = new AbortController();

  try {
    await api.streamChat({ userId: store.userId, threadId, message: msg, history }, {
      onEvent: (event) => {
        if (event.type === "thinking" || event.type === "tool_start" || event.type === "plan") {
          thinking.value = event.content || "思考中...";
        } else if (event.type === "refine") {
          // 空结果恢复回环：换查询/换工具重试。流式过程中让用户看到"正在换个思路重试"。
          thinking.value = event.content ? `重试中：${event.content}` : "换个思路重新检索…";
        } else if (event.type === "eval") {
          // 自省/核对节点：剔除违反约束的候选。仅在思考态提示，不污染最终气泡。
          if (event.content) thinking.value = event.content;
        } else if (event.type === "candidates") {
          for (const c of event.payload?.cards || []) botMsg.cards.push(c);
          if (event.payload?.taste_experiment) botMsg.tasteExperiment = event.payload.taste_experiment;
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          scrollDown();
        } else if (event.type === "song_card") {
          botMsg.cards.push(event.payload || {});
        } else if (event.type === "album_card") {
          botMsg.albums.push(normalizeAlbumCard(event.payload));
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          scrollDown();
        } else if (event.type === "artist_card") {
          botMsg.artists.push(event.payload || {});
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          scrollDown();
        } else if (event.type === "dossier") {
          botMsg.dossier = event.payload?.dossier || event.payload || null;
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          scrollDown();
        } else if (event.type === "sample_relations") {
          botMsg.sampleDossier = event.payload?.sample_dossier || event.payload || null;
          const sourceCards = event.payload?.source_cards || [];
          if (Array.isArray(sourceCards) && sourceCards.length) {
            botMsg.cards.splice(0, botMsg.cards.length, ...sourceCards);
          }
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          scrollDown();
        } else if (event.type === "token") {
          // 真流式：答案正文边生成边追加。首 token 清掉「思考中」，并把气泡入列（chat 等无候选卡片的意图靠这里首次出现）。
          if (thinking.value) thinking.value = "";
          botMsg.text += event.content || "";
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          scrollDown();
        } else if (event.type === "final") {
          finalText = event.content || "";
          const finalCards = event.payload?.cards;
          if (Array.isArray(finalCards)) {
            botMsg.cards.splice(0, botMsg.cards.length, ...finalCards);
            if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          }
          botMsg.traceSummary = event.payload?.trace_summary || null;
          botMsg.tasteExperiment = event.payload?.taste_experiment || botMsg.tasteExperiment;
          botMsg.dossier = event.payload?.dossier || botMsg.dossier;
          botMsg.sampleDossier = event.payload?.sample_dossier || botMsg.sampleDossier;
          if (Array.isArray(event.payload?.artists)) botMsg.artists.splice(0, botMsg.artists.length, ...event.payload.artists);
        } else if (event.type === "confirmation_required") {
          botMsg.pendingActions.push({ ...event.payload, text: event.content, resolved: false });
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
        } else if (event.type === "error") {
          finalText = "⚠️ " + (event.content || "出错了，请重试");
        }
      },
    }, abortController.signal);
    thinking.value = "";
    // final 事件是权威文本（可能经 guard_answer 清理过幻觉歌名），覆盖流式预览。
    botMsg.text = finalText || botMsg.text;
    if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
    history.push({ role: "assistant", content: finalText });
  } catch (err) {
    thinking.value = "";
    if (err.name !== "AbortError") {
      toast("⚠️ 连接失败，请检查服务是否启动。");
    }
  } finally {
    isStreaming.value = false;
    abortController = null;
    scrollDown();
    saveToStorage();
  }
}

async function resolveAction(action, approved) {
  if (action.resolved) return;
  action.resolved = true;
  action.approved = approved;
  try {
    await api.resumeAgent({ userId: store.userId, threadId, actionId: action.action_id, approved }, {
      onEvent: (event) => {
        if (event.type === "tool_result" || event.type === "error") toast(event.content || "操作已处理");
      },
    });
  } catch {
    action.resolved = false;
    toast("确认操作失败，请重试。");
  }
}

function cancelStream() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  isStreaming.value = false;
  thinking.value = "";
}

function onKey(e) {
  if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); send(); }
}
</script>

<template>
  <div class="agent-chat" :class="{ 'player-visible': store.player.visible }">
    <div ref="scroller" class="chat-scroll">

      <!-- ── Welcome State ── -->
      <div v-if="messages.length === 0 && !thinking" class="welcome">
        <div class="welcome-glow"></div>
        <div class="welcome-logo">
          <div class="logo-ring"></div>
          <span class="logo-inner">S</span>
        </div>
        <h1 class="welcome-title">你好，我是 SonicMind</h1>
        <p class="welcome-sub">你的私人音乐 Agent。告诉我心情、场景或歌手，我来找到真实可追溯的音乐。</p>
        <div class="quick-grid">
          <button v-for="(q, i) in QUICK" :key="i" class="quick-card"
            :style="{ animationDelay: `${i * 70}ms` }"
            @click="send(q.prompt)">
            <span class="quick-text">{{ q.text }}</span>
            <svg class="quick-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </button>
        </div>
      </div>

      <!-- ── Messages ── -->
      <div class="messages-wrap">
        <!-- New Chat button when history exists -->
        <div v-if="messages.length > 0" class="history-bar">
          <span class="history-label">{{ messages.length }} 条对话</span>
          <button class="clear-btn" @click="clearHistory">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
            新对话
          </button>
        </div>

        <TransitionGroup name="msg-list">
          <div v-for="m in messages" :key="m.id" class="msg" :class="m.role === 'user' ? 'msg-user' : 'msg-bot'">
            <div class="avatar" v-html="m.role === 'user'
              ? `<svg width='16' height='16' viewBox='0 0 24 24' fill='currentColor'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>`
              : `<svg width='16' height='16' viewBox='0 0 24 24' fill='currentColor'><path d='M12 3v10.55c-.59-.34-1.27-.55-2-.55C7.79 13 6 14.79 6 17s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z'/></svg>`
            "></div>
            <div class="body">
              <div class="role-label">{{ m.role === "user" ? "你" : "SonicMind" }}</div>
              <div v-if="m.text" class="text">{{ m.text }}</div>
              <div v-for="action in m.pendingActions || []" :key="action.action_id" class="confirmation-card">
                <span>{{ action.text || "需要确认账号操作" }}</span>
                <div v-if="!action.resolved">
                  <button @click="resolveAction(action, false)">取消</button>
                  <button class="confirm" @click="resolveAction(action, true)">确认</button>
                </div>
                <small v-else>{{ action.approved ? "已确认" : "已取消" }}</small>
              </div>
              <div v-if="m.tasteExperiment" class="taste-preview">
                <div class="taste-preview-head">
                  <span>Taste Lab</span>
                  <strong>{{ m.tasteExperiment.status || "collecting" }}</strong>
                </div>
                <p>{{ m.tasteExperiment.hypothesis }}</p>
                <div class="taste-preview-buckets">
                  <span v-for="segment in m.tasteExperiment.segments || []" :key="segment.name">
                    {{ segment.label || segment.name }} · {{ (segment.tracks || []).length }}
                  </span>
                </div>
                <small>已保存到“实验室”，可以在那里播放整列、打反馈并生成报告。</small>
              </div>
              <div v-if="m.dossier" class="dossier-card" :class="{ partial: m.dossier.partial }">
                <div class="dossier-head">
                  <div>
                    <span class="dossier-kicker">{{ m.dossier.entity?.type || "music" }}</span>
                    <h3>{{ m.dossier.entity?.name || "音乐档案" }}</h3>
                    <p v-if="m.dossier.entity?.artist">{{ m.dossier.entity.artist }}</p>
                  </div>
                  <img v-if="m.dossier.entity?.image" :src="m.dossier.entity.image" alt="" loading="lazy" />
                  <div v-else class="dossier-disc">◎</div>
                </div>
                <div v-if="m.dossier.partial" class="dossier-warning">
                  资料不完整：{{ m.dossier.degraded_reason || "部分来源在时间预算内未返回" }}
                </div>
                <p v-if="m.dossier.summary" class="dossier-summary">{{ m.dossier.summary }}</p>
                <div v-if="m.dossier.style_tags?.length" class="dossier-tags">
                  <span v-for="tag in m.dossier.style_tags.slice(0, 8)" :key="tag">{{ tag }}</span>
                </div>
                <div v-if="m.dossier.critical_consensus" class="review-consensus">
                  <strong>乐评/资料共识</strong>
                  <p>{{ m.dossier.critical_consensus }}</p>
                </div>
                <div v-if="m.dossier.listening_guide?.length" class="listening-guide">
                  <strong>聆听路线</strong>
                  <ol>
                    <li v-for="item in m.dossier.listening_guide.slice(0, 4)" :key="item">{{ item }}</li>
                  </ol>
                </div>
                <div v-if="m.dossier.citations?.length" class="citation-list">
                  <strong>来源</strong>
                  <a v-for="(c, idx) in m.dossier.citations.slice(0, 6)" :key="(c.url || c.title || c.source || 'citation') + '-' + idx" :href="c.url || '#'" target="_blank" rel="noreferrer">
                    <span>{{ c.kind }}</span>{{ c.title || c.source || c.url }}
                  </a>
                </div>
              </div>
              <div v-if="m.sampleDossier" class="sample-card" :class="{ partial: m.sampleDossier.partial }">
                <div class="sample-head">
                  <span>Sample Trace</span>
                  <strong>{{ m.sampleDossier.target?.title || "采样溯源" }}</strong>
                </div>
                <div v-if="m.sampleDossier.partial" class="sample-warning">
                  资料不完整：{{ m.sampleDossier.degraded_reason || "部分来源未能确认" }}
                </div>
                <div v-if="m.sampleDossier.relations?.length" class="sample-relations">
                  <div v-for="(rel, idx) in m.sampleDossier.relations" :key="idx" class="sample-relation">
                    <div class="sample-type">{{ rel.relation_type || "unknown" }} · {{ Math.round((rel.confidence || 0) * 100) }}%</div>
                    <div class="sample-title">
                      <span>{{ rel.target_track?.title || "目标曲" }}</span>
                      <em>←</em>
                      <strong>{{ rel.source_track?.title || "未知源曲" }}</strong>
                    </div>
                    <p v-if="rel.source_track?.artist">{{ rel.source_track.artist }}</p>
                    <small v-if="rel.note">{{ rel.note }}</small>
                  </div>
                </div>
                <div v-else class="sample-empty">没有找到可核实采样关系，我不会硬编源曲。</div>
                <div v-if="m.sampleDossier.citations?.length" class="citation-list">
                  <strong>采样证据</strong>
                  <a v-for="(c, idx) in m.sampleDossier.citations.slice(0, 6)" :key="(c.url || c.title || c.source || 'sample') + '-' + idx" :href="c.url || '#'" target="_blank" rel="noreferrer">
                    <span>{{ c.source_tier || "C" }} · {{ c.source }}</span>{{ c.title || c.url }}
                  </a>
                </div>
              </div>
              <details v-if="m.traceSummary" class="trace-summary">
                <summary>决策摘要</summary>
                <div class="trace-grid">
                  <span>意图</span><strong>{{ m.traceSummary.intent }}</strong>
                  <span>策略</span><strong>{{ m.traceSummary.strategy }}</strong>
                  <span>工具状态</span><strong>{{ ({ ok: "执行成功", empty: "已执行，0 个候选", error: "执行失败", not_planned: "本轮未规划工具", planned_not_executed: "已规划但未执行" })[m.traceSummary.tool_execution_state] || "未知" }}</strong>
                  <span>已规划</span><strong>{{ (m.traceSummary.tools_planned || []).join(" / ") || "—" }}</strong>
                  <span>已执行</span><strong>{{ (m.traceSummary.tools_executed || m.traceSummary.tools || []).join(" / ") || "—" }}</strong>
                  <span v-if="m.traceSummary.empty_results?.length">空结果</span><strong v-if="m.traceSummary.empty_results?.length">{{ m.traceSummary.empty_results.join(" / ") }}</strong>
                  <span v-if="m.traceSummary.tool_errors?.length">失败</span><strong v-if="m.traceSummary.tool_errors?.length">{{ m.traceSummary.tool_errors.join(" / ") }}</strong>
                  <span v-if="m.traceSummary.tool_error_details?.length">错误详情</span><strong v-if="m.traceSummary.tool_error_details?.length">{{ m.traceSummary.tool_error_details.map(item => `${item.tool}: ${item.message}`).join("；") }}</strong>
                  <span>来源</span><strong>{{ (m.traceSummary.sources || []).join(" / ") || "无候选来源" }}</strong>
                  <span>卡片</span><strong>{{ m.traceSummary.final_cards }}</strong>
                  <span v-if="m.traceSummary.latency_budget">耗时预算</span><strong v-if="m.traceSummary.latency_budget">{{ m.traceSummary.latency_budget.elapsed_seconds }}s / {{ m.traceSummary.latency_budget.budget_seconds }}s<span v-if="m.traceSummary.latency_budget.partial"> · 部分降级</span></strong>
                </div>
              </details>
              <div v-if="m.cards.length" class="cards">
                <button v-if="m.cards.length > 1" class="play-all-btn" @click="store.playAll(m.cards)">
                  ▶ 全部播放（{{ m.cards.length }}首）
                </button>
                <SongCard v-for="(c, j) in m.cards" :key="`${m.id}-${j}`" :card="c" @toast="toast" />
              </div>
              <div v-if="m.artists?.length" class="artist-cards">
                <button v-for="artist in m.artists" :key="artist.name" class="artist-card" @click="send(`介绍 ${artist.name}`)">
                  <strong>{{ artist.name }}</strong>
                  <span>{{ artist.reason || (artist.genres || []).join(" · ") }}</span>
                  <small v-if="artist.representative_tracks?.length">{{ artist.representative_tracks.slice(0, 2).join(" / ") }}</small>
                </button>
              </div>
              <div v-if="m.albums?.length" class="album-cards">
                <div v-for="(album, j) in m.albums" :key="`${m.id}-album-${album.id || album.name}-${j}`" class="album-card">
                  <div class="album-cover-wrap">
                    <img v-if="album.image" class="album-cover" :src="album.image" alt="" loading="lazy" />
                    <div v-else class="album-cover-ph">💿</div>
                  </div>
                  <div class="album-main">
                    <div class="album-title">{{ album.name || "未知专辑" }}</div>
                    <div class="album-meta">{{ album.artist || "未知歌手" }}<span v-if="album.track_count"> · {{ album.track_count }} 首</span></div>
                    <div v-if="album.loading" class="album-track-status">正在按专辑顺序加载曲目...</div>
                    <div v-if="album.tracks?.length" class="album-track-list">
                      <span v-for="(track, idx) in album.tracks" :key="`${album.id}-${track.source_id || track.title}-${idx}`">
                        {{ idx + 1 }}. {{ track.title }}<em v-if="track.artist"> - {{ track.artist }}</em>
                      </span>
                    </div>
                  </div>
                  <div class="album-actions">
                    <button class="album-action primary" :disabled="album.loading" @click="playAlbum(album)">
                      {{ album.loading ? "加载中" : "播放" }}
                    </button>
                    <button class="album-action" :class="{ saved: album.saved }" :disabled="album.saving || album.loading" @click="toggleAlbumSave(album)">
                      {{ album.saved ? "已收藏" : "收藏" }}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </TransitionGroup>

        <!-- ── Thinking ── -->
        <Transition name="thinking">
          <div v-if="thinking" class="msg msg-bot thinking-msg">
            <div class="avatar thinking-avatar" v-html="`<svg width='16' height='16' viewBox='0 0 24 24' fill='currentColor'><path d='M12 3v10.55c-.59-.34-1.27-.55-2-.55C7.79 13 6 14.79 6 17s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z'/></svg>`"></div>
            <div class="body">
              <div class="role-label">SonicMind</div>
              <div class="thinking">
                <span class="thinking-text">{{ thinking }}</span>
                <span class="thinking-dots">
                  <span class="dot"></span>
                  <span class="dot"></span>
                  <span class="dot"></span>
                </span>
              </div>
            </div>
          </div>
        </Transition>
      </div>
    </div>

    <!-- ── Floating Composer ── -->
    <div class="composer-wrap">
      <div class="composer" :class="{ streaming: isStreaming }">
        <textarea
          v-model="input" class="composer-input" rows="1"
          placeholder="描述你想听的音乐…"
          @keydown="onKey" :disabled="isStreaming"
        ></textarea>
        <div class="composer-actions">
          <span class="composer-hint">Ctrl+Enter</span>
          <button v-if="!isStreaming" class="send-btn" :disabled="!input.trim()" @click="send()">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
          </button>
          <button v-else class="stop-btn" @click="cancelStream">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
          </button>
        </div>
      </div>
      <div class="composer-footer">SonicMind 可能会犯错，请核实重要信息。</div>
    </div>
  </div>
</template>

<style scoped>
.agent-chat {
  display: flex; flex-direction: column; height: 100%;
  position: relative;
}

.chat-scroll {
  flex: 1; overflow-y: auto;
  scroll-behavior: smooth;
  padding-bottom: 8px;
  transition: padding-bottom 0.35s var(--ease-out);
}

.agent-chat.player-visible .chat-scroll {
  padding-bottom: calc(var(--player-h) + 18px + env(safe-area-inset-bottom, 0px));
}

/* ── Welcome ── */
.welcome {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; min-height: calc(100dvh - var(--topbar-h) - 180px);
  padding: 60px 24px 40px; position: relative;
  animation: fadeInUp 0.7s var(--ease-out);
}
.welcome-glow {
  position: absolute; top: 10%; left: 50%; transform: translateX(-50%);
  width: 300px; height: 300px; border-radius: 50%;
  background: radial-gradient(circle, rgba(29,185,84,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.welcome-logo {
  position: relative; width: 64px; height: 64px; margin-bottom: 24px;
}
.logo-ring {
  position: absolute; inset: 0; border-radius: 18px;
  border: 2px solid var(--accent);
  animation: ring-breathe 3s ease-in-out infinite;
}
@keyframes ring-breathe {
  0%, 100% { transform: scale(1); opacity: 0.3; }
  50% { transform: scale(1.1); opacity: 0.6; }
}
.logo-inner {
  position: absolute; inset: 4px; border-radius: 14px;
  background: var(--accent-grad);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-display); font-weight: 800;
  font-size: 1.6rem; color: #000;
}
.welcome-title {
  font-family: var(--font-display);
  font-size: 1.8rem; font-weight: 800;
  letter-spacing: -0.03em;
  margin-bottom: 10px;
}
.welcome-sub {
  color: var(--text-sub); font-size: 0.95rem;
  text-align: center; max-width: 480px; line-height: 1.5;
  margin-bottom: 36px;
}
.quick-grid {
  display: grid; grid-template-columns: repeat(2, 1fr);
  gap: 10px; max-width: 520px; width: 100%;
}
.quick-card {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 18px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-sub); font-size: 0.88rem;
  text-align: left;
  transition: all var(--dur-norm) var(--ease-out);
  animation: fadeInUp 0.5s var(--ease-out) both;
}
.quick-card:hover {
  background: var(--bg-hover); color: var(--text);
  border-color: var(--border-light);
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.quick-text { flex: 1; }
.quick-arrow { flex-shrink: 0; opacity: 0.3; transition: all var(--transition); }
.quick-card:hover .quick-arrow { opacity: 0.7; transform: translateX(3px); }

/* ── Messages ── */
.messages-wrap {
  max-width: var(--chat-max);
  margin: 0 auto; padding: 24px 24px 8px;
  width: 100%;
}

/* ── History Bar ── */
.history-bar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 20px; padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
.history-label {
  font-size: 0.78rem; color: var(--text-muted);
  font-family: var(--font-display); font-weight: 600;
}
.clear-btn {
  display: flex; align-items: center; gap: 5px;
  padding: 5px 12px; border-radius: var(--radius-pill);
  font-size: 0.76rem; color: var(--text-muted);
  font-family: var(--font-display); font-weight: 600;
  transition: all var(--transition);
}
.clear-btn:hover { background: var(--bg-hover); color: var(--text); }

.msg {
  display: flex; gap: 14px; margin-bottom: 28px;
  animation: fadeInUp 0.35s var(--ease-out) both;
}
.msg-user { flex-direction: row-reverse; }

.avatar {
  width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; margin-top: 2px;
}
.msg-user .avatar { color: var(--text-sub); }
.msg-bot .avatar { color: var(--accent); }

.body { min-width: 0; flex: 1; }
.role-label {
  font-family: var(--font-display);
  font-size: 0.75rem; font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.04em;
  margin-bottom: 6px;
}
.msg-user .role-label { text-align: right; }

.text {
  font-size: 0.95rem; line-height: 1.7;
  white-space: pre-wrap; color: var(--text);
}
.msg-user .text {
  background: var(--accent-dim);
  padding: 12px 16px;
  border-radius: var(--radius);
  border: 1px solid rgba(29,185,84,0.1);
  display: inline-block;
}

.cards { margin-top: 14px; }

.confirmation-card {
  margin-top: 12px; padding: 12px 14px; max-width: 560px;
  border: 1px solid rgba(245, 166, 35, 0.45); border-radius: var(--radius);
  background: rgba(245, 166, 35, 0.08); color: var(--text);
}
.confirmation-card > div { display: flex; gap: 8px; margin-top: 10px; }
.confirmation-card button {
  padding: 6px 12px; border-radius: var(--radius-pill); background: var(--bg-hover); color: var(--text);
}
.confirmation-card button.confirm { background: var(--accent); color: #07130b; }
.confirmation-card small { display: block; margin-top: 8px; color: var(--text-muted); }

.taste-preview {
  margin-top: 12px;
  width: min(100%, 620px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  padding: 12px 14px;
}

.taste-preview-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: var(--text-muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  font-weight: 800;
}

.taste-preview-head strong {
  color: var(--accent);
  font-size: 0.72rem;
}

.taste-preview p {
  margin: 8px 0 10px;
  color: var(--text-sub);
  font-size: 0.86rem;
  line-height: 1.55;
}

.taste-preview-buckets {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.taste-preview-buckets span {
  padding: 5px 9px;
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  color: var(--text);
  font-size: 0.78rem;
}

.taste-preview small {
  color: var(--text-muted);
  line-height: 1.45;
}

.dossier-card {
  margin-top: 12px;
  width: min(100%, 660px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
  padding: 14px;
}

.dossier-card.partial {
  border-color: rgba(245, 166, 35, 0.35);
}

.dossier-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}

.dossier-kicker {
  color: var(--accent);
  font-size: 0.72rem;
  font-family: var(--font-display);
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.dossier-head h3 {
  font-family: var(--font-display);
  font-size: 1.08rem;
  margin: 3px 0;
}

.dossier-head p {
  color: var(--text-sub);
  font-size: 0.84rem;
}

.dossier-head img,
.dossier-disc {
  width: 58px;
  height: 58px;
  border-radius: 12px;
  object-fit: cover;
  flex-shrink: 0;
  background: var(--bg-elevated);
}

.dossier-disc {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 1.8rem;
}

.dossier-warning {
  margin: 8px 0;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  background: rgba(245, 166, 35, 0.09);
  color: #f5b84d;
  font-size: 0.8rem;
  line-height: 1.45;
}

.dossier-summary,
.review-consensus p {
  color: var(--text-sub);
  font-size: 0.86rem;
  line-height: 1.55;
}

.dossier-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 10px 0;
}

.dossier-tags span {
  padding: 4px 9px;
  border-radius: var(--radius-pill);
  background: var(--accent-dim);
  color: var(--accent);
  font-size: 0.75rem;
  font-weight: 700;
}

.review-consensus,
.listening-guide,
.citation-list {
  margin-top: 12px;
}

.review-consensus strong,
.listening-guide strong,
.citation-list strong {
  display: block;
  margin-bottom: 6px;
  color: var(--text);
  font-family: var(--font-display);
  font-size: 0.83rem;
}

.listening-guide ol {
  margin: 0;
  padding-left: 18px;
  color: var(--text-sub);
  font-size: 0.84rem;
  line-height: 1.55;
}

.citation-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.citation-list a {
  color: var(--text-sub);
  font-size: 0.8rem;
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.citation-list a:hover { color: var(--accent); }

.citation-list span {
  display: inline-block;
  margin-right: 6px;
  color: var(--text-muted);
  font-size: 0.7rem;
  text-transform: uppercase;
}

.sample-card {
  margin-top: 12px;
  width: min(100%, 660px);
  border: 1px solid rgba(120, 160, 255, 0.22);
  border-radius: var(--radius);
  background: linear-gradient(135deg, rgba(80,120,255,0.08), rgba(255,255,255,0.025));
  padding: 14px;
}

.sample-card.partial {
  border-color: rgba(245, 166, 35, 0.35);
}

.sample-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.sample-head span {
  color: #8ea7ff;
  font-size: 0.72rem;
  font-family: var(--font-display);
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.sample-head strong {
  font-family: var(--font-display);
  font-size: 0.96rem;
}

.sample-warning,
.sample-empty {
  margin: 8px 0;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  background: rgba(245, 166, 35, 0.09);
  color: #f5b84d;
  font-size: 0.8rem;
  line-height: 1.45;
}

.sample-relations {
  display: grid;
  gap: 8px;
}

.sample-relation {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.035);
}

.sample-type {
  color: var(--text-muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  font-weight: 800;
  margin-bottom: 5px;
}

.sample-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-sub);
  font-size: 0.9rem;
}

.sample-title strong {
  color: var(--text);
}

.sample-title em {
  color: #8ea7ff;
  font-style: normal;
}

.sample-relation p,
.sample-relation small {
  display: block;
  margin-top: 5px;
  color: var(--text-muted);
  font-size: 0.78rem;
  line-height: 1.45;
}

.trace-summary {
  margin-top: 10px;
  width: min(100%, 520px);
  color: var(--text-muted);
  font-size: 0.76rem;
}

.trace-summary summary {
  cursor: pointer;
  color: var(--text-sub);
  font-family: var(--font-display);
  font-weight: 700;
  list-style: none;
}

.trace-summary summary::-webkit-details-marker { display: none; }

.trace-summary summary::before {
  content: "+";
  display: inline-flex;
  width: 16px;
  height: 16px;
  align-items: center;
  justify-content: center;
  margin-right: 6px;
  border-radius: 50%;
  background: var(--bg-card);
  color: var(--accent);
}

.trace-summary[open] summary::before { content: "-"; }

.trace-grid {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  gap: 6px 12px;
  margin-top: 8px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
}

.trace-grid strong {
  color: var(--text-sub);
  font-weight: 600;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.album-cards {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.artist-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.artist-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 5px;
  padding: 13px 14px;
  text-align: left;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: border-color var(--transition), transform var(--transition);
}

.artist-card:hover { border-color: var(--accent); transform: translateY(-1px); }
.artist-card strong { color: var(--text); font-family: var(--font-display); }
.artist-card span { color: var(--text-sub); font-size: 0.82rem; }
.artist-card small { color: var(--text-muted); font-size: 0.74rem; }

.album-card {
  display: grid;
  grid-template-columns: 64px minmax(0, 1fr) auto;
  align-items: center;
  gap: 12px;
  padding: 10px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.album-cover-wrap {
  width: 64px;
  height: 64px;
  border-radius: 8px;
  overflow: hidden;
  background: var(--bg-hover);
  flex-shrink: 0;
}

.album-cover {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.album-cover-ph {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
}

.album-main { min-width: 0; }

.album-title {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.album-meta {
  margin-top: 3px;
  font-size: 0.78rem;
  color: var(--text-muted);
}

.album-track-list {
  display: grid;
  gap: 4px;
  margin-top: 8px;
  color: var(--text-sub);
  font-size: 0.74rem;
  line-height: 1.3;
  max-height: 190px;
  overflow: auto;
  padding-right: 4px;
}

.album-track-list span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.album-track-list em {
  color: var(--text-muted);
  font-style: normal;
}

.album-track-status {
  margin-top: 8px;
  color: var(--text-muted);
  font-size: 0.74rem;
}

.album-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

.album-action {
  min-width: 58px;
  height: 32px;
  padding: 0 10px;
  border-radius: 8px;
  border: 1px solid var(--border);
  color: var(--text-sub);
  background: rgba(255,255,255,0.03);
  font-size: 0.76rem;
  font-family: var(--font-display);
  font-weight: 700;
  transition: all var(--transition);
}

.album-action:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--border-light);
  background: var(--bg-hover);
}

.album-action.primary,
.album-action.saved {
  background: var(--accent-dim);
  border-color: rgba(29,185,84,0.24);
  color: var(--accent);
}

.album-action:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ── Message list transition ── */
.msg-list-enter-active { animation: fadeInUp 0.35s var(--ease-out); }
.msg-list-leave-active { transition: all 0.2s ease; }
.msg-list-leave-to { opacity: 0; transform: translateX(-10px); }

/* ── Thinking ── */
.thinking {
  display: inline-flex; align-items: center; gap: 10px;
  background: var(--bg-card); padding: 10px 16px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.thinking-text { color: var(--text-sub); font-size: 0.85rem; }
.thinking-dots { display: flex; gap: 4px; }
.thinking-dots .dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--accent);
  animation: dot-bounce 1.4s ease-in-out infinite;
}
.thinking-dots .dot:nth-child(2) { animation-delay: 0.16s; }
.thinking-dots .dot:nth-child(3) { animation-delay: 0.32s; }
.thinking-avatar { animation: pulse-glow 2s ease infinite; }
.thinking-enter-active { animation: fadeInUp 0.3s var(--ease-out); }
.thinking-leave-active { transition: all 0.2s ease; }
.thinking-leave-to { opacity: 0; transform: translateY(-8px); }

/* ── Floating Composer ── */
.composer-wrap {
  max-width: var(--chat-max);
  width: 100%; margin: 0 auto;
  padding: 0 24px 24px;
  flex-shrink: 0;
  transition: padding-bottom 0.35s var(--ease-out);
}

.agent-chat.player-visible .composer-wrap {
  padding-bottom: calc(24px + var(--player-h) + env(safe-area-inset-bottom, 0px));
}
.composer {
  display: flex; align-items: flex-end; gap: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 10px 12px 10px 18px;
  transition: all var(--dur-norm) var(--ease-out);
  box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}
.composer:focus-within {
  border-color: var(--accent);
  box-shadow: 0 4px 24px rgba(0,0,0,0.2), 0 0 0 3px var(--accent-dim);
}
.composer.streaming {
  border-color: rgba(231,76,60,0.3);
}

.composer-input {
  flex: 1; background: transparent; border: none; outline: none;
  color: var(--text); font-size: 0.95rem;
  resize: none; max-height: 120px;
  line-height: 1.5; padding: 4px 0;
}
.composer-input::placeholder { color: var(--text-muted); }

.composer-actions {
  display: flex; align-items: center; gap: 8px; flex-shrink: 0;
}
.composer-hint {
  font-size: 0.65rem; color: var(--text-muted);
  letter-spacing: 0.02em; opacity: 0.5; white-space: nowrap;
}
.composer:focus-within .composer-hint { opacity: 0.8; }

.send-btn {
  width: 36px; height: 36px; border-radius: 50%;
  background: var(--accent); color: #000;
  display: flex; align-items: center; justify-content: center;
  transition: all var(--dur-norm) var(--ease-out); flex-shrink: 0;
}
.send-btn:hover:not(:disabled) { background: var(--accent-hover); transform: scale(1.08); box-shadow: var(--glow-sm); }
.send-btn:disabled { opacity: 0.25; cursor: not-allowed; }

.stop-btn {
  width: 36px; height: 36px; border-radius: 50%;
  background: rgba(231,76,60,0.15); color: #e74c3c;
  display: flex; align-items: center; justify-content: center;
  transition: all var(--dur-norm) var(--ease-out); flex-shrink: 0;
}
.stop-btn:hover { background: rgba(231,76,60,0.25); transform: scale(1.08); }

.composer-footer {
  text-align: center; font-size: 0.7rem;
  color: var(--text-muted); opacity: 0.4; margin-top: 8px;
}

/* ── Responsive ── */
@media (max-width: 768px) {
  .welcome { min-height: calc(100dvh - var(--topbar-h) - 160px); padding: 40px 16px; }
  .quick-grid { grid-template-columns: 1fr; max-width: 100%; }
  .messages-wrap { padding: 16px 12px 8px; }
  .composer-wrap { padding: 0 12px 16px; }
  .agent-chat.player-visible .composer-wrap {
    padding-bottom: calc(16px + var(--player-h) + env(safe-area-inset-bottom, 0px));
  }
  .welcome-title { font-size: 1.5rem; }
  .album-card { grid-template-columns: 52px minmax(0, 1fr); }
  .album-cover-wrap { width: 52px; height: 52px; }
  .album-actions { grid-column: 1 / -1; justify-content: flex-end; }
}
</style>
