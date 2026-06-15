<script setup>
import { ref, reactive, nextTick, watch, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const STORAGE_KEY = `sonicmind_chat_${store.userId}`;

const messages = ref([]);
const input = ref("");
const isStreaming = ref(false);
const thinking = ref("");
const scroller = ref(null);
const history = [];
let msgId = 0;
let abortController = null;

const QUICK = [
  { text: "🎵 推荐几首适合深夜的歌", prompt: "推荐几首适合深夜的歌" },
  { text: "🏃 帮我做 20 首跑步歌单", prompt: "帮我做 20 首跑步歌单" },
  { text: "🔍 找 The Weeknd 的歌", prompt: "找 The Weeknd 的歌" },
  { text: "🌅 清晨到深夜的音乐旅程", prompt: "做一个清晨到深夜的音乐旅程" },
];

function scrollDown() {
  nextTick(() => { if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight; });
}

function toast(text) {
  messages.value.push({ id: ++msgId, role: "bot", text, cards: [] });
  scrollDown();
}

// ── Persistence ──
function saveToStorage() {
  try {
    const data = {
      messages: messages.value.map(m => ({
        id: m.id, role: m.role, text: m.text,
        cards: m.cards.map(c => ({
          title: c.title, artist: c.artist, source: c.source,
          source_id: c.source_id, cover_url: c.cover_url,
          playback_url: c.playback_url, reason: c.reason,
        })),
      })),
      history: history.slice(-20),
      msgId,
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
      messages.value = data.messages;
      msgId = data.msgId || data.messages.length;
    }
    if (data.history?.length) {
      history.push(...data.history);
    }
  } catch { /* corrupt data */ }
}

function clearHistory() {
  messages.value = [];
  history.length = 0;
  msgId = 0;
  localStorage.removeItem(STORAGE_KEY);
}

// Auto-save whenever messages change
watch(messages, saveToStorage, { deep: true });

onMounted(() => {
  loadFromStorage();
  scrollDown();
});

// ── Send ──
async function send(text) {
  const msg = (text ?? input.value).trim();
  if (!msg || isStreaming.value) return;
  input.value = "";
  messages.value.push({ id: ++msgId, role: "user", text: msg, cards: [] });
  history.push({ role: "user", content: msg });
  isStreaming.value = true;
  thinking.value = "思考中...";
  scrollDown();

  // 必须用 reactive：后续 candidates/song_card/final 事件会持续 push/splice botMsg.cards，
  // 若是普通对象，这些改动绕过响应式代理、Vue 检测不到，导致流式阶段只显示第一个
  // candidates 批次（约 5 张），final 的完整列表不刷新——只能靠刷新页面从 storage 重建。
  const botMsg = reactive({ id: ++msgId, role: "bot", text: "", cards: [] });
  let finalText = "";
  abortController = new AbortController();

  try {
    await api.streamChat({ userId: store.userId, message: msg, history }, {
      onEvent: (event) => {
        if (event.type === "thinking" || event.type === "tool_start" || event.type === "plan") {
          thinking.value = event.content || "思考中...";
        } else if (event.type === "candidates") {
          for (const c of event.payload?.cards || []) botMsg.cards.push(c);
          if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          scrollDown();
        } else if (event.type === "song_card") {
          botMsg.cards.push(event.payload || {});
        } else if (event.type === "final") {
          finalText = event.content || "";
          const finalCards = event.payload?.cards;
          if (Array.isArray(finalCards)) {
            botMsg.cards.splice(0, botMsg.cards.length, ...finalCards);
            if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
          }
        } else if (event.type === "error") {
          finalText = "⚠️ " + (event.content || "出错了，请重试");
        }
      },
    }, abortController.signal);
    thinking.value = "";
    botMsg.text = finalText;
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
  <div class="agent-chat">
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
              <div v-if="m.cards.length" class="cards">
                <button v-if="m.cards.length > 1" class="play-all-btn" @click="store.playAll(m.cards)">
                  ▶ 全部播放（{{ m.cards.length }}首）
                </button>
                <SongCard v-for="(c, j) in m.cards" :key="`${m.id}-${j}`" :card="c" @toast="toast" />
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
  .welcome-title { font-size: 1.5rem; }
}
</style>
