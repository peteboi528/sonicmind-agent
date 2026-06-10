<script setup>
import { ref, nextTick } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import SongCard from "./SongCard.vue";

const messages = ref([]); // {role, text, cards: []}
const input = ref("");
const isStreaming = ref(false);
const thinking = ref("");
const scroller = ref(null);
const history = [];

const QUICK = [
  "推荐几首适合深夜的歌",
  "帮我做 20 首跑步歌单",
  "找 The Weeknd 的歌",
  "做一个清晨到深夜的音乐旅程",
];

function scrollDown() {
  nextTick(() => { if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight; });
}

function toast(text) {
  messages.value.push({ role: "bot", text, cards: [] });
  scrollDown();
}

async function send(text) {
  const msg = (text ?? input.value).trim();
  if (!msg || isStreaming.value) return;
  input.value = "";
  messages.value.push({ role: "user", text: msg, cards: [] });
  history.push({ role: "user", content: msg });
  isStreaming.value = true;
  thinking.value = "思考中...";
  scrollDown();

  const botMsg = { role: "bot", text: "", cards: [] };
  let finalText = "";

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
        } else if (event.type === "error") {
          finalText = "⚠️ " + (event.content || "出错了，请重试");
        }
      },
    });
    thinking.value = "";
    botMsg.text = finalText;
    if (!messages.value.includes(botMsg)) messages.value.push(botMsg);
    history.push({ role: "assistant", content: finalText });
  } catch {
    thinking.value = "";
    toast("⚠️ 连接失败，请检查服务是否启动。");
  } finally {
    isStreaming.value = false;
    scrollDown();
  }
}

function onKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
}
</script>

<template>
  <div class="chat-wrap">
    <div ref="scroller" class="chat-scroll">
      <div v-if="messages.length === 0" class="welcome">
        <div class="welcome-title">🎵 你想听点什么？</div>
        <div class="welcome-sub">告诉我心情、场景或歌手，我来找真实可追溯的音乐。</div>
        <div class="quick-row">
          <button v-for="q in QUICK" :key="q" class="quick-chip" @click="send(q)">{{ q }}</button>
        </div>
      </div>

      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role === 'user' ? 'msg-user' : 'msg-bot'">
        <div class="avatar">{{ m.role === "user" ? "👤" : "🎵" }}</div>
        <div class="body">
          <div v-if="m.text" class="text">{{ m.text }}</div>
          <div v-if="m.cards.length" class="cards">
            <SongCard v-for="(c, j) in m.cards" :key="j" :card="c" @toast="toast" />
          </div>
        </div>
      </div>

      <div v-if="thinking" class="msg msg-bot">
        <div class="avatar">🎵</div>
        <div class="body"><div class="thinking">{{ thinking }}</div></div>
      </div>
    </div>

    <div class="input-bar">
      <textarea
        v-model="input" class="input" rows="1" placeholder="说点什么…（Enter 发送，Shift+Enter 换行）"
        @keydown="onKey" :disabled="isStreaming"
      ></textarea>
      <button class="btn" :disabled="isStreaming || !input.trim()" @click="send()">发送</button>
    </div>
  </div>
</template>

<style scoped>
.chat-wrap { display: flex; flex-direction: column; height: 100%; }
.chat-scroll { flex: 1; overflow-y: auto; padding: 24px 0; }
.welcome { text-align: center; padding: 60px 20px; }
.welcome-title { font-size: 1.8rem; font-weight: 800; }
.welcome-sub { color: var(--text-sub); margin: 10px 0 28px; }
.quick-row { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
.quick-chip {
  padding: 10px 18px; background: var(--bg-card); border-radius: var(--radius-pill);
  color: var(--text-sub); font-size: 0.9rem; transition: var(--transition);
}
.quick-chip:hover { background: var(--bg-hover); color: var(--text); }
.msg { display: flex; gap: 12px; margin-bottom: 20px; max-width: 760px; }
.msg-user { flex-direction: row-reverse; margin-left: auto; }
.avatar {
  width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; background: var(--bg-card);
}
.body { min-width: 0; }
.text { background: var(--bg-card); padding: 12px 16px; border-radius: var(--radius); white-space: pre-wrap; line-height: 1.5; }
.msg-user .text { background: var(--accent-dim); }
.cards { margin-top: 10px; }
.thinking { background: var(--bg-card); padding: 12px 16px; border-radius: var(--radius); color: var(--text-sub); }
.input-bar { display: flex; gap: 10px; padding: 16px 0; align-items: flex-end; }
textarea.input { resize: none; max-height: 120px; }
</style>
