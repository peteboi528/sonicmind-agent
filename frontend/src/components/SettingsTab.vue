<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const rules = ref([]);
const taste = ref(null);
const newRule = ref("");
const msg = ref("");
const loading = ref(false);
let toastTimer = null;

function setToast(text) {
  clearTimeout(toastTimer);
  msg.value = text;
  toastTimer = setTimeout(() => { msg.value = ""; }, 2600);
}

async function load() {
  loading.value = true;
  try {
    const [exclData, tasteData] = await Promise.all([
      api.listExclusions(store.userId),
      api.getTaste(store.userId).catch(() => null),
    ]);
    rules.value = exclData.rules || [];
    taste.value = tasteData;
  } catch { setToast("加载失败"); }
  finally { loading.value = false; }
}

async function addRule() {
  const r = newRule.value.trim();
  if (!r) return;
  try {
    const data = await api.addExclusion(store.userId, r);
    rules.value = data.rules || [];
    newRule.value = "";
    setToast(`已添加排除规则：${r}`);
  } catch { setToast("添加失败"); }
}

async function removeRule(rule) {
  try {
    const data = await api.removeExclusion(store.userId, rule);
    rules.value = data.rules || [];
    setToast(`已移除：${rule}`);
  } catch { setToast("移除失败"); }
}

onMounted(load);
</script>

<template>
  <div>
    <div class="section-title">偏好设置</div>
    <div class="section-sub">管理你的音乐偏好和排除规则。</div>

    <!-- 口味档案摘要 -->
    <div v-if="taste" class="card">
      <div class="card-title">你的口味档案</div>
      <div v-if="taste.top_genres?.length" class="tag-group">
        <span class="tag-label">偏好的风格：</span>
        <span v-for="[g, w] in taste.top_genres" :key="g" class="tag">{{ g }} ({{ w.toFixed(1) }})</span>
      </div>
      <div v-if="taste.top_moods?.length" class="tag-group">
        <span class="tag-label">偏好的情绪：</span>
        <span v-for="[m, w] in taste.top_moods" :key="m" class="tag">{{ m }} ({{ w.toFixed(1) }})</span>
      </div>
      <div v-if="!taste.top_genres?.length && !taste.top_moods?.length" class="empty-hint">
        还没有口味数据，多听几首歌后会自动生成。
      </div>
    </div>

    <!-- 排除规则 -->
    <div class="card">
      <div class="card-title">排除规则</div>
      <div class="section-sub">这些关键词会从推荐和搜索结果中过滤掉。你也可以在对话中说"不要抖音热歌"来自动添加。</div>

      <div class="input-row">
        <input v-model="newRule" class="input" placeholder="输入排除词，如：抖音热歌" @keyup.enter="addRule" />
        <button class="btn" :disabled="!newRule.trim()" @click="addRule">添加</button>
      </div>

      <TransitionGroup name="chip-list">
        <div v-if="rules.length" class="chip-row" :key="'row'">
          <span v-for="rule in rules" :key="rule" class="chip">
            {{ rule }}
            <button class="chip-x" @click="removeRule(rule)">×</button>
          </span>
        </div>
      </TransitionGroup>
      <div v-if="!rules.length" class="empty-hint">暂无排除规则。</div>
    </div>

    <div v-if="loading" class="loading-hint">加载中…</div>
    <Transition name="toast-slide">
      <div v-if="msg" class="toast-msg">{{ msg }}</div>
    </Transition>
  </div>
</template>

<style scoped>
.card {
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: 22px;
  margin-bottom: 18px;
  border: 1px solid var(--border);
}
.card-title {
  font-family: var(--font-display);
  font-weight: 700; font-size: 1.05rem;
  margin-bottom: 14px;
}
.tag-group {
  display: flex; flex-wrap: wrap; align-items: center;
  gap: 6px; margin-bottom: 10px;
}
.tag-label { color: var(--text-sub); font-size: 0.85rem; margin-right: 4px; }
.tag {
  display: inline-block; padding: 4px 12px;
  background: var(--accent-dim); color: var(--accent);
  border-radius: var(--radius-pill);
  font-size: 0.78rem; font-family: var(--font-display);
  font-weight: 600;
}
.input-row { display: flex; gap: 10px; margin-bottom: 18px; max-width: 480px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 7px 14px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  font-size: 0.84rem; color: var(--text);
  transition: all var(--transition);
}
.chip:hover { border-color: var(--border-light); }
.chip-x {
  background: none; border: none;
  color: var(--text-muted); font-size: 1.05rem;
  cursor: pointer; padding: 0; line-height: 1;
  transition: all var(--transition);
}
.chip-x:hover { color: var(--danger); transform: scale(1.2); }

/* ── Chip Transition ── */
.chip-list-enter-active { animation: fadeInUp 0.3s var(--ease-out); }
.chip-list-leave-active { transition: all 0.2s ease; }
.chip-list-leave-to { opacity: 0; transform: scale(0.9); }

/* ── Toast ── */
.toast-msg {
  position: fixed;
  bottom: calc(var(--player-h) + 20px);
  left: 50%; transform: translateX(-50%);
  background: var(--bg-glass);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  color: var(--text); padding: 10px 22px;
  border-radius: var(--radius-pill);
  font-size: 0.84rem; border: 1px solid var(--border);
  box-shadow: var(--shadow); z-index: 100;
}
.toast-slide-enter-active { animation: fadeInUp 0.3s var(--ease-out); }
.toast-slide-leave-active { transition: all 0.2s ease; }
.toast-slide-leave-to { opacity: 0; transform: translateX(-50%) translateY(10px); }
</style>
