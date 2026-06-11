<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const rules = ref([]);
const taste = ref(null);
const newRule = ref("");
const msg = ref("");
const loading = ref(false);

async function load() {
  loading.value = true;
  try {
    const [exclData, tasteData] = await Promise.all([
      api.listExclusions(store.userId),
      api.getTaste(store.userId).catch(() => null),
    ]);
    rules.value = exclData.rules || [];
    taste.value = tasteData;
  } catch { msg.value = "加载失败"; }
  finally { loading.value = false; }
}

async function addRule() {
  const r = newRule.value.trim();
  if (!r) return;
  try {
    const data = await api.addExclusion(store.userId, r);
    rules.value = data.rules || [];
    newRule.value = "";
    msg.value = `已添加排除规则：${r}`;
  } catch { msg.value = "添加失败"; }
}

async function removeRule(rule) {
  try {
    const data = await api.removeExclusion(store.userId, rule);
    rules.value = data.rules || [];
    msg.value = `已移除：${rule}`;
  } catch { msg.value = "移除失败"; }
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

      <div v-if="rules.length" class="chip-row">
        <span v-for="rule in rules" :key="rule" class="chip">
          {{ rule }}
          <button class="chip-x" @click="removeRule(rule)">×</button>
        </span>
      </div>
      <div v-else class="empty-hint">暂无排除规则。</div>
    </div>

    <div v-if="loading" class="loading-hint">加载中…</div>
    <div v-if="msg" class="toast-msg">{{ msg }}</div>
  </div>
</template>

<style scoped>
.card {
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 18px;
}
.card-title {
  font-weight: 700;
  font-size: 1.05rem;
  margin-bottom: 12px;
}
.tag-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}
.tag-label {
  color: var(--text-sub);
  font-size: 0.85rem;
  margin-right: 4px;
}
.tag {
  display: inline-block;
  padding: 3px 10px;
  background: var(--accent-dim);
  color: var(--accent);
  border-radius: var(--radius-pill);
  font-size: 0.8rem;
}
.input-row {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
  max-width: 480px;
}
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--bg-elevated);
  border-radius: var(--radius-pill);
  font-size: 0.85rem;
  color: var(--text);
}
.chip-x {
  background: none;
  border: none;
  color: var(--text-muted);
  font-size: 1rem;
  cursor: pointer;
  padding: 0;
  line-height: 1;
  transition: var(--transition);
}
.chip-x:hover {
  color: var(--danger);
}
.toast-msg {
  position: fixed;
  bottom: calc(var(--player-h) + 16px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-elevated);
  color: var(--text);
  padding: 10px 20px;
  border-radius: var(--radius-pill);
  font-size: 0.85rem;
  box-shadow: var(--shadow);
  z-index: 100;
  animation: fadeInOut 2.5s ease forwards;
}
@keyframes fadeInOut {
  0% { opacity: 0; transform: translateX(-50%) translateY(10px); }
  15% { opacity: 1; transform: translateX(-50%) translateY(0); }
  80% { opacity: 1; }
  100% { opacity: 0; }
}
</style>
