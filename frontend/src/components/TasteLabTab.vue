<script setup>
import { computed, onMounted, ref } from "vue";
import { api } from "../api.js";
import { store } from "../store.js";
import SongCard from "./SongCard.vue";

const prompt = ref("推荐点不一样的，做个品味实验");
const loading = ref(false);
const historyLoading = ref(false);
const experiments = ref([]);
const current = ref(null);
const toast = ref("");
const reportLoading = ref(false);
const regenLoading = ref("");

const bucketTone = {
  safe: "safe",
  stretch: "stretch",
  bold: "bold",
};

const BUCKET_LABEL = { safe: "安全区", stretch: "轻微越界", bold: "大胆探索" };
function bucketLabel(name) {
  return BUCKET_LABEL[name] || name;
}
const STATUS_LABEL = { collecting: "收集中", ready: "就绪", reported: "已出报告" };
const statusLabel = (s) => STATUS_LABEL[s] || s || "";
// 反馈信号 → 中文（用于按钮高亮/提示）
const SIGNAL_LABEL = {
  completed: "听完", liked: "喜欢", skipped: "跳过",
  disliked: "不喜欢", saved: "收藏", rated: "评分",
  too_safe: "太稳", too_far: "太远",
};
const signalLabel = (s) => SIGNAL_LABEL[s] || s || "";
const pct = (x) => `${Math.round((Number(x) || 0) * 100)}%`;

// 每首歌的三锚打分明细：让用户看懂 safe/stretch/bold 是怎么算出来的
function anchorScores(item) {
  const c = item.components || item.track?.components || {};
  return [
    { key: "semantic", label: "语义", value: Number(c.semantic) || 0, help: "跟你的描述有多相关" },
    { key: "personalize", label: "口味", value: Number(c.personalize) || 0, help: "跟你历史画像有多像" },
    { key: "behavior", label: "行为", value: Number(c.behavior) || 0, help: "按你听完/秒跳的反馈" },
  ];
}

const activeTracks = computed(() => {
  if (!current.value) return 0;
  return (current.value.segments || []).reduce((sum, segment) => sum + (segment.tracks || []).length, 0);
});

function showToast(message) {
  toast.value = message;
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => { toast.value = ""; }, 2400);
}

function trackKey(item) {
  const track = item.track || item;
  if (track.source_id) return `${track.source || "netease"}:${track.source_id}`;
  return `title:${(track.title || "").toLowerCase()}:${(track.artist || "").toLowerCase()}`;
}

function itemToCard(item) {
  const track = item.track || item;
  return {
    title: track.title || "",
    artist: track.artist || "",
    source: track.source || "netease",
    source_id: track.source_id || track.external_id || "",
    genre: track.genre || [],
    mood: track.mood || [],
    score: track.score,
    components: item.components || track.components || {},
    reason: item.reason || "",
  };
}

async function loadExperiments() {
  historyLoading.value = true;
  try {
    const data = await api.listTasteExperiments(store.userId);
    experiments.value = data.experiments || [];
    if (!current.value && experiments.value.length) current.value = experiments.value[experiments.value.length - 1];
  } catch {
    showToast("实验历史加载失败");
  } finally {
    historyLoading.value = false;
  }
}

async function generate() {
  const text = prompt.value.trim() || "探索我的口味";
  loading.value = true;
  try {
    current.value = await api.generateTasteExperiment(store.userId, text, 12);
    await loadExperiments();
    showToast("已生成新的品味实验");
  } catch {
    showToast("实验生成失败，请稍后再试");
  } finally {
    loading.value = false;
  }
}

function playSegment(segment) {
  const cards = (segment.tracks || []).map(itemToCard);
  if (!cards.length) return;
  store.playAll(cards);
  showToast(`播放${segment.label || segment.name}：${cards.length} 首`);
}

async function sendFeedback(item, signal) {
  if (!current.value) return;
  try {
    current.value = await api.tasteExperimentFeedback(
      store.userId,
      current.value.experiment_id,
      trackKey(item),
      signal,
    );
    const idx = experiments.value.findIndex(exp => exp.experiment_id === current.value.experiment_id);
    if (idx >= 0) experiments.value[idx] = current.value;
    showToast("反馈已记录");
  } catch {
    showToast("反馈记录失败");
  }
}

async function buildReport() {
  if (!current.value) return;
  reportLoading.value = true;
  try {
    const report = await api.tasteExperimentReport(store.userId, current.value.experiment_id);
    current.value = { ...current.value, report, result_summary: report.summary };
    showToast("实验报告已更新");
    await loadExperiments();
  } catch {
    showToast("报告生成失败");
  } finally {
    reportLoading.value = false;
  }
}

function selectExperiment(exp) {
  current.value = exp;
}

async function regenerateBucket(name) {
  if (!current.value || regenLoading.value) return;
  regenLoading.value = name;
  try {
    current.value = await api.regenerateTasteBucket(store.userId, current.value.experiment_id, name);
    const idx = experiments.value.findIndex(exp => exp.experiment_id === current.value.experiment_id);
    if (idx >= 0) experiments.value[idx] = current.value;
    showToast(`已重做「${bucketLabel(name)}」档`);
  } catch {
    showToast("重做失败，请稍后再试");
  } finally {
    regenLoading.value = "";
  }
}

const deleting = ref(false);
async function deleteExperiment(exp, ev) {
  if (ev) ev.stopPropagation();   // 历史条上的 × 不能触发选中
  if (deleting.value || !exp) return;
  deleting.value = true;
  try {
    await api.deleteTasteExperiment(store.userId, exp.experiment_id);
    experiments.value = experiments.value.filter(e => e.experiment_id !== exp.experiment_id);
    if (current.value?.experiment_id === exp.experiment_id) {
      current.value = experiments.value.length ? experiments.value[experiments.value.length - 1] : null;
    }
    showToast("已删除该实验");
  } catch {
    showToast("删除失败");
  } finally {
    deleting.value = false;
  }
}

onMounted(loadExperiments);
</script>

<template>
  <div class="taste-lab">
    <div class="lab-head">
      <div>
        <p class="eyebrow">Taste Lab</p>
        <h1>品味实验室</h1>
        <p class="subtitle">用安全区、轻微越界和大胆探索三档推荐，验证你真正喜欢什么。</p>
      </div>
      <div class="lab-stats">
        <span>{{ experiments.length }} 次实验</span>
        <strong>{{ activeTracks }} 首候选</strong>
      </div>
    </div>

    <section class="how-to">
      <details>
        <summary>
          <span class="ht-icon">💡</span>
          怎么用品味实验室？
          <span class="ht-hint">（点开看说明）</span>
        </summary>
        <div class="ht-body">
          <ol>
            <li><b>三档候选</b>：<em class="c-safe">安全区</em>＝你大概率喜欢；<em class="c-stretch">轻微越界</em>＝相邻的新风格；<em class="c-bold">大胆探索</em>＝明显越界，但已避开你不喜欢的。</li>
            <li><b>听一首、点一下</b>：每首歌告诉系统你的真实反应——<b>听完 / 喜欢 / 跳过</b>。觉得这档放错了就点 <b>太稳</b>（无聊）或 <b>太远</b>（太怪）。</li>
            <li><b>攒够反馈 → 生成报告</b>：系统会算出哪一档你最买账，并给出下一轮该往哪探索。不满意某一档可以单独<b>重做</b>。</li>
          </ol>
          <p class="ht-legend">
            每首歌下方的
            <span class="legend-chip">语义</span><b>＝跟你的描述有多相关</b>、
            <span class="legend-chip">口味</span><b>＝跟你历史画像有多像</b>、
            <span class="legend-chip">行为</span><b>＝按你听完/秒跳的反馈</b>。
            三档就是按这三个分的综合高低排出来的。
          </p>
        </div>
      </details>
    </section>

    <section class="control-row">
      <div class="generator">
        <input v-model="prompt" type="text" placeholder="例如：推荐点不一样的，别太吵" @keydown.enter="generate" />
        <button class="primary-btn" :disabled="loading" @click="generate">
          <svg v-if="!loading" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          <span>{{ loading ? "生成中" : "生成实验" }}</span>
        </button>
      </div>
      <button class="ghost-btn" :disabled="!current || reportLoading" @click="buildReport">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5V4a2 2 0 0 1 2-2h10l4 4v13.5A2.5 2.5 0 0 1 17.5 22h-11A2.5 2.5 0 0 1 4 19.5z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h6"/></svg>
        <span>{{ reportLoading ? "统计中" : "生成报告" }}</span>
      </button>
    </section>

    <div v-if="toast" class="toast">{{ toast }}</div>

    <section v-if="current" class="exp-toolbar">
      <span class="status-badge" :class="current.status">{{ statusLabel(current.status) }}</span>
      <span class="exp-prompt" :title="current.prompt">{{ current.prompt || "品味实验" }}</span>
      <button class="del-btn" :disabled="deleting" @click="deleteExperiment(current)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>
        <span>{{ deleting ? "删除中" : "删除本轮" }}</span>
      </button>
    </section>

    <section v-if="current" class="experiment-meta">
      <div>
        <span class="meta-label">本轮假设</span>
        <p>{{ current.hypothesis }}</p>
      </div>
      <div v-if="current.result_summary" class="result-summary">
        <span class="meta-label">结果</span>
        <p>{{ current.result_summary }}</p>
      </div>
    </section>

    <section v-if="current" class="bucket-grid">
      <div
        v-for="segment in current.segments"
        :key="segment.name"
        class="bucket"
        :class="bucketTone[segment.name]"
      >
        <div class="bucket-head">
          <div>
            <h2>{{ segment.label }}</h2>
            <p>{{ segment.description }}</p>
          </div>
          <div class="bucket-actions">
            <button class="icon-btn" title="播放这一档" @click="playSegment(segment)">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
            </button>
            <button class="icon-btn" :title="`重做${segment.label}档`" :disabled="!!regenLoading" @click="regenerateBucket(segment.name)">
              <svg v-if="regenLoading === segment.name" class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>
              <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>
            </button>
          </div>
        </div>

        <div class="track-list">
          <div v-for="item in segment.tracks" :key="trackKey(item)" class="experiment-track">
            <SongCard :card="itemToCard(item)" :show-reason="false" @toast="showToast" />
            <div class="anchor-breakdown">
              <span class="ab-title">三锚打分</span>
              <span v-for="a in anchorScores(item)" :key="a.key" class="anchor-chip" :title="`${a.label}：${a.help}`">
                <em>{{ a.label }}</em>
                <i class="bar"><b :style="{ width: pct(a.value) }"></b></i>
                <var>{{ a.value.toFixed(2) }}</var>
              </span>
            </div>
            <div class="feedback-row">
              <span class="fb-label">你的反应</span>
              <button class="fb-btn" :class="{ active: item.feedback?.last_signal === 'completed' }" @click="sendFeedback(item, 'completed')">听完</button>
              <button class="fb-btn" :class="{ active: item.feedback?.last_signal === 'liked' }" @click="sendFeedback(item, 'liked')">喜欢</button>
              <button class="fb-btn" :class="{ active: item.feedback?.last_signal === 'skipped' }" @click="sendFeedback(item, 'skipped')">跳过</button>
              <span class="fb-sep" title="觉得这档放错了？"></span>
              <button class="fb-btn fb-warn" :class="{ active: item.feedback?.last_signal === 'too_safe' }" title="放在这档太无聊/太安全" @click="sendFeedback(item, 'too_safe')">太稳</button>
              <button class="fb-btn fb-warn" :class="{ active: item.feedback?.last_signal === 'too_far' }" title="偏离太远、太怪" @click="sendFeedback(item, 'too_far')">太远</button>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section v-if="current?.report" class="report">
      <h2>实验报告</h2>
      <p>{{ current.report.summary }}</p>
      <div class="report-grid">
        <div v-for="(stats, name) in current.report.bucket_stats" :key="name" class="report-card">
          <span class="rc-name">{{ bucketLabel(name) }}</span>
          <div class="rc-rates">
            <div><strong>{{ pct(stats.liked_rate) }}</strong><em>喜欢</em></div>
            <div><strong>{{ pct(stats.completed_rate) }}</strong><em>听完</em></div>
            <div><strong>{{ pct(stats.skip_rate) }}</strong><em>跳过</em></div>
          </div>
          <p class="rc-n">{{ stats.feedback_count || 0 }} 条反馈 / {{ stats.tracks || 0 }} 首</p>
        </div>
      </div>
      <p class="strategy">{{ current.report.next_recommendation_strategy }}</p>
    </section>

    <aside class="history-strip">
      <div
        v-for="exp in experiments.slice().reverse().slice(0, 8)"
        :key="exp.experiment_id"
        class="history-item"
        :class="{ active: current?.experiment_id === exp.experiment_id }"
      >
        <button class="history-select" @click="selectExperiment(exp)">
          <span>{{ statusLabel(exp.status) }}</span>
          <strong>{{ exp.prompt || "品味实验" }}</strong>
        </button>
        <button class="history-del" title="删除这个实验" @click="deleteExperiment(exp, $event)">✕</button>
      </div>
      <span v-if="historyLoading" class="history-loading">加载中</span>
    </aside>

    <div v-if="!current && !historyLoading" class="empty-state">
      <p>还没有实验。先生成一组候选，听几首以后就能看到口味边界。</p>
    </div>
  </div>
</template>

<style scoped>
.taste-lab {
  min-height: 100%;
  padding: 28px clamp(18px, 4vw, 42px) calc(var(--player-h) + 34px);
  color: var(--text);
}

.lab-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 22px;
}

.eyebrow {
  margin: 0 0 7px;
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: clamp(1.55rem, 3vw, 2.25rem);
  letter-spacing: 0;
}

.subtitle {
  max-width: 640px;
  margin: 8px 0 0;
  color: var(--text-sub);
  line-height: 1.55;
}

.lab-stats {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  color: var(--text-sub);
}

.lab-stats strong { color: var(--text); font-size: 1.1rem; }

.control-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 18px;
}

.generator {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  flex: 1;
  min-width: 0;
}

.generator input {
  width: 100%;
  min-height: 44px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  color: var(--text);
  padding: 0 14px;
  font: inherit;
}

.primary-btn,
.ghost-btn,
.icon-btn {
  border: 0;
  cursor: pointer;
  font: inherit;
}

.primary-btn,
.ghost-btn {
  min-height: 44px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border-radius: var(--radius);
  padding: 0 16px;
  white-space: nowrap;
}

.primary-btn {
  background: var(--accent);
  color: #07120b;
  font-weight: 800;
}

.ghost-btn {
  background: var(--bg-card);
  color: var(--text);
  border: 1px solid var(--border);
}

.primary-btn:disabled,
.ghost-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.toast {
  margin-bottom: 14px;
  color: var(--accent);
  font-weight: 700;
}

.experiment-meta {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.8fr);
  gap: 14px;
  margin-bottom: 18px;
}

.experiment-meta > div,
.report {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  padding: 14px 16px;
}

.meta-label {
  display: block;
  color: var(--text-muted);
  font-size: 0.72rem;
  margin-bottom: 5px;
}

.experiment-meta p,
.report p {
  margin: 0;
  color: var(--text-sub);
  line-height: 1.55;
}

.bucket-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  align-items: start;
}

.bucket {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  overflow: hidden;
}

.bucket.safe { border-top: 3px solid #46d483; }
.bucket.stretch { border-top: 3px solid #4ca8ff; }
.bucket.bold { border-top: 3px solid #ff6b6b; }

.bucket-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  padding: 14px;
  border-bottom: 1px solid var(--border);
}

.bucket-head h2 {
  margin: 0;
  font-size: 1rem;
  letter-spacing: 0;
}

.bucket-head p {
  margin: 5px 0 0;
  color: var(--text-sub);
  font-size: 0.82rem;
  line-height: 1.45;
}

.icon-btn {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  color: var(--text);
  background: var(--bg-elevated);
}

.track-list {
  padding: 10px;
}

.experiment-track {
  padding-bottom: 10px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--border);
}

.experiment-track:last-child {
  margin-bottom: 0;
  border-bottom: 0;
}

.experiment-track :deep(.song-card) {
  margin-bottom: 6px;
  padding: 10px;
  background: var(--bg-elevated);
}

.experiment-track :deep(.actions) {
  opacity: 1;
}

.feedback-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  padding: 0 4px;
}

.fb-label {
  color: var(--text-muted);
  font-size: 0.72rem;
  margin-right: 2px;
}

.fb-btn {
  min-height: 28px;
  padding: 0 12px;
  border-radius: var(--radius-pill);
  background: var(--bg-elevated);
  color: var(--text-sub);
  font-size: 0.76rem;
  font-weight: 600;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all var(--transition);
}

.fb-btn:hover {
  color: var(--accent);
  border-color: var(--border-light);
}

.fb-btn.active {
  background: var(--accent);
  color: #07120b;
  border-color: var(--accent);
}

.fb-btn.fb-warn.active {
  background: #ff6b6b;
  color: #fff;
  border-color: #ff6b6b;
}

.fb-sep {
  width: 1px;
  height: 18px;
  background: var(--border);
  margin: 0 2px;
}

.report {
  margin-top: 18px;
}

.report h2 {
  margin: 0 0 8px;
  font-size: 1rem;
}

.report-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin: 14px 0;
}

.report-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px;
}

.rc-name {
  display: block;
  color: var(--text);
  font-size: 0.82rem;
  font-weight: 700;
  margin-bottom: 8px;
}

.rc-rates {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}

.rc-rates div {
  text-align: center;
}

.rc-rates strong {
  display: block;
  font-size: 1.15rem;
  color: var(--accent);
}

.rc-rates em {
  display: block;
  color: var(--text-muted);
  font-size: 0.68rem;
  font-style: normal;
  margin-top: 2px;
}

.rc-n {
  margin: 8px 0 0;
  color: var(--text-muted);
  font-size: 0.7rem;
}

/* 每首歌的三锚打分明细 */
.anchor-breakdown {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 6px 4px 2px;
}

.ab-title {
  color: var(--text-muted);
  font-size: 0.68rem;
  letter-spacing: 0.02em;
}

.anchor-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.7rem;
  color: var(--text-muted);
}

.anchor-chip em {
  font-style: normal;
  color: var(--text-sub);
  min-width: 22px;
}

.anchor-chip .bar {
  display: inline-block;
  width: 42px;
  height: 4px;
  border-radius: 2px;
  background: var(--bg-elevated);
  overflow: hidden;
}

.anchor-chip .bar b {
  display: block;
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
}

.anchor-chip var {
  font-style: normal;
  color: var(--text-sub);
  font-variant-numeric: tabular-nums;
}

/* 分桶头的播放/重做按钮组 */
.bucket-actions {
  display: flex;
  gap: 6px;
}

.icon-btn.spin {
  animation: lab-spin 0.9s linear infinite;
}

@keyframes lab-spin { to { transform: rotate(360deg); } }

.strategy {
  color: var(--text) !important;
}

/* 怎么用 说明（折叠） */
.how-to {
  margin-bottom: 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
}

.how-to details {
  padding: 0;
}

.how-to summary {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  cursor: pointer;
  font-size: 0.86rem;
  font-weight: 600;
  color: var(--text);
  list-style: none;
}

.how-to summary::-webkit-details-marker { display: none; }
.how-to summary::marker { display: none; }

.ht-icon { font-size: 1rem; }
.ht-hint { color: var(--text-muted); font-weight: 400; font-size: 0.76rem; }

.ht-body {
  padding: 0 16px 14px;
  color: var(--text-sub);
  font-size: 0.82rem;
  line-height: 1.6;
}

.ht-body ol {
  margin: 0 0 10px;
  padding-left: 18px;
}

.ht-body li { margin-bottom: 6px; }

.ht-body em { font-style: normal; font-weight: 700; }
.c-safe { color: #46d483; }
.c-stretch { color: #4ca8ff; }
.c-bold { color: #ff6b6b; }

.ht-legend {
  margin: 8px 0 0;
  padding: 10px 12px;
  background: var(--bg-elevated);
  border-radius: var(--radius-sm);
  font-size: 0.78rem;
}

.legend-chip {
  display: inline-block;
  padding: 1px 7px;
  margin: 0 2px;
  border-radius: var(--radius-pill);
  background: var(--accent-dim);
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 700;
}

/* 本轮实验工具条：状态 + 删除 */
.exp-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 14px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
}

.status-badge {
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  font-size: 0.72rem;
  font-weight: 700;
  background: var(--bg-elevated);
  color: var(--text-sub);
}

.status-badge.collecting { background: rgba(255, 196, 0, 0.14); color: #ffc400; }
.status-badge.ready { background: rgba(70, 212, 131, 0.14); color: #46d483; }
.status-badge.reported { background: var(--accent-dim); color: var(--accent); }

.exp-prompt {
  flex: 1;
  min-width: 120px;
  color: var(--text-sub);
  font-size: 0.84rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.del-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 34px;
  padding: 0 12px;
  border-radius: var(--radius);
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-sub);
  font-size: 0.8rem;
  cursor: pointer;
  transition: all var(--transition);
}

.del-btn:hover:not(:disabled) {
  color: #ff6b6b;
  border-color: #ff6b6b;
}

.del-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.history-strip {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 16px 0 4px;
}

.history-item {
  position: relative;
  min-width: 180px;
  max-width: 240px;
}

.history-select {
  width: 100%;
  min-width: 0;
  text-align: left;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  color: var(--text-sub);
  padding: 10px 12px;
  cursor: pointer;
  font: inherit;
}

.history-item.active .history-select {
  border-color: var(--accent);
  color: var(--text);
}

.history-del {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-elevated);
  color: var(--text-muted);
  font-size: 0.72rem;
  line-height: 1;
  border: 0;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--transition), color var(--transition);
}

.history-item:hover .history-del,
.history-item.active .history-del {
  opacity: 1;
}

.history-del:hover {
  color: #ff6b6b;
}

.history-select span,
.history-loading {
  display: block;
  color: var(--text-muted);
  font-size: 0.7rem;
  margin-bottom: 4px;
}

.history-select strong {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.84rem;
}

.empty-state {
  min-height: 260px;
  display: grid;
  place-items: center;
  color: var(--text-sub);
  text-align: center;
}

@media (max-width: 980px) {
  .lab-head,
  .control-row,
  .experiment-meta {
    grid-template-columns: 1fr;
    flex-direction: column;
    align-items: stretch;
  }

  .lab-stats {
    align-items: flex-start;
  }

  .bucket-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .taste-lab {
    padding-inline: 14px;
  }

  .generator {
    grid-template-columns: 1fr;
  }

  .ghost-btn,
  .primary-btn {
    width: 100%;
  }

  .report-grid {
    grid-template-columns: 1fr;
  }
}
</style>
