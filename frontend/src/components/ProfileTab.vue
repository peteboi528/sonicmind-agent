<script setup>
import { computed, nextTick, onActivated, onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "../api.js";
import { store } from "../store.js";

const loading = ref(false);
const profile = ref(null);
const toast = ref("");
const busyInsight = ref("");

// 排除规则（从「偏好」页并入）——这是画像页原本没有的独有交互功能。
const rules = ref([]);
const newRule = ref("");

function showToast(message) {
  toast.value = message;
  window.clearTimeout(showToast._t);
  showToast._t = window.setTimeout(() => { toast.value = ""; }, 2200);
}

const pct = (x) => `${Math.round((Number(x) || 0) * 100)}%`;

const BAND_LABEL = { high: "高置信", medium: "中置信", low: "低置信" };
const bandLabel = (b) => BAND_LABEL[b] || "";

const RELATION_LABEL = {
  core: "核心艺人", rising: "近期上升", explore: "探索方向",
  occasional: "偶尔喜欢", avoid: "不再推荐",
};
const relationLabel = (r) => RELATION_LABEL[r] || r;

const isEmpty = computed(() => !profile.value || profile.value.is_empty);
const soundDims = computed(() => profile.value?.sound_fingerprint?.dimensions || []);
const moodPoints = computed(() => profile.value?.mood_landscape?.global_points || []);
const scenes = computed(() => profile.value?.scenes || []);
const artistGroups = computed(() => {
  const groups = {};
  for (const a of profile.value?.artists || []) {
    (groups[a.relation_type] ||= []).push(a);
  }
  // 展示顺序：核心 → 上升 → 探索 → 偶尔 → 回避
  const order = ["core", "rising", "explore", "occasional", "avoid"];
  return order.filter(k => groups[k]?.length).map(k => ({ type: k, items: groups[k] }));
});
const insights = computed(() => profile.value?.insights || []);
const discovery = computed(() => profile.value?.discovery_style || {});
const confidence = computed(() => profile.value?.summary?.confidence || 0);

// 情绪地图：valence(明亮↔阴郁) 映射到 X，arousal(平静↔激昂) 映射到 Y。
// SVG 视口 280×280，中心 (140,140)，坐标 [-1,1] → [20,260]。
function moodX(valence) { return 140 + (Number(valence) || 0) * 120; }
function moodY(arousal) { return 140 - (Number(arousal) || 0) * 120; }
function moodR(weight) { return 8 + (Number(weight) || 0) * 18; }

// 置信度环形仪表：r=52，周长 ≈ 326.73；dashoffset = 周长 × (1 - 置信度)。
const RING_R = 52;
const RING_CIRC = 2 * Math.PI * RING_R;
const ringOffset = computed(() => RING_CIRC * (1 - confidence.value));

async function load(force = false) {
  if (loading.value) return;
  if (profile.value && !force) return;
  loading.value = true;
  try {
    // 画像与排除规则并行拉取；排除失败不阻塞画像展示。
    const [prof, excl] = await Promise.all([
      api.getProfile(store.userId),
      api.listExclusions(store.userId).catch(() => ({ rules: [] })),
    ]);
    profile.value = prof;
    rules.value = excl.rules || [];
    nextTick(setupObserver);
  } catch {
    showToast("画像加载失败，请稍后再试");
  } finally {
    loading.value = false;
  }
}

async function addRule() {
  const r = newRule.value.trim();
  if (!r) return;
  try {
    const data = await api.addExclusion(store.userId, r);
    rules.value = data.rules || [];
    newRule.value = "";
    showToast(`已添加排除规则：${r}`);
  } catch { showToast("添加失败"); }
}

async function removeRule(rule) {
  try {
    const data = await api.removeExclusion(store.userId, rule);
    rules.value = data.rules || [];
    showToast(`已移除：${rule}`);
  } catch { showToast("移除失败"); }
}

async function sendFeedback(insight, action) {
  if (busyInsight.value) return;
  busyInsight.value = insight.insight_id;
  try {
    profile.value = await api.profileInsightFeedback(store.userId, insight.insight_id, action);
    const LABEL = { confirm: "已确认准确", reject: "已标记不准确", temporary: "已标记为最近喜欢", reset: "已恢复默认" };
    showToast(LABEL[action] || "反馈已记录");
  } catch {
    showToast("反馈记录失败");
  } finally {
    busyInsight.value = "";
  }
}

const refreshing = ref(false);
async function refresh() {
  refreshing.value = true;
  await load(true);
  refreshing.value = false;
  showToast("画像已刷新");
}

// ── 粘性分节导航 + scroll-spy ──
const navItems = computed(() => {
  if (isEmpty.value) return [];
  const items = [
    { id: "summary", label: "摘要" },
    { id: "control", label: "控制" },
    { id: "sound", label: "声音" },
    { id: "mood", label: "情绪" },
  ];
  if (scenes.value.length) items.push({ id: "scene", label: "场景" });
  if (artistGroups.value.length) items.push({ id: "artist", label: "艺术家" });
  items.push({ id: "discovery", label: "探索" });
  if (insights.value.length) items.push({ id: "insight", label: "理解" });
  return items;
});
const activeNav = ref("summary");
let navObserver = null;

function scrollToSection(id) {
  const el = document.getElementById(id);
  el?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setupObserver() {
  navObserver?.disconnect();
  const root = document.querySelector(".tab-content");
  if (!root) return;
  navObserver = new IntersectionObserver((entries) => {
    const hit = entries
      .filter(e => e.isIntersecting)
      .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (hit) activeNav.value = hit.target.getAttribute("data-nav-target");
  }, { root, rootMargin: "-15% 0px -70% 0px", threshold: 0 });
  document.querySelectorAll("[data-nav-target]").forEach(el => navObserver.observe(el));
}

onMounted(load);
onActivated(load);
onBeforeUnmount(() => navObserver?.disconnect());
</script>

<template>
  <div class="profile-view">
    <!-- Hero -->
    <div class="pf-head">
      <div class="pf-head-text">
        <p class="eyebrow">Your Profile</p>
        <h1>音乐画像</h1>
        <p class="subtitle">不是标签墙，而是可解释、可纠错、能驱动推荐的品味模型。</p>
      </div>
      <div class="pf-head-right">
        <!-- 置信度环形仪表：签名视觉，取代原文字药丸 -->
        <div class="conf-ring-wrap" v-if="profile && !isEmpty" :title="`整体置信度 ${pct(confidence)}`">
          <svg viewBox="0 0 120 120" class="conf-ring">
            <defs>
              <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#1DB954" />
                <stop offset="100%" stop-color="#1a9f8f" />
              </linearGradient>
            </defs>
            <circle cx="60" cy="60" :r="RING_R" class="ring-track" />
            <circle cx="60" cy="60" :r="RING_R" class="ring-fill"
              :stroke-dasharray="RING_CIRC" :stroke-dashoffset="ringOffset"
              transform="rotate(-90 60 60)" />
          </svg>
          <div class="ring-center">
            <span class="ring-pct">{{ pct(confidence) }}</span>
            <span class="ring-cap">置信度</span>
          </div>
        </div>
        <button class="ghost-btn" :disabled="refreshing" @click="refresh">
          <svg :class="{ spin: refreshing }" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>
          <span>{{ refreshing ? "刷新中" : "刷新画像" }}</span>
        </button>
      </div>
    </div>

    <div v-if="toast" class="toast">{{ toast }}</div>

    <div v-if="loading && !profile" class="loading-hint">正在分析你的音乐品味…</div>

    <!-- 空状态：数据不足，引导而非空标签 -->
    <section v-else-if="isEmpty" class="empty-card">
      <h2>SonicMind 还在认识你</h2>
      <p>{{ profile?.empty_hint || "多给一些反馈，我会逐渐理解你的音乐品味。" }}</p>
      <div class="empty-actions">
        <span class="hint-chip">导入歌单</span>
        <span class="hint-chip">对推荐点喜欢 / 不喜欢</span>
        <span class="hint-chip">告诉我你想听什么</span>
        <span class="hint-chip">设定排除规则</span>
      </div>
    </section>

    <template v-else>
      <!-- 粘性分节导航 -->
      <nav class="pf-nav" v-if="navItems.length">
        <button v-for="n in navItems" :key="n.id"
          :class="['nav-pill', { active: activeNav === n.id }]"
          @click="scrollToSection(n.id)">
          {{ n.label }}
        </button>
      </nav>

      <!-- 1. 品味摘要 -->
      <section id="summary" data-nav-target="summary" class="card summary-card nav-target stagger-item">
        <div class="summary-chips">
          <span v-for="c in profile.summary.chips" :key="c" class="chip">{{ c }}</span>
        </div>
        <p class="headline">{{ profile.summary.headline }}</p>
        <ul class="core-prefs">
          <li v-for="(p, i) in profile.summary.core_preferences" :key="i">{{ p }}</li>
        </ul>
        <p class="rec-hint">{{ profile.summary.recommendation_hint }}</p>
      </section>

      <!-- 2. 推荐控制 · 排除规则（从「偏好」页并入） -->
      <section id="control" data-nav-target="control" class="card nav-target stagger-item">
        <h2 class="card-title">推荐控制 · 排除规则</h2>
        <p class="card-sub">这些关键词会从推荐与搜索结果中过滤掉。你也可以在对话里说「不要抖音热歌」自动添加。</p>
        <div class="excl-input">
          <input v-model="newRule" class="input excl-field" placeholder="输入排除词，如：抖音热歌" @keyup.enter="addRule" />
          <button class="excl-add" :disabled="!newRule.trim()" @click="addRule">添加</button>
        </div>
        <TransitionGroup v-if="rules.length" name="excl-pop" tag="div" class="excl-chips">
          <span v-for="rule in rules" :key="rule" class="excl-chip">
            {{ rule }}
            <button class="excl-x" @click="removeRule(rule)" title="移除">×</button>
          </span>
        </TransitionGroup>
        <p v-else class="excl-empty">暂无排除规则——所有风格都参与推荐。</p>
      </section>

      <div class="grid-2">
        <!-- 3. 声音指纹 -->
        <section id="sound" data-nav-target="sound" class="card nav-target stagger-item">
          <h2 class="card-title">声音指纹</h2>
          <p class="card-sub">{{ profile.sound_fingerprint.explanation }}</p>
          <div class="fingerprint">
            <div v-for="d in soundDims" :key="d.key" class="fp-row" :title="d.explanation">
              <span class="fp-label">{{ d.label }}</span>
              <span class="fp-bar"><b :style="{ width: pct(d.value) }"></b></span>
              <var class="fp-val">{{ Math.round(d.value * 100) }}</var>
            </div>
          </div>
        </section>

        <!-- 4. 情绪地图 -->
        <section id="mood" data-nav-target="mood" class="card nav-target stagger-item">
          <h2 class="card-title">情绪地图</h2>
          <p class="card-sub">{{ profile.mood_landscape.summary }}</p>
          <div class="mood-map-wrap" v-if="moodPoints.length">
            <svg viewBox="0 0 280 280" class="mood-map">
              <circle cx="140" cy="140" r="120" class="mood-guide" />
              <circle cx="140" cy="140" r="60" class="mood-guide" />
              <line x1="140" y1="20" x2="140" y2="260" class="axis" />
              <line x1="20" y1="140" x2="260" y2="140" class="axis" />
              <text x="264" y="144" class="axis-label">激昂</text>
              <text x="2" y="144" class="axis-label">平静</text>
              <text x="140" y="14" class="axis-label" text-anchor="middle">明亮</text>
              <text x="140" y="276" class="axis-label" text-anchor="middle">阴郁</text>
              <g v-for="p in moodPoints" :key="p.mood">
                <circle :cx="moodX(p.valence)" :cy="moodY(-p.arousal)" :r="moodR(p.weight)" class="mood-dot" />
                <text :x="moodX(p.valence)" :y="moodY(-p.arousal) + 3" text-anchor="middle" class="mood-text">{{ p.mood }}</text>
              </g>
            </svg>
          </div>
        </section>
      </div>

      <!-- 5. 场景偏好 -->
      <section v-if="scenes.length" id="scene" data-nav-target="scene" class="card nav-target stagger-item">
        <h2 class="card-title">场景偏好</h2>
        <p class="card-sub">不同场景下你的音乐需求不同。</p>
        <div class="scene-grid">
          <div v-for="s in scenes" :key="s.scene" class="scene-card">
            <div class="scene-head">
              <h3>{{ s.label || s.scene }}</h3>
              <span class="conf-mini">{{ pct(s.confidence) }}</span>
            </div>
            <p class="scene-strategy">{{ s.recommendation_strategy }}</p>
            <div class="scene-tags" v-if="s.preferred_genres.length">
              <span class="t-label">偏好</span>
              <span v-for="g in s.preferred_genres" :key="g" class="mini-chip">{{ g }}</span>
            </div>
            <div class="scene-tags avoid" v-if="s.avoid_features.length">
              <span class="t-label">避免</span>
              <span v-for="a in s.avoid_features" :key="a" class="mini-chip avoid">{{ a }}</span>
            </div>
          </div>
        </div>
      </section>

      <!-- 6. 艺术家星系 -->
      <section v-if="artistGroups.length" id="artist" data-nav-target="artist" class="card nav-target stagger-item">
        <h2 class="card-title">艺术家关系</h2>
        <p class="card-sub">展示你和艺人之间的关系，而非简单罗列。</p>
        <div class="artist-groups">
          <div v-for="grp in artistGroups" :key="grp.type" class="artist-group">
            <span class="grp-label" :class="grp.type">{{ relationLabel(grp.type) }}</span>
            <div class="artist-cards">
              <div v-for="a in grp.items" :key="a.artist" class="artist-card" :title="a.reasons.join('；')">
                <strong>{{ a.artist }}</strong>
                <span class="artist-reason">{{ a.reasons[0] }}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 7. 探索倾向 -->
      <section id="discovery" data-nav-target="discovery" class="card nav-target stagger-item">
        <h2 class="card-title">探索倾向 · {{ discovery.label }}</h2>
        <p class="card-sub">{{ discovery.explanation }}</p>
        <div class="discovery-bars">
          <div class="db-row"><span>新鲜度容忍</span><span class="fp-bar"><b :style="{ width: pct(discovery.novelty_tolerance) }"></b></span><var>{{ Math.round((discovery.novelty_tolerance||0)*100) }}</var></div>
          <div class="db-row"><span>主流偏好</span><span class="fp-bar"><b :style="{ width: pct(discovery.mainstream_preference) }"></b></span><var>{{ Math.round((discovery.mainstream_preference||0)*100) }}</var></div>
          <div class="db-row"><span>冷门接受度</span><span class="fp-bar"><b :style="{ width: pct(discovery.niche_openness) }"></b></span><var>{{ Math.round((discovery.niche_openness||0)*100) }}</var></div>
          <div class="db-row"><span>语言开放度</span><span class="fp-bar"><b :style="{ width: pct(discovery.language_openness) }"></b></span><var>{{ Math.round((discovery.language_openness||0)*100) }}</var></div>
        </div>
      </section>

      <!-- 8. 画像置信度与纠错 -->
      <section v-if="insights.length" id="insight" data-nav-target="insight" class="card nav-target stagger-item">
        <h2 class="card-title">系统对你的理解</h2>
        <p class="card-sub">每条判断都标了置信度，理解错了可以纠正——纠正会影响后续推荐。</p>
        <div class="insight-list">
          <div v-for="ins in insights" :key="ins.insight_id" class="insight" :class="[`band-${ins.confidence_band}`, { rejected: ins.status === 'rejected' }]">
            <div class="ins-top">
              <span class="ins-band" :class="ins.confidence_band">{{ bandLabel(ins.confidence_band) }}</span>
              <strong class="ins-title">{{ ins.title }}</strong>
              <span v-if="ins.status !== 'active'" class="ins-status" :class="ins.status">
                {{ { confirmed: "已确认", rejected: "已忽略", temporary: "最近喜欢" }[ins.status] }}
              </span>
            </div>
            <p class="ins-exp">{{ ins.explanation }}</p>
            <ul class="ins-evidence" v-if="ins.evidence.length">
              <li v-for="(e, i) in ins.evidence" :key="i">{{ e }}</li>
            </ul>
            <div class="ins-actions">
              <button class="fb-btn" :class="{ active: ins.status === 'confirmed' }" :disabled="busyInsight === ins.insight_id" @click="sendFeedback(ins, 'confirm')">准确</button>
              <button class="fb-btn warn" :class="{ active: ins.status === 'rejected' }" :disabled="busyInsight === ins.insight_id" @click="sendFeedback(ins, 'reject')">不准确</button>
              <button class="fb-btn" :class="{ active: ins.status === 'temporary' }" :disabled="busyInsight === ins.insight_id" @click="sendFeedback(ins, 'temporary')">只是最近</button>
              <button v-if="ins.status !== 'active'" class="fb-btn ghost" :disabled="busyInsight === ins.insight_id" @click="sendFeedback(ins, 'reset')">撤销</button>
            </div>
          </div>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
.profile-view {
  min-height: 100%;
  padding: 28px clamp(18px, 4vw, 42px) calc(var(--player-h) + 34px);
  color: var(--text);
}

/* ── Hero ── */
.pf-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 22px;
  margin-bottom: 18px;
}
.pf-head-text { min-width: 0; }
.eyebrow {
  margin: 0 0 7px;
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: clamp(1.55rem, 3vw, 2.25rem);
  letter-spacing: -0.01em;
}
.subtitle {
  max-width: 620px;
  margin: 8px 0 0;
  color: var(--text-sub);
  line-height: 1.55;
}
.pf-head-right {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}

/* 置信度环形仪表 */
.conf-ring-wrap {
  position: relative;
  width: 88px;
  height: 88px;
  flex-shrink: 0;
}
.conf-ring { width: 100%; height: 100%; display: block; }
.ring-track { fill: none; stroke: var(--border-light); stroke-width: 8; }
.ring-fill {
  fill: none;
  stroke: url(#ringGrad);
  stroke-width: 8;
  stroke-linecap: round;
  transition: stroke-dashoffset 0.8s var(--ease-out);
  filter: drop-shadow(0 0 6px var(--accent-glow));
}
.ring-center {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 1px;
}
.ring-pct {
  font-family: var(--font-display);
  font-size: 1.25rem; font-weight: 800;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.ring-cap {
  font-size: 0.6rem; color: var(--text-muted);
  letter-spacing: 0.04em;
}

.ghost-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 40px;
  padding: 0 16px;
  border-radius: var(--radius);
  background: var(--bg-card);
  color: var(--text);
  border: 1px solid var(--border);
  white-space: nowrap;
  transition: all var(--transition);
}
.ghost-btn:hover:not(:disabled) { border-color: var(--border-light); background: var(--bg-hover); }
.ghost-btn:disabled { opacity: 0.55; cursor: not-allowed; }
.spin { animation: pf-spin 0.9s linear infinite; }
@keyframes pf-spin { to { transform: rotate(360deg); } }

.toast {
  margin-bottom: 14px;
  color: var(--accent);
  font-weight: 700;
}

/* ── 粘性分节导航 ── */
.pf-nav {
  position: sticky;
  top: 0;
  z-index: 30;
  display: flex;
  gap: 4px;
  margin: 0 -6px 18px;
  padding: 8px 6px;
  background: linear-gradient(to bottom, var(--bg) 62%, transparent);
  overflow-x: auto;
  scrollbar-width: none;
}
.pf-nav::-webkit-scrollbar { display: none; }
.nav-pill {
  flex-shrink: 0;
  padding: 6px 14px;
  border-radius: var(--radius-pill);
  color: var(--text-muted);
  font-family: var(--font-display);
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
  transition: all var(--transition);
}
.nav-pill:hover { color: var(--text-sub); background: var(--bg-hover); }
.nav-pill.active { color: var(--accent); background: var(--accent-dim); }

/* 锚点目标预留粘性导航高度，避免平滑滚动时被遮挡 */
.nav-target { scroll-margin-top: 64px; }

/* ── 卡片通用 ── */
.card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  padding: 18px 20px;
  margin-bottom: 16px;
  transition: border-color var(--transition);
}
.card:hover { border-color: var(--border-light); }
.card-title {
  margin: 0 0 4px;
  font-size: 1.05rem;
  font-family: var(--font-display);
}
.card-sub {
  margin: 0 0 14px;
  color: var(--text-sub);
  font-size: 0.84rem;
  line-height: 1.5;
}

.grid-2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.grid-2 .card { margin-bottom: 0; }

/* 品味摘要 */
.summary-card { border-left: 3px solid var(--accent); }
.summary-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.chip {
  padding: 4px 12px;
  border-radius: var(--radius-pill);
  background: var(--accent-dim);
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
}
.headline {
  margin: 0 0 12px;
  font-size: 1.1rem;
  line-height: 1.6;
  color: var(--text);
}
.core-prefs {
  margin: 0 0 12px;
  padding-left: 18px;
  color: var(--text-sub);
  line-height: 1.7;
}
.rec-hint {
  margin: 0;
  padding: 10px 14px;
  background: var(--bg-elevated);
  border-radius: var(--radius-sm);
  color: var(--text-sub);
  font-size: 0.84rem;
  line-height: 1.5;
}

/* 排除规则（并入自偏好页） */
.excl-input {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
  max-width: 520px;
}
.excl-field { flex: 1; }
.excl-add {
  flex-shrink: 0;
  padding: 0 20px;
  min-height: 46px;
  border-radius: var(--radius-sm);
  background: var(--accent);
  color: #000;
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 0.86rem;
  transition: all var(--transition);
}
.excl-add:hover:not(:disabled) { background: var(--accent-hover); transform: translateY(-1px); box-shadow: var(--glow-sm); }
.excl-add:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }
.excl-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.excl-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px 7px 14px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  font-size: 0.84rem;
  color: var(--text);
  transition: all var(--transition);
}
.excl-chip:hover { border-color: var(--border-light); }
.excl-x {
  width: 18px; height: 18px;
  border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  color: var(--text-muted);
  font-size: 0.95rem; line-height: 1;
  transition: all var(--transition);
}
.excl-x:hover { color: #fff; background: #ff6b6b; }
.excl-empty { color: var(--text-muted); font-size: 0.84rem; }
.excl-pop-enter-active { animation: excl-in 0.25s var(--ease-out); }
.excl-pop-leave-active { transition: all 0.2s ease; position: absolute; }
.excl-pop-leave-to { opacity: 0; transform: scale(0.85); }
.excl-pop-move { transition: transform 0.25s var(--ease-out); }
@keyframes excl-in { from { opacity: 0; transform: scale(0.85); } to { opacity: 1; transform: scale(1); } }

/* 声音指纹 / 探索条 */
.fingerprint, .discovery-bars { display: flex; flex-direction: column; gap: 11px; }
.fp-row, .db-row {
  display: grid;
  grid-template-columns: 84px 1fr 28px;
  align-items: center;
  gap: 10px;
}
.fp-label, .db-row > span:first-child {
  color: var(--text-sub);
  font-size: 0.82rem;
}
.fp-bar {
  height: 7px;
  border-radius: 4px;
  background: var(--bg-elevated);
  overflow: hidden;
}
.fp-bar b {
  display: block;
  height: 100%;
  background: var(--accent-grad);
  border-radius: 4px;
  transition: width 0.5s var(--ease-out);
}
.fp-val, .db-row var {
  font-style: normal;
  text-align: right;
  color: var(--text-sub);
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
}

/* 情绪地图 */
.mood-map-wrap { display: flex; justify-content: center; }
.mood-map { width: 100%; max-width: 340px; height: auto; }
.mood-map .mood-guide { fill: none; stroke: var(--border); stroke-width: 1; stroke-dasharray: 2 4; }
.mood-map .axis { stroke: var(--border-light); stroke-width: 1; }
.mood-map .axis-label { fill: var(--text-muted); font-size: 9px; }
.mood-dot { fill: var(--accent-dim); stroke: var(--accent); stroke-width: 1.2; transition: r 0.3s var(--ease-out); }
.mood-text { fill: var(--text); font-size: 9px; font-weight: 600; }

/* 场景卡 */
.scene-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}
.scene-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  padding: 14px;
}
.scene-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.scene-head h3 { margin: 0; font-size: 0.95rem; }
.conf-mini {
  font-size: 0.7rem;
  color: var(--accent);
  font-weight: 700;
}
.scene-strategy {
  margin: 0 0 10px;
  color: var(--text-sub);
  font-size: 0.82rem;
  line-height: 1.5;
}
.scene-tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px;
  margin-top: 6px;
}
.t-label { color: var(--text-muted); font-size: 0.72rem; margin-right: 2px; }
.mini-chip {
  padding: 2px 9px;
  border-radius: var(--radius-pill);
  background: var(--bg-card);
  color: var(--text-sub);
  font-size: 0.72rem;
}
.mini-chip.avoid { color: #ff8a8a; background: rgba(255, 107, 107, 0.1); }

/* 艺术家星系 */
.artist-groups { display: flex; flex-direction: column; gap: 14px; }
.grp-label {
  display: inline-block;
  margin-bottom: 8px;
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  font-size: 0.74rem;
  font-weight: 700;
  background: var(--bg-elevated);
  color: var(--text-sub);
}
.grp-label.core { background: var(--accent-dim); color: var(--accent); }
.grp-label.rising { background: rgba(76, 168, 255, 0.14); color: #4ca8ff; }
.grp-label.explore { background: rgba(186, 130, 255, 0.14); color: #ba82ff; }
.grp-label.avoid { background: rgba(255, 107, 107, 0.14); color: #ff6b6b; }
.artist-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}
.artist-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  background: var(--bg-elevated);
}
.artist-card strong { display: block; font-size: 0.9rem; }
.artist-reason {
  display: block;
  margin-top: 3px;
  color: var(--text-muted);
  font-size: 0.74rem;
  line-height: 1.4;
}

/* Insight 列表 */
.insight-list { display: flex; flex-direction: column; gap: 12px; }
.insight {
  border: 1px solid var(--border);
  border-left: 3px solid var(--text-muted);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  background: var(--bg-elevated);
}
.insight.band-high { border-left-color: var(--accent); }
.insight.band-medium { border-left-color: #ffc400; }
.insight.band-low { border-left-color: var(--text-muted); }
.insight.rejected { opacity: 0.55; }
.ins-top {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.ins-band {
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  font-size: 0.68rem;
  font-weight: 700;
  background: var(--bg-card);
}
.ins-band.high { color: var(--accent); }
.ins-band.medium { color: #ffc400; }
.ins-band.low { color: var(--text-muted); }
.ins-title { font-size: 0.92rem; }
.ins-status {
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  font-size: 0.68rem;
  font-weight: 700;
}
.ins-status.confirmed { background: var(--accent-dim); color: var(--accent); }
.ins-status.rejected { background: rgba(255,107,107,0.14); color: #ff6b6b; }
.ins-status.temporary { background: rgba(255,196,0,0.14); color: #ffc400; }
.ins-exp {
  margin: 0 0 6px;
  color: var(--text-sub);
  font-size: 0.84rem;
  line-height: 1.5;
}
.ins-evidence {
  margin: 0 0 10px;
  padding-left: 16px;
  color: var(--text-muted);
  font-size: 0.76rem;
  line-height: 1.5;
}
.ins-actions { display: flex; flex-wrap: wrap; gap: 6px; }
.fb-btn {
  min-height: 28px;
  padding: 0 12px;
  border-radius: var(--radius-pill);
  background: var(--bg-card);
  color: var(--text-sub);
  font-size: 0.76rem;
  font-weight: 600;
  border: 1px solid transparent;
  transition: all var(--transition);
}
.fb-btn:hover:not(:disabled) { color: var(--accent); border-color: var(--border-light); }
.fb-btn.active { background: var(--accent); color: #07120b; border-color: var(--accent); }
.fb-btn.warn.active { background: #ff6b6b; color: #fff; border-color: #ff6b6b; }
.fb-btn.ghost { color: var(--text-muted); }
.fb-btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* 空状态 */
.empty-card {
  border: 1px dashed var(--border-light);
  border-radius: var(--radius);
  background: var(--bg-card);
  padding: 40px 28px;
  text-align: center;
}
.empty-card h2 { margin: 0 0 10px; font-family: var(--font-display); }
.empty-card p { margin: 0 auto 18px; max-width: 520px; color: var(--text-sub); line-height: 1.6; }
.empty-actions { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.hint-chip {
  padding: 7px 14px;
  border-radius: var(--radius-pill);
  background: var(--bg-elevated);
  color: var(--text-sub);
  font-size: 0.82rem;
}

@media (max-width: 880px) {
  .pf-head { flex-direction: column; align-items: stretch; }
  .pf-head-right { justify-content: space-between; }
  .grid-2 { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .profile-view { padding-inline: 14px; }
  .fp-row, .db-row { grid-template-columns: 72px 1fr 26px; }
  .conf-ring-wrap { width: 72px; height: 72px; }
  .ring-pct { font-size: 1.05rem; }
}
</style>
