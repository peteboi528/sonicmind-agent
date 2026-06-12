<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const assets = ref([]);
const loading = ref(false);
const msg = ref("");
const ingestUrl = ref("");
const ingesting = ref(false);

async function load() {
  loading.value = true;
  try {
    const data = await api.listAssets();
    assets.value = data.assets || [];
    try {
      const ratingsData = await api.getRatings(store.userId);
      const ratingMap = {};
      for (const r of ratingsData.ratings || []) {
        ratingMap[r.asset_id] = r.score;
      }
      for (const a of assets.value) {
        if (ratingMap[a.asset_id] != null) {
          a._rated = ratingMap[a.asset_id];
        }
      }
    } catch { /* 评分加载失败不影响主流程 */ }
  } catch { msg.value = "加载库失败。"; }
  finally { loading.value = false; }
}

async function rate(asset, score) {
  try {
    await api.rate(store.userId, asset.asset_id, score);
    asset._rated = score;
    msg.value = `已为《${asset.title}》评 ${score} 分。`;
  } catch { msg.value = "评分失败。"; }
}

async function remove(asset) {
  if (!confirm(`确定删除《${asset.title}》？`)) return;
  try {
    await api.deleteAsset(asset.asset_id, store.userId);
    assets.value = assets.value.filter((a) => a.asset_id !== asset.asset_id);
  } catch { msg.value = "删除失败。"; }
}

async function ingest() {
  const url = ingestUrl.value.trim();
  if (!url) return;
  ingesting.value = true;
  msg.value = "";
  try {
    const asset = await api.ingest(url);
    msg.value = `已入库：《${asset.title || "新资源"}》`;
    ingestUrl.value = "";
    await load();
  } catch { msg.value = "入库失败，请检查链接。"; }
  finally { ingesting.value = false; }
}

onMounted(load);
</script>

<template>
  <div>
    <div class="section-title">我的库</div>
    <div class="section-sub">管理已入库的曲目，评分会实时更新你的口味画像。</div>

    <div class="ingest-row">
      <input v-model="ingestUrl" class="input" placeholder="粘贴音乐链接入库（网易云/B站/YouTube）" @keyup.enter="ingest" />
      <button class="btn" :disabled="ingesting || !ingestUrl.trim()" @click="ingest">
        {{ ingesting ? "入库中…" : "添加" }}
      </button>
    </div>

    <Transition name="toast-slide">
      <div v-if="msg" class="toast">{{ msg }}</div>
    </Transition>
    <div v-if="loading" class="loading-hint">加载中…</div>

    <div v-if="!loading && !assets.length" class="empty-hint">库还是空的，先添加点音乐吧。</div>

    <div v-for="(a, idx) in assets" :key="a.asset_id" class="lib-row stagger-item" :style="{ animationDelay: `${idx * 40}ms` }">
      <div class="info">
        <div class="title">{{ a.title || "未命名" }}</div>
        <div class="artist">
          {{ a.artist || "未知" }}
          <span v-for="g in (a.genre || []).slice(0,2)" :key="g" class="tag">{{ g }}</span>
        </div>
      </div>
      <div class="rate">
        <button v-for="s in [2,4,6,8,10]" :key="s"
          class="star" :class="{ on: a._rated >= s }" @click="rate(a, s)">★</button>
      </div>
      <button class="del" @click="remove(a)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.ingest-row { display: flex; gap: 10px; margin-bottom: 20px; max-width: 640px; }
.toast {
  background: var(--accent-dim); color: var(--accent); padding: 10px 14px;
  border-radius: var(--radius-sm); margin-bottom: 16px;
  border: 1px solid rgba(29,185,84,0.12);
}
.toast-slide-enter-active { animation: fadeInUp 0.25s var(--ease-out); }
.toast-slide-leave-active { transition: all 0.15s ease; }
.toast-slide-leave-to { opacity: 0; transform: translateY(-6px); }

.lib-row {
  display: flex; align-items: center; gap: 14px;
  background: var(--bg-card); padding: 14px 18px;
  border-radius: var(--radius); margin-bottom: 8px;
  border: 1px solid var(--border);
  transition: all var(--dur-norm) var(--ease-out);
}
.lib-row:hover {
  background: var(--bg-hover);
  border-color: var(--border-light);
  transform: translateY(-1px);
}
.info { flex: 1; min-width: 0; }
.title {
  font-family: var(--font-display);
  font-weight: 600; font-size: 0.93rem;
}
.artist { color: var(--text-sub); font-size: 0.82rem; margin-top: 3px; }
.tag {
  display: inline-block; margin-left: 6px; padding: 2px 8px;
  background: var(--bg-elevated); border-radius: var(--radius-pill);
  font-size: 0.68rem; color: var(--text-muted);
  font-family: var(--font-display); font-weight: 600;
}

.rate { display: flex; gap: 2px; }
.star {
  font-size: 1.15rem; color: var(--text-muted);
  transition: all 0.2s var(--ease-spring);
}
.star.on {
  color: var(--accent);
  text-shadow: 0 0 8px var(--accent-glow);
}
.star:hover {
  color: var(--accent-hover);
  transform: scale(1.25);
}

.del {
  width: 34px; height: 34px; border-radius: 50%;
  color: var(--text-muted); display: flex; align-items: center; justify-content: center;
  transition: all var(--transition);
}
.del:hover { background: rgba(231,76,60,0.1); color: var(--danger); }
</style>
