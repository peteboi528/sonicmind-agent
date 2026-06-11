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
    // 从后端加载已保存的评分，合并到 asset._rated 让星星点亮
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

    <div v-if="msg" class="toast">{{ msg }}</div>
    <div v-if="loading" class="loading-hint">加载中…</div>

    <div v-if="!loading && !assets.length" class="empty-hint">库还是空的，先添加点音乐吧。</div>

    <div v-for="a in assets" :key="a.asset_id" class="lib-row">
      <div class="info">
        <div class="title">{{ a.title || "未命名" }}</div>
        <div class="artist">{{ a.artist || "未知" }}
          <span v-for="g in (a.genre || []).slice(0,2)" :key="g" class="tag">{{ g }}</span>
        </div>
      </div>
      <div class="rate">
        <button v-for="s in [2,4,6,8,10]" :key="s"
          class="star" :class="{ on: a._rated >= s }" @click="rate(a, s)">★</button>
      </div>
      <button class="del" @click="remove(a)">🗑</button>
    </div>
  </div>
</template>

<style scoped>
.ingest-row { display: flex; gap: 10px; margin-bottom: 18px; max-width: 640px; }
.toast { background: var(--accent-dim); color: var(--accent); padding: 10px 14px; border-radius: var(--radius-sm); margin-bottom: 14px; }
.lib-row { display: flex; align-items: center; gap: 14px; background: var(--bg-card); padding: 12px 16px; border-radius: var(--radius); margin-bottom: 8px; }
.info { flex: 1; min-width: 0; }
.title { font-weight: 600; }
.artist { color: var(--text-sub); font-size: 0.85rem; margin-top: 2px; }
.tag { display: inline-block; margin-left: 6px; padding: 1px 7px; background: var(--bg-elevated); border-radius: var(--radius-pill); font-size: 0.7rem; color: var(--text-muted); }
.rate { display: flex; gap: 2px; }
.star { font-size: 1.1rem; color: var(--text-muted); transition: var(--transition); }
.star.on { color: var(--accent); }
.star:hover { color: var(--accent-hover); }
.del { width: 36px; height: 36px; border-radius: 50%; color: var(--text-sub); }
.del:hover { background: var(--bg-hover); color: var(--danger); }
</style>
