<script setup>
import { ref, onMounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";
import NeteaseLogin from "./NeteaseLogin.vue";

const userInput = ref(store.userId);
const showLogin = ref(false);
const importRef = ref("");
const importLimit = ref(200);
const importing = ref(false);
const prefEvent = ref("");
const learning = ref(false);
const msg = ref("");

function saveUser() {
  store.setUser(userInput.value);
  refreshAccount();
  msg.value = "已切换用户：" + store.userId;
}

async function refreshAccount() {
  try {
    const info = await api.neteaseAccount(store.userId);
    store.setNetease(info);
  } catch { /* ignore */ }
}

async function unbind() {
  try {
    await api.neteaseUnbind(store.userId);
    store.setNetease({ bound: false });
    msg.value = "已解绑网易云";
  } catch { msg.value = "解绑失败"; }
}

async function doImport() {
  const ref_ = importRef.value.trim();
  if (!ref_) return;
  importing.value = true;
  msg.value = "";
  try {
    const r = await api.importNetease(store.userId, ref_, importLimit.value);
    if (r.error) msg.value = r.error;
    else { msg.value = `导入《${r.name}》：新增 ${r.imported}/${r.total} 首`; importRef.value = ""; }
  } catch { msg.value = "导入失败"; }
  finally { importing.value = false; }
}

async function learn() {
  const ev = prefEvent.value.trim();
  if (!ev) return;
  learning.value = true;
  msg.value = "";
  try {
    await api.updateMemory(store.userId, ev);
    msg.value = "已学习你的偏好";
    prefEvent.value = "";
  } catch { msg.value = "学习失败"; }
  finally { learning.value = false; }
}

function onBound() { showLogin.value = false; refreshAccount(); }

onMounted(refreshAccount);
</script>

<template>
  <aside class="sidebar">
    <div class="brand">SONICMIND<span class="dot">·</span><small>为你而听</small></div>

    <!-- 用户 -->
    <div class="block">
      <div class="block-title">用户</div>
      <div class="user-row">
        <input v-model="userInput" class="input sm" @keyup.enter="saveUser" />
        <button class="btn-ghost sm" @click="saveUser">切换</button>
      </div>
    </div>

    <!-- 网易云 -->
    <div class="block">
      <div class="block-title">网易云</div>
      <div v-if="store.netease.bound" class="netease-on">
        <img v-if="store.netease.avatar" :src="store.netease.avatar" class="avatar" alt="" />
        <div class="ne-info">
          <div class="ne-name">{{ store.netease.nickname || "已登录" }}</div>
          <div class="ne-vip">{{ store.netease.vipLabel }}</div>
        </div>
        <button class="btn-ghost xs" @click="unbind">解绑</button>
      </div>
      <button v-else class="btn full" @click="showLogin = true">扫码登录</button>
    </div>

    <!-- 导入歌单 -->
    <div class="block">
      <div class="block-title">导入网易云歌单</div>
      <input v-model="importRef" class="input sm" placeholder="歌单链接或 ID" />
      <div class="import-row">
        <input v-model.number="importLimit" type="number" class="input sm num" min="10" max="500" />
        <button class="btn-ghost sm" :disabled="importing || !importRef.trim()" @click="doImport">
          {{ importing ? "导入中…" : "导入" }}
        </button>
      </div>
    </div>

    <!-- 训练偏好 -->
    <div class="block">
      <div class="block-title">训练偏好</div>
      <input v-model="prefEvent" class="input sm" placeholder="如：我喜欢慵懒的爵士" @keyup.enter="learn" />
      <button class="btn-ghost sm full" :disabled="learning || !prefEvent.trim()" @click="learn">
        {{ learning ? "学习中…" : "学习" }}
      </button>
    </div>

    <div v-if="msg" class="msg">{{ msg }}</div>

    <NeteaseLogin v-if="showLogin" @close="showLogin = false" @bound="onBound" />
  </aside>
</template>

<style scoped>
.sidebar {
  width: var(--sidebar-w); flex-shrink: 0; background: #000;
  border-right: 1px solid var(--border); padding: 20px 16px;
  overflow-y: auto; display: flex; flex-direction: column; gap: 22px;
}
.brand { font-weight: 800; font-size: 1.15rem; }
.brand .dot { color: var(--accent); margin: 0 4px; }
.brand small { color: var(--text-sub); font-weight: 400; font-size: 0.7rem; }
.block-title { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.input.sm { padding: 8px 11px; font-size: 0.85rem; margin-bottom: 6px; }
.input.num { width: 80px; }
.user-row { display: flex; gap: 6px; }
.btn-ghost.sm, .btn.full { padding: 7px 14px; font-size: 0.82rem; }
.btn-ghost.xs { padding: 4px 10px; font-size: 0.75rem; }
.full { width: 100%; }
.import-row { display: flex; gap: 6px; align-items: center; }
.netease-on { display: flex; align-items: center; gap: 10px; }
.avatar { width: 36px; height: 36px; border-radius: 50%; }
.ne-info { flex: 1; min-width: 0; }
.ne-name { font-size: 0.85rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ne-vip { font-size: 0.72rem; color: var(--accent); }
.msg { font-size: 0.8rem; color: var(--accent); background: var(--accent-dim); padding: 8px 10px; border-radius: var(--radius-sm); }
@media (max-width: 768px) {
  .sidebar { width: 100%; flex-direction: row; flex-wrap: wrap; gap: 12px; }
  .block { flex: 1; min-width: 140px; }
}
</style>
