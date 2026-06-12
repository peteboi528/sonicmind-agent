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
const emit = defineEmits(["close"]);

// 我的歌单
const myPlaylists = ref([]);
const loadingPlaylists = ref(false);
const selectedPlaylistId = ref("");

async function loadMyPlaylists() {
  loadingPlaylists.value = true;
  msg.value = "";
  try {
    const data = await api.neteasePlaylistList(store.userId);
    if (data.error) { msg.value = data.error; return; }
    myPlaylists.value = data.playlists || [];
    if (myPlaylists.value.length) selectedPlaylistId.value = String(myPlaylists.value[0].id);
  } catch { msg.value = "加载歌单失败"; }
  finally { loadingPlaylists.value = false; }
}

async function importSelected() {
  if (!selectedPlaylistId.value) return;
  importing.value = true;
  msg.value = "";
  try {
    const r = await api.importNetease(store.userId, selectedPlaylistId.value, importLimit.value);
    if (r.error) msg.value = r.error;
    else msg.value = `导入《${r.name}》：新增 ${r.imported}/${r.total} 首`;
  } catch { msg.value = "导入失败"; }
  finally { importing.value = false; }
}

function saveUser() {
  store.setUser(userInput.value);
  refreshAccount();
  myPlaylists.value = [];
  selectedPlaylistId.value = "";
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
    myPlaylists.value = [];
    selectedPlaylistId.value = "";
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
    <button class="drawer-close" @click="$emit('close')">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
    <div class="brand">
      <span class="brand-mark">S</span>
      <span class="brand-text">SONICMIND</span>
      <span class="brand-tag">为你而听</span>
    </div>

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

      <!-- 已登录时：加载我的歌单 -->
      <template v-if="store.netease.bound">
        <button class="btn-ghost sm full" :disabled="loadingPlaylists" @click="loadMyPlaylists">
          {{ loadingPlaylists ? "加载中…" : "加载我的歌单" }}
        </button>
        <template v-if="myPlaylists.length">
          <select v-model="selectedPlaylistId" class="input sm select-pl">
            <option v-for="pl in myPlaylists" :key="pl.id" :value="String(pl.id)">
              {{ pl.name }}（{{ pl.count }} 首）
            </option>
          </select>
          <div class="import-row">
            <input v-model.number="importLimit" type="number" class="input sm num" min="10" max="500" />
            <button class="btn-ghost sm" :disabled="importing || !selectedPlaylistId" @click="importSelected">
              {{ importing ? "导入中…" : "导入选中" }}
            </button>
          </div>
        </template>
        <div class="divider-or">或手动输入</div>
      </template>

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

    <Transition name="msg-slide">
      <div v-if="msg" class="msg">{{ msg }}</div>
    </Transition>

    <NeteaseLogin v-if="showLogin" @close="showLogin = false" @bound="onBound" />
  </aside>
</template>

<style scoped>
.sidebar {
  width: 100%; min-height: 100%;
  background: #040406;
  padding: 24px 18px;
  overflow-y: auto; display: flex; flex-direction: column; gap: 24px;
  position: relative;
}

/* ── Drawer Close ── */
.drawer-close {
  position: absolute; top: 16px; right: 16px;
  width: 32px; height: 32px; border-radius: 50%;
  color: var(--text-muted);
  display: flex; align-items: center; justify-content: center;
  transition: all var(--transition);
}
.drawer-close:hover { background: var(--bg-hover); color: var(--text); }

/* ── Brand ── */
.brand {
  display: flex; align-items: center; gap: 8px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.brand-mark {
  width: 32px; height: 32px; border-radius: 8px;
  background: var(--accent-grad);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-display);
  font-weight: 800; font-size: 1.1rem; color: #000;
  flex-shrink: 0;
}
.brand-text {
  font-family: var(--font-display);
  font-weight: 800; font-size: 1rem;
  letter-spacing: 0.04em;
}
.brand-tag {
  color: var(--text-muted); font-size: 0.65rem;
  margin-left: auto;
}

.block-title {
  font-family: var(--font-display);
  font-size: 0.7rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 10px; font-weight: 600;
}

.input.sm { padding: 9px 12px; font-size: 0.84rem; margin-bottom: 6px; }
.input.num { width: 80px; }
.user-row { display: flex; gap: 6px; }
.btn-ghost.sm, .btn.full { padding: 8px 14px; font-size: 0.82rem; }
.btn-ghost.xs { padding: 5px 10px; font-size: 0.74rem; }
.full { width: 100%; }
.import-row { display: flex; gap: 6px; align-items: center; }
.select-pl { width: 100%; margin-bottom: 6px; appearance: auto; }

/* ── Netease Bound ── */
.netease-on {
  display: flex; align-items: center; gap: 10px;
  padding: 8px; border-radius: var(--radius-sm);
  background: var(--bg-card);
  border: 1px solid var(--border);
}
.avatar {
  width: 38px; height: 38px; border-radius: 50%;
  border: 2px solid var(--accent-dim);
}
.ne-info { flex: 1; min-width: 0; }
.ne-name {
  font-family: var(--font-display);
  font-size: 0.84rem; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ne-vip { font-size: 0.7rem; color: var(--accent); font-weight: 600; }

.divider-or {
  font-size: 0.7rem; color: var(--text-muted); text-align: center;
  margin: 10px 0 8px; position: relative;
}
.divider-or::before, .divider-or::after {
  content: ""; position: absolute; top: 50%; width: 36%; height: 1px;
  background: var(--border);
}
.divider-or::before { left: 0; }
.divider-or::after { right: 0; }

.msg {
  font-size: 0.78rem; color: var(--accent);
  background: var(--accent-dim); padding: 9px 12px;
  border-radius: var(--radius-sm);
  border: 1px solid rgba(29,185,84,0.12);
}

.msg-slide-enter-active { animation: fadeInUp 0.25s var(--ease-out); }
.msg-slide-leave-active { transition: all 0.15s ease; }
.msg-slide-leave-to { opacity: 0; transform: translateY(-6px); }

@media (max-width: 768px) {
  .sidebar { width: 100%; flex-direction: row; flex-wrap: wrap; gap: 12px; padding: 16px; }
  .block { flex: 1; min-width: 140px; }
  .brand { display: none; }
}
</style>
