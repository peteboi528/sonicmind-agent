<script setup>
import { ref, onUnmounted } from "vue";
import { store } from "../store.js";
import { api } from "../api.js";

const emit = defineEmits(["close", "bound"]);
const qrImg = ref("");
const status = ref("正在获取二维码…");
const polling = ref(false);
let timer = null;
let unikey = "";

async function start() {
  status.value = "正在获取二维码…";
  qrImg.value = "";
  try {
    const data = await api.neteaseQrKey();
    if (!data.unikey) { status.value = data.error || "获取失败"; return; }
    unikey = data.unikey;
    qrImg.value = data.qr_img;
    status.value = "请用网易云音乐 App 扫码";
    poll();
  } catch { status.value = "获取二维码失败，请重试"; }
}

function poll() {
  polling.value = true;
  timer = setInterval(async () => {
    try {
      const r = await api.neteaseQrStatus(unikey, store.userId);
      if (r.code === 801) status.value = "等待扫码中…";
      else if (r.code === 802) status.value = "📱 已扫描，请在手机上确认";
      else if (r.code === 803) {
        stop();
        status.value = `✅ 绑定成功，欢迎 ${r.nickname || ""}`;
        store.setNetease({ bound: true, ...r });
        setTimeout(() => emit("bound"), 800);
      } else if (r.code === 800) {
        stop();
        status.value = "二维码已过期";
      }
    } catch { /* 轮询容错，继续 */ }
  }, 2000);
}

function stop() {
  if (timer) { clearInterval(timer); timer = null; }
  polling.value = false;
}

onUnmounted(stop);
start();
</script>

<template>
  <div class="overlay" @click.self="emit('close')">
    <div class="modal">
      <button class="close" @click="emit('close')">✕</button>
      <div class="title">扫码登录网易云</div>
      <div class="qr-box">
        <img v-if="qrImg" :src="qrImg" alt="QR" />
        <div v-else class="qr-ph">…</div>
      </div>
      <div class="status">{{ status }}</div>
      <button class="btn-ghost" @click="start">刷新二维码</button>
    </div>
  </div>
</template>

<style scoped>
.overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 300; }
.modal { background: var(--bg-elevated); border-radius: var(--radius); padding: 28px 32px; width: 320px; text-align: center; position: relative; }
.close { position: absolute; top: 12px; right: 14px; color: var(--text-sub); font-size: 1.1rem; }
.title { font-weight: 700; font-size: 1.1rem; margin-bottom: 18px; }
.qr-box { width: 200px; height: 200px; margin: 0 auto 16px; background: #fff; border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center; overflow: hidden; }
.qr-box img { width: 100%; height: 100%; }
.qr-ph { color: #999; }
.status { color: var(--text-sub); font-size: 0.9rem; margin-bottom: 16px; min-height: 20px; }
</style>
