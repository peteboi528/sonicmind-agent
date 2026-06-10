<script setup>
import { ref, watch } from "vue";
import { store } from "../store.js";

const audio = ref(null);
const spinning = ref(false);

// URL 变化时自动播放
watch(() => store.player.url, (url) => {
  if (url && audio.value) {
    audio.value.src = url;
    audio.value.play().catch(() => {});
  }
});

function close() {
  if (audio.value) { audio.value.pause(); audio.value.src = ""; }
  spinning.value = false;
  store.closePlayer();
}
</script>

<template>
  <div class="player-bar" :class="{ visible: store.player.visible }">
    <div class="disc" :class="{ spinning }">
      <img v-if="store.player.cover" :src="store.player.cover" alt="" />
    </div>
    <div class="meta">
      <div class="ptitle">{{ store.player.title || "未知" }}</div>
      <div class="partist">{{ store.player.artist }}</div>
    </div>
    <audio
      ref="audio" controls
      @play="spinning = true" @pause="spinning = false" @ended="spinning = false"
    ></audio>
    <button class="close" @click="close">✕</button>
  </div>
</template>

<style scoped>
.player-bar {
  position: fixed; bottom: 0; left: 0; right: 0; height: var(--player-h);
  display: flex; align-items: center; gap: 14px; padding: 0 20px;
  background: var(--bg-elevated); border-top: 1px solid var(--border);
  transform: translateY(100%); transition: transform 0.3s ease; z-index: 200;
}
.player-bar.visible { transform: translateY(0); }
.disc {
  width: 48px; height: 48px; border-radius: 50%; flex-shrink: 0;
  background: var(--bg-hover); overflow: hidden;
}
.disc.spinning { animation: spin 4s linear infinite; }
.disc img { width: 100%; height: 100%; object-fit: cover; }
@keyframes spin { to { transform: rotate(360deg); } }
.meta { min-width: 120px; }
.ptitle { font-weight: 600; font-size: 0.9rem; }
.partist { color: var(--text-sub); font-size: 0.8rem; }
audio { flex: 1; max-width: 520px; height: 36px; }
.close { width: 32px; height: 32px; border-radius: 50%; color: var(--text-sub); }
.close:hover { background: var(--bg-hover); color: var(--text); }
@media (max-width: 768px) {
  audio { max-width: none; }
  .meta { display: none; }
}
</style>
