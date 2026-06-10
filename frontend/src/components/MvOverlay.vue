<script setup>
import { store } from "../store.js";
</script>

<template>
  <div v-if="store.mv.visible" class="mv-overlay" @click.self="store.closeMv()">
    <div class="mv-box">
      <button class="mv-close" @click="store.closeMv()">✕</button>
      <div class="mv-frame">
        <iframe :src="store.mv.url" allowfullscreen allow="autoplay; encrypted-media"></iframe>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mv-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.85);
  display: flex; align-items: center; justify-content: center;
  z-index: 300; padding: 20px;
}
.mv-box { width: min(900px, 100%); position: relative; }
.mv-close {
  position: absolute; top: -40px; right: 0; width: 36px; height: 36px;
  border-radius: 50%; background: var(--bg-card); color: var(--text); font-size: 1.1rem;
}
.mv-frame { position: relative; padding-bottom: 56.25%; height: 0; border-radius: var(--radius); overflow: hidden; }
.mv-frame iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }
</style>
