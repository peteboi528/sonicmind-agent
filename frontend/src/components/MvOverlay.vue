<script setup>
import { store } from "../store.js";
</script>

<template>
  <Transition name="overlay">
    <div v-if="store.mv.visible" class="mv-overlay" @click.self="store.closeMv()">
      <div class="mv-box">
        <button class="mv-close" @click="store.closeMv()">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
        <div class="mv-frame">
          <iframe :src="store.mv.url" allowfullscreen allow="autoplay; encrypted-media"></iframe>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.mv-overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.88);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  display: flex; align-items: center; justify-content: center;
  z-index: 300; padding: 24px;
}
.mv-box {
  width: min(900px, 100%); position: relative;
}
.mv-close {
  position: absolute; top: -44px; right: 0;
  width: 38px; height: 38px; border-radius: 50%;
  background: var(--bg-card); color: var(--text);
  display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--border);
  transition: all var(--transition);
}
.mv-close:hover { background: var(--bg-hover); transform: rotate(90deg); }
.mv-frame {
  position: relative; padding-bottom: 56.25%; height: 0;
  border-radius: var(--radius); overflow: hidden;
  box-shadow: 0 24px 64px rgba(0,0,0,0.6);
}
.mv-frame iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }

/* ── Overlay Transition ── */
.overlay-enter-active { transition: all 0.35s var(--ease-out); }
.overlay-leave-active { transition: all 0.25s ease; }
.overlay-enter-from, .overlay-leave-to {
  opacity: 0;
}
.overlay-enter-from .mv-box { transform: scale(0.92) translateY(20px); }
.overlay-leave-to .mv-box { transform: scale(0.95); }
</style>
