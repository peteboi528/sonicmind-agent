<script setup>
import { ref, computed, watch } from "vue";
import { store } from "./store.js";
import Sidebar from "./components/Sidebar.vue";
import ChatTab from "./components/ChatTab.vue";
import DiscoverTab from "./components/DiscoverTab.vue";
import TasteLabTab from "./components/TasteLabTab.vue";
import ProfileTab from "./components/ProfileTab.vue";
import DailyTab from "./components/DailyTab.vue";
import LibraryTab from "./components/LibraryTab.vue";
import HistoryTab from "./components/HistoryTab.vue";
import PlaylistTab from "./components/PlaylistTab.vue";
import PlayerBar from "./components/PlayerBar.vue";
import MvOverlay from "./components/MvOverlay.vue";

const TABS = [
  { id: "chat",     label: "对话",   comp: ChatTab,     icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>` },
  { id: "daily",    label: "今日",   comp: DailyTab,    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>` },
  { id: "discover", label: "浏览",   comp: DiscoverTab, icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>` },
  { id: "taste-lab", label: "探索", comp: TasteLabTab, icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>` },
  { id: "profile",  label: "画像",   comp: ProfileTab,  icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M4 21v-1a7 7 0 0 1 14 0v1"/></svg>` },
  { id: "library",  label: "我的库", comp: LibraryTab,  icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>` },
  { id: "history",  label: "历史",   comp: HistoryTab,  icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/></svg>` },
  { id: "playlist", label: "歌单",   comp: PlaylistTab, icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>` },
];

const active = ref("chat");
const drawerOpen = ref(false);

const activeTab = computed(() => TABS.find(t => t.id === active.value)?.comp || ChatTab);
const isChat = computed(() => active.value === "chat");

// 跨 tab 导航：历史 tab 点「最近对话」→ store.navigateTo('chat', threadId) → 切到对话 tab。
watch(() => store.navigate?.nonce, () => {
  if (store.navigate) active.value = store.navigate.tab;
});
</script>

<template>
  <div class="app-shell">
    <!-- Drawer Overlay -->
    <Transition name="drawer-mask">
      <div v-if="drawerOpen" class="drawer-mask" @click="drawerOpen = false"></div>
    </Transition>

    <!-- Slide-out Sidebar Drawer -->
    <Transition name="drawer">
      <aside v-if="drawerOpen" class="drawer-panel">
        <Sidebar @close="drawerOpen = false" />
      </aside>
    </Transition>

    <!-- Main -->
    <div class="main-area">
      <!-- Top Bar -->
      <header class="top-bar">
        <button class="top-icon" @click="drawerOpen = !drawerOpen" title="设置面板">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
        <div class="brand-mark">S</div>
        <span class="brand-name">SONICMIND</span>

        <nav class="tab-pills">
          <button v-for="t in TABS" :key="t.id"
            :class="['pill', { active: active === t.id }]"
            @click="active = t.id"
          >
            <span class="pill-icon" v-html="t.icon"></span>
            <span class="pill-label">{{ t.label }}</span>
          </button>
        </nav>
      </header>

      <div class="tab-content" :class="{ 'chat-mode': isChat }">
        <!-- 缓存全部 tab：去掉 :key（与 :is 冗余且会干扰缓存键），:max 覆盖全部
             tab 避免 LRU 淘汰导致切回发现页时 DiscoverTab 重新挂载、onMounted 重跑
             loadForYou/loadTrending（"重复加载"的根因）。-->
        <KeepAlive :max="8">
          <component :is="activeTab" />
        </KeepAlive>
      </div>

      <PlayerBar />
      <MvOverlay />
    </div>
  </div>
</template>
