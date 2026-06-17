<script setup>
import { ref, computed } from "vue";
import Sidebar from "./components/Sidebar.vue";
import ChatTab from "./components/ChatTab.vue";
import DiscoverTab from "./components/DiscoverTab.vue";
import TasteLabTab from "./components/TasteLabTab.vue";
import DailyTab from "./components/DailyTab.vue";
import LibraryTab from "./components/LibraryTab.vue";
import PlaylistTab from "./components/PlaylistTab.vue";
import SettingsTab from "./components/SettingsTab.vue";
import PlayerBar from "./components/PlayerBar.vue";
import MvOverlay from "./components/MvOverlay.vue";

const TABS = [
  { id: "chat",     label: "对话",   comp: ChatTab,     icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>` },
  { id: "daily",    label: "今日",   comp: DailyTab,    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>` },
  { id: "discover", label: "发现",   comp: DiscoverTab, icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>` },
  { id: "taste-lab", label: "实验室", comp: TasteLabTab, icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 2v7.31"/><path d="M14 9.3V2"/><path d="M8.5 2h7"/><path d="M14 9.3a6 6 0 1 1-4 0"/></svg>` },
  { id: "library",  label: "我的库", comp: LibraryTab,  icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>` },
  { id: "playlist", label: "歌单",   comp: PlaylistTab, icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>` },
  { id: "settings", label: "偏好",   comp: SettingsTab, icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>` },
];

const active = ref("chat");
const drawerOpen = ref(false);

const activeTab = computed(() => TABS.find(t => t.id === active.value)?.comp || ChatTab);
const isChat = computed(() => active.value === "chat");
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
        <!-- 缓存全部 tab：去掉 :key（与 :is 冗余且会干扰缓存键），:max 覆盖 6 个
             tab 避免 LRU 淘汰导致切回发现页时 DiscoverTab 重新挂载、onMounted 重跑
             loadForYou/loadTrending（"重复加载"的根因）。-->
        <KeepAlive :max="7">
          <component :is="activeTab" />
        </KeepAlive>
      </div>

      <PlayerBar />
      <MvOverlay />
    </div>
  </div>
</template>
