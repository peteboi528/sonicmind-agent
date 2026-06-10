<script setup>
import { ref } from "vue";
import Sidebar from "./components/Sidebar.vue";
import ChatTab from "./components/ChatTab.vue";
import DiscoverTab from "./components/DiscoverTab.vue";
import DailyTab from "./components/DailyTab.vue";
import LibraryTab from "./components/LibraryTab.vue";
import PlaylistTab from "./components/PlaylistTab.vue";
import SettingsTab from "./components/SettingsTab.vue";
import PlayerBar from "./components/PlayerBar.vue";
import MvOverlay from "./components/MvOverlay.vue";

const TABS = [
  { id: "daily", label: "今日推荐", comp: DailyTab },
  { id: "discover", label: "发现", comp: DiscoverTab },
  { id: "library", label: "我的库", comp: LibraryTab },
  { id: "playlist", label: "我的歌单", comp: PlaylistTab },
  { id: "chat", label: "对话", comp: ChatTab },
  { id: "settings", label: "偏好", comp: SettingsTab },
];
const active = ref("chat");
</script>

<template>
  <div class="app-shell">
    <Sidebar />
    <div class="main-area">
      <nav class="tab-nav">
        <button
          v-for="t in TABS" :key="t.id"
          :class="{ active: active === t.id }"
          @click="active = t.id"
        >{{ t.label }}</button>
      </nav>
      <div class="tab-content">
        <!-- keep-alive 保留各 Tab 状态，切换不丢失已加载数据 -->
        <KeepAlive>
          <component :is="TABS.find((t) => t.id === active).comp" />
        </KeepAlive>
      </div>
      <PlayerBar />
      <MvOverlay />
    </div>
  </div>
</template>
