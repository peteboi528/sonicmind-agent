// 轻量全局状态：用户 ID（localStorage 持久化）+ 网易云绑定态 + 播放器。
// 用 Vue 的 reactive，避免引入 Pinia 这种重依赖。
import { reactive } from "vue";

const STORED_USER = localStorage.getItem("sonicmind_user") || "web_user";

export const store = reactive({
  userId: STORED_USER,
  netease: { bound: false, nickname: null, avatar: null, vipLabel: "" },
  player: { visible: false, title: "", artist: "", cover: "", url: "" },
  mv: { visible: false, url: "" },

  setUser(id) {
    this.userId = (id || "").trim() || "web_user";
    localStorage.setItem("sonicmind_user", this.userId);
  },
  setNetease(info) {
    this.netease = {
      bound: !!info?.bound,
      nickname: info?.nickname || null,
      avatar: info?.avatar || null,
      vipLabel: info?.vip_label || "",
    };
  },
  playTrack({ title, artist, cover, url }) {
    this.player = { visible: true, title, artist, cover, url };
  },
  closePlayer() { this.player.visible = false; this.player.url = ""; },
  showMv(url) { this.mv = { visible: true, url }; },
  closeMv() { this.mv = { visible: false, url: "" }; },
});
