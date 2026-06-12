// 轻量全局状态：用户 ID（localStorage 持久化）+ 网易云绑定态 + 播放器 + 播放队列。
// 用 Vue 的 reactive，避免引入 Pinia 这种重依赖。
import { reactive } from "vue";
import { api } from "./api.js";

const STORED_USER = localStorage.getItem("sonicmind_user") || "web_user";

export const store = reactive({
  userId: STORED_USER,
  netease: { bound: false, nickname: null, avatar: null, vipLabel: "" },
  player: { visible: false, title: "", artist: "", cover: "", url: "" },
  mv: { visible: false, url: "" },

  // ── 播放队列 ──
  queue: [],          // [{...card}] 完整卡片对象，URL 延迟加载
  queueIndex: -1,     // 当前播放索引
  _playSeq: 0,        // 防并发：递增序号，旧请求返回时丢弃

  // ── 队列 toast ──
  toastMsg: "",
  toastKey: 0,

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
    // 同步队列索引：如果这首歌在队列中，更新 queueIndex
    if (this.queue.length) {
      const idx = this.queue.findIndex(
        c => c.title === title && (c.artist || "") === (artist || ""),
      );
      if (idx >= 0) {
        this.queueIndex = idx;
      } else {
        // 不在当前队列 → 用户切换了上下文，清空旧队列并提示
        this.showToast(`已切换到单曲播放，原队列（${this.queue.length}首）已清除`);
        this.queue = [];
        this.queueIndex = -1;
      }
    }
  },

  closePlayer() {
    this.player.visible = false;
    this.player.url = "";
  },
  showMv(url) { this.mv = { visible: true, url }; },
  closeMv() { this.mv = { visible: false, url: "" }; },

  // ── 队列 toast ──
  showToast(msg) {
    this.toastMsg = msg;
    this.toastKey++;
  },

  // ── 队列方法 ──

  /** 把卡片列表加入队列，从第 0 首开始播放 */
  playAll(cards) {
    if (!cards?.length) return;
    this.queue = cards.map(c => ({ ...c }));
    this.queueIndex = 0;
    this._playQueueItem(0, 0);
  },

  /** 添加单首到队列尾部 */
  enqueueNext(card) {
    this.queue.push({ ...card });
  },

  /** 跳到队列中指定位置 */
  playQueueIndex(idx) {
    if (idx < 0 || idx >= this.queue.length) return;
    this.queueIndex = idx;
    this._playQueueItem(idx, 0);
  },

  /** 下一首（越界则停止） */
  nextTrack() {
    if (this.queue.length && this.queueIndex < this.queue.length - 1) {
      this.queueIndex++;
      this._playQueueItem(this.queueIndex, 0);
    }
  },

  /** 上一首 */
  prevTrack() {
    if (this.queue.length && this.queueIndex > 0) {
      this.queueIndex--;
      this._playQueueItem(this.queueIndex, 0);
    }
  },

  /** 内部：获取真实 URL 并播放。failCount 记录连续失败次数。 */
  async _playQueueItem(idx, failCount = 0) {
    const card = this.queue[idx];
    if (!card) return;
    const seq = ++this._playSeq;
    try {
      const data = await api.playbackAudio(this.userId, card);
      if (seq !== this._playSeq) return; // 被新请求取代
      if (!data.url) {
        const hints = {
          vip_required: `⚠️《${card.title}》需要 VIP，已跳过`,
          not_found:   `⚠️《${card.title}》无音频链接，已跳过`,
          error:       `⚠️《${card.title}》取流失败，已跳过`,
        };
        this.showToast(hints[data.reason] || `⚠️《${card.title}》暂无试听，已跳过`);
        this._skipToNext(idx, failCount + 1);
        return;
      }
      this.playTrack({
        title: card.title,
        artist: card.artist || "",
        cover: card.cover_url || "",
        url: data.url,
      });
    } catch {
      if (seq !== this._playSeq) return;
      this.showToast(`⚠️《${card.title}》播放失败，已跳过`);
      this._skipToNext(idx, failCount + 1);
    }
  },

  /** 内部：跳到下一首。连续失败 ≥3 次则停止队列。 */
  _skipToNext(currentIdx, failCount) {
    if (failCount >= 3) {
      this.showToast("⚠️ 多首歌曲无法播放，已停止队列");
      return;
    }
    if (currentIdx + 1 < this.queue.length) {
      this.queueIndex = currentIdx + 1;
      this._playQueueItem(currentIdx + 1, failCount);
    }
  },
});
