// 媒体键控制：接管键盘/外设的「上一首 · 下一首 · 播放/暂停」键，并接入系统 Now Playing。
//
// 两条通路，互补：
//   1) Media Session API（navigator.mediaSession）—— 标准通路：硬件媒体键 + OS 媒体控件
//      （macOS 锁屏/控制中心、Windows 媒体覆盖层、Linux MPRIS）。音频成为活动媒体会话后，
//      硬件键由系统直接派发到这里。同时设置 MediaMetadata，让系统控件显示标题/艺人/封面。
//   2) window keydown 兜底——部分浏览器/外设不经过 mediaSession，直接发 MediaTrackNext 等
//      键码；tab 在前台时这里兜住。
//
// 上一首/下一首直接走 store 队列（store.prevTrack/nextTrack，越界自动停）；播放/暂停操作
// PlayerBar 的 <audio> 元素（通过 audioGetter 注入）。幂等：installMediaKeys 重复调用安全。
import { watch } from "vue";
import { store } from "./store.js";

const hasMediaSession = typeof navigator !== "undefined" && "mediaSession" in navigator;

let getAudio = () => null;
let installed = false;

function play() {
  const a = getAudio();
  if (a && a.paused) a.play().catch(() => {});
}
function pause() {
  const a = getAudio();
  if (a && !a.paused) a.pause();
}
function toggle() {
  const a = getAudio();
  if (!a) return;
  if (a.paused) a.play().catch(() => {});
  else a.pause();
}

/** 由 PlayerBar 在 onPlay/onPause 调用，同步 OS 媒体控件的播放态图标（▶/⏸）。 */
export function setMediaPlayState(playing) {
  if (hasMediaSession) navigator.mediaSession.playbackState = playing ? "playing" : "paused";
}

/** 把当前曲目写进系统 Now Playing（标题/艺人/封面）。 */
function syncMetadata() {
  if (!hasMediaSession) return;
  const p = store.player;
  if (!p?.visible || !p.title) {
    navigator.mediaSession.metadata = null;
    return;
  }
  try {
    navigator.mediaSession.metadata = new MediaMetadata({
      title: p.title,
      artist: p.artist || "",
      album: "SonicMind",
      artwork: p.cover ? [{ src: p.cover, sizes: "512x512", type: "image/jpeg" }] : [],
    });
  } catch { /* MediaMetadata 不可用，忽略：键控仍由 keydown 兜底 */ }
}

function onKeydown(e) {
  switch (e.code) {
    case "MediaTrackNext":
    case "MediaFastForward":
      e.preventDefault();
      store.nextTrack();
      break;
    case "MediaTrackPrevious":
    case "MediaRewind":
      e.preventDefault();
      store.prevTrack();
      break;
    case "MediaPlayPause":
      e.preventDefault();
      toggle();
      break;
    case "MediaPlay":
      e.preventDefault();
      play();
      break;
    case "MediaPause":
      e.preventDefault();
      pause();
      break;
    default:
      return;
  }
}

/**
 * 安装媒体键接管。
 * @param {() => HTMLAudioElement | null} audioGetter 返回当前 <audio> 元素（播放/暂停由它执行）。
 */
export function installMediaKeys(audioGetter) {
  if (installed) return;
  installed = true;
  getAudio = audioGetter || (() => null);

  window.addEventListener("keydown", onKeydown);

  if (hasMediaSession) {
    const set = (action, fn) => {
      try { navigator.mediaSession.setActionHandler(action, fn); }
      catch { /* 个别 action 在某些浏览器不支持，跳过；keydown 仍兜底 */ }
    };
    set("play", play);
    set("pause", pause);
    set("previoustrack", () => store.prevTrack());
    set("nexttrack", () => store.nextTrack());
    // 曲目切换时刷新 OS Now Playing 元数据（immediate：首次安装若已在播也立即同步一次）。
    watch(
      () => [store.player.visible, store.player.title, store.player.artist, store.player.cover],
      syncMetadata,
      { immediate: true },
    );
  }
}
