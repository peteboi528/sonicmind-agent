from __future__ import annotations

import streamlit as st

from app.agent import AudioVisualAgent
from app.models import MemoryUpdateRequest

st.set_page_config(page_title="SONICMIND", page_icon="♬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg: #121212;
  --bg-card: #181818;
  --bg-hover: #282828;
  --bg-elevated: #1f1f1f;
  --accent: #1DB954;
  --accent-dim: rgba(29,185,84,0.12);
  --text: #FFFFFF;
  --text-sub: #B3B3B3;
  --text-muted: #6A6A6A;
  --border: #282828;
}

.stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
h1,h2,h3,h4,h5,h6 {
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 1.5rem 2.5rem !important; max-width: 1320px; }
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; height: 0; }
/* Sidebar 展开/收起按钮始终高亮可见。
   Streamlit 1.57 真实 testid: stExpandSidebarButton(收起后用于展开),
   stSidebarCollapseButton(侧栏内用于收起)。旧版 collapsedControl 一并兼容。 */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  visibility: visible !important;
  display: flex !important;
  opacity: 1 !important;
  z-index: 999999 !important;
  background: var(--accent) !important;
  border-radius: 8px !important;
  padding: 4px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}
[data-testid="stExpandSidebarButton"] {
  position: fixed !important;
  top: 12px !important;
  left: 12px !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] button svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] button svg,
[data-testid="collapsedControl"] svg {
  color: #000 !important;
  fill: #000 !important;
  width: 22px !important;
  height: 22px !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover { background: #1ed760 !important; }

section[data-testid="stSidebar"] {
  background: #000000 !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stDivider {
  color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stCaption p {
  color: #B3B3B3 !important;
}

.stTextInput input, .stTextArea textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-size: 14px !important;
}
.stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: none !important;
}

/* 紧凑播放按钮（🎵 只听 / 📺 MV）：通过 keyed 容器 st-key-play* 精准命中 */
[class*="st-key-play"] .stButton > button {
  background: rgba(255,255,255,0.05) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 6px 4px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  min-height: 32px !important;
  letter-spacing: 0 !important;
}
[class*="st-key-play"] .stButton > button:hover {
  background: var(--accent) !important;
  color: #000 !important;
  border-color: var(--accent) !important;
  transform: none !important;
}

/* MV 关闭/打开控制条：固定在视口顶部，盖在全屏 overlay(z=1000) 之上 */
.st-key-mv_controls {
  position: fixed !important;
  top: 18px; left: 50%; transform: translateX(-50%);
  width: 420px; max-width: 90vw; z-index: 1001 !important;
}
.st-key-mv_controls .stButton > button,
.st-key-mv_controls .stLinkButton > a {
  background: rgba(20,20,20,0.92) !important;
  border: 1px solid #3a3a3a !important;
  color: #fff !important;
  border-radius: 500px !important;
  padding: 8px 16px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}
.st-key-mv_controls .stButton > button:hover { background: #e22134 !important; border-color: #e22134 !important; transform: none !important; }
.st-key-mv_controls .stLinkButton > a:hover { background: var(--accent) !important; color: #000 !important; }

/* 关闭音频按钮：小巧次要样式 */
.st-key-audio_close .stButton > button {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text-sub) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 6px 16px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
}
.st-key-audio_close .stButton > button:hover { background: #e22134 !important; color: #fff !important; border-color: #e22134 !important; transform: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stButton > button,
.stFormSubmitButton > button {
  background: var(--accent) !important;
  color: #000 !important;
  border: none !important;
  border-radius: 500px !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 12px 32px !important;
  letter-spacing: 0.02em;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
  background: #1ed760 !important;
  transform: scale(1.02);
}

.stTabs [data-baseweb="tab-list"] {
  background: transparent;
  gap: 8px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
  color: var(--text-sub) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 12px 16px !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  background: transparent !important;
  border-bottom: 2px solid var(--accent) !important;
}

.sm-hero {
  padding: 32px 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.sm-brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}
.sm-brand-mark {
  position: relative;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background:
    radial-gradient(circle at 50% 50%, #050505 0 16%, transparent 17%),
    conic-gradient(from 210deg, #1ed760, #1DB954, #8cffb5, #1DB954);
  box-shadow: 0 0 0 1px rgba(29,185,84,0.32), 0 8px 22px rgba(29,185,84,0.22);
  flex: 0 0 auto;
}
.sm-brand-mark::before,
.sm-brand-mark::after {
  content: '';
  position: absolute;
  top: 9px;
  width: 2px;
  border-radius: 999px;
  background: #07130c;
  opacity: 0.85;
}
.sm-brand-mark::before {
  right: 9px;
  height: 9px;
  box-shadow: -5px 2px 0 #07130c, -10px 4px 0 #07130c;
}
.sm-brand-mark::after {
  right: 5px;
  height: 13px;
}
.sm-brand-word {
  color: var(--text);
  font-size: 15px;
  font-weight: 900;
  letter-spacing: 0.14em;
  line-height: 1;
}
.sm-brand-sub {
  color: var(--text-sub);
  font-size: 11px;
  margin-top: 5px;
}
.sm-brand-small .sm-brand-mark {
  width: 24px;
  height: 24px;
}
.sm-brand-small .sm-brand-word {
  color: var(--accent);
  font-size: 15px;
}
.sm-logo {
  font-family: 'Inter', sans-serif;
  font-weight: 900;
  font-size: 11px;
  letter-spacing: 0.2em;
  color: var(--text-sub);
  text-transform: uppercase;
}
.sm-headline {
  font-size: 28px;
  font-weight: 800;
  color: var(--text);
  margin: 8px 0 4px;
  line-height: 1.2;
}
.sm-sub {
  font-size: 14px;
  color: var(--text-sub);
}
.sm-status {
  display: inline-block;
  font-size: 11px;
  color: var(--accent);
  font-weight: 600;
  margin-top: 8px;
}
.sm-status::before {
  content: '';
  display: inline-block;
  width: 6px; height: 6px;
  background: var(--accent);
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
}

.sm-track {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 10px 14px;
  border-radius: 6px;
  margin-bottom: 2px;
  transition: background 0.15s;
}
.sm-track:hover { background: var(--bg-hover); }
.sm-play {
  width: 32px; height: 32px;
  border-radius: 50%;
  background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; flex-shrink: 0;
}
.sm-play::after {
  content: '';
  border-style: solid;
  border-width: 5px 0 5px 9px;
  border-color: transparent transparent transparent #000;
}
.sm-num { color: var(--text-muted); font-size: 14px; min-width: 20px; font-variant-numeric: tabular-nums; }
.sm-info { flex: 1; min-width: 0; }
.sm-title { color: var(--text); font-size: 15px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sm-artist { color: var(--text-sub); font-size: 13px; }
.sm-reason { color: var(--text-muted); font-size: 12px; margin-top: 1px; }
.sm-tag {
  font-size: 10px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.tag-f { background: var(--accent-dim); color: var(--accent); }
.tag-d { background: rgba(59,130,246,0.12); color: #60a5fa; }
.tag-m { background: rgba(251,146,60,0.12); color: #fb923c; }

.sm-card {
  background: var(--bg-card);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
}

.sm-chat-user {
  background: var(--bg-hover);
  border-radius: 16px 16px 4px 16px;
  padding: 12px 16px;
  margin: 8px 0 8px 20%;
  font-size: 14px;
}
.sm-chat-agent {
  background: var(--bg-card);
  border-radius: 16px 16px 16px 4px;
  padding: 12px 16px;
  margin: 8px 20% 8px 0;
  font-size: 14px;
  border-left: 2px solid var(--accent);
}

/* Agent markdown reply: keep generated content readable inside chat.
   Streamlit markdown defaults make ### headings huge and tables too dark on
   this Spotify-like theme, so scope typography to assistant chat messages. */
[data-testid="stChatMessage"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-left: 2px solid var(--accent) !important;
  border-radius: 12px !important;
  padding: 14px 16px !important;
  margin: 10px 12% 14px 0 !important;
}
[data-testid="stChatMessageAvatar"] {
  background:
    radial-gradient(circle at 50% 50%, #0b0b0b 0 24%, transparent 25%),
    linear-gradient(135deg, #1ed760, #1DB954) !important;
  color: #07130c !important;
  border: 1px solid rgba(29,185,84,0.45) !important;
  box-shadow: 0 0 0 3px rgba(29,185,84,0.12), 0 10px 24px rgba(0,0,0,0.35) !important;
  font-size: 17px !important;
  font-weight: 900 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  color: var(--text-sub) !important;
  font-size: 14px !important;
  line-height: 1.72 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
  color: var(--text-sub) !important;
  font-size: 14px !important;
  line-height: 1.72 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {
  color: var(--text) !important;
  font-size: 17px !important;
  line-height: 1.35 !important;
  font-weight: 800 !important;
  margin: 14px 0 8px !important;
  letter-spacing: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3:first-child {
  margin-top: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
  color: var(--text) !important;
  font-weight: 700 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
  margin: 8px 0 10px 18px !important;
  padding-left: 14px !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
  border-left: 2px solid var(--accent) !important;
  color: var(--text-sub) !important;
  background: rgba(255,255,255,0.035) !important;
  margin: 12px 0 !important;
  padding: 10px 14px !important;
  border-radius: 0 8px 8px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] hr {
  border: 0 !important;
  border-top: 1px solid var(--border) !important;
  margin: 14px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
  width: 100% !important;
  border-collapse: collapse !important;
  margin: 12px 0 !important;
  font-size: 13px !important;
  color: var(--text-sub) !important;
  background: transparent !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
  border: 1px solid var(--border) !important;
  padding: 9px 10px !important;
  color: var(--text-sub) !important;
  background: rgba(255,255,255,0.015) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
  color: var(--text) !important;
  font-weight: 700 !important;
  background: rgba(255,255,255,0.045) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
  color: var(--accent) !important;
  background: rgba(29,185,84,0.12) !important;
  border-radius: 4px !important;
  padding: 1px 5px !important;
}

/* Number input - Pitchfork circle style */
div[data-testid="stNumberInput"] { display: none !important; }

.pfk-score {
  width: 52px;
  height: 52px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  font-weight: 700;
  margin: 0 auto;
  transition: all 0.2s;
}
.pfk-none {
  border: 2px dashed var(--text-muted);
  color: var(--text-muted);
}
.pfk-low {
  border: 2px solid #B3B3B3;
  color: #B3B3B3;
  background: transparent;
}
.pfk-mid {
  border: 2px solid #FFFFFF;
  color: #FFFFFF;
  background: transparent;
}
.pfk-good {
  border: 2px solid #FFFFFF;
  color: #000000;
  background: #FFFFFF;
}
.pfk-best {
  border: 2px solid #FFFFFF;
  color: #000000;
  background: #FFFFFF;
  font-weight: 800;
}

/* Delete circle — matches pfk-score exactly, different color only */
.stButton:has(> button[aria-label="✕"]) {
  display: flex !important;
  justify-content: center !important;
}
.stButton > button[aria-label="✕"] {
  width: 52px !important;
  height: 52px !important;
  border-radius: 50% !important;
  padding: 0 !important;
  background: transparent !important;
  border: 2px solid #ff4444 !important;
  color: #ff4444 !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  font-family: 'Inter', sans-serif !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  letter-spacing: normal !important;
  margin: 0 auto !important;
  transition: all 0.2s;
}
.stButton > button[aria-label="✕"]:hover {
  background: rgba(255, 68, 68, 0.12) !important;
  border-color: #ff6666 !important;
  color: #ff6666 !important;
  transform: scale(1.05);
}

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  width: 14px !important;
  height: 14px !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] { display: none; }
.stSlider p { display: none !important; }

div[data-testid="stMetric"] {
  background: var(--bg-card);
  border-radius: 8px;
  padding: 14px;
}

/* ── Spotify-style fixed bottom bar ──────────────────────────────── */
@keyframes discSpin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
.sm-bottom-bar {
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
  height: 80px;
  background: rgba(8,8,8,0.96);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border-top: 1px solid var(--border);
  display: flex; align-items: center;
  padding: 0 28px; gap: 18px;
}
.sm-bar-disc {
  width: 56px; height: 56px; flex-shrink: 0;
  border-radius: 50%;
  background: radial-gradient(circle,
    #1a1a1a 30%, #222 31%, #1a1a1a 40%,
    #282828 41%, #1a1a1a 60%, #333 100%);
  display: flex; align-items: center; justify-content: center;
  animation: discSpin 3s linear infinite;
  box-shadow: 0 0 12px rgba(0,0,0,0.4);
}
.sm-bar-cover {
  width: 32px; height: 32px; border-radius: 50%;
  object-fit: cover; border: 1px solid #444;
}
.sm-bar-info { flex: 1; min-width: 0; }
.sm-bar-title {
  color: var(--text); font-size: 13px; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sm-bar-artist {
  color: var(--text-sub); font-size: 11px; margin-top: 1px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sm-bar-audio {
  flex-shrink: 0; min-width: 280px;
}
.sm-bar-audio audio {
  height: 36px;
  filter: invert(1) hue-rotate(180deg) brightness(0.85) contrast(1.1);
  border-radius: 8px;
}
.sm-bar-close {
  flex-shrink: 0;
  background: none; border: none; color: var(--text-muted);
  font-size: 16px; cursor: pointer; padding: 4px 8px;
}
.sm-bar-close:hover { color: var(--text); }
.sm-bottom-spacer { height: 92px; }

/* ── Floating MV overlay (Quark-style) ────────────────────────────── */
.sm-mv-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  z-index: 1000;
  background: rgba(0,0,0,0.45);
  backdrop-filter: blur(28px);
  -webkit-backdrop-filter: blur(28px);
  display: flex; align-items: center; justify-content: center;
}
.sm-mv-card {
  width: 860px; max-width: 92vw;
  background: #0a0a0a;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 12px 64px rgba(0,0,0,0.7);
  border: 1px solid #222;
}
.sm-mv-top {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px;
}
.sm-mv-song {
  color: var(--text); font-size: 14px; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sm-mv-close {
  background: none; border: none; color: var(--text-muted);
  font-size: 16px; cursor: pointer; flex-shrink: 0; margin-left: 16px;
  padding: 4px 8px;
}
.sm-mv-close:hover { color: var(--text); }
.sm-mv-bottom {
  padding: 10px 20px;
  border-top: 1px solid #1a1a1a;
}
.sm-mv-artist {
  color: var(--text-sub); font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── 关闭播放器（兼容旧的 query param 信号，正常走 st.button） ──
if st.query_params.get("close_video"):
    for k in ("video_url", "video_title", "video_ext_url"):
        st.session_state.pop(k, None)
    st.query_params.clear()
    st.rerun()
if st.query_params.get("close_audio"):
    for k in ("audio_url", "audio_title", "audio_cover"):
        st.session_state.pop(k, None)
    st.query_params.clear()
    st.rerun()

@st.cache_resource
def get_agent() -> AudioVisualAgent:
    return AudioVisualAgent()


agent = get_agent()


def play_audio(t) -> None:
    """只听歌：拿网易云音频直链，放进底部播放条。"""
    _ck = st.session_state.get("netease_cookie", "")
    audio_url = agent.get_audio_url(t, netease_cookie=_ck)
    if audio_url:
        st.session_state["audio_url"] = audio_url
        st.session_state["audio_title"] = f"{t.title} - {getattr(t, 'artist', '') or ''}"
        st.session_state["audio_cover"] = getattr(t, "cover_url", "") or ""
        st.rerun()
    else:
        st.toast("没找到可播放的音频，试试看 MV", icon="⚠️")


def play_mv(t) -> None:
    """看 MV：拿视频嵌入地址，弹出浮动播放器。"""
    mv_url = agent.get_mv_url(t)
    ext_url = agent.get_external_url(t)
    if mv_url:
        st.session_state["video_url"] = mv_url
        st.session_state["video_ext_url"] = ext_url or ""
        st.session_state["video_title"] = f"{t.title} - {getattr(t, 'artist', '') or ''}"
        st.rerun()
    elif ext_url:
        st.session_state["video_url"] = ""
        st.session_state["video_ext_url"] = ext_url
        st.session_state["video_title"] = f"{t.title} - {getattr(t, 'artist', '') or ''}"
        st.rerun()
    else:
        st.toast("没找到 MV，试试只听歌", icon="⚠️")


def _card_to_obj(card: dict):
    """把流式 candidates 事件里的卡片 dict 适配成 play_audio/play_mv 需要的对象。"""
    from types import SimpleNamespace

    return SimpleNamespace(
        title=card.get("title", ""),
        artist=card.get("artist", "") or "",
        source=card.get("source", "local"),
        source_id=card.get("source_id", "") or "",
        external_id=card.get("source_id", "") or "",
        playback_url=card.get("playback_url"),
        source_url=card.get("playback_url"),
        cover_url=card.get("cover_url", "") or "",
        genre=card.get("genre", []) or [],
        mood=card.get("mood", []) or [],
    )


def render_chat_cards(cards: list[dict], turn_key: str) -> None:
    """在聊天回复下渲染推荐歌曲卡片，每张带 只听/MV/不喜欢 按钮。"""
    if not cards:
        return
    for i, card in enumerate(cards):
        obj = _card_to_obj(card)
        meta = " · ".join(filter(None, [", ".join(card.get("genre", [])), ", ".join(card.get("mood", []))]))
        score = card.get("score")
        score_html = f"<span class='sm-tag tag-d'>{score:.2f}</span>" if isinstance(score, (int, float)) else ""
        c1, c2 = st.columns([0.72, 0.28])
        with c1:
            st.markdown(f"""<div class="sm-track">
              <span class="sm-num">{i + 1}</span>
              <div class="sm-info">
                <div class="sm-title">{obj.title}</div>
                <div class="sm-artist">{obj.artist or '未知'} · {obj.source}{(' · ' + meta) if meta else ''}</div>
                <div class="sm-reason">{card.get('reason', '') or ''}</div>
              </div>
              {score_html}
            </div>""", unsafe_allow_html=True)
        with c2:
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🎵", key=f"ca_{turn_key}_{i}", help="只听歌"):
                    play_audio(obj)
            with b2:
                if st.button("📺", key=f"cm_{turn_key}_{i}", help="看 MV"):
                    play_mv(obj)
            with b3:
                if st.button("👎", key=f"cd_{turn_key}_{i}", help="不喜欢（负反馈给 Thompson + 加入屏蔽）"):
                    from app.models import DislikeRequest
                    agent.record_dislike(DislikeRequest(
                        user_id=user_id,
                        title=obj.title,
                        artist=obj.artist,
                        source=obj.source,
                        source_id=obj.source_id,
                        reason="ui_dislike",
                    ))
                    st.toast(f"已记录不喜欢：{obj.title}", icon="👎")
                    st.rerun()


def render_transparency_panel(meta: dict, turn_key: str) -> None:
    """Agent 透明度面板：决策过程 / 记忆变化 / 三锚打分明细。

    数据来源：trace（Phase 0-2 各节点写入）、cards.components（Phase 1 三锚精排）。
    """
    trace_text = meta.get("trace", "") or ""
    cards = meta.get("cards", []) or []
    trace_lines = [ln for ln in trace_text.splitlines() if ln.strip()]
    has_components = any(c.get("components") for c in cards)
    if not trace_lines and not has_components:
        return

    with st.expander("🔬 Agent 透明度面板", expanded=False):
        # 1) 决策过程：把 trace 按节点归类
        st.markdown("**🧠 决策过程**")
        stage_icons = {
            "plan": "🎯", "stream:plan": "🎯", "load_context": "📂", "gssc": "📐",
            "web_music_search": "🌐", "stream:candidates": "🎵", "recommend": "🎁",
            "playlist": "📋", "search": "🔍", "import": "📥", "web_fallback": "🔄",
            "eval": "⚖️", "guard": "🛡️", "final": "✅",
        }
        for ln in trace_lines:
            icon = next((v for k, v in stage_icons.items() if k in ln.lower()), "•")
            st.caption(f"{icon} {ln}")

        # 2) 记忆变化 / 预算：从 trace 里抽 GSSC 预算行与 guard 行
        budget_lines = [ln for ln in trace_lines if "gssc" in ln.lower() or "预算" in ln]
        guard_lines = [ln for ln in trace_lines if "guard" in ln.lower() or "移除" in ln]
        if budget_lines or guard_lines:
            st.markdown("**📊 上下文预算 / 反幻觉**")
            for ln in budget_lines + guard_lines:
                st.caption(ln.strip())

        # 3) 三锚打分明细
        if has_components:
            st.markdown("**🧮 三锚精排打分**")
            rows = []
            for c in cards:
                comp = c.get("components", {})
                if not comp:
                    continue
                rows.append({
                    "歌曲": c.get("title", "")[:20],
                    "总分": c.get("score"),
                    "语义": comp.get("semantic"),
                    "口味": comp.get("personalize"),
                    "行为": comp.get("behavior"),
                    "权重(语/味/为)": f"{comp.get('w_semantic', 0):.2f}/{comp.get('w_personalize', 0):.2f}/{comp.get('w_behavior', 0):.2f}",
                })
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.caption("总分 = 语义×w_语义 + 口味×w_口味 + 行为×w_行为（缺锚时权重自动重分配）")


def render_trace(title: str, trace: list[str]) -> None:
    if not trace:
        return
    st.markdown(
        f"<div class='sm-card'><span style='color:var(--accent);font-size:11px;font-weight:700;letter-spacing:0.1em'>{title}</span>"
        f"<p style='font-size:13px;color:var(--text-sub);margin:8px 0 0'>{'<br/>'.join(trace)}</p></div>",
        unsafe_allow_html=True,
    )


def render_evidences(evidences: list) -> None:
    if not evidences:
        return
    st.markdown("**Evidence**")
    for evidence in evidences[:4]:
        asset_title = evidence.metadata.get("asset_title", "")
        label = f"{asset_title} · " if asset_title else ""
        st.caption(f"{label}{evidence.timestamp} · {evidence.modality} · score {evidence.similarity}")
        st.write(evidence.content)

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        "<div class='sm-brand sm-brand-small'>"
        "<div class='sm-brand-mark'></div>"
        "<div><div class='sm-brand-word'>SONICMIND</div>"
        "<div class='sm-brand-sub'>你的音乐智能体</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    user_id = st.text_input("当前用户", value="demo-user",
                            help="用于区分不同用户的音乐库、歌单、偏好和网易云绑定。切换名称即切换档案。")
    st.caption(f"当前档案：{user_id}")
    st.divider()

    st.markdown("### 🎧 添加音乐")
    url = st.text_input("链接", placeholder="YouTube / B站链接", label_visibility="collapsed")
    force_refresh = st.checkbox("强制刷新同链接缓存", value=False)
    skip_enrich = st.checkbox("跳过联网识别（仅离线分析）", value=False,
                              help="勾选后只用 URL 解析标题，不调 LLM 识别歌名歌手")
    if st.button("入库", use_container_width=True):
        if url:
            try:
                with st.spinner("步骤 1/3 入库..."):
                    asset = agent.ingest_video(url, force_refresh=force_refresh)
                if not skip_enrich:
                    with st.spinner("步骤 2/3 识别歌名歌手..."):
                        enrich = agent.enrich_asset(asset.asset_id, use_network=True)
                        asset = enrich.asset
                with st.spinner("步骤 3/3 生成片段..."):
                    asset, _ = agent.analyze_media(asset.asset_id, force_refresh=force_refresh)
                title = asset.title
                artist = asset.artist or "?"
                genre = "、".join(asset.genre) if asset.genre else "未知风格"
                st.success(f"✅ {title} — {artist}（{genre}）")
            except Exception as exc:
                st.error(f"入库失败: {exc}")
    st.caption("入库流程：解析 URL → LLM 识别元数据 → 生成片段。可勾选跳过联网识别。")

    st.divider()
    st.markdown("**导入网易云歌单**")
    pl_url = st.text_input(
        "歌单链接", placeholder="粘贴网易云歌单链接或 id",
        label_visibility="collapsed", key="ne_pl_url",
    )
    pl_limit = st.slider("最多导入", 10, 500, 200, step=10, key="ne_pl_limit")
    if st.button("导入歌单", use_container_width=True, key="ne_pl_btn"):
        if pl_url:
            _ck = st.session_state.get("netease_cookie", "")
            try:
                with st.spinner("拉取歌单并导入中..."):
                    res = agent.import_netease_playlist(
                        pl_url, cookie=_ck, user_id=user_id, limit=pl_limit,
                    )
                st.success(
                    f"✅ 《{res['name'] or '歌单'}》共 {res['total']} 首，"
                    f"新增 {res['imported']} 首，跳过 {res['skipped']} 首已存在。"
                )
            except Exception as exc:
                st.error(f"导入失败: {exc}")
        else:
            st.warning("请先粘贴歌单链接")

    # 登录后：列出"我的歌单"供直接选择
    _ck = st.session_state.get("netease_cookie", "")
    if _ck:
        from app.netease_auth import fetch_user_playlists
        if st.button("加载我的歌单", use_container_width=True, key="ne_my_pl_btn"):
            with st.spinner("加载中..."):
                st.session_state["_my_playlists"] = fetch_user_playlists(_ck)
        my_pls = st.session_state.get("_my_playlists") or []
        if my_pls:
            options = {f"{p['name']}（{p['count']}首）": p["id"] for p in my_pls}
            chosen = st.selectbox("我的歌单", list(options.keys()), key="ne_my_pl_sel")
            if st.button("导入选中歌单", use_container_width=True, key="ne_my_pl_imp"):
                try:
                    with st.spinner("导入中..."):
                        res = agent.import_netease_playlist(
                            options[chosen], cookie=_ck, user_id=user_id, limit=pl_limit,
                        )
                    st.success(
                        f"✅ 《{res['name'] or '歌单'}》新增 {res['imported']} 首，"
                        f"跳过 {res['skipped']} 首。"
                    )
                except Exception as exc:
                    st.error(f"导入失败: {exc}")
    st.divider()
    st.markdown("**训练偏好**")
    pref = st.text_input("偏好", placeholder="我喜欢...", label_visibility="collapsed")
    if st.button("学习", use_container_width=True):
        if pref:
            agent.update_memory(MemoryUpdateRequest(user_id=user_id, event=pref))
            st.success("已记住")
    st.divider()
    st.markdown("**网易云账号**")
    import io

    import qrcode as _qr

    from app.netease_auth import check_qr_status, clear_cookie, get_qr_key, load_cookie, save_cookie

    _saved = load_cookie(user_id)
    if _saved and _saved.get("cookie"):
        st.session_state["netease_cookie"] = _saved["cookie"]
        if _saved.get("avatar"):
            st.image(_saved["avatar"], width=36)
        if _saved.get("nickname"):
            st.caption(f"已绑定：{_saved['nickname']}")
        else:
            st.caption("已绑定网易云账号")
        _vip_label = _saved.get("vip_label", "")
        if _saved.get("vip_type", 0) > 0:
            st.caption(f"🎫 {_vip_label or '黑胶 VIP'} · 可播放会员歌曲")
        elif _vip_label:
            st.caption(f"⚪ {_vip_label} · VIP 歌曲可能无法播放")
        if st.button("解除绑定", use_container_width=True):
            clear_cookie(user_id)
            st.session_state.pop("netease_cookie", None)
            st.rerun()
    else:
        if st.button("扫码登录网易云", use_container_width=True):
            with st.spinner("获取二维码..."):
                try:
                    unikey = get_qr_key()
                    st.session_state["_qr_key"] = unikey
                    qr_url = f"https://music.163.com/login?codekey={unikey}"
                    img = _qr.make(qr_url)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.session_state["_qr_img"] = buf.getvalue()
                except Exception as e:
                    st.error(f"获取失败: {e}")
        if "_qr_img" in st.session_state:
            st.image(st.session_state["_qr_img"], width=180)
            st.caption("打开网易云 APP 扫码登录")

            # Auto-polling: check every time the page reruns
            if "_qr_key" in st.session_state:
                import time
                _last_check = st.session_state.get("_qr_last_check", 0)
                _now = time.time()
                if _now - _last_check >= 2:
                    st.session_state["_qr_last_check"] = _now
                    try:
                        result = check_qr_status(st.session_state["_qr_key"])
                        code = result.get("code", 800)

                        if code == 803:
                            music_u = result.get("cookie") or ""
                            if music_u:
                                save_cookie(
                                    user_id, music_u,
                                    result.get("nickname"), result.get("avatar"),
                                    vip_type=result.get("vip_type", 0),
                                    vip_label=result.get("vip_label", ""),
                                )
                                st.session_state["netease_cookie"] = music_u
                                for k in ("_qr_key", "_qr_img", "_qr_last_check"):
                                    st.session_state.pop(k, None)
                                _vt = result.get("vip_type", 0)
                                _vl = result.get("vip_label", "")
                                if _vt > 0:
                                    st.success(f"✅ 绑定成功！{_vl}，可播放会员歌曲")
                                else:
                                    st.success(f"✅ 绑定成功！当前为{_vl or '非会员'}")
                                st.rerun()
                            else:
                                # 803 but no cookie — show debug info
                                raw = result.get("raw", {})
                                cookies = raw.get("_cookies", {})
                                st.warning("登录成功但未获取到 MUSIC_U cookie")
                                with st.expander("调试信息"):
                                    st.json({k: v for k, v in raw.items() if k != "_cookies"})
                                    st.caption(f"Set-Cookie: {cookies}")
                        elif code == 802:
                            st.info("📱 已扫描，请在手机上确认...")
                        elif code == 800:
                            st.warning("二维码已过期，请重新获取")
                            for k in ("_qr_key", "_qr_img", "_qr_last_check"):
                                st.session_state.pop(k, None)
                        else:
                            st.caption("🔄 等待扫码中...")
                    except Exception as e:
                        st.caption(f"⚠️ 检查失败: {e}")

                # 用 sleep+rerun 轮询，而不是 meta 整页刷新。
                # meta refresh 会新建会话、清空 session_state（含 _qr_key），导致扫码永远无法完成。
                import time as _time
                _time.sleep(2)
                st.rerun()
    st.divider()
    mem = agent.memory.get_memory(user_id)
    if mem.preferences:
        st.markdown("**已学习**")
        for p in mem.preferences[-4:]:
            st.caption(f"• {p}")

# --- Hero ---
library = agent.list_assets()
mem = agent.memory.get_memory(user_id)
taste = mem.taste_profile

genre_tags = " ".join([f"<span class='sm-tag tag-f'>{g}</span>" for g, _ in (taste.top_genres[:4] if taste else [])])
rated_count = len(mem.ratings)
listened_count = len(mem.listening_history)

st.markdown(f"""
<div class="sm-hero">
  <div class="sm-brand">
    <div class="sm-brand-mark"></div>
    <div class="sm-brand-word">SONICMIND</div>
  </div>
  <div class="sm-headline">为你而听</div>
  <div class="sm-sub">AI Agent · 持续学习你的品味 · 每次交互都在进化</div>
  <div class="sm-status">Agent 在线 · 已学习 {listened_count} 次收听 · {rated_count} 条评分 · {len(library)} 首入库</div>
  <div style="margin-top:12px">{genre_tags}</div>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["今日推荐", "发现", "我的库", "歌单", "对话"])

TAG_CLS = {"familiar": "tag-f", "discovery": "tag-d", "mood_match": "tag-m"}
TAG_TEXT = {"familiar": "SIMILAR", "discovery": "DISCOVER", "mood_match": "MOOD"}

with tab1:
    c1, c2 = st.columns([0.8, 0.2])
    with c2:
        tod = st.selectbox("", ["morning", "focus", "afternoon", "evening", "night"], index=3,
                          format_func=lambda x: {"morning": "早晨", "focus": "专注", "afternoon": "午后", "evening": "傍晚", "night": "深夜"}[x], label_visibility="collapsed")
    with c1:
        if st.button("生成歌单", use_container_width=True):
            with st.spinner("分析品味中..."):
                st.session_state["daily_rec"] = agent.daily_recommend(user_id, time_of_day=tod)
    rec = st.session_state.get("daily_rec")
    if rec:
        st.markdown(f"""<div class='sm-card'>
          <span style='color:var(--accent);font-size:11px;font-weight:700;letter-spacing:0.1em'>AGENT REASONING</span>
          <p style='font-size:14px;margin:6px 0 0;color:var(--text-sub)'>{rec.reason_summary}</p>
        </div>""", unsafe_allow_html=True)
        render_trace("RECOMMEND TRACE", rec.agent_trace)
        render_evidences(rec.evidences)
        for i, tr in enumerate(rec.tracks, 1):
            t = tr.asset
            tag_cls = TAG_CLS.get(tr.category, "tag-f")
            tag_text = TAG_TEXT.get(tr.category, "")
            c1, c2 = st.columns([0.78, 0.22])
            with c1:
                st.markdown(f"""<div class="sm-track">
                  <span class="sm-num">{i}</span>
                  <div class="sm-info">
                    <div class="sm-title">{t.title}</div>
                    <div class="sm-artist">{getattr(t,'artist','') or ''} · {', '.join(getattr(t,'genre',[]))}</div>
                    <div class="sm-reason">{tr.reason}</div>
                  </div>
                  <span class="sm-tag {tag_cls}">{tag_text}</span>
                </div>""", unsafe_allow_html=True)
            with c2:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🎵 只听", key=f"playaud_rec_{i}", help="只听歌（网易云音频）"):
                        play_audio(t)
                with b2:
                    if st.button("📺 MV", key=f"playmv_rec_{i}", help="看 MV（视频）"):
                        play_mv(t)

with tab2:
    q = st.text_input("", placeholder="搜索歌手、风格、情绪...", label_visibility="collapsed")
    if st.button("搜索", key="s_btn", use_container_width=True) and q:
        with st.spinner(""):
            st.session_state["sr"] = agent.search(user_id, q, include_external=True, top_k=12)
    sr = st.session_state.get("sr")
    if sr:
        st.markdown(f"<div class='sm-card'><span style='color:var(--accent);font-size:12px;font-weight:700'>SEARCH</span><br/>{sr.summary}</div>", unsafe_allow_html=True)
        render_trace("SEARCH TRACE", sr.agent_trace)
        render_evidences(sr.evidences)
        for a in sr.local[:8]:
            st.markdown(f"<div class='sm-track'><div class='sm-info'><div class='sm-title'>{a.title}</div><div class='sm-artist'>{a.artist or ''} · {', '.join(a.genre)}</div></div></div>", unsafe_allow_html=True)
        for t in sr.external[:8]:
            st.markdown(f"<div class='sm-track'><div class='sm-info'><div class='sm-title'>{t.title}</div><div class='sm-artist'>{t.artist} · {', '.join(t.genre)}</div></div><span class='sm-tag tag-d'>EXT</span></div>", unsafe_allow_html=True)

with tab3:
    assets = agent.list_assets()
    if not assets:
        st.caption("添加音乐链接开始建库")
    else:
        mem_lib = agent.memory.get_memory(user_id)
        rmap = {r.asset_id: r.score for r in mem_lib.ratings}
        taste_lib = mem_lib.taste_profile
        if taste_lib and taste_lib.top_genres:
            genres_str = " → ".join([f"{g} ({w:.0f})" for g, w in taste_lib.top_genres[:4]])
            st.markdown(f"<div class='sm-card'><span style='color:var(--accent);font-size:11px;font-weight:700;letter-spacing:0.1em'>TASTE MODEL</span><p style='font-size:13px;color:var(--text-sub);margin:6px 0 0'>{genres_str}</p></div>", unsafe_allow_html=True)
        st.caption(f"{len(assets)} 首 · 评分越高，该风格推荐权重越大")
        for a in assets:
            c1, c2, c3, c4 = st.columns([0.56, 0.2, 0.12, 0.12])
            with c1:
                st.markdown(f"<div class='sm-track'><div class='sm-info'><div class='sm-title'>{a.title}</div><div class='sm-artist'>{a.artist or '?'} · {', '.join(a.genre)}</div></div></div>", unsafe_allow_html=True)
            with c2:
                cur = rmap.get(a.asset_id, 0.0)
                sc = st.slider("", min_value=0.0, max_value=10.0, value=float(cur), step=0.1,
                    key=f"r_{a.asset_id}", label_visibility="collapsed")
                if sc != cur and sc > 0:
                    agent.rate_asset(user_id, a.asset_id, sc)
                    st.toast(f"{a.title} → {sc:.1f}")
            with c3:
                display_score = rmap.get(a.asset_id, 0.0)
                if sc > 0:
                    display_score = sc
                if display_score == 0:
                    cls = "pfk-none"
                    label = "—"
                elif display_score < 4.0:
                    cls = "pfk-low"
                    label = f"{display_score:.1f}"
                elif display_score < 7.0:
                    cls = "pfk-mid"
                    label = f"{display_score:.1f}"
                elif display_score < 9.0:
                    cls = "pfk-good"
                    label = f"{display_score:.1f}"
                else:
                    cls = "pfk-best"
                    label = f"{display_score:.1f}"
                st.markdown(f"<div class='pfk-score {cls}'>{label}</div>", unsafe_allow_html=True)
            with c4:
                pending_delete = st.session_state.get("delete_asset_id") == a.asset_id
                if not pending_delete and st.button("✕", key=f"delete_asset_{a.asset_id}", use_container_width=True):
                    st.session_state["delete_asset_id"] = a.asset_id
                    st.rerun()

            if st.session_state.get("delete_asset_id") == a.asset_id:
                st.warning(f"确认从库中删除《{a.title}》？评分和收听记录也会同步移除。")
                confirm_col, cancel_col = st.columns([0.5, 0.5])
                with confirm_col:
                    if st.button("确认删除", key=f"confirm_delete_asset_{a.asset_id}", use_container_width=True):
                        if agent.delete_asset(a.asset_id, user_id=user_id):
                            st.toast(f"已删除：{a.title}")
                        st.session_state.pop("delete_asset_id", None)
                        st.rerun()
                with cancel_col:
                    if st.button("取消", key=f"cancel_delete_asset_{a.asset_id}", use_container_width=True):
                        st.session_state.pop("delete_asset_id", None)
                        st.rerun()

with tab4:
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        pl_instruction = st.text_input("", placeholder="帮我生成一个深夜放松歌单 / 说唱合集...", key="pl_input", label_visibility="collapsed")
    with c2:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("生成歌单", key="gen_pl", use_container_width=True) and pl_instruction:
                with st.spinner("Agent 生成中..."):
                    pl = agent.generate_playlist(user_id, pl_instruction)
                    st.session_state["last_playlist"] = pl
                st.rerun()
        with col_b:
            if st.button("自动分类", key="auto_pl", use_container_width=True):
                with st.spinner("Agent 分析中..."):
                    pls = agent.auto_playlists(user_id)
                    if pls:
                        st.toast(f"已生成 {len(pls)} 个歌单")
                st.rerun()

    saved = agent.list_playlists(user_id)
    if saved:
        for pl in saved:
            with st.expander(f"**{pl.name}** · {len(pl.tracks)} 首 · {pl.generated_by}"):
                st.caption(pl.description)
                for j, t in enumerate(pl.tracks):
                    title = t.title if hasattr(t, "title") else ""
                    artist = getattr(t, "artist", "") or ""
                    c1, c2 = st.columns([0.7, 0.3])
                    with c1:
                        st.markdown(f"<div class='sm-track'><div class='sm-info'><div class='sm-title'>{title}</div><div class='sm-artist'>{artist}</div></div></div>", unsafe_allow_html=True)
                    with c2:
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("🎵", key=f"playaud_{pl.playlist_id}_{j}", help="只听歌"):
                                play_audio(t)
                        with b2:
                            if st.button("📺", key=f"playmv_{pl.playlist_id}_{j}", help="看 MV"):
                                play_mv(t)
                                st.rerun()
                if st.button("删除歌单", key=f"del_{pl.playlist_id}"):
                    agent.delete_playlist(user_id, pl.playlist_id)
                    st.rerun()
    else:
        st.caption("还没有歌单，输入指令或点击自动分类来生成")

with tab5:
    if "ch" not in st.session_state:
        st.session_state["ch"] = [("a", agent.generate_greeting(user_id), {
            "trace": "[greeting] 读取用户记忆、活跃目标、最近播放和曲库状态。 → 主动生成开场建议。",
            "evidence": "",
            "goal": "",
        })]
    for turn_idx, entry in enumerate(st.session_state["ch"]):
        role, txt = entry[0], entry[1]
        if role == "u":
            st.markdown(f"<div class='sm-chat-user'>{txt}</div>", unsafe_allow_html=True)
            continue
        # agent 正文用原生 markdown 渲染（解析 **加粗** / ## 标题 / --- 等 GFM 语法）
        with st.chat_message("assistant", avatar="🎧"):
            st.markdown(txt)
            meta = entry[2] if len(entry) > 2 else {}
            if meta.get("cards"):
                render_chat_cards(meta["cards"], turn_key=str(turn_idx))
            render_transparency_panel(meta, turn_key=str(turn_idx))
            if meta.get("evidence"):
                with st.expander("📎 Evidence"):
                    st.text(meta["evidence"])
            if meta.get("goal"):
                with st.expander("🎯 Goal"):
                    st.text(meta["goal"])
    msg = None
    with st.form(key="chat_form", clear_on_submit=True):
        msg = st.text_input("", placeholder="分析我的品味 / 推荐类似的歌手...", key="ci", label_visibility="collapsed")
        sent = st.form_submit_button("发送", use_container_width=True)
    if sent and msg:
        # 在追加本轮用户输入前，用已有对话构造多轮上下文 history
        # assistant 文本里带 [Trace]/[Evidence] 块，需剥掉只保留正文，避免污染 LLM 上下文
        history = []
        for _entry in st.session_state["ch"][-10:]:
            _role, _txt = _entry[0], _entry[1]
            clean = _txt.split("\n\n[Trace]")[0].split("\n\n[Evidence]")[0].split("\n\n[Goal]")[0]
            history.append({"role": "user" if _role == "u" else "assistant", "content": clean})
        st.session_state["ch"].append(("u", msg))
        with st.spinner("Agent 思考中..."):
            stream_box = st.empty()
            stream_events = []
            stream_cards = []
            ans = None
            for event in agent.stream_chat(user_id, msg, history=history or None):
                stream_events.append(event)
                if event.type == "candidates":
                    stream_cards = event.payload.get("cards", []) or stream_cards
                if event.type != "final":
                    stream_box.caption(f"{event.type}: {event.content}")
                else:
                    from app.models import AgentAnswer
                    ans = AgentAnswer.model_validate(event.payload)
            if ans is None:
                ans = agent.chat(user_id, msg, history=history or None)
            stream_box.empty()
        trace_block = "\n".join(ans.agent_trace)
        if stream_events:
            stream_trace = "\n".join(f"[stream:{event.type}] {event.content}" for event in stream_events if event.type != "final")
            trace_block = stream_trace + ("\n" + trace_block if trace_block else "")
        evidence_block = "\n".join(
            [f"{e.timestamp} · {e.metadata.get('asset_title', '')} · {e.content}" for e in ans.evidences[:3]]
        )
        goal_block = "\n".join(ans.goal_progress) if ans.goal_progress else ""
        # 正文与 Trace/Evidence/Goal 分开存：正文走原生 markdown 渲染，元信息收进折叠区
        st.session_state["ch"].append(("a", ans.answer, {
            "trace": trace_block,
            "evidence": evidence_block,
            "goal": goal_block,
            "cards": stream_cards,
        }))
        st.rerun()

# --- 固定底部音频播放器 (Spotify style) ---
_audio_url = st.session_state.get("audio_url")
if _audio_url:
    _atitle = st.session_state.get("audio_title", "")
    _acover = st.session_state.get("audio_cover", "")
    _asong = _atitle.split(" - ")[0] if " - " in _atitle else _atitle
    _aartist = _atitle.split(" - ")[1] if " - " in _atitle else ""
    _cover_html = f'<img class="sm-bar-cover" src="{_acover}">' if _acover else '<div class="sm-bar-cover" style="background:#333;display:flex;align-items:center;justify-content:center;font-size:14px;">&#9835;</div>'
    st.markdown(f"""
    <div class="sm-bottom-bar">
      <div class="sm-bar-disc">{_cover_html}</div>
      <div class="sm-bar-info">
        <div class="sm-bar-title">{_asong}</div>
        <div class="sm-bar-artist">{_aartist}</div>
      </div>
      <div class="sm-bar-audio">
        <audio controls autoplay src="{_audio_url}" style="width:100%;"></audio>
      </div>
    </div>
    <div class="sm-bottom-spacer"></div>
    """, unsafe_allow_html=True)
    with st.container(key="audio_close"):
        if st.button("✕ 关闭播放", key="close_audio_btn"):
            for k in ("audio_url", "audio_title", "audio_cover"):
                st.session_state.pop(k, None)
            st.rerun()

# --- 浮动 MV 播放器 (Quark-style) ---
_video_url = st.session_state.get("video_url")
if _video_url:
    _vtitle = st.session_state.get("video_title", "")
    _vext = st.session_state.get("video_ext_url", "")
    _vsong = _vtitle.split(" - ")[0] if " - " in _vtitle else _vtitle
    _vartist = _vtitle.split(" - ")[1] if " - " in _vtitle else ""
    st.markdown(f"""
    <div class="sm-mv-overlay">
      <div class="sm-mv-card">
        <div class="sm-mv-top">
          <div class="sm-mv-song">{_vsong} · {_vartist}</div>
        </div>
        <iframe src="{_video_url}" width="100%" height="480" frameborder="0"
          allow="autoplay; encrypted-media; fullscreen" allowfullscreen></iframe>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # 关闭/打开控制条：放进 keyed 容器，CSS 把它固定在视口顶部、盖在 overlay 之上
    with st.container(key="mv_controls"):
        cclose, copen = st.columns(2)
        with cclose:
            if st.button("✕ 关闭视频", key="close_video_btn", use_container_width=True):
                for k in ("video_url", "video_title", "video_ext_url"):
                    st.session_state.pop(k, None)
                st.rerun()
        with copen:
            if _vext:
                st.link_button("在浏览器打开", _vext, use_container_width=True)
