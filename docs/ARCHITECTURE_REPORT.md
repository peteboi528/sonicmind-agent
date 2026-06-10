# SonicMind (MusicAgent) — 项目架构全景报告

> 生成日期：2026-06-10 | 代码总量：~10,100 行 Python + ~990 行前端 | 测试：20 个文件 / 149 用例全绿

---

## 一、项目定位

**一句话**：可解释、反幻觉、记忆驱动的 AI 音乐推荐 Agent，支持网易云/B站/YouTube 三平台，零依赖可运行。

**技术栈**：Python 3.11+ / FastAPI / Pydantic v2 / SQLite + JSON 文件存储 / Vanilla HTML+CSS+JS / 可选 LangGraph + sentence-transformers

**设计哲学**：

| 原则 | 实现方式 |
|------|---------|
| **零依赖可用** | MockLLM + MockSource + DemoAnalyzer 替代所有外部服务 |
| **反幻觉** | 候选回查真实元数据 + Answer Guard 白名单扫描 + `source="llm"` 标记 |
| **优雅降级** | LangGraph → ReAct → 关键词三层编排 |
| **在线学习** | Thompson Sampling Beta 后验 + BaRT 行为奖励 + 曝光衰减 |
| **可解释** | 每个回答附带 `agent_trace` + 三锚打分明细 + GSSC 预算报告 |

---

## 二、目录结构

```
MusicAgent/
├── pyproject.toml                          # 项目配置 & 依赖
├── .env                                    # 环境变量
├── README.md
│
├── app/
│   ├── agent.py              ★ 核心协调器 (1,571 行)
│   ├── react_loop.py         ★ ReAct 推理循环 (984 行)
│   ├── models.py             ★ 全部 Pydantic 数据模型 (422 行)
│   ├── config.py             ★ 配置中心 (51 行)
│   ├── memory.py             ★ 记忆管理 + BaRT 行为评分 (449 行)
│   ├── storage.py            JSON 文件持久化 (74 行)
│   ├── library.py            SQLite 资源库 + Thompson Sampling (265 行)
│   ├── similarity.py         资产/片段相似度 (81 行)
│   ├── netease_auth.py       网易云 QR 登录 + 歌单导入 (291 行)
│   │
│   ├── api/
│   │   ├── main.py           ★ FastAPI 主应用 (260 行, 33 端点)
│   │   ├── web_routes.py     Web 前端路由 + 播放代理 (113 行)
│   │   └── bot_routes.py     飞书/微信 Webhook 路由 (160 行)
│   │
│   ├── adapters/
│   │   ├── __init__.py       包导出
│   │   ├── protocol.py       BotAdapter 协议 + 数据类 (62 行)
│   │   ├── base.py           AgentAnswer → BotResponse 转换 (77 行)
│   │   ├── feishu_adapter.py 飞书 Bot 实现 (272 行)
│   │   └── wechat_adapter.py 微信公众号实现 (193 行)
│   │
│   ├── graph/
│   │   ├── builder.py        LangGraph StateGraph 构建 (132 行)
│   │   ├── nodes.py          图节点实现 (483 行)
│   │   ├── routing.py        工具路由辅助 (14 行)
│   │   ├── state.py          AgentState TypedDict (21 行)
│   │   └── tag_rules.py      确定性标签规则 (73 行)
│   │
│   ├── llm/
│   │   ├── protocol.py       LLMProvider 协议 + LLMResponse (49 行)
│   │   ├── client.py         OpenAI 兼容客户端 (137 行)
│   │   ├── mock.py           MockLLM 零依赖演示 (245 行)
│   │   ├── tools.py          13 个工具 schema (179 行)
│   │   └── structured.py     JSON 提取工具 (88 行)
│   │
│   ├── recommend/
│   │   ├── engine.py         评分公式 + 口味计算 (196 行)
│   │   ├── daily.py          每日推荐器 (241 行)
│   │   └── rerank.py         三锚精排 + MMR 多样性 (248 行)
│   │
│   ├── retrieval/
│   │   ├── vector_store.py   混合检索器 (116 行)
│   │   └── embeddings.py     sentence-transformers 后端 (93 行)
│   │
│   ├── context/
│   │   └── gssc.py           GSSC 上下文预算管理 (133 行)
│   │
│   ├── media/
│   │   ├── pipeline.py       媒体入库+分析管线 (185 行)
│   │   └── analyzer.py       DemoAnalyzer 合成片段 (100 行)
│   │
│   ├── sources/
│   │   ├── protocol.py       ExternalSource 协议 (11 行)
│   │   ├── mock_source.py    50 首模拟曲库 (136 行)
│   │   ├── netease.py        网易云 API (128 行)
│   │   ├── bilibili.py       B站视频搜索 (78 行)
│   │   └── youtube.py        YouTube 搜索 (70 行)
│   │
│   ├── prompts/
│   │   ├── agent_system.py   Agent 系统 prompt (v4)
│   │   ├── intent.py         意图分类 prompt
│   │   ├── query_plan.py     结构化意图规划 prompt
│   │   ├── recommend.py      推荐生成 prompt
│   │   ├── search.py         搜索 prompt
│   │   ├── playlist.py       歌单生成 prompt
│   │   ├── identify.py       URL 歌曲识别 prompt
│   │   └── reflect.py        ReAct 反思 prompt
│   │
│   ├── ui/
│   │   └── streamlit_app.py  Streamlit 全功能 UI (1,330 行)
│   │
│   └── web/
│       ├── index.html        Web 聊天 SPA (80 行)
│       ├── style.css         Spotify 深色主题 (550 行)
│       └── app.js            交互逻辑 (358 行)
│
├── tests/                                  # 20 个测试文件, ~1,966 行
├── docs/                                   # 5 个文档
├── scripts/
│   └── demo_flow.py         CLI 演示脚本
│
└── data/
    ├── resource_library.sqlite             # SQLite 资源库
    └── store/                              # JSON 持久化
        ├── assets/          (20 个)
        ├── segments/        (20 个)
        ├── memory/          (3 个)
        ├── goals/           (2 个)
        ├── playlists/       (11 个)
        └── netease_auth/    (1 个)
```

---

## 三、核心架构

### 3.1 请求处理流程

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────────────┐
│  AudioVisualAgent.chat() / stream_chat()              │
│  ├── [主路径] LangGraph StateGraph                    │
│  │   load_context → plan_intent → execute_tools       │
│  │   → [web_fallback?] → evaluate → finalize          │
│  │                                                     │
│  ├── [备选路径] ReActLoop (984 行)                     │
│  │   think → act(13个工具) → observe → 循环 (≤5步)    │
│  │   → Answer Guard (反幻觉扫描)                       │
│  │                                                     │
│  └── [兜底路径] 关键词匹配 → 顺序执行                   │
└──────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────┐
│  输出: AgentAnswer                                     │
│  ├── answer: str            回答文本                   │
│  ├── evidences: list        RAG 证据                   │
│  ├── recommended_segments:  推荐片段                   │
│  ├── agent_trace: list      决策过程                   │
│  ├── pending_goal: str      进行中的目标               │
│  └── memory_updated: bool   记忆是否更新               │
└──────────────────────────────────────────────────────┘
```

### 3.2 推荐评分公式

```
score = 0.30×genre_match + 0.25×mood_match + 0.20×energy_proximity
      + 0.15×tempo_fit + 0.10×novelty + 0.25×behavior_reward
```

三锚精排权重（自动归一化，缺项重分配）：
```
final = 0.45×semantic + 0.30×personalize + 0.25×behavior
```

MMR 多样性：`mmr = λ×rel - (1-λ)×max_overlap×rel` (λ=0.7)

### 3.3 记忆系统三层架构

| 层 | 数据结构 | 来源 |
|----|---------|------|
| **显式偏好** | `MemoryEntry(text, frequency, last_used)` | 用户直接表述 + `auto_learn_from_turn()` |
| **行为记忆** | `ListeningEvent` + `RatingEntry` | 听歌时长/完成率/评分 |
| **派生口味** | `TasteProfile(genres, moods, energy, tempo, openness)` | 从库中资产+评分+行为自动计算 |

### 3.4 反幻觉管线

```
搜索结果 → 候选回查真实元数据(_search_netease_detail/_search_bilibili_detail)
         → _valid_external_track 过滤(query≠title)
         → LLM 生成曲目标记 source="llm"
         → Answer Guard: _collect_known_titles() 建白名单
         → guard_answer() 扫描《》标记, 剔除不在白名单的曲目
```

---

## 四、完整 API 接口清单

### 4.1 核心业务端点（`app/api/main.py`）

| 方法 | 路径 | 请求体 | 返回 | 说明 |
|------|------|--------|------|------|
| GET | `/health` | — | `{status, checks, details}` | 健康检查 |
| GET | `/assets` | — | `{assets: [...]}` | 列出所有资产 |
| POST | `/assets/ingest` | `{url, force_refresh}` | `Asset` | 入库音乐 URL |
| POST | `/assets/{id}/enrich` | `{use_network}` | `EnrichResponse` | 元数据识别 |
| POST | `/assets/{id}/analyze` | `?force_refresh` | `{asset, segments}` | 媒体分析 |
| DELETE | `/assets/{id}` | `?user_id` | `{deleted, asset_id}` | 删除资产 |
| DELETE | `/cache` | `?preserve_memory` | `{cleared, preserve_memory}` | 清缓存 |
| POST | `/rate` | `{user_id, asset_id, score}` | `{rated, taste_updated, top_genres}` | 评分 |
| GET | `/ratings/{user_id}` | — | `{ratings: [...]}` | 查询评分 |
| POST | `/recommend/daily` | `{user_id, time_of_day}` | `DailyRecommendation` | 每日推荐 |
| GET | `/recommend/daily/{user_id}` | — | `DailyRecommendation` | 获取每日推荐 |
| GET | `/assets/{id}/similar` | `?top_k` | `{similar_assets: [...]}` | 相似资产 |
| POST | `/search` | `{user_id, query, include_external, top_k}` | `SearchResponse` | 搜索音乐 |
| POST | `/listen` | `{user_id, asset_id, duration, completed}` | `{memory_updated, history_count}` | 记录收听 |
| POST | `/chat` | `{user_id, message, history}` | `AgentAnswer` | 对话 |
| POST | `/agent/run` | `{user_id, message, history}` | `AgentAnswer` | 运行 Agent |
| POST | `/agent/stream` | `{user_id, message, history}` | SSE `StreamEvent` | 流式对话 |
| GET | `/taste/{user_id}` | — | `TasteProfile` | 口味分析 |
| POST | `/memory/update` | `{user_id, event, asset_id?}` | `{memory, updated}` | 更新记忆 |
| POST | `/memory/feedback` | `{user_id, segment_id, accepted}` | `{memory, updated}` | 片段反馈 |
| POST | `/feedback/dislike` | `{user_id, title, artist, source, source_id}` | `{updated, memory}` | 不喜欢 |
| GET | `/memory/{user_id}` | — | `UserMemory` | 完整记忆 |
| GET | `/library/tracks` | `?limit` | `{tracks: [...]}` | 资源库曲目 |
| POST | `/playlist/generate` | `{user_id, instruction}` | `Playlist` | 生成歌单 |
| POST | `/journey/generate` | `{user_id, instruction}` | `{phases, tracks}` | 音乐旅程 |
| POST | `/playlist/auto/{user_id}` | — | `{playlists: [...]}` | 自动分类歌单 |
| GET | `/playlists/{user_id}` | — | `{playlists: [...]}` | 列出歌单 |
| DELETE | `/playlist/{uid}/{pid}` | — | `{deleted}` | 删除歌单 |

### 4.2 Web 前端端点（`app/api/web_routes.py`）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/web` | 聊天主页 |
| GET | `/web/style.css` | 样式 |
| GET | `/web/app.js` | 交互逻辑 |
| POST | `/api/playback/audio` | 音频播放代理 |
| POST | `/api/playback/mv` | MV 播放代理 |

### 4.3 Bot Webhook 端点（`app/api/bot_routes.py`）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/webhook/feishu` | 飞书事件回调 |
| GET | `/webhook/wechat` | 微信签名验证 |
| POST | `/webhook/wechat` | 微信消息回调 |

### 4.4 SSE 流式事件类型

| type | 含义 | payload 关键字段 |
|------|------|-----------------|
| `plan` | 意图规划结果 | `intent`, `strategy` |
| `thinking` | 推理中间步骤 | `content` |
| `tool_start` | 开始执行工具 | `tool_name` |
| `tool_result` | 工具执行结果 | `tool_name`, `data` |
| `candidates` | 推荐候选列表 | `cards: [{title, artist, ...}]` |
| `song_card` | 单张歌曲卡片 | `title`, `artist`, `cover_url` |
| `eval` | 评估结果 | `score` |
| `final` | 最终回答 | `content`, `payload` |
| `guard` | 反幻觉过滤 | `removed` |
| `error` | 错误 | `content` |

---

## 五、数据模型

### 5.1 核心实体

```
Asset (资产/曲目)
├── asset_id, source_url, title, duration_seconds
├── artist, album, cover_url
├── genre[], mood[], tempo_bpm, energy_level
├── source: "local" | "external"
└── status: INGESTED | ANALYZED | FAILED

ExternalTrack (外部平台曲目)
├── external_id, title, artist, album
├── genre[], mood[], tempo_bpm, energy_level
├── cover_url, preview_url, playback_url
└── source: "netease" | "bilibili" | "youtube" | "mock"

TrackEntity (候选池契约)
├── title, artist, source, source_id
├── verified: bool (是否经真实来源回查)
├── origin: local | netease | bilibili | youtube | mock | llm_guess
└── evidence_ref: str | None

UserMemory (用户记忆)
├── preferences[], structured_preferences: MemoryEntry[]
├── listening_history: ListeningEvent[] (max 200)
├── ratings: RatingEntry[]
├── dislikes: str[]
├── taste_profile: TasteProfile | None
└── daily_rec_last_generated: str | None
```

### 5.2 推荐管线模型

```
AgentPlan
├── intent: recommend | search | playlist | taste | import | journey | chat
├── strategy: online_first | library_first | memory_only | no_search
├── tools_needed: str[]
├── target_count: int | None
└── retrieval_plan: RetrievalPlan

RankingBreakdown
├── title, source, score, reason
└── components: { semantic, personalize, behavior }

StreamEvent
├── type: plan | thinking | tool_start | tool_result | candidates
│         song_card | eval | final | guard | error
├── content: str
└── payload: dict
```

---

## 六、外部服务集成

### 6.1 网易云音乐

| API | 用途 | 认证 |
|-----|------|------|
| `/api/search/get/web` | 搜索歌曲 | 无 |
| `/api/song/detail` | 歌曲元数据 | 无 |
| `/api/song/enhance/player/url/v1` | VIP 高品质音频 | MUSIC_U cookie |
| `/api/song/enhance/player/url` | 320kbps MP3 | 无 (部分歌曲不可播) |
| `/api/login/qrcode/unikey` | QR 登录 | 无 |
| `/api/login/qrcode/client/login` | QR 状态轮询 | 无 |
| `/api/nuser/account/get` | 账号信息+VIP | MUSIC_U cookie |
| `/api/v6/playlist/detail` | 歌单详情 | cookie 可选 |
| `/api/v3/song/detail` | 批量歌曲详情 | cookie 可选 |

**关键限制**：`fee=1` 的付费歌曲需登录+VIP 才能获取音频 URL（API 返回 `code:-110, url:null`）。

### 6.2 Bilibili

| API | 用途 |
|-----|------|
| `/x/web-interface/search/type` | 视频搜索 |
| HTML `<title>` 抓取 | 标题提取 |

**注意**：B 站只返回视频，不提供音频直链。播放使用 embed iframe。

### 6.3 YouTube

| API | 用途 |
|-----|------|
| oEmbed `/oembed` | 标题提取 |
| `ytInitialData` 解析 | 搜索结果 |

**注意**：YouTube 不提供音频直链。播放使用 embed iframe。

---

## 七、前端能力矩阵

### 7.1 Streamlit UI（完整功能）

| 功能 | 实现状态 |
|------|---------|
| 💬 对话聊天（SSE 流式） | ✅ |
| 🎵 歌曲卡片 + 播放 | ✅ |
| 📱 网易云扫码登录 | ✅ |
| 📥 导入网易云歌单 | ✅ |
| 📋 我的歌单列表 | ✅ |
| 📥 添加音乐（入库） | ✅ |
| 🎯 今日推荐 | ✅ |
| 🔍 发现/搜索 | ✅ |
| 📚 我的库 + 评分 | ✅ |
| 🎵 歌单管理 | ✅ |
| 🧠 训练偏好 | ✅ |
| 👎 不喜欢反馈 | ✅ |
| 📊 透明度面板 | ✅ |
| 🎧 底部播放器 | ✅ |
| 📺 MV 浮层播放器 | ✅ |

### 7.2 Web 前端（当前状态）

| 功能 | 实现状态 | 备注 |
|------|---------|------|
| 💬 对话聊天（SSE 流式） | ✅ | 已完成 |
| 🎵 歌曲卡片 + 播放 | ✅ | 已完成 |
| 🎧 底部播放器 | ✅ | 已完成 |
| 📺 MV 浮层播放器 | ✅ | 已完成 |
| 📱 网易云扫码登录 | ❌ | 需新增后端 API + 前端 UI |
| 📥 导入网易云歌单 | ❌ | 后端逻辑已有 |
| 🎯 今日推荐 | ❌ | 后端端点已有 |
| 🔍 发现/搜索 | ❌ | 后端端点已有 |
| 📚 我的库 + 评分 | ❌ | 后端端点已有 |
| 🎵 歌单管理 | ❌ | 后端端点已有 |
| 🧠 训练偏好 | ❌ | 后端端点已有 |
| 👎 不喜欢反馈 | ❌ | 后端端点已有 |
| 📊 透明度面板 | ❌ | |
| 侧边栏 | ❌ | |

---

## 八、已知问题与技术债

### 8.1 当前最高优先级问题

| # | 问题 | 影响 | 根因 |
|---|------|------|------|
| P0 | **推荐 The Weeknd 只展示 B站/YouTube 视频合集** | 用户体验严重退化 | `search_web_music()` 不过滤视频类型；网易云付费歌曲无 cookie 无法播放 |
| P1 | **Web 前端功能不完整** | 无法替代 Streamlit | 只有聊天，缺少侧边栏/Tab/扫码等 |
| P2 | **播放接口缺少结构化失败原因** | 前端无法区分"VIP 不可播"和"找不到" | `/api/playback/audio` 只返回 `{url: null}` |

### 8.2 技术债

| 项目 | 说明 | 影响 |
|------|------|------|
| `JsonStore` 全表扫描 | `list_assets()` 每次读所有 JSON | 资产 >200 时变慢 |
| `DemoAnalyzer` 合成数据 | 无真实音频分析 | 标签完全依赖 LLM/规则 |
| 无 CORS 安全策略 | `allow_origins=["*"]` | 仅适合内网/开发 |
| 无用户认证 | `user_id` 客户端自填 | 生产环境不可用 |
| `Embedding` 默认关闭 | TF cosine 作为降级 | 检索质量受限 |
| `ExternalTrack` 无 `candidate_kind` | 无法区分歌曲/视频/MV/合集 | 推荐质量控制无法实现 |

---

## 九、配置项完整清单

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI 兼容 LLM 端点（Ollama） |
| `LLM_API_KEY` | `""` | API Key（空 = MockLLM） |
| `LLM_MODEL` | `qwen2.5` | 模型名 |
| `LLM_TIMEOUT_SECONDS` | `45` | 请求超时 |
| `LLM_MAX_TOKENS` | `1024` | 最大输出 token |
| `EXTERNAL_SOURCE` | `mock` | 外部源类型 |
| `STORE_ROOT` | `data/store` | JSON 存储根目录 |
| `MEDIA_ROOT` | `data/media` | 媒体文件目录 |
| `RESOURCE_LIBRARY_PATH` | `data/resource_library.sqlite` | SQLite 路径 |
| `DAILY_REC_COUNT` | `25` | 每日推荐数量 |
| `ENABLE_ONLINE_ENRICH` | `false` | 在线元数据识别 |
| `ENABLE_EMBEDDINGS` | `false` | 向量检索（需 sentence-transformers） |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 嵌入模型 |
| `TRI_ANCHOR_W_SEMANTIC` | `0.45` | 语义锚权重 |
| `TRI_ANCHOR_W_PERSONAL` | `0.30` | 个性化锚权重 |
| `TRI_ANCHOR_W_BEHAVIOR` | `0.25` | 行为锚权重 |
| `MMR_LAMBDA` | `0.7` | MMR 相关性-多样性权衡 |
| `EXPLORATION_RATIO` | `0.2` | Thompson Sampling 探索比例 |
| `ENABLE_RERANK` | `true` | 启用三锚精排 |
| `FEISHU_APP_ID` | `""` | 飞书应用 ID |
| `FEISHU_APP_SECRET` | `""` | 飞书应用密钥 |
| `FEISHU_VERIFICATION_TOKEN` | `""` | 飞书验证 Token |
| `FEISHU_ENCRYPT_KEY` | `""` | 飞书加密 Key |
| `WECHAT_TOKEN` | `""` | 微信 Token |
| `WECHAT_APP_ID` | `""` | 微信 AppID |
| `WECHAT_APP_SECRET` | `""` | 微信 AppSecret |

---

## 十、依赖关系

### 核心依赖

| 包 | 版本 | 用途 |
|---|------|------|
| `fastapi` | ≥0.115.0 | Web 框架 |
| `pydantic` | ≥2.8.0 | 数据模型 |
| `uvicorn` | ≥0.30.0 | ASGI 服务器 |
| `httpx` | ≥0.27.0 | HTTP 客户端（Bot API 调用） |
| `python-dotenv` | ≥1.0.0 | 环境变量加载 |
| `qrcode` | ≥7.0 | 网易云 QR 码生成 |
| `langgraph` | ≥0.2.0 | 状态图编排（可选，缺则降级） |
| `cryptography` | ≥42.0.0 | 飞书 AES 加解密 |
| `streamlit` | ≥1.36.0 | 开发者 UI |

### 可选依赖

| 包 | 版本 | 用途 |
|---|------|------|
| `yt-dlp` | ≥2024.8.6 | YouTube 标题提取兜底 |
| `sentence-transformers` | ≥2.2.0 | 向量检索 |

### 零 SDK 策略

所有外部平台（网易云/B站/YouTube/飞书/微信）均通过 `urllib.request` 或 `httpx` 直接调 HTTP API，不依赖任何平台 SDK。

---

## 十一、测试覆盖

```
tests/
├── conftest.py                 # 全局 fixtures：MockLLM + fake search
├── test_agent_flow.py          # 完整生命周期 (8 tests)
├── test_react.py               # ReAct 意图分类 (5 tests)
├── test_react_tool_calling.py  # 工具调用循环 (10 tests)
├── test_answer_guard.py        # 反幻觉守卫 (6 tests)
├── test_api.py                 # FastAPI 端点 (1 test)
├── test_web_routes.py          # Web 路由 (9 tests)
├── test_memory_enhanced.py     # 结构化偏好 (6 tests)
├── test_daily_recommend.py     # 每日推荐 (6 tests)
├── test_behavior_scoring.py    # BaRT 行为评分 (6 tests)
├── test_rerank.py              # 三锚精排 + MMR (9 tests)
├── test_gssc.py                # 上下文预算 (4 tests)
├── test_embeddings.py          # 向量检索 (4 tests)
├── test_similarity.py          # 相似度 (3 tests)
├── test_phase0_orchestration.py # LangGraph 编排 (10 tests)
├── test_graph_resource_library.py # 资源库集成 (7 tests)
├── test_import_classification.py # 导入分类 (5 tests)
├── test_stream.py              # SSE 流式 (5 tests)
├── test_adapters.py            # Bot 适配器 (19 tests)
└── test_eval_cases.py          # LLM-as-Judge eval (21 tests)

总计: 149 tests, 全绿, ~1.2s
```

---

## 十二、未来开发建议

### 阶段 1：修复核心体验（1-2 天）

**目标**：让"推荐 The Weeknd"能正确展示歌曲并播放。

1. **候选质量分类**：`ExternalTrack` 新增 `candidate_kind` 字段（`track`/`mv`/`filtered`）
2. **`search_web_music()` 过滤**：B站/YouTube 标题含"合集"/"playlist"/"歌词版"则丢弃
3. **播放接口增强**：`/api/playback/audio` 返回 `{url, reason: "ok"|"vip_required"|"not_found"}`
4. **前端适配**：VIP 歌曲展示"扫码登录后可播放"提示

### 阶段 2：Web 前端补全（2-3 天）

**目标**：Web 前端功能与 Streamlit 对齐。

1. **侧边栏**：用户 ID、网易云扫码登录、歌单导入、添加音乐、训练偏好
2. **5 个 Tab**：推荐、发现、我的库、歌单、对话
3. **新增后端 API**：6 个 `/api/netease/*` 端点（复用已有 `netease_auth.py`）
4. **前端 JS**：所有交互模块（扫码、导入、评分、搜索等）

### 阶段 3：工程质量提升（可选）

| 项目 | 预估 | 价值 |
|------|------|------|
| SQLite 替换 JsonStore | 1 天 | 解决全表扫描性能问题 |
| 用户认证 | 1 天 | 生产环境必需 |
| 运行时配置热更新 | 0.5 天 | 运维便利 |
| 日志/监控 | 1 天 | 可观测性 |
| Docker 化 | 0.5 天 | 部署标准化 |

### 阶段 4：智能深度提升（长期）

| 项目 | 依赖 | 价值 |
|------|------|------|
| Neo4j GraphRAG | Neo4j | 关系推理推荐 |
| M2D-CLAP 音频嵌入 | PyTorch | 真实音频理解 |
| sentence-transformers 默认开启 | GPU/内存 | 检索质量提升 |
| 协同过滤 | 大量用户行为数据 | 跨用户推荐 |
