<div align="center">

# SonicMind —— 智能音乐推荐 Agent

**把一句听歌需求变成一份自己挑过、下次还能打开的场景歌单。**

[![CI](https://github.com/peteboi528/sonicmind-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/peteboi528/sonicmind-agent/actions/workflows/ci.yml)
[![Nightly LLM Eval](https://github.com/peteboi528/sonicmind-agent/actions/workflows/nightly-llm-eval.yml/badge.svg)](https://github.com/peteboi528/sonicmind-agent/actions/workflows/nightly-llm-eval.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Version](https://img.shields.io/badge/version-0.3.0-blue)

FastAPI · LangGraph · Vue 3 · DeepSeek/Qwen

</div>

---

打开 SonicMind，首页就是「场景歌单」。例如下班坐地铁，想听几首放松的 R&B，可以这样做。

1. 输入「晚高峰通勤，想听 8 首放松的 R&B」，点「预览歌单」。
2. 查看实际找到的歌曲和歌手，取消不想听的曲目，把草稿命名为「回家路上」。
3. 想再挑一轮，可以只勾选要保留的歌，填上「少一点人声，偏爵士」等调整要求，再点「保留已选，其余换歌」。也可以整份换一批。已看过的曲目会从后续候选中排除。
4. 满意后点「保存选中的曲目」。在「我的歌单」里重新打开并按列表顺序播放。播放可用性取决于曲源和授权。

每个用户保留最近生成的一份草稿。预览不会出现在已保存歌单中。确认时，服务端按生成时的候选快照保存选曲，无需再次请求模型或音乐平台。模型生成失败时可以回退到已有候选，页面按实际曲数展示结果。

「对话」支持追问歌手、专辑和推荐理由。「我的库」和偏好反馈为个性化推荐提供信息。当前的节奏和能量估计来自元数据标签。

## 目录

- [它解决了什么问题](#它解决了什么问题)
- [核心技术决策](#核心技术决策)
- [系统架构](#系统架构)
- [能做什么](#能做什么)
- [快速开始](#快速开始)
- [工程原则](#工程原则)
- [质量保障](#质量保障)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [文档](#文档)

## 它解决了什么问题

| 使用时的问题 | 系统如何处理 |
|---|---|
| 知道听歌场景，但不知道从哪首找起 | 根据场景从音乐平台和曲库召回候选，生成可编辑的歌单草稿 |
| 只喜欢推荐中的几首 | 用户勾选并命名，同一草稿重复确认相同选择只保存一份 |
| 想知道推荐从哪来 | 推荐对话展示来源和工具调用轨迹，答案生成后检查带引号的疑似曲名 |

## 核心技术决策

下面是几个最值得展开讲的工程决策。每一条都来自真实踩过的坑，不是纸上谈兵。

### 1. 反幻觉：把"答案"和"来源"彻底解耦

**问题**：LLM 会幻觉歌名——有时编一首不存在的，有时把 A 的歌安到 B 头上。光靠 prompt 约束不可靠。

候选由音乐工具返回。推荐对话输出后，`guard_answer` 会对带书名号和部分引号的疑似曲名做匹配，移除未命中的名称。这个检查允许标题部分匹配，覆盖范围取决于文案格式。场景歌单确认使用另一种约束，客户端只提交草稿 ID 和所选位置，服务端从快照中取出对应曲目。

### 2. 推荐排序：三锚归一化 + 缺锚自动重分配

**问题**：语义、口味、行为三类信号，在不同环境下可用性完全不同——没装 embedding 模型时语义锚消失，新用户没有行为数据时行为锚消失。简单加权会让分数失真。

**方案**：`final = w·语义 + w·口味 + w·行为`，**某锚缺失时其权重按比例转移给其余锚**（而不是把分数拉平），再用 MMR（`λ·相关性 − (1−λ)·候选间标签重叠`）抑制连续推荐同质曲目。

**权衡**：没有上图数据库。4 维 Jaccard（genre / mood / scenario / theme）对用户偏好的区分度已经够用，换来的是零额外基础设施——这个判断在个人项目尺度上是成立的。

### 3. 探索与利用：Thompson Sampling 在线学习

**问题**：纯利用会困在已知喜好，纯探索体验差；冷启动还缺数据。

**方案**：每个候选维护一个 Beta(α, β) 后验。听完 → α+1，秒跳 → β+0.5，高分 → α+，负反馈 / 不喜欢 → β+3，曝光衰减 → β+0.3。采样时从后验里抽，尾部候选有概率被捞回——于是高潜力的冷门歌不会被埋没。

**效果**：一套**无需离线训练**的轻量在线学习，边用边校准口味。

### 4. 意图识别：LLM 与确定性规则分工

**问题**：让 LLM 同时判意图、抽实体、打曲风 / 心情标签，又贵又容易幻觉，还很难加新意图。

**方案**：让 LLM 只做它擅长的——判意图 + 抽实体名；`genre / mood / scenario` 标签交给确定性规则（`app/graph/tag_rules.py`）。所有意图的工具链、策略、关键词、优先级集中在 registry（`app/intents.py`），**新增意图只改一处**。LLM 误判时（比如把"介绍"判成讨论），关键词安全网还能把它纠正回来。

**效果**：降本、降幻觉，还把"加新功能"的成本压到了最低。这条决策背后是 Reddit 上那句 *"boring architecture wins in production"*——能确定化的就别交给概率模型。

### 5. 踩过的坑：推理模型的"空答案"

**问题**：接入 DeepSeek-V4 后，对话一度变得僵硬、答案经常是空的。第一反应是去调 prompt——没用。

**根因**：推理模型把**真正的答案吐在了 `reasoning_content` 字段，`content` 经常是空的**。这是模型用法问题，不是 prompt 问题。（这条教训后来被证实是对话僵硬的总根因。）

**方案**：识别推理模型、默认关掉 thinking 模式（单次调用快约 35%、输出 token 少约 3 倍）、真正需要推理时再流式读取 `reasoning_content`。

**教训**：debug 诡异输出时，先核对**"模型到底把答案放在了哪个字段"**，别一上来就怪 prompt。这个坑我后来写进了项目的记忆库，避免重蹈。

### 6. 性能：先定位真正的瓶颈，再动手

**问题**：随曲库变大，单次请求越来越慢。直觉以为是网络 / 外部接口限流。

**根因**：不是网络，是 **per-request 重算**——`list_assets` 每次全量扫库、embedding 每次重算，开销随库增长线性放大。

**方案**：`list_assets` / embedding 进程内缓存，SQLite 作为资源库的单一事实源（根治双写漂移）。同样的思路也治了推荐超时：根因是前序已搜过、`online_fallback` 却**二次联网**撞上限流吃光预算——加一个"已搜过就别重搜"的短路就解决了。

**教训**：性能问题先量化和定位瓶颈，别凭直觉堆超时和重试——那只会把症状盖住。

## 系统架构

所有聊天请求走**单一的 Graph 编排**（`app/graph/`），生产环境只有这一套决策逻辑，没有并行的第二套：

```
用户输入
  │
  ▼
┌─────────────────── Graph 编排 (app/graph/) ───────────────────┐
│                                                                │
│  load_context ─▶ plan_intent ─▶ execute_tools                  │
│       │              │               │                         │
│   GSSC 预算     LLM 判意图        调用对应工具链                │
│   (按优先级      + 规则打标签     search / recommend /          │
│    分 token)                     knowledge / video / ...       │
│       │              │               │                         │
│       │              │      ◀─ 候选不足？→ web_fallback 联网补搜 │
│       │              ▼               │                         │
│       │         evaluate ─▶ reflect ─▶ finalize                │
│                                                                │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
                   ┌─────────── Answer Guard ───────────┐
                   │ 答案里每个《歌名》/《专辑名》必须    │
                   │ 命中工具返回的真实候选白名单，        │
                   │ 否则在出答案前剔除                   │
                   └───────────────────┬────────────────┘
                                       ▼
          SSE: plan / candidates / album_card / eval / final
                                       ▼
                          Vue 3 前端（对话 / 今日 / 浏览 /
                          探索 / 画像 / 我的库 / 历史 / 歌单）
```

**编排要点**

- **load_context**：读取记忆、历史、资源库，GSSC 按优先级分配 token 预算，**绝不同步调 LLM 压缩**（会阻塞主流程）。
- **plan_intent**：LLM 或 MockLLM 输出结构化 `AgentPlan`，关键词 registry 兜底 + 安全升级。
- **execute_tools**：按 intent 调用对应工具链；一句双意图时两条链可并行（`ENABLE_MULTI_INTENT`）。
- **evaluate / reflect**：候选不足或约束不满足时触发补救，知识线还有确定性核对 + 可选回环重试。
- **finalize**：统一生成 grounded answer，跑 Answer Guard，写回记忆和对话状态。

**关键模块**

| 模块 | 职责 |
|---|---|
| `app/graph/` | 主编排：planning / execution / recovery / finalize 分文件，nodes.py 是纯门面 |
| `app/intents.py` | 意图 registry：工具链、策略、关键词、优先级集中声明 |
| `app/answer.py` | 候选收集、Answer Guard、卡片 / 进度辅助 |
| `app/recommend/` | 三锚精排、MMR、Thompson Sampling、tempo/energy 估算 |
| `app/knowledge.py` | 音乐知识线：实体消歧、乐评正文抓取、专辑/艺人档案、对比、采样溯源 |
| `app/context/gssc.py` | 确定性上下文预算分配 |
| `app/security/` | 凭证 Fernet 加密、入库 URL 的 SSRF 阻断、非可信内容定界 |
| `app/sources/` | 网易云 / B站 / YouTube / Last.fm / web search，真实源 + mock 兜底 |

## 能做什么

Agent 自动识别意图，路由到对应工具链。下面是代表性场景（完整意图元数据集中在 `app/intents.py`）：

| 类别 | 触发示例 | 行为 |
|------|---------|------|
| 推荐 | 「推荐几首歌」「来点 chill 的」「适合跑步的歌」 | 网易云 + LLM 候选 + Last.fm 多路搜索 → 三锚精排 + MMR 重排 |
| 搜索 | 「找一首歌」「帮我搜 Drake」 | 网易云歌曲搜索为主，本地库补充 |
| 音乐知识 | 「讲讲 *Blonde*」「介绍 The Weeknd 的生涯」「Drake vs Kendrick」「这首歌采样了谁」 | 实体消歧 → 元数据 + 乐评正文（Tavily / Discogs / Last.fm）→ DeepSeek 参数化直答 + web 兜底 → 档案卡片 |
| 视频 | 「找 The Weeknd 的 MV」「Adele 现场」 | 直搜 B 站 + YouTube，不走网易云 |
| 歌单 | 「做 20 首 chill 歌单」「修一下这个歌单」 | 联网扩展 + LLM 精选；`playlist_repair` 清洗脏数据 |
| 品味实验 | 「推荐点不一样的」「做个品味实验」 | 生成 safe / stretch / bold 三档，收反馈出报告 |
| 品味洞察 | 「分析我的品味」「我最近口味变了吗」 | 只读记忆 / 行为画像；`taste_shift_detector` 检测漂移 |
| 导入 | 「导入网易云歌单」 | 一键导入，入库前过质量闸门拒脏数据 |

**多轮对话**：「多来几首」「换一批」继承上轮实体并去重；「找他的 MV」「介绍这个歌手」检测到新意图就不再延续；「他的歌」自动消解指代。

## 快速开始

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest                 # 全量测试，零依赖即可跑

# 可选增强
python3 -m pip install -e ".[embeddings]"   # 语义检索（sentence-transformers）
python3 -m pip install -e ".[vision]"       # 封面识别（视觉模型 / 本地 OCR 兜底）
# langgraph 已在主依赖，缺失时自动降级为等价同步执行

uvicorn app.api.main:app --reload --port 8000
```

- API 文档：`http://127.0.0.1:8000/docs`
- 前端：`http://127.0.0.1:8000/web`（源码 `frontend/`，构建产物 `app/web/dist/`，部署无需 node）

**无任何外部 key 时**，系统用 MockLLM / MockSource 跑通完整链路，作为一份稳定的离线 demo。主要环境变量（均可选，完整见 `.env.example`）：

```bash
LLM_API_KEY=...              # 缺省走 MockLLM
TAVILY_API_KEY=...           # 歌手百科 / 乐评正文；无 key 时 DuckDuckGo 兜底
LASTFM_API_KEY=...           # 音乐发现与标签
DISCOGS_TOKEN=...            # 乐评与发行信息
ENABLE_MULTI_INTENT=false    # 一句双意图并行
TURN_BUDGET_SECONDS=15       # 普通任务与截图推荐的单轮预算（秒）
KNOWLEDGE_TURN_BUDGET_SECONDS=50  # 艺人与专辑解读的单轮预算（秒）
AUTH_ENABLED=false           # 部署时开启 API key 鉴权
RATE_LIMIT_ENABLED=false     # 令牌桶限流（按 user_id / IP keying）
```

### 截图推荐演示

打开 `/web/` 的「对话」页，添加一张带有曲名和艺人署名的播放页截图。发送「推荐三首这个歌手的歌，别和截图重复」，再发「我指的是合作歌手，收藏过的也去掉」。收藏过滤读取 SonicMind 已保存的歌单和专辑。决策摘要显示工具候选数、实际卡片数、超时阶段和来源。

图片功能默认开启。可用 `AGENT_IMAGE_ENABLED=false` 关闭。配置 `VISION_LLM_API_KEY` 可使用视觉模型。Mac 本地也可安装 `pip install -e '.[vision]'`，对有清晰曲名和艺人署名的截图使用 OCR。确定性回归使用已采集的真实曲目元数据，运行 `python -m pytest -q tests/test_image_recommendation.py`。

## 工程原则

几条贯穿全项目的设计取舍，体现的是"宁可朴素、不要侥幸"的判断：

- **确定性优先**：能确定化的就别交给概率模型——标签走规则、报告先算数再让 LLM 润色、知识线有确定性核对。LLM 只做它真正擅长的部分。
- **零依赖可跑**：langgraph / sentence-transformers / vision 缺失都有等价降级路径。完整链路在零配置下能跑，这条特性由 CI 持续守护。
- **诚实的数据观**：tempo / energy 没有真实音频分析（取流路径被禁），就从曲风 / 情绪标签估算并标注 `estimated`；识别不出的标「未分类」。好看永远让位于准确。
- **拦截性功能默认关**：鉴权、限流等会改变本地体验的能力默认关闭，部署时显式打开——和"零依赖可跑"保持一致。
- **零风险边界**：不碰逆向与风控路径（网易云写接口 / 取流逆向、已关闭的 Spotify audio-features），音频特征维持标签估算。

## 质量保障

| 检查 | 命令 | 覆盖范围 |
|---|---|---|
| 单元与集成测试 | `python3 -m pytest` | 意图路由、工具结果、推荐排序、会话状态与 API |
| 长对话 smoke | `python3 scripts/long_dialogue_smoke.py` | 离线验证多轮交互、SSE 与 API，输出 Markdown 和 JSON |
| 跨会话回归 | `python3 scripts/realistic_memory_eval.py` | 固定数据的 16 轮任务，验证长期偏好、临时约束、偏好纠正和新会话隔离。外部联网尝试会使评测失败 |
| LLM-as-judge | `python3 -m tests.eval.run` | 使用真实模型评估回答质量。使用 `--offline` 可将候选固定为音乐源夹具 |

长对话 smoke 和跨会话回归随 CI 运行，报告可以从该次工作流下载。Nightly 使用已配置的真实 LLM 与固定音乐候选做质量评估，真实音乐源可独立验证。

推荐延续状态（上轮实体、标签、已展示曲目）随聊天线程保存，长期偏好在同一用户的各线程共享。未指定线程 ID 的旧客户端继续使用原有默认会话。

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | FastAPI · Pydantic v2 · LangGraph · SQLite |
| LLM | DeepSeek / Qwen（OpenAI 兼容协议）· 原生异步 + 流式 |
| 检索 | sentence-transformers（可选，回退 TF cosine） |
| 前端 | Vue 3 · Vite · 原生 SSE |
| 数据源 | 网易云 · B站 / YouTube · Last.fm · Tavily / DuckDuckGo · Discogs |
| 视觉 | DashScope Qwen-VL · rapidocr 兜底（封面识别） |
| 安全 | Fernet 凭证加密 · 令牌桶限流 · SSRF / 注入边界 |

## 项目结构

```
app/
├── api/            # FastAPI 路由：main / web / auth / bot
├── graph/          # LangGraph 编排：planning / execution / recovery / finalize
├── recommend/      # 三锚精排、MMR、Thompson、特征估算
├── knowledge.py    # 音乐知识线（专辑/艺人/乐评/对比/采样）
├── sources/        # 网易云 / B站 / YouTube / Last.fm / web search
├── security/       # secret_box / ingest_guard / untrusted_boundary
├── context/        # GSSC 上下文预算
├── llm/            # 异步客户端 + 流式 + 可观测
└── intents.py      # 意图 registry（单一事实源）
frontend/src/       # Vue 3 SPA：8 个 Tab + 播放器 + 歌词同步
tests/              # pytest 单元/集成 + eval（LLM-as-judge）
scripts/            # long_dialogue_smoke / realistic_memory_eval
docs/               # 架构报告 / 原理讲解 / 部署 / 前端
```

## 文档

- [docs/EXPLAINER.md](docs/EXPLAINER.md) —— 原理讲解（系统定位、编排、记忆与推荐、反幻觉）
- [docs/ARCHITECTURE_REPORT.md](docs/ARCHITECTURE_REPORT.md) —— 架构全景报告（子系统、API、部署）
- [docs/FRONTEND.md](docs/FRONTEND.md) —— Vue 前端开发与构建
- [docs/BOT_DEPLOYMENT.md](docs/BOT_DEPLOYMENT.md) —— 飞书 / 微信 Bot 部署
- [tests/eval/README.md](tests/eval/README.md) —— LLM-as-judge 评估说明

---

<div align="center">

*这是一个持续打磨中的个人项目。每一处设计背后都有具体的坑和权衡，欢迎在 issues 里讨论。*

</div>
