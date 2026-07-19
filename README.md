<div align="center">

# SonicMind

**一个会"较真"的音乐推荐 Agent —— 它不会推荐一首不存在的歌。**

[![CI](https://github.com/peteboi528/sonicmind-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/peteboi528/sonicmind-agent/actions/workflows/ci.yml)
[![Nightly LLM Eval](https://github.com/peteboi528/sonicmind-agent/actions/workflows/nightly-llm-eval.yml/badge.svg)](https://github.com/peteboi528/sonicmind-agent/actions/workflows/nightly-llm-eval.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-1009%20collected-brightgreen)
![Version](https://img.shields.io/badge/version-0.3.0-blue)

FastAPI · LangGraph · Vue 3 · DeepSeek/Qwen

</div>

---

SonicMind 是一个围绕用户品味持续工作的音乐推荐 Agent。它不是把问题丢给 LLM 然后等一句话回复的聊天壳，而是一条完整的工程链路：识别意图、调用真实音乐数据源、维护记忆与行为反馈、给出**每一条都能追溯到真实来源**的推荐，并把整个决策过程通过 SSE 交给前端展示。

和直接用 LLM 做推荐相比，它最在意一件事——**诚实**：歌名、专辑名必须能在工具返回的真实候选里核对到，否则在出答案前就被剔除。换句话说，它宁可说"没找到"，也不编一首交差。

> 媒体分析当前以**元数据富化**为主，不声称已具备 Whisper / CLIP 级别的真实音视频理解。tempo / energy 从曲风与情绪标签估算并如实标注为 `estimated`；未识别的 genre / mood 标「未分类」，不随机伪造——这是同一个原则的延伸。

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

把"推荐音乐"这件事直接交给 LLM，会撞上三个绕不过去的毛病。SonicMind 的每一步设计，几乎都是在堵其中一个：

| 毛病 | 表现 | SonicMind 的应对 |
|---|---|---|
| **编造** | 一本正经地推荐一首根本不存在的歌，或张冠李戴 | 工具先返回真实候选，答案里的歌名必须命中候选白名单，否则剔除 |
| **不可追溯** | "为什么推这首？"答不上来 | 每条推荐附节点级 `agent_trace` + 稳定 `trace_summary`，意图/工具/来源全程留痕 |
| **千篇一律** | 推荐永远困在已知喜好里，或全是热歌 | Thompson Sampling 探索冷门 + Taste Lab 用实验验证品味边界 |

不是"让 LLM 小心一点"，而是**从架构上把幻觉和偷懒关在答案门外**。

## 核心技术决策

下面是几个最值得展开讲的工程决策。每一条都来自真实踩过的坑，不是纸上谈兵。

### 1. 反幻觉：把"答案"和"来源"彻底解耦

**问题**：LLM 会幻觉歌名——有时编一首不存在的，有时把 A 的歌安到 B 头上。光靠 prompt 约束不可靠。

**方案**：把候选生成和答案生成分成两步。工具链先返回**真实候选**（网易云、Last.fm 等平台数据），生成回答时，每个书名号歌名 / 专辑名必须命中这个候选白名单；`guard_answer` 在出答案前把无法追溯的内容剔除。候选不足时，系统会如实说"没找到足够的歌"，而不是硬编。

**效果**：每条推荐都能追溯到真实平台来源。这甚至可以被自动化 smoke 校验——"有没有走对工具、有没有发出卡片、有没有 final"都成了可断言的结构化检查。

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
TURN_BUDGET_SECONDS=25       # 单轮墙钟预算，超时降级而非挂死
AUTH_ENABLED=false           # 部署时开启 API key 鉴权
RATE_LIMIT_ENABLED=false     # 令牌桶限流（按 user_id / IP keying）
```

## 工程原则

几条贯穿全项目的设计取舍，体现的是"宁可朴素、不要侥幸"的判断：

- **确定性优先**：能确定化的就别交给概率模型——标签走规则、报告先算数再让 LLM 润色、知识线有确定性核对。LLM 只做它真正擅长的部分。
- **零依赖可跑**：langgraph / sentence-transformers / vision 缺失都有等价降级路径。完整链路在零配置下能跑，这条特性由 CI 持续守护。
- **诚实的数据观**：tempo / energy 没有真实音频分析（取流路径被禁），就从曲风 / 情绪标签估算并标注 `estimated`；识别不出的标「未分类」。好看永远让位于准确。
- **拦截性功能默认关**：鉴权、限流等会改变本地体验的能力默认关闭，部署时显式打开——和"零依赖可跑"保持一致。
- **零风险边界**：不碰逆向与风控路径（网易云写接口 / 取流逆向、已关闭的 Spotify audio-features），音频特征维持标签估算。

## 质量保障

三层回归把"Agent 没跑偏"变成可重复检查：

| 层 | 命令 | 作用 |
|---|---|---|
| 单元 + 集成 | `python3 -m pytest` | 1009 用例，零依赖可跑 |
| 长对话 smoke | `python3 scripts/long_dialogue_smoke.py` | 无需真实 key 的结构化回归，输出 MD/JSON 报告 |
| LLM-as-judge | `python3 -m tests.eval.run` | 真实 key 下的端到端评分（推荐 / 歌单 / 反幻觉 / 多样性 / 旅程） |

CI 跑 ruff、pytest coverage 和 long dialogue smoke；真实源 eval 走 nightly，避免外部接口波动阻塞本地回归。另有 `scripts/realistic_memory_eval.py` 做三段跨会话、16 轮的真实记忆压力测试。

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
