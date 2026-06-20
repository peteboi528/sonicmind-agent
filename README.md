# 智能影音推荐 Agent（MusicAgent / SonicMind）

一个**可解释、记忆驱动、反幻觉**的音乐推荐 Agent。核心不是单次 LLM 对话，而是一个围绕用户品味持续工作、决策全程可追溯的智能体。

```text
用户输入 → Graph 编排（context → intent plan → tools → evaluate/reflect → finalize）
   → Answer Guard（只保留可追溯歌名/专辑名）
   → SSE（plan / candidates / album_card / eval / final）
   → Vue 前端（聊天 / 探索工作台 / 播放器 / 曲库 / 歌单 / 偏好）
```

> 「分段分析」当前为**确定性占位**（demo/离线演示用，保证 RAG 链路可跑通）；真实 Whisper 转写 / CLIP 视觉标签是后续扩展点（见 [docs/EXPLAINER.md](docs/EXPLAINER.md) §7）。未识别出的 genre/mood 标「未分类」、tempo/energy 留空，**不随机伪造**——与本项目反幻觉原则一致。

## 四大特色

| 特色 | 说明 |
|---|---|
| 🛡️ **反幻觉** | Answer Guard 在出答案前移除未核实歌名；候选不足时**诚实说明**而非编造。所有推荐可追溯到真实平台来源。 |
| 🔌 **零依赖可跑** | 无 `LLM_API_KEY` 时用 `MockLLM`，无 langgraph 时图编排自动降级到等价同步执行，无 sentence-transformers 时语义检索回退 TF cosine。开箱即跑的稳定 demo。 |
| 🎵 **多源搜索生态** | 网易云歌曲搜索 + B站/YouTube MV/现场视频 + Tavily 联网百科。不同场景自动路由最优数据源。 |
| 🧪 **Taste Lab 品味实验** | 把推荐拆成 safe / stretch / bold 三档：先提出品味假设，再用播放、跳过、喜欢、不喜欢等反馈验证探索边界，最后生成实验报告。 |
| 📈 **BaRT 行为奖励 + 在线学习** | 听完/秒跳/评分/负反馈实时喂给 Thompson Sampling，探索冷门同时在线学习用户口味。 |
| 🧪 **质量闭环** | `pytest` + 长对话 smoke + eval/regress，把“Agent 没跑偏”变成可重复检查。 |

## 对话场景

Agent 自动识别意图，路由到最合适的工具链：

| 意图 | 触发示例 | 行为 |
|------|---------|------|
| 🎧 **recommend** | 「推荐几首歌」「来点chill的」「适合跑步的歌」 | 网易云 + LLM候选 + Last.fm 多路搜索 → 三锚精排 + MMR 重排 |
| 🔍 **search** | 「找一首歌」「帮我搜Drake」 | 网易云歌曲搜索为主，本地库补充 |
| 🎬 **video** | 「找The Weeknd的MV」「Adele现场演唱会」 | **直搜 B站 + YouTube**，不走网易云。B站优先（华语命中率高），YouTube 补位 |
| 📖 **artist_info** | 「介绍NewJeans」「Drake是谁」「Coldplay出道经历」 | **Tavily 搜索引擎查百科**，LLM 基于真实搜索结果组织 200-400 字介绍 + 来源链接 |
| 💬 **discuss** | 「The Weeknd怎么样」「Drake和Kendrick谁更牛」 | 搜真实曲目作为论据，LLM 基于事实讨论（反幻觉：不确定的说不知道） |
| 📋 **playlist** | 「做20首chill歌单」 | 联网扩展 + LLM 精选，保留歌单名和描述 |
| 📥 **import** | 「导入网易云歌单」 | 网易云「我的歌单」一键导入 |
| 🎯 **taste** | 「分析我的品味」 | 只读记忆/行为画像，无需联网 |
| 🧪 **taste_experiment** | 「推荐点不一样的」「做个品味实验」「我听腻了」 | 生成 safe / stretch / bold 三档实验，收集反馈后输出报告 |
| 🏔️ **journey** | 「做个从热身到冲刺的音乐旅程」 | 多阶段情绪曲线编排 |
| 💭 **chat** | 「你好」「谢谢」 | 自然寒暄 |

### 多轮对话

- **延续指令**：「多来几首」「换一批」「还有吗」→ 继承上一轮实体和意图，自动去重已展示歌曲
- **话题切换**：「找他的MV」「介绍这个歌手」→ 自动检测新意图信号，不再延续
- **指代消解**：「他的歌」「只要他的」→ 短句指代自动继承上一轮实体

## 架构概览

### 单一 Graph 编排

聊天请求走统一的图编排（`app/graph/`），不再有多套并行的决策逻辑：

```text
load_context → plan_intent → execute_tools
       └→ web_fallback? → evaluate → reflect? → finalize
```

- **意图识别分工**：LLM 只判意图 + 抽实体名，`genre/mood/scenario` 标签交给**确定性规则**（`app/graph/tag_rules.py`），降幻觉降成本。
- **意图 Registry**（`app/intents.py`）：所有意图元数据集中声明——工具链、策略、关键词、优先级。新增意图只改一处。
- **条件路由**：本地/检索候选不足时自动触发 `web_fallback` 联网补搜。
- **单一编排**：生产对话只走异步 LangGraph；复合任务由同一 compiled graph 作为子图执行。
- **安全网**：LLM 误判意图时（如把「介绍」判成 discuss），按关键词信号自动升级到正确意图。
- **稳定摘要**：`final` SSE payload 附带 `trace_summary`，前端可展示意图、工具、来源、fallback 和最终卡片数。

### 三锚归一化精排（`app/recommend/rerank.py`）

```text
final = w_语义·semantic + w_口味·personalize + w_行为·behavior
```

- **语义锚**：sentence-transformers dense 向量（可选），不可用时回退 TF 词项重叠。
- **口味锚**：4 维 Jaccard（genre/mood/scenario/theme）对用户偏好，轻量版 Graph Affinity，无需图数据库。
- **行为锚**：BaRT 收听奖励信号。
- **缺锚自动重分配**：某锚不可用时其权重转移给其余锚，不把分数拉平。
- **MMR 多样性重排**：`λ·相关性 − (1−λ)·候选间标签重叠`，避免连续推荐同质曲目。

### GSSC 上下文管理（`app/context/gssc.py`）

多源上下文按优先级（用户输入 > 记忆 > 历史）分配 token 预算，min_tokens 保底 + 剩余按优先级分配 + 按行截断兜底。**绝不同步调 LLM 压缩**（会阻塞主流程）。每轮产出 before/after/saved 追踪报告。

### Thompson Sampling 探索 + 反馈环（`app/library.py`）

- 每个候选维护 Beta(α, β) 后验，尾部候选采样捞回高潜力冷门歌。
- **在线学习反馈环**：听完 → α+1，秒跳 → β+0.5，高分 → α+，负反馈/不喜欢 → β+3，曝光衰减 → β+0.3。

### Taste Lab 品味实验（`app/agent.py`）

Taste Lab 是项目的特色主线：Agent 不只回答“我猜你会喜欢什么”，而是生成一个可验证的小实验。

```text
品味画像/行为信号 → 候选池 → rerank components → safe / stretch / bold
        → 播放/跳过/喜欢/不喜欢/太安全/太远
        → bucket 统计 → 实验报告 → 下一轮推荐策略
```

- **safe**：高 personalize / behavior，验证稳定画像。
- **stretch**：相邻风格或能量轻微越界，验证可扩展空间。
- **bold**：明显探索但仍遵守 dislike/exclusion，验证边界。
- **报告确定性优先**：完成率、跳过率、喜欢率、收藏率和评分先算清楚，LLM 只可作为自然语言增强，不负责编造结论。

### Agent 透明度（UI / 报告）

每条流式最终回复会携带：
- **原始 `agent_trace`**：节点级调试轨迹。
- **稳定 `trace_summary`**：意图、策略、工具、来源、fallback、guard、最终卡片数。
- **候选卡片**：与答案文本实际列出的曲目对齐，避免“文本 5 首、卡片 12 张”的漂移。

## 快速开始

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest                 # 全量测试（零依赖即可跑）

# 可选增强
python3 -m pip install -e ".[embeddings]"   # 语义检索（sentence-transformers）
# langgraph 已在主依赖，缺失时自动降级

# 环境变量（可选，增强联网能力）
# LLM_API_KEY=...           # LLM 服务密钥
# TAVILY_API_KEY=...         # Tavily 搜索引擎（歌手百科查询，无 key 时 DuckDuckGo 兜底）
# LASTFM_API_KEY=...         # Last.fm 音乐发现
# AUTH_ENABLED=true          # 部署时开启 API key 鉴权
# USER_API_KEYS=u1:key1      # per-user key，服务端覆盖客户端 user_id
# ALLOWED_ORIGINS=https://your.domain

uvicorn app.api.main:app --reload --port 8000
```

- API 文档：`http://127.0.0.1:8000/docs`
- Vue 3 前端：`http://127.0.0.1:8000/web`（源码 `frontend/`，构建产物 `app/web/dist/`）

## 主要 API

| 端点 | 用途 |
|---|---|
| `POST /chat` | 主对话入口（走 LangGraph 编排） |
| `POST /agent/stream` | 流式：plan → 候选卡片 → 理由 → final |
| `POST /recommend/daily` | 每日推荐（三锚精排） |
| `POST /playlist/generate` | 歌单生成 |
| `POST /search` | 搜索（本地 + 联网真实候选） |
| `POST /listen` `POST /rate` | 收听/评分（喂 Thompson 反馈环） |
| `POST /taste/experiment/generate` | 生成 Taste Lab 三档品味实验 |
| `GET /taste/experiments/{user_id}` | 查看实验历史 |
| `POST /taste/experiment/feedback` | 记录实验内单曲反馈 |
| `POST /taste/experiment/report` | 生成/刷新实验报告 |
| `POST /assets/ingest` `/analyze` `/enrich` | 入库 / 分析 / 联网补全 |

## Demo 路线

1. 导入一个网易云歌单（自动三层兜底分类 genre/mood）。
2. 教 Agent 一个偏好：`我喜欢电子音乐和放松的氛围`。
3. 推荐 / 生成歌单，展开**透明度面板**看三锚打分和决策过程。
4. 对推荐点 👎，观察 Thompson 探索分下降、后续不再选中。
5. 问一个虚构查询，验证反幻觉守卫诚实拒答。
6. 试 `介绍NewJeans`，看 Tavily 搜索引擎实时查百科。
7. 试 `帮我找The Weeknd的MV`，看 B站/YouTube 视频搜索。
8. 试 `推荐点不一样的，做个品味实验`，打开“实验室”听三档候选并标记“太安全/太远”。
9. 说 `多来几首`，验证跨轮去重（不再重复推荐）。

## 测试与评估

- `python3 -m pytest`：单元 + 集成测试（零依赖可跑）。
- `python3 scripts/long_dialogue_smoke.py`：长对话结构化回归，输出 `artifacts/long_dialogue_smoke_report.md/json`。
- `python3 -m tests.eval.run`：LLM-as-judge 端到端评分（需真实 key），10 个 case 覆盖推荐/歌单/反幻觉/多样性/旅程/目标跟踪。详见 [tests/eval/README.md](tests/eval/README.md)。

CI 会跑 ruff、pytest coverage 和 long dialogue smoke；真实源 eval 不进默认 CI，避免外部接口波动阻塞本地回归。

## 工程取舍

- **所有增强依赖可选**：langgraph / sentence-transformers 缺失都有等价降级路径，零依赖 demo 能力贯穿始终。
- **原生异步 LLM**：SSE 的规划、自省、恢复和最终 token 使用共享异步 HTTP 客户端；GSSC 继续负责确定性上下文预算。
- **标签走规则不走 LLM**：降低幻觉和成本，LLM 只做它擅长的意图判断。
- **关键词 fallback + LLM 双保险**：LLM 不可用时关键词信号兜底；LLM 误判时安全网自动纠正。

更多面试向架构讲解见 [docs/EXPLAINER.md](docs/EXPLAINER.md)。
