# 智能影音推荐 Agent（MusicAgent / SonicMind）

一个**可解释、记忆驱动、反幻觉**的音乐推荐 Agent。核心不是单次 LLM 对话，而是一个围绕用户品味持续工作、决策全程可追溯的智能体。

```text
入库 → 分段分析 → 证据库 → 记忆/偏好/收听/评分 → 品味模型
   → LangGraph 编排（意图识别 → 工具执行 → web 兜底 → 反幻觉守卫 → grounded answer）
   → 三锚精排 + MMR 多样性重排 → Thompson 探索/在线学习 → 透明度面板
```

## 四大特色

| 特色 | 说明 |
|---|---|
| 🛡️ **反幻觉** | Answer Guard 在出答案前移除未核实歌名；候选不足时**诚实说明**而非编造。所有推荐可追溯到真实平台来源。 |
| 🔌 **零依赖可跑** | 无 `LLM_API_KEY` 时用 `MockLLM`，无 langgraph 时图编排自动降级到等价同步执行，无 sentence-transformers 时语义检索回退 TF cosine。开箱即跑的稳定 demo。 |
| 🎵 **多源搜索生态** | 网易云歌曲搜索 + B站/YouTube MV/现场视频 + Tavily 联网百科。不同场景自动路由最优数据源。 |
| 📈 **BaRT 行为奖励 + 在线学习** | 听完/秒跳/评分/负反馈实时喂给 Thompson Sampling，探索冷门同时在线学习用户口味。 |

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
| 🏔️ **journey** | 「做个从热身到冲刺的音乐旅程」 | 多阶段情绪曲线编排 |
| 💭 **chat** | 「你好」「谢谢」 | 自然寒暄 |

### 多轮对话

- **延续指令**：「多来几首」「换一批」「还有吗」→ 继承上一轮实体和意图，自动去重已展示歌曲
- **话题切换**：「找他的MV」「介绍这个歌手」→ 自动检测新意图信号，不再延续
- **指代消解**：「他的歌」「只要他的」→ 短句指代自动继承上一轮实体

## 架构概览

### 单一 LangGraph 编排

聊天请求走统一的图编排（`app/graph/`），不再有多套并行的决策逻辑：

```text
load_context → plan_intent → execute_tools ─┬─（候选充足）→ evaluate → finalize
   (GSSC预算)   (LLM判意图)    (调用工具)      └─（候选不足）→ web_fallback ↗
```

- **意图识别分工**：LLM 只判意图 + 抽实体名，`genre/mood/scenario` 标签交给**确定性规则**（`app/graph/tag_rules.py`），降幻觉降成本。
- **意图 Registry**（`app/intents.py`）：所有意图元数据集中声明——工具链、策略、关键词、优先级。新增意图只改一处。
- **条件路由**：本地/检索候选不足时自动触发 `web_fallback` 联网补搜。
- **降级链**：LangGraph → 同步等价执行 → ReAct 循环兜底。
- **安全网**：LLM 误判意图时（如把「介绍」判成 discuss），按关键词信号自动升级到正确意图。

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

### Agent 透明度面板（UI）

每条推荐回复下可展开：
- 🧠 **决策过程**：各节点 trace 结构化（意图 → 工具 → 兜底 → 守卫）。
- 📊 **上下文预算 / 反幻觉**：GSSC 预算报告 + 守卫移除记录。
- 🧮 **三锚打分明细**：每首歌的语义/口味/行为分 + 权重表格。

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

uvicorn app.api.main:app --reload --port 8000
streamlit run app/ui/streamlit_app.py --server.port 8501
```

- API 文档：`http://127.0.0.1:8000/docs`
- Vue 3 前端：`app/web/dist/index.html`
- Streamlit UI：`http://127.0.0.1:8501`

## 主要 API

| 端点 | 用途 |
|---|---|
| `POST /chat` | 主对话入口（走 LangGraph 编排） |
| `POST /agent/stream` | 流式：plan → 候选卡片 → 理由 → final |
| `POST /recommend/daily` | 每日推荐（三锚精排） |
| `POST /playlist/generate` | 歌单生成 |
| `POST /search` | 搜索（本地 + 联网真实候选） |
| `POST /listen` `POST /rate` | 收听/评分（喂 Thompson 反馈环） |
| `POST /assets/ingest` `/analyze` `/enrich` | 入库 / 分析 / 联网补全 |

## Demo 路线

1. 导入一个网易云歌单（自动三层兜底分类 genre/mood）。
2. 教 Agent 一个偏好：`我喜欢电子音乐和放松的氛围`。
3. 推荐 / 生成歌单，展开**透明度面板**看三锚打分和决策过程。
4. 对推荐点 👎，观察 Thompson 探索分下降、后续不再选中。
5. 问一个虚构查询，验证反幻觉守卫诚实拒答。
6. 试 `介绍NewJeans`，看 Tavily 搜索引擎实时查百科。
7. 试 `帮我找The Weeknd的MV`，看 B站/YouTube 视频搜索。
8. 说 `多来几首`，验证跨轮去重（不再重复推荐）。

## 测试与评估

- `python3 -m pytest`：单元 + 集成测试（零依赖可跑）。
- `python3 -m tests.eval.run`：LLM-as-judge 端到端评分（需真实 key），10 个 case 覆盖推荐/歌单/反幻觉/多样性/旅程/目标跟踪。详见 [tests/eval/README.md](tests/eval/README.md)。

## 工程取舍

- **所有增强依赖可选**：langgraph / sentence-transformers 缺失都有等价降级路径，零依赖 demo 能力贯穿始终。
- **不做异步 LLM 压缩**：同步架构下会阻塞主流程，GSSC 用确定性截断兜底。
- **标签走规则不走 LLM**：降低幻觉和成本，LLM 只做它擅长的意图判断。
- **关键词 fallback + LLM 双保险**：LLM 不可用时关键词信号兜底；LLM 误判时安全网自动纠正。

更多面试向架构讲解见 [docs/EXPLAINER.md](docs/EXPLAINER.md)。
