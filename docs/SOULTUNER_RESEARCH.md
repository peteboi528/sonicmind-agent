# SoulTuner-Agent 调研与对比分析

> 调研时间：2026-06-09 · 仓库地址：https://github.com/hgsanyang/SoulTuner-Agent
> 本文档目的：记录 SoulTuner-Agent 的优秀设计，对比分析差距，规划 MusicAgent 的升级路线

---

## 目录

- [1. SoulTuner-Agent 项目概览](#1-soultuner-agent-项目概览)
- [2. 核心架构对比](#2-核心架构对比)
- [3. SoulTuner 的 7 大优秀设计（值得学习）](#3-soultuner-的-7-大优秀设计值得学习)
- [4. MusicAgent 的 4 大优势（SoulTuner 做不到的）](#4-musicagent-的-4-大优势soultuner-做不到的)
- [5. LangGraph 重构工作量评估](#5-langgraph-重构工作量评估)
- [6. 差异化方向规划](#6-差异化方向规划)
- [7. 分阶段升级路线图](#7-分阶段升级路线图)
- [8. 决策记录与待定事项](#8-决策记录与待定事项)

---

## 1. SoulTuner-Agent 项目概览

### 基本信息

| 维度 | 说明 |
|------|------|
| 定位 | 音乐推荐 Agent（中文场景） |
| 技术栈 | LangGraph + LangChain + Neo4j + sentence-transformers |
| 重型依赖 | Neo4j 图数据库、M2D-CLAP（语义嵌入）、OMAR-RQ（声学嵌入）、GraphZep（长期记忆） |
| 模型要求 | GPU 推荐，模型总计 ~2.4GB |
| 部署方式 | Docker Compose（Neo4j + GraphZep + 主服务） |

### 架构图

```
用户输入
  ↓
recall_graphzep_memory（双阶段记忆召回：粗召回 20 条 → 精排 top 5）
  ↓
analyze_intent（结构化意图分析 → MusicQueryPlan Pydantic 对象）
  ↓
route_by_intent（7 类意图路由）
  ├── search_songs → route_after_search → web_fallback / generate_explanation
  ├── web_fallback → generate_explanation
  ├── generate_recommendations → generate_explanation
  ├── general_chat → persist_memory → END
  ├── acquire_music → generate_explanation
  ├── analyze_user_preferences → enhanced_recommendations → create_playlist / generate_explanation
  └── recommend_by_favorites（两层：Seeds 收藏 + Discoveries 向量发现）→ generate_explanation
       ↓
generate_explanation（SSE 流式：thinking → 歌曲卡片 → 推荐理由）
       ↓
extract_preferences（异步偏好提取 → Neo4j）
       ↓
persist_to_graphzep（异步对话持久化 → GraphZep → Neo4j）
       ↓
END
```

### 检索管线（7 步漏斗）

```
Step 1: 解析各引擎 JSON → 标准化列表
Step 2: 平等合并去重（双引擎交叉命中标记 🔥）
Step 3: DISLIKES 过滤（排除用户明确不喜欢的歌曲）
Step 4: Artist 多样性初筛（每个歌手最多 N 首）
Step 5: Graph Affinity 粗排（图距离 + 4 维 Jaccard 偏好加分）+ Thompson Sampling 探索槽
Step 6: 三锚归一化精排（语义 + 声学 + 个性化，归一化到 [0,1]）
Step 7: MMR 多维多样性重排（genre + mood + theme + scenario）+ FinalCut
```

---

## 2. 核心架构对比

### 2.1 全景对比表

| 维度 | SoulTuner | MusicAgent（现状） | 差距等级 |
|------|-----------|---------------------|---------|
| **Agent 框架** | LangGraph StateGraph（图节点路由） | 自写 ReAct Loop + if-elif 链 | 🔴 大 |
| **意图识别** | 7 类 + Pydantic 结构化输出 `MusicQueryPlan` | 13 工具由 LLM 自由选择（function calling） | 🟡 中 |
| **上下文管理** | GSSC 4 阶段 Token 预算 + 异步预压缩缓存 | `obs_text[:1500]` 硬截断 | 🔴 大 |
| **检索管线** | 7 步漏斗（合并→去重→DISLIKES→多样性→粗排→三锚精排→MMR） | TF 余弦 + 关键词混合 | 🔴 大 |
| **精排算法** | 三锚归一化（语义 M2D-CLAP + 声学 OMAR-RQ + 个性化 Graph Affinity） | 手工权重打分（5 个固定权重） | 🔴 大 |
| **探索-利用** | Thompson Sampling（Beta 分布采样 + 曝光衰减） | 固定 `discovery_openness` 比例 | 🟡 中 |
| **长期记忆** | GraphZep 双阶段召回（粗 20 条 → 精 top 5） | JSON 偏好列表 + 时间衰减 | 🟡 中 |
| **流式输出** | SSE（thinking → 歌曲卡片 → 推荐理由） | 无（等整个 ReAct 循环结束） | 🔴 大 |
| **反幻觉** | 无 | ✅ Answer Guard（白名单校验 + 歌名剥离） | 🟢 **领先** |
| **离线 Demo** | 必须有 Neo4j + GPU | ✅ MockLLM 零依赖即开即用 | 🟢 **领先** |
| **网易云深度集成** | 仅搜索 API | ✅ QR 登录 + 歌单导入 + 音频播放 + MV 播放 | 🟢 **领先** |
| **行为评分** | 无（只有点赞/跳过） | ✅ BaRT 行为评分 + discovery_openness | 🟢 **领先** |
| **UI 层** | 前端（React + TailwindCSS） | ✅ Streamlit Spotify 主题 UI | 🟡 各有千秋 |
| **代码量** | ~15,000+ 行（分散） | ~4,570 行（精简） | 🟡 各有千秋 |

### 2.2 SoulTuner 的关键依赖（我们没有且不需要的）

| 依赖 | 用途 | 我们是否需要 |
|------|------|------------|
| Neo4j | 图数据库、知识图谱、向量索引 | ❌ 个人项目太重 |
| M2D-CLAP | 音乐文本联合嵌入（~1.4GB） | ❌ 需要 GPU |
| OMAR-RQ | 声学嵌入（~800MB） | ❌ 需要 GPU |
| GraphZep | 长期记忆服务（独立 Docker） | ❌ 个人项目太重 |
| langchain-openai | LLM 调用 | ❌ 有自定义 client |
| sentence-transformers | 语义嵌入 | ✅ 已有（可选） |

---

## 3. SoulTuner 的 7 大优秀设计（值得学习）

### 设计 1：LangGraph StateGraph（架构层面）

**SoulTuner 的做法：**
- 每个能力是一个独立的 `async def node(state) -> dict` 函数
- 节点间通过条件边路由：`route_by_intent`、`route_after_search`
- 状态通过 TypedDict 在节点间流动
- 支持 MemorySaver checkpoint（对话状态持久化）
- 三个 LLM 角色分离：意图分析 / 解释生成 / 通用聊天

**可以学习的：**
- 将 `_execute_tool` 的 200 行 if-elif 链拆为独立节点函数
- 条件路由替代固定的 for 循环
- 状态 TypedDict 替代 `list[dict[str, Any]]` 传递

**学习成本：** 新增 ~590 行代码，依赖 langgraph

### 设计 2：GSSC 上下文管理（Token 预算）

**SoulTuner 的做法（4 阶段）：**
```
Stage 1: Gather — 收集所有上下文源（记忆、对话历史、检索结果）
Stage 2: Select — 按优先级排序（用户输入 > 记忆 > 对话历史 > 检索结果）
Stage 3: Structure — 按 min_tokens 保证 + 剩余预算按优先级分配
Stage 4: Compress — 超预算时 LLM 摘要压缩（替代硬截断）+ 异步预压缩缓存
```

**我们的现状：** `obs_text[:1500]` — 直接砍掉尾部，丢失信息

**可以学习的（简化版）：**
```python
def smart_truncate(contexts: dict[str, str], budget: int = 2000) -> dict[str, str]:
    PRIORITY = {"user_query": 0, "memory": 1, "tool_results": 2, "chat_history": 3}
    # 按优先级分配 Token 预算，低优先级的先被截断
```

**学习成本：** ~80 行新代码，无新依赖

### 设计 3：结构化意图输出（MusicQueryPlan）

**SoulTuner 的做法：**
```python
class MusicQueryPlan(BaseModel):
    intent_type: Literal["graph_search", "hybrid_search", "vector_search",
                          "web_search", "general_chat", "acquire_music", "recommend_by_favorites"]
    parameters: dict
    reasoning: str
    retrieval_plan: RetrievalPlan  # 明确的检索策略

class RetrievalPlan(BaseModel):
    use_graph: bool
    use_vector: bool
    use_web_search: bool
    graph_entities: list[str]
    graph_genre_filter: str | None
    graph_mood_filter: str | None
    vector_acoustic_query: str  # HyDE 声学描述
```

**我们的现状：** LLM 通过 function calling 自由选工具，无结构化检索计划

**可以学习的：**
```python
class AgentPlan(BaseModel):
    intent: str
    tools_needed: list[str]
    reasoning: str
    search_strategy: Literal["local_only", "local_then_online", "online_first", "no_search"]
```

**学习成本：** ~50 行新代码（Pydantic 模型 + prompt 模板调整）

### 设计 4：Thompson Sampling（优雅的探索-利用平衡）

**SoulTuner 的做法：**
```python
# 每首歌维护 Beta(α, β) 参数
# α → 用户喜欢的信号，β → 曝光衰减信号
# 推荐一次 → ts_beta += 0.3（降低未来被采样概率）
# 用户喜欢 → ts_alpha += 1（提升采样分数）
# 粗排后尾部歌曲用 TS 采样捞回冷门
```

**我们的现状：** `discovery_openness` 是一个固定比例（0.1 ~ 0.6）

**可以学习的（简化版）：**
```python
import random

def thompson_sample(candidates: list[dict], explore_ratio: float = 0.2) -> list[dict]:
    main = candidates[:int(len(candidates) * (1 - explore_ratio))]
    tail = candidates[int(len(candidates) * (1 - explore_ratio)):]
    for c in tail:
        alpha, beta = c.get("ts_alpha", 1), c.get("ts_beta", 1)
        c["_ts_score"] = random.betavariate(alpha, beta)
    tail.sort(key=lambda x: x["_ts_score"], reverse=True)
    return main + tail[:max(1, int(len(candidates) * explore_ratio))]
```

**学习成本：** ~40 行新代码，无新依赖

### 设计 5：SSE 流式输出（用户体验质变）

**SoulTuner 的做法：**
```
前端实时收到：
1. thinking... → "正在分析你的需求..."
2. songs_card → 渲染歌曲卡片（不等文字完成）
3. explanation → 逐字流式输出推荐理由
```

**我们的现状：** 等整个 ReAct 循环结束才返回，用户可能等 10-30 秒

**可以学习的（FastAPI SSE）：**
```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        yield f"data: {json.dumps({'type': 'thinking', 'content': '正在思考...'})}\n\n"
        for step in react_steps:
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': step.name})}\n\n"
        yield f"data: {json.dumps({'type': 'final', 'content': final_answer})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**学习成本：** ~100 行新代码（FastAPI SSE + Streamlit 前端适配）

### 设计 6：DISLIKES 过滤（显式负面偏好）

**SoulTuner 的做法：**
```python
# Neo4j 查询 DISLIKES 关系
MATCH (u:User {id: $uid})-[:DISLIKES]->(s:Song)
RETURN collect(s.title) AS titles

# 推荐结果中过滤掉
final_list = [item for item in final_list if title not in disliked_titles]
```

**我们的现状：** 只有正面偏好（喜欢/评分），没有显式"不喜欢"机制

**可以学习的（简化版）：**
```python
# memory.py 中增加 dislikes 列表
# 每次推荐前过滤
disliked = {d.asset_id for d in memory.dislikes}
candidates = [c for c in candidates if c.asset_id not in disliked]
```

**学习成本：** ~30 行新代码，在 MemoryManager 中加一个列表

### 设计 7：异步预压缩缓存（消除阻塞）

**SoulTuner 的做法：**
```python
# 每轮对话结束后，异步预压缩对话历史
asyncio.create_task(pre_compress_and_cache(user_id, chat_history_text))
# 下一轮请求直接读取缓存，跳过 LLM 压缩（节省 15-20s）
```

**我们的现状：** 无压缩机制

**可以学习的：** 这是 GSSC 的一部分，在实现上下文管理时一并考虑

---

## 4. MusicAgent 的 4 大优势（SoulTuner 做不到的）

### 优势 1：Answer Guard（反幻觉机制）

SoulTuner **完全没有**反幻觉机制。我们有：
- 收集所有工具结果中的"已知歌名" → 白名单
- 扫描最终回答中 `《》` 包裹的歌名
- 不在白名单中 → 剥离（歌单/专辑名豁免）
- Trace 中记录被移除的幻觉歌名

**价值：** 这是面试加分项，SoulTuner 没有这个能力

### 优势 2：零依赖离线 Demo

SoulTuner 需要：
- Neo4j 服务
- GraphZep 服务
- GPU（M2D-CLAP + OMAR-RQ）
- 至少 3 个 LLM API key

我们只需要：
- 一个 Python 环境
- 无 API Key 时自动切换 MockLLM
- `pip install -e .` 即可运行

**价值：** 面试时现场 demo 零风险

### 优势 3：网易云深度生态集成

SoulTuner 只有网易云 API 搜索。我们有：
- QR 码登录（`netease_auth.py`）
- 歌单批量导入（`import_netease_playlist`）
- 音频 URL 获取（含 VIP）
- B站 MV 嵌入播放
- 元数据自动丰富

**价值：** 差异化核心，SoulTuner 无法复制

### 优势 4：BaRT 行为评分

SoulTuner 只有简单的喜欢/跳过。我们有：
- Spotify BaRT 论文启发的行为奖励信号
- 完整播放 +1，秒跳 -1，部分播放线性插值
- 时间指数衰减累积
- `discovery_openness` 动态调整

**价值：** 理论基础扎实，面试时可以展开讲

---

## 5. LangGraph 重构工作量评估

### 5.1 数据总览

| 指标 | 数值 |
|------|------|
| 现有代码总量 | ~4,570 行 |
| 可原样复用 | ~3,890 行（**85%**） |
| 需要重写 | ~470 行（10%） |
| 需要丢弃 | ~210 行（5%） |
| **需要新写** | **~590 行** |
| 新建文件 | 5 个 |
| 修改文件 | 2 个 |
| 新增依赖 | 3 个 |

### 5.2 新文件清单

```
app/graph/
├── __init__.py     (~150行) — 图构建：节点注册 + 边连接 + 编译
├── state.py        (~30行)  — AgentState TypedDict
├── nodes.py        (~300行) — 16 个节点函数
├── routing.py      (~60行)  — 条件边：should_continue / route_to_tool
└── guard.py        (~50行)  — Answer Guard（从 react_loop.py 搬过来）
```

### 5.3 图拓扑设计

```
__start__
  ↓
load_context（加载记忆 + 目标状态）
  ↓
build_messages（拼 system prompt + history + context）
  ↓
llm_decide（调 LLM，让它选工具）
  ↓
should_continue ──── 无 tool_calls ───→ finalize
  │                                        ↓
  ↓ 有 tool_calls                     guard_answer
route_to_tool                             ↓
  │                                  update_goal
  ├→ recommend          ──┐          ↓
  ├→ search              │       compose_output
  ├→ playlist            │          ↓
  ├→ taste               │       __end__
  ├→ similar_cross       │
  ├→ retrieve            │
  ├→ memory_update       │
  ├→ web_music_search    │
  ├→ fetch_metadata      │
  ├→ import_netease      │
  └→ ...                 │
                         ↓
                   check_completion ──→ 未满足 → build_messages（循环）
                                      满足 → finalize
```

### 5.4 逐文件改动清单

#### 完全不动（0 改动）

| 文件 | 行数 | 原因 |
|------|------|------|
| `app/models.py` | 351 | Pydantic 模型不变 |
| `app/memory.py` | 441 | MemoryManager 作为服务注入节点 |
| `app/recommend/engine.py` | 197 | 推荐引擎不变 |
| `app/recommend/daily.py` | 242 | 每日推荐不变 |
| `app/retrieval/vector_store.py` | 117 | 检索引擎不变 |
| `app/retrieval/embeddings.py` | ~60 | 嵌入模块不变 |
| `app/llm/client.py` | 138 | 保持自定义 LLM 客户端 |
| `app/llm/tools.py` | 180 | OpenAI 工具 schema 直接可用 |
| `app/prompts/*` | ~100 | 所有 prompt 不变 |

#### 需要改动

| 文件 | 改动 | 工作量 |
|------|------|--------|
| `app/react_loop.py` (985行) | 拆分为 5 个 graph 文件 | 最大 |
| `app/agent.py` (1485行) | 删 `chat()` 和 `__init__`，业务逻辑保留 | 小 |
| `app/api/main.py` (214行) | 改 5 行：`agent.chat()` → `graph.invoke()` | 最小 |
| `pyproject.toml` | 加 3 行依赖 | 最小 |

### 5.5 时间估算

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| Day 1 | 新建 `app/graph/`，写 `state.py` + `routing.py` + `guard.py` | 3-4h |
| Day 2 | 写 `nodes.py`（从 `_execute_tool` 提取 13 个节点） | 4-5h |
| Day 3 上午 | 写 `__init__.py`（图构建 + 边连接） | 2-3h |
| Day 3 下午 | 改 `api/main.py` 接入 graph，调通基本流程 | 2-3h |
| Day 4 | 跑通全部测试 + 修复兼容性问题 | 4-6h |
| Day 5 | 加 MemorySaver checkpoint + SSE 流式（可选） | 3-4h |
| **总计** | | **5-7 天** |

---

## 6. 差异化方向规划

### SoulTuner 的弱点（我们的机会）

| SoulTuner 弱点 | 我们的机会 |
|---------------|-----------|
| 部署复杂（Neo4j + GPU + Docker） | 零依赖离线 demo，面试即开即用 |
| 无反幻觉机制 | Answer Guard 是独有优势 |
| 网易云只做搜索 | 深度生态集成（登录/导入/播放/MV） |
| 无行为评分 | BaRT 评分有论文基础 |
| 重型系统，个人开发者难以复现 | 轻量精简，代码量仅 1/3 |

### 差异化方向 1：音乐旅程编排（真正的多步规划）

```
用户："帮我编一段跑步时的音乐，先热身，中间冲起来，最后放松"
  ↓
Agent 拆解为 3 个情绪阶段：
  Phase 1: 温暖/轻快（热身）→ 推荐 3 首
  Phase 2: 激烈/高能量（冲刺）→ 推荐 3 首
  Phase 3: 舒缓/放松（冷却）→ 推荐 3 首
  ↓
每阶段附带推荐理由和过渡说明
```

**实现方式：** 用现有的推荐引擎 + prompt 拆解，代码量约 100 行

### 差异化方向 2：Agent 透明度面板（面试杀手锏）

```
Streamlit 聊天界面，每条回复下方可展开：

📋 Agent 决策过程
├─ [Think] 用户想找适合跑步的歌，需要高能量 + 节拍快
├─ [Act] 调用 search_music("高能量 跑步 快节拍")
├─ [Observe] 找到 5 首，但只有 2 首符合 BPM>130
├─ [Reflect] 不够，换用 recommend_music 补充
├─ [Act] 调用 recommend_music(genre=electronic, energy=0.9)
├─ [Observe] 又找到 3 首
└─ [Guard] 检查歌名真实性 → 全部通过 ✅

📊 记忆变化
├─ 新增偏好：高能量音乐 (freq=1)
└─ discovery_openness: 0.3 → 0.35 (听完了一首新流派)

🧮 推荐打分
├─ 晴天: genre=0.30 mood=0.25 energy=0.18 → 0.73
├─ 双截棍: genre=0.15 mood=0.10 energy=0.20 → 0.45 (探索推荐 🆕)
└─ ...
```

### 差异化方向 3：网易云深度集成扩展

```
现有能力：
├─ ✅ QR 码登录
├─ ✅ 歌单导入
├─ ✅ 音频 URL 获取
└─ ✅ MV 播放

可扩展：
├─ 📋 "把这个歌单同步到我的网易云" ← 一键导出
├─ 📊 "我网易云最近在听什么？"     ← 读取最近播放记录
└─ 👥 "和我品味相似的用户在听什么" ← 社交推荐
```

---

## 7. 分阶段升级路线图

### Phase 1：基础质感（1 周，优先级最高）

| # | 任务 | 来源 | 预计耗时 | 产出 |
|---|------|------|---------|------|
| 1 | 加 `logging` + 替换 15 处 `except: pass` | 工程质量 | 1h | 全局可观测 |
| 2 | 原子写入（`os.replace`） | 工程质量 | 0.5h | 存储安全 |
| 3 | 错误异常化（替代 `startswith("LLM 请求失败")`） | 工程质量 | 0.5h | 正确的错误处理 |
| 4 | 魔数提取为 `ScoringWeights` dataclass | 工程质量 | 0.5h | 可配置打分 |
| 5 | `_execute_tool` if-elif → 策略模式 | 代码组织 | 2-3h | 可维护性提升 |

### Phase 2：LangGraph 重构（1-2 周，核心升级）

| # | 任务 | 来源 | 预计耗时 | 产出 |
|---|------|------|---------|------|
| 6 | 新建 `app/graph/` + `state.py` + `routing.py` | SoulTuner 设计 1 | 3-4h | 图状态定义 |
| 7 | 从 `_execute_tool` 提取 13 个节点函数 | SoulTuner 设计 1 | 4-5h | 节点解耦 |
| 8 | 图构建（`__init__.py`）+ API 接入 | SoulTuner 设计 1 | 3-4h | LangGraph 运行 |
| 9 | 跑通全部测试 + 兼容性修复 | — | 4-6h | 测试全绿 |

### Phase 3：Agent 智能化（1 周，用户体验提升）

| # | 任务 | 来源 | 预计耗时 | 产出 |
|---|------|------|---------|------|
| 10 | 智能上下文管理（替代硬截断） | SoulTuner 设计 2 | 2-3h | 长对话不丢信息 |
| 11 | 结构化意图输出（`AgentPlan`） | SoulTuner 设计 3 | 2-3h | 决策可追踪 |
| 12 | Thompson Sampling 替代固定 explore | SoulTuner 设计 4 | 1-2h | 冷门歌有出头机会 |
| 13 | SSE 流式输出 | SoulTuner 设计 5 | 3-4h | 实时体验 |

### Phase 4：差异化打磨（1 周，面试加分）

| # | 任务 | 来源 | 预计耗时 | 产出 |
|---|------|------|---------|------|
| 14 | Agent 透明度面板 | 原创差异化 | 2-3h | 决策可视化 |
| 15 | 音乐旅程编排 | SoulTuner 启发 | 3-4h | 真正的多步规划 |
| 16 | 网易云扩展（同步/历史） | 原创差异化 | 3-4h | 生态深度 |
| 17 | README + 架构文档重写 | — | 2-3h | 项目门面 |

---

## 8. 决策记录与待定事项

### 已确认的决策

| # | 决策 | 选项 | 理由 |
|---|------|------|------|
| 1 | LLM 客户端 | **保持自定义 `OpenAICompatibleLLM`** | 不引入 langchain-openai，减少迁移风险 |
| 2 | 工具 schema | **保持现有 dict 格式** | 兼容现有 LLM client，不换 @tool |
| 3 | 状态持久化 | **`MemorySaver`（内存级）** | 个人项目够用 |
| 4 | agent.py 拆分 | **保持原样，节点调用 agent 方法** | 先跑通再重构 |
| 5 | 不引入 Neo4j | **保持 JSON 存储** | 个人项目不需要图数据库 |
| 6 | 不引入 GPU 模型 | **保持 TF 余弦 + 可选 sentence-transformers** | 保持轻量 |

### 待定事项（需进一步思考）

| # | 问题 | 选项 | 影响 |
|---|------|------|------|
| 1 | 是否引入 `langchain-core`？ | A) 只为 langgraph 依赖引入 / B) 完全不用 langgraph，自写图框架 | 决定架构方向 |
| 2 | agent.py 拆到 services 的时机 | A) Phase 2 一起做 / B) Phase 4 再做 | 影响重构范围 |
| 3 | SSE 流式的 Streamlit 适配 | A) 用 `st.write_stream` / B) 换前端框架 | 影响 UI 工作量 |
| 4 | 音乐旅程是否作为独立工具 | A) 新工具 `music_journey` / B) 在 playlist 中实现 | 影响工具架构 |
| 5 | DISLIKES 是否实现 | A) Phase 3 实现 / B) 暂不实现 | 影响推荐精度 |

---

## 附录：SoulTuner 关键代码片段（供参考）

### A. GSSC 上下文构建核心逻辑

```python
# 摘自 SoulTuner retrieval/gssc_context_builder.py

PRIORITY_USER_INPUT = 0        # 不可截断
PRIORITY_GRAPHZEP_FACTS = 1    # 高优先级
PRIORITY_CHAT_HISTORY = 2      # 中等
PRIORITY_RETRIEVAL = 3         # 可完全省略

async def build_context(graphzep_facts, chat_history, retrieval_context, user_input, total_budget):
    # Stage 1: Gather — 收集所有源
    sources = [ContextSource(name, content, priority, min_tokens) for ...]
    # Stage 2: Select — 按优先级排序
    sources.sort(key=lambda s: s.priority)
    # Stage 3: Structure — 分配预算
    for src in sources:
        extra_needed = max(0, src.estimated_tokens - src.min_tokens)
        allocations[src.name] = src.min_tokens + min(extra_needed, remaining_budget)
    # Stage 4: Compress — 智能压缩
    if src.estimated_tokens > budget * 1.5:
        cached = get_cached_compression(user_id, src.estimated_tokens)
        if cached: result[src.name] = cached
        else: result[src.name] = src.truncate_to(budget)
```

### B. 三锚精排核心公式

```python
# 摘自 SoulTuner retrieval/hybrid_retrieval.py

final_score = w_semantic * semantic + w_acoustic * acoustic + w_personalize * personalize

# semantic: M2D-CLAP cosine(song_emb, query_text_emb) → (x+1)/2 归一化到 [0,1]
# acoustic:  OMAR-RQ cosine(song_emb, centroid) → (x+1)/2 归一化到 [0,1]
# personalize: graph_affinity (图距离 + Jaccard 偏好) → MinMax 归一化到 [0,1]
```

### C. Thompson Sampling 探索槽

```python
# 摘自 SoulTuner retrieval/hybrid_retrieval.py

# 粗排后尾部歌曲用 TS 采样
tail_with_scores = sorted(zip(tail_candidates, ts_scores), key=lambda x: x[1], reverse=True)
explore_picks = [item for item, _ in tail_with_scores[:n_explore]]

# 每推荐一次，ts_beta += 0.3（曝光衰减）
# 用户喜欢，ts_alpha += 1（提升分数）
```

---

> 📄 本文档基于 SoulTuner-Agent 源码（2026-06-09 版本）与 MusicAgent 代码的深度对比分析。
> 作为后续升级工作的参考依据。
