# MusicAgent 全面升级计划：追平 SoulTuner + 保留自身优势

> 基于 SoulTuner-Agent **真实源码**（非文档转述）的对比，2026-06-09 制定。
> 目标：把所有架构/算法差距追回，同时保留 4 大优势（反幻觉、零依赖 demo、网易云生态、BaRT）。

## 已确认的关键决策（用户拍板）

| # | 决策点 | 选择 |
|---|--------|------|
| 1 | 两套编排收敛 | **统一到 `app/graph/`**，把 react_loop 的 LLM function-calling / Answer Guard / eval / goal 全部吸收进图节点 |
| 2 | 流式实现 | **同步生成器流式**（generator + yield），不改 async，零迁移风险 |
| 3 | 精排范围 | **全做**：三锚归一化 + MMR + Graph Affinity + Thompson，**引入 sentence-transformers** 做语义锚 |
| 4 | 上下文管理 | 同步版 GSSC：优先级预算分配 + 按行截断 + Token 追踪报告（不做 LLM 压缩） |
| 5 | 在线学习 | **补齐 Thompson 反馈环**（听完→α+1，秒跳→β+，超越 SoulTuner 的未完成项） |
| 6 | 面试加分 | 透明度面板 + README/架构文档重写 + eval 扩展 + **UI 美化**（参考 SoulTuner Spotify 深色主题） |

## 起点现状（已开工，比旧文档领先）

- `app/graph/`：5 节点线性链（load_context→plan_intent→execute_tools→evaluate→finalize），但 `build_agent_plan` 是**纯关键词**，无 LLM 意图、无条件路由、无 web_fallback 回环。
- `app/react_loop.py`：**真 LLM function-calling 迭代循环**，已有 Answer Guard / eval / goal / grounded answer。这是要被吸收进图的核心资产。
- `app/library.py`：SQLite 候选库 + DISLIKES（已追平 SoulTuner）。
- `models.py`：`AgentPlan` / `StreamEvent` / `RankingBreakdown` / `ResourceTrack` 已定义。
- `engine.py`：手工 5 权重打分 + BaRT 行为分 + discovery_openness 自适应。
- 测试 61/61 通过。
- `agent.chat()` 当前主走 graph（弱），fallback 走 react（强）—— **本末倒置，需收敛**。

---

## Phase 0：编排收敛（地基，最先做）

**目标**：消灭双编排债，让 `app/graph/` 成为唯一、且具备 react_loop 全部能力的编排层。

1. **`AgentPlan` 模型升级**（`models.py`）：对齐 SoulTuner 的"检索策略型"意图分类。
   - `intent` 改为 `Literal["graph_search","hybrid_search","vector_search","web_search","recommend_by_favorites","journey","general_chat"]`（5 检索 + journey + chat）。
   - 新增 `RetrievalPlan` 子模型（`use_local/use_vector/use_web/entities/genre_filter/mood_filter/...`）。
   - 保留 `target_count/online_required/reasoning_summary`。
2. **LLM 结构化意图节点**（替换 `plan_intent` 的关键词 `build_agent_plan`）：
   - 新增 `app/prompts/query_plan.py`：让 LLM 产出 `AgentPlan`（用 `chat_with_tools` 的 `tool_choice=required` 强制单工具，或 `generate`+`extract_json_dict` 解析）。
   - **学 SoulTuner 的分工**：LLM 只判意图 + 抽实体名；genre/mood/scenario 标签由**确定性规则** `app/graph/tag_rules.py` 关键词映射填充。
   - 关键词 `build_agent_plan` **降级为 fallback**（LLM 失败时兜底，保留零依赖能力）。
   - MockLLM 增加 query_plan 的结构化模拟输出。
3. **`execute_tools` 节点重构为带条件路由的工具执行**：
   - 把 react_loop 的 `_execute_tool` 13 个工具分支搬进图节点（或复用：节点内调用 `agent` 方法）。
   - 引入 **web_fallback 回环**：本地检索无命中/实体未匹配 → 设 `state["_need_web_fallback"]` → 条件边路由到 web 搜索节点（对应 SoulTuner `route_after_search`）。
4. **finalize 节点接入完整 Answer Guard + goal**：`guard_answer` / `_collect_known_titles` 已可复用，evaluate 节点接 react 的 `_evaluate_progress` 诚实性检查。
5. **`builder.py` 图拓扑升级**：从直线改为带条件边（`add_conditional_edges`）的图，langgraph 可用时走真图、不可用时走 `_fallback_invoke`（保持零依赖）。
6. **`react_loop.py` 退役策略**：保留为 `agent.chat()` 的终极 fallback（graph 整体异常时），但不再是主路径默认。可保留其纯函数工具（guard_answer 等）供图复用，避免重复代码。

**验证**：现有 61 测试全绿；`agent.chat()` 主路径走 graph 且能力不退化。

---

## Phase 1：检索与精排升级（质量核心）

**目标**：用轻量等价实现复刻 SoulTuner 的 7 步漏斗 + 三锚精排 + MMR + Thompson，跑在你的 SQLite 候选池上。

1. **语义锚（sentence-transformers）**：
   - `pyproject.toml` 加 `sentence-transformers` 为**可选依赖**（extras `[semantic]`），无模型时降级到现有 TF cosine（保零依赖 demo）。
   - 扩展 `app/retrieval/embeddings.py`：候选/query 编码 → cosine → `(x+1)/2` 归一化到 [0,1]。
2. **三锚精排框架**（新增 `app/recommend/rerank.py`）：
   - `final = w_sem·semantic + w_personal·personalize + w_behavior·behavior`（你无 GPU 声学模型，用 BaRT 行为分替代 SoulTuner 的声学锚）。
   - 权重自动归一化 + 缺项重分配（学 SoulTuner 缺 OMAR 时的降级逻辑）。
   - 权重提到 `config.py` 可配置（`tri_anchor_w_*`）。
   - 每首歌产出 `RankingBreakdown.components`（透明度面板要用）。
3. **Graph Affinity 轻量版**（无 Neo4j）：用 4 维 Jaccard（genre 0.30 / mood 0.30 / scenario 0.25 / theme 0.15）算 personalize，偏好集合来自 taste_profile + 记忆。
4. **Thompson Sampling 探索槽**（`library.py` 加 `ts_alpha/ts_beta` 列）：
   - 粗排尾部用 `random.betavariate(α,β)` 采样捞冷门。
   - **曝光衰减**：推荐时 `ts_beta += 0.3`。
   - **反馈环（超越 SoulTuner）**：`record_listen` 听完/高分 → `ts_alpha += 1`；秒跳 → `ts_beta += 0.5`。
5. **MMR 多样性重排**（`rerank.py`）：λ=0.7，4 维标签 Jaccard 惩罚 `mmr = λ·rel − (1−λ)·max_overlap`。
6. **DISLIKES 过滤接入检索管线**：候选池过一遍 `library.is_disliked`（已实现，只需接线）。

**验证**：新增 `tests/test_rerank.py`（三锚归一化、MMR 去重、Thompson 衰减/反馈环各覆盖）。

---

## Phase 2：上下文管理 + 真流式（体验核心）

1. **同步版 GSSC**（新增 `app/context/gssc.py`）：
   - `estimate_tokens`（中英混合估算，照搬 SoulTuner 公式）。
   - 4 源优先级：用户输入(0) > 记忆(1) > 历史(2) > 检索(3)，按 min_tokens 保底 + 剩余预算按优先级分配 + 按行截断。
   - **Token 追踪报告**（before/after/saved 表格日志，面试展示用）。
   - 替换 react/graph 里的 `[:1500]` / `[:1800]` 硬截断。
2. **同步生成器流式**（`builder.py` 的 `stream()` 真正实现）：
   - 每个图节点 `yield` `StreamEvent`：`plan` → `tool_start` → `candidates`（先吐歌曲卡片）→ `tool_result` → `eval` → `final`。
   - 先推候选卡片 payload，再推理由文本（对齐 SoulTuner `__songs__` 先于 explanation）。
   - `api/main.py` 的 `/agent/stream` 已是 SSE，确认事件流贯通。

**验证**：`tests/test_gssc.py`（预算分配、优先级截断）；`tests/test_stream.py`（事件顺序：candidates 先于 final）。

---

## Phase 3：面试加分项（门面）

1. **Agent 透明度面板**（`streamlit_app.py`）：每条回复下可展开
   - 🧠 决策过程（agent_trace 的 Think/Act/Observe/Eval/Guard）
   - 📊 记忆变化（新增偏好、discovery_openness 变动、dislike）
   - 🧮 三锚打分明细（RankingBreakdown.components 表格）
2. **UI 美化**（参考 SoulTuner Spotify 深色主题 `theme.ts`）：
   - 主题：背景 `#000`/卡片 `#121212`/强调 `#1db954`，圆角卡片 + 阴影。
   - 歌曲卡片化（封面 + 标题 + 来源徽章 + 播放/不喜欢按钮）。
   - 思考中指示器（流式 thinking 状态）。
   - 左侧导航（推荐/搜索/歌单/旅程/品味）。
3. **README + 架构文档重写**：突出 4 大优势 + 新架构图（单一 LangGraph + 三锚精排 + GSSC + 透明度）。
4. **eval 扩展**（`tests/eval/cases.py`）：给意图路由、三锚精排、流式、web_fallback 加 LLM-as-judge case，量化升级前后效果。

---

## 修正旧研究文档的事实错误（顺带）

`docs/SOULTUNER_RESEARCH.md` 有 3 处与真实源码不符，一并更正：
1. 意图分类是 **V3 的 5+2**（不是 7 类），核心是"LLM 判意图 + 规则抽标签"的分工。
2. GSSC **已放弃同步 LLM 压缩**，缓存未命中直接按行截断，压缩只在轮后异步预算。
3. Thompson **反馈环 SoulTuner 自己也没接全**（只有曝光衰减）——这是我们的超越点。

---

## 执行顺序与依赖

```
Phase 0（编排收敛）──→ Phase 1（精排）──→ Phase 2（GSSC+流式）──→ Phase 3（面试加分）
     ↑ 地基,必须先做      ↑ 依赖统一编排      ↑ 依赖图节点产出事件     ↑ 依赖前三阶段产物
```

每个 Phase 结束跑全量测试，保持绿。新依赖（sentence-transformers / langgraph）一律可选，**零依赖 demo 能力贯穿始终不破坏**。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 收敛编排时打断现有 chat 行为 | react_loop 保留为终极 fallback；先让测试全绿再切主路径 |
| sentence-transformers 重依赖破坏零依赖 demo | 设为 optional extra，未安装自动降级 TF cosine |
| 同步流式在 Streamlit 卡顿 | 节点粒度 yield，逐事件刷新，不阻塞 |
| 三锚/MMR 改动影响推荐稳定性 | 新增 rerank 模块旁路，可配置开关，eval 量化对比 |
