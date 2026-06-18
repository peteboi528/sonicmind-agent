# 推荐引擎范式重构设计文档（Recommendation Revolution）

> 状态：设计稿 · 待评审
> 范围：(A) 检索 + 排序范式从「词法召回 + 确定性利用」换到「语义召回 + 不确定性探索」；(B) Agent Graph 编排从「串行 + 整轮易崩」升级到「并发 + 单点降级」
> 底线：现有 455 测试零回归；每根支柱 / 每块编排升级独立交付、独立测试、独立提交

---

## 1. 背景与诊断

用户反馈两个体感问题，根因同源：

1. **推荐没有真正的 explore** —— 每轮来回那几首，没有新鲜感/发现感。
2. **打错字就不匹配** —— "Emenem" 搜不到 Eminem。

扒到底后确认：这不是 bug，是**范式问题**。当前管线表面叫「三锚精排 + 语义锚 + MMR」，
实际运行是一条**纯词法、纯利用（exploit）**的流水线，披着语义和智能的外衣。

### 三个硬事实（file:line 实证）

| 问题 | 证据 | 后果 |
|------|------|------|
| Explore 是死的 | `library.sample_ts_scores()` 定义了**全项目从不调用**；`ts_beta+=0.3` 衰减、`update_ts_feedback` 反馈环都在跑，但没人采样 beta 先验 | 永远推「最稳妥、相关性最高」那几首 |
| 匹配是纯子串 | `agent.py:2767` `_query_matches_track` 核心是 `e in searchable`，无任何编辑距离 | 打错字 → 网易云即使召回对的歌也被这层过滤筛掉 |
| 语义锚不语义 | `enable_embeddings="auto"` 但运行环境未装 → `semantic_scores` 返回 None → 回退 TF 词项 Jaccard + 手写同义词表 | 跨不了「说唱→rap」，处理不了错字，召回侧完全无语义 |

**一句话**：系统是「词法召回 + 确定性利用」，用户想要的是「语义召回 + 不确定性探索」。这是范式差，补丁补不出来。

---

## 2. 目标与非目标

### 目标
- 推荐结果每轮有新鲜度：在「相关」与「新发现」之间可调平衡，由不确定性驱动而非随机。
- 检索对错字/近义/跨语言鲁棒：用户拼错、用中文描述英文曲风，都能召回。
- 召回侧引入语义：候选池从「单关键词 top-N」升级到「多视角语义并集」。
- 最大化复用已建好但没接线的设施（Thompson Sampling、embedding 后端、归一化函数）。

### 非目标（本次不做）
- 不引入外部向量数据库服务（继续用进程内 `HybridRetriever`）。
- 不重写 LLM 意图规划器（query_plan 仅增量加多查询字段）。
- 不动 bot 适配器 / Web 前端 / 鉴权。
- 不在本设计内做 agent.py 拆分（那是独立的可维护性轨道）。

---

## 3. 现状架构（As-Is）

```
用户 query
  └─> query_plan(LLM) ── 产出单个 search_query + entities + intent
        └─> recommend_for_query / search / search_web_music
              └─> 网易云关键词 API 单查询 → top-N
                    └─> _query_matches_track 子串过滤  ← 错字在此被筛掉
                          └─> _rerank_tracks
                                └─> tri_anchor_rerank(语义TF/口味/行为/CF)  ← 确定性，无探索
                                      └─> mmr_rerank  ← 仅批内多样性
                                            └─> top_k（永远最稳妥那批）
```

死设施：`sample_ts_scores`（探索）、dense embedding（语义召回）、`exposure_count` 读取（去重）。

---

## 4. 目标架构（To-Be）：三根支柱

```
用户 query
  └─> query_plan(LLM) ── 产出 search_query + entities + **search_variants[]**  ← 支柱三
        └─> 多查询并发召回（原词 + 语义变体 + 纠错词）  ← 支柱二/三
              └─> 鲁棒匹配过滤（归一化 + 编辑距离模糊）  ← 支柱二
                    └─> 候选池并集去重
                          └─> tri_anchor_rerank + **uncertainty 锚(TS采样)**  ← 支柱一
                                └─> **explore/exploit 分槽选择**  ← 支柱一
                                      └─> mmr_rerank（保留）
                                            └─> top_k（相关 + 新发现混合）
```

三根支柱相互独立，可分批落地。下面逐根给精确设计。

---

## 5. 支柱一 · Explore 赌博机（接活死掉的探索）

### 问题
`tri_anchor_rerank`（rerank.py:267）纯利用：分数全来自「已知信号」（语义/口味/行为/CF）。
对系统**没见过或不确定**的候选，没有任何机制让它有机会冒头。`sample_ts_scores` 建好没人调。

### 设计

**5.1 接线 TS 不确定性锚（rerank.py）**
- `tri_anchor_rerank` 新增可选入参 `ts_scores: dict[str,float] | None`（候选 key → Beta 采样值 [0,1]）。
- 作为第五锚加进 `base`，权重 `tri_anchor_w_explore`（新配置，默认 0.15）。
- TS 锚不可用（mock/无库记录）时走 `_normalized_weights` 缺项重分配，行为与现状一致。
- 记进 `RankingBreakdown.components["explore"]`，打分透明。

**5.2 explore/exploit 分槽（rerank.py 新增 `bandit_select`）**
- top_k 槽位按 `settings.explore_ratio`（默认 0.3）切分：70% exploit 槽给最高 `final_score`，
  30% explore 槽给「高 TS 不确定性 + 低 exposure」的尾部候选捞回。
- 与 MMR 协作：exploit 槽内部仍跑 MMR 多样性；explore 槽按 TS 降序取。

**5.3 接线（agent.py:_rerank_tracks 2288）**
- 调 `rerank_candidates` 前：`ts = self.library.sample_ts_scores(tracks)` 传入。
- `record_exposure` / `decay_exposure_ts` 保持精排后调用（已有），反馈环闭合。
- **吸收原 TODO 轨道一**：曝光去重不再单独做加性惩罚——TS 的 `ts_beta+=0.3` 衰减
  已天然实现「推得越多、不确定性越低、explore 槽越不容易再选中」，去重融进探索。

**5.4 在线学习闭合（已大部分就绪）**
- Taste Lab + PlayerBar 的 listen 上报 → `update_ts_feedback(positive=听完/liked)`。
- 确认这条已接（见 [[three-anchor-revival-2026-06]]），补 explore 槽命中后的反馈记录。

### 配置项（config.py 新增）
```python
self.enable_explore: bool   = env("ENABLE_EXPLORE", "true")
self.explore_ratio: float   = env("EXPLORE_RATIO", "0.3")      # explore 槽占比
self.tri_anchor_w_explore: float = env("TRI_ANCHOR_W_EXPLORE", "0.15")
```

### 测试（tests/test_rerank.py / test_behavior_scoring.py）
- `test_ts_anchor_demotes_overexposed`：同候选 ts_beta 高（推过多次）→ explore 槽不再选中。
- `test_explore_slot_surfaces_novel`：构造高相关老歌 + 低相关新歌，验证 explore 槽捞回新歌。
- `test_bandit_select_ratio`：top_k=10、explore_ratio=0.3 → 恰 3 个 explore 槽。
- `test_explore_disabled_equals_legacy`：`enable_explore=false` 时排序与现状逐位一致（零回归保险）。

---

## 6. 支柱二 · 鲁棒匹配（打错字也能命中）

### 问题
`_query_matches_track`（agent.py:2723）用 `e in searchable` 纯子串。错字、变体、大小写边界全挂。
讽刺：`netease.py` 已有 `_normalize_music_name()` 归一化函数，只用于专辑校验，搜索过滤不用。

### 设计

**6.1 引入 rapidfuzz（轻量、纯 C、无 torch 级体积）**
- pyproject 新增 `rapidfuzz>=3.0`。纯算法库，冷启动无成本。

**6.2 升级 `_query_matches_track`**
- 先对 query token 和 `title+artist` 都过 `_normalize_music_name`（提到公共模块复用）。
- entity token 命中判定从 `e in searchable` 升级为：
  子串命中 **或** `rapidfuzz.fuzz.partial_ratio(e, searchable) >= settings.fuzzy_threshold`（默认 82）。
- 泛化 token 保持严格全命中（防 API 垃圾），但也走归一化后比较。

**6.3 搜索词进网易云前纠错（可选增强）**
- 在 query_plan 阶段，让 LLM 在 `search_query` 里顺手纠正明显拼写错误（prompt 已在改写 query，零额外调用）。
- 兜底：纯算法层不做西文拼写词典纠错（成本高），靠 6.2 的模糊匹配 + 6.4 多查询覆盖。

**6.4 与支柱三协同**
- 错字最终保险是多查询召回：原词 + LLM 纠错变体并发搜，任一命中即可。

### 配置项
```python
self.fuzzy_threshold: int = env("FUZZY_THRESHOLD", "82")  # rapidfuzz partial_ratio 阈值
```

### 测试（tests/test_search_modules.py / test_candidate_quality.py）
- `test_typo_artist_matches`："Emenem" 对 title="Without Me" artist="Eminem" → 放行。
- `test_fuzzy_threshold_rejects_unrelated`：阈值下不会把无关歌放进来（防过度宽松）。
- `test_normalized_match_punctuation`："R&B" vs "RnB"、全半角差异归一化后命中。
- `test_exact_match_still_works`：原有精确匹配用例零回归。

---

## 7. 支柱三 · 语义召回（不只是语义排序）

### 问题
召回侧只有网易云**单关键词查询**。语义全压在排序阶段的语义锚上，而那个锚还退化成 TF。
候选池天花板 = 单查询能搜到的，语义广度从源头就被卡死。

### 设计

**7.1 启用真 dense embedding（用户已确认装）**
- 确保运行环境 `pip install sentence-transformers`（pyproject 已声明 `>=2.2.0`）。
- `embeddings.embeddings_available()` 返回 True 后：
  - `semantic_scores`（rerank 语义锚）自动走 dense 余弦——锚立刻变「真语义」。
  - `HybridRetriever`（本地库检索）dense 分支自动激活（vector_store.py:38 已写好）。
- 模型单例懒加载已实现（embeddings.py:24），首次加载成本一次性。
- 默认模型 `paraphrase-multilingual-MiniLM-L12-v2`（多语言，支持中英跨语言）。

**7.2 多查询语义召回（核心新增）**
- query_plan 输出新增字段 `search_variants: list[str]`（2-4 个语义变体）。
  - 例："深夜慵懒爵士" → ["late night jazz", "smooth jazz chill", "慵懒爵士 深夜", "lounge jazz"]
  - 跨语言 + 近义 + 相关曲风。LLM 已在做 query 改写，扩成数组成本极低。
- `search_web_music` / `recommend_for_query` 用 `run_parallel`（concurrency.py 已有）并发搜
  原词 + 各变体，结果并集去重 → 候选池广度成倍提升。
- 限流保护：变体数封顶 4，并发复用现有网易云多端点轮询 + 重试（见 [[netease-search-rate-limit]]）。

**7.3 dense 召回兜底（二期）**
- 对入库的 ExternalTrack 预编码向量，query 端 embedding 做向量检索补充关键词召回的盲区。
- 一期先靠多查询拿语义广度，dense 召回作为二期增强（避免一次改太多）。

### 配置项
```python
self.enable_embeddings = "true"   # 从 auto 显式打开（确认环境已装）
self.max_search_variants: int = env("MAX_SEARCH_VARIANTS", "4")
```

### 测试（tests/test_query_rewrite.py / test_embeddings.py / test_recommended_tracks.py）
- `test_search_variants_parsed`：query_plan 输出 variants 被正确解析进 plan。
- `test_multi_query_union_dedupe`：多变体召回结果并集去重正确，无重复无丢失。
- `test_embeddings_available_uses_dense`：装了模型后 `semantic_scores` 走 dense 而非 TF。
- `test_cross_lingual_recall`：中文意图召回到英文曲目（多语言模型验证）。
- `test_variants_capped`：变体数超限被截断到 max_search_variants。

---

# Part B · Agent Graph 编排升级

> 与 Part A（三根支柱）正交：A 改「推荐质量」，B 改「编排效率与稳定性」。可并行推进。

## 8. 编排现状（As-Is）

```
load_context → plan_intent → execute_tools → [web_fallback?] → evaluate → reflect → [refine↺execute_tools | finalize] → END
```

双份实现且优雅降级（有 langgraph 走 `StateGraph` builder.py:217；无则 `_fallback_invoke` builder.py:195 手写复刻同样的边）。
但编排本身有四块明确优化空间，按收益排序如下（8.1 收益最高）。

### 8.1 工具 / 子任务并发（收益最高）
- **现状**：`execute_tools`(nodes.py:234) 是 `for tool in plan.tools_needed: _run_tool(...)` 纯串行；
  `invoke_compound`(builder.py:57) 子任务逐个 `for` 循环跑。
- **问题**：两处都在白等网络 IO。`concurrency.py` 已有 `run_parallel`（`search_videos` 在用），编排层没用。
- **改法**：`execute_tools` 把互不依赖的工具用 `run_parallel` 并发（有依赖的留串行，如 playlist 需先
  `_collect_tracks(results)` 拿 seed）；`invoke_compound` 按 `depends_on_prev` 分组，False 批并发、True 串行等前置。
- **保序**：流式卡片按工具固定顺序合并结果，并发不打乱 candidates 事件顺序。

### 8.2 节点容错隔离（治 chat 稳定性）
- **现状**：只有 `stream` 路径有整体 try/except(builder.py:150)；`compiled.invoke` / `_fallback_invoke` 无节点级隔离。
- **问题**：任一工具抛异常 → 整轮崩。这是「chat bug 层出不穷」的来源之一（某源超时本该降级，结果整轮挂）。
- **改法**：`_run_tool` 包一层单工具失败降级（记 trace + 发降级事件 + 跳过该工具，不冒泡）；
  关键节点(execute/reflect)加 try/except 走安全默认值，保证 finalize 永远能出回复。

### 8.3 懒加载上下文（减浪费）
- **现状**：`load_context`(nodes.py:54) 无脑做满（memory + `recall_episodes` 跨会语义召回 + GSSC 预算），
  chat 也照跑全套；它在 `plan_intent` 之前，拿不到意图。`reflect`(nodes.py:512) 对每个会列曲意图都发一次 LLM 约束核对。
- **问题**：闲聊不需要语义召回却照跑；reflect 在无用户约束时也进节点空转。
- **改法**：`load_context` 内部独立 IO（memory/goal/recall/resource_count）用 `run_parallel` 并发；
  昂贵的 `recall_episodes` 延迟到知道意图后按需触发（chat 跳过）；`reflect` 在 `_gather_constraints` 为空时直接跳过 LLM 往返。

### 8.4 回环 / 空节点清理（结构简化）
- **现状**：`route_after_reflect→refine` 回到 `execute_tools`(builder.py:238) 重跑 `plan.tools_needed` 全部工具；
  `evaluate`(nodes.py:500) 只算候选数文案、发个事件，夹在 execute 和 reflect 间多一跳。
- **问题**：refine 回环重复劳动（已成功工具又跑一遍）；evaluate 是近乎空的节点徒增一跳。
- **改法**：refine 只补量缺口（记下已成功工具，回环只跑能补量的，如翻页再搜一批，不重跑全部）；
  `evaluate` 逻辑折进 `reflect` 头部，去掉独立节点。

## 9. 编排目标架构（To-Be）

```
load_context(并发IO) → plan_intent → execute_tools(独立工具并发 + 单点降级)
  → [web_fallback?] → reflect(吸收evaluate + 无约束跳过LLM) → [refine↺只补量 | finalize] → END
```

`_fallback_invoke` 与 langgraph 两条路径同步改造，保持等价。建议先 B1（容错，纯增益）再 B2（并发）。

## 10. 编排测试（tests/test_graph_flow.py / test_agent_flow.py / test_compound.py）
- `test_independent_tools_run_parallel`：多工具 plan 并发执行，结果与串行逐位一致（仅时序变）。
- `test_compound_independent_subtasks_parallel`：`depends_on_prev=False` 子任务并发；True 的仍串行等前置。
- `test_tool_failure_isolated`：单工具抛异常 → 该工具跳过、其余照常、finalize 出降级回复（不整轮崩）。
- `test_chat_skips_recall`：chat 意图不触发 `recall_episodes`。
- `test_reflect_skipped_without_constraints`：无用户约束时 reflect 不发 LLM 调用。
- `test_refine_only_backfills`：refine 回环不重跑已成功工具。
- `test_orchestration_disabled_equals_legacy`：并发/精简全关时，节点序列与输出逐位等于现状（零回归保险）。

---

# Part C · 迁移、风险与落地

## 11. 风险与权衡

- **explore 引入「看起来不那么相关」的歌**：这是设计意图（发现感），但需平衡。`explore_ratio`
  默认 0.3 偏保守，可按用户反馈（explore 槽命中率）动态调；命中差则调低。
- **模糊匹配过宽**：`fuzzy_threshold=82` 偏严，宁可漏召回也不放垃圾进来；配合多查询补漏。
- **embedding 体积/冷启动**：模型约 100–400MB，首次加载几秒。单例缓存后无重复成本；
  部署需确认磁盘/内存。不接受则退回多查询（语义广度打折但可用）。
- **多查询放大限流**：变体封顶 4 + 复用多端点轮询；监控 `netease-fallback` 占比，升高则降变体数。
- **与去重的关系**：原 TODO 轨道一（曝光加性惩罚）被支柱一 TS 吸收，不再单独实现，避免双重惩罚。
- **编排并发打乱卡片顺序**：流式 candidates 事件按工具固定顺序合并，并发不影响展示次序。
- **编排时序变化影响测试**：每块编排升级带 `disabled 时逐位等于现状` 的保险测试兜底。

## 12. 迁移顺序与回滚

Part A（推荐质量）与 Part B（编排）正交，可并行。各阶段独立开关、可灰度回滚：

| 阶段 | 内容 | 风险 | 回滚 |
|------|------|------|------|
| P1 | 支柱二（鲁棒匹配） | 低（纯算法，阈值可调） | 降 `fuzzy_threshold` 或开关 |
| P2 | 支柱一（Explore 赌博机） | 中（改排序输出） | `enable_explore=false` 逐位回现状 |
| P3 | 支柱三 7.1+7.2（装 embedding + 多查询） | 中（依赖体积 + 召回变宽） | `enable_embeddings=false` / `max_search_variants=1` |
| P4 | 支柱三 7.3（dense 召回兜底） | 中高 | 单独评估，独立提交 |
| B1 | 编排 8.2 节点容错隔离 | 低（只加保护，不改逻辑） | 去掉 try/except 包装 |
| B2 | 编排 8.1 工具/子任务并发 | 中（时序变化） | `enable_parallel_tools=false` 回串行 |
| B3 | 编排 8.3+8.4 懒加载 + 回环精简 | 中 | 各自开关回退 |

**全程硬门槛**：每个 PR 跑全量 455 测试零回归；每块带「disabled 时逐位等于现状」的保险测试。

## 13. 对现有 TODO / 其他轨道的影响

- 轨道一（曝光去重 4 项）→ **并入支柱一**，TS 锚天然处理跨轮去重，删除独立的加性惩罚实现。
- 轨道二（可播性门）→ **保留不变**，与本重构正交（虚拟歌过滤是数据质量，不是范式）。
- 轨道三（意图收口 / 拆 agent.py）→ **保留不变**，可维护性轨道，独立推进。

## 14. 落地检查清单（实施进度）

**Part A · 推荐质量**
- [x] P1 支柱二：rapidfuzz 依赖 + `_query_matches_track` 模糊化 + 回归测试
- [x] P2 支柱一：TS 锚接线 + `bandit_select` 分槽 + config + 回归测试
- [x] P3a 支柱三一期：`search_variants` 字段 + 多查询并发召回 + cap/路由回归测试
- [x] P3b 支柱三 7.1：确认 embeddings extra，并补 dense 模式冒烟测试
- [x] P4 支柱三二期：ResourceLibrary dense 召回兜底（独立评估）

**Part B · 编排**
- [x] B1a 单工具容错隔离（`_run_tool` 降级，execute/web_fallback 不整轮崩）+ 回归测试
- [x] B1b 关键节点级 try/except（evaluate/reflect/finalize 安全默认值）+ 回归测试
- [x] B2 工具/子任务并发（execute_tools + invoke_compound）+ 回归测试
- [x] B3 懒加载上下文 + evaluate 空节点折叠 + 回归测试

- [x] 每阶段：全量测试零回归 + disabled/降级路径回归保险


