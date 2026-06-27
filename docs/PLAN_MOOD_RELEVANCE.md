# 计划：场景-情绪相关性（治「深夜推下午」）

> 起因：用户问「适合深夜的歌」，返回列表里既有假歌（轻音乐钢琴曲），也有真 R&B（Sober / If You Let Me），
> 但**整张单子是「下午轻松」vibe，不是「深夜」vibe**。hygiene 能杀假歌，但治不了 vibe 跑偏——
> 这是推荐**相关性**的核心问题，不是 bug，没有正则能修。本计划分三层把它从「拍脑袋」变成「可度量、可治本」。

---

## 问题定性

- **现象**：场景 query（深夜/下午/跑步/学习…）的推荐结果 vibe 与场景不符。
- **根因**：
  1. 网易云**歌曲无情绪/能量特征**（无 valence/energy），系统分不清「深夜 R&B」和「下午 R&B」——标签层都叫「放松/慵懒」。
  2. 网易云「深夜」歌单本身偏水，多是「放松/慵懒/chill」杂烩，下午也能听；Route B 歌单搜索召回的就是这种偏轻合辑。
  3. 推荐路径**没有把候选情绪对齐到场景目标情绪**——`_is_playlist_context_compatible` 只做粗 genre/mood 兼容，不区分时段。
- **结论**：得给场景建「目标情绪画像」，给候选推断情绪，再按契合度过滤/精排。hygiene 是另一条线（杀假歌），别混。

---

## M0 — 治标：标题时段反匹配（~0.5 天）

**目标**：深夜 query 不返回标题明写白天时段的歌（*Sunny Afternoon*、*慵懒的午后*）。**注意：只挡得住「标题露馅」的，挡不住 vibe 偏轻的真歌（Sober）**——这点必须诚实，M0 不是终点。

**动作（文件级）**：
- `app/agent.py` 的 `_is_playlist_context_compatible(goal, track)`（或新增 `_is_scenario_time_compatible`）：
  - 从 goal 抽场景（复用 `app/graph/tag_rules.py::extract_scenario`）。
  - 场景→反时段词表：`深夜/夜晚` ↔ `午后/下午/午间/白天/早晨/清晨/早上`；`早晨/清晨` ↔ `深夜/夜晚`；`下午/午后` ↔ `深夜/早晨`。
  - 候选标题命中「当前场景的反时段词」→ 不兼容（剔除/降权）。
- 接到 Route B/C 的 verified 过滤链（和 `_is_recommendation_quality_track` 同级）。

**验收**：
- 「推荐几首适合深夜的歌」结果里**不含**标题带 午后/下午/早晨 的歌。
- 「适合早晨的歌」结果里不含 深夜/夜晚 标题。
- 单测：`tests/test_scenario_time_compat.py`（深夜 query + 午后标题 → 剔除；深夜 query + 无时段标题 → 保留）。

**风险**：词表别过激——只反「明确反义时段」，不反中性词。误伤风险低（标题明写时段的歌本就少）。

---

## M1 — 度量：做成 P1 golden 的第一条场景相关性退化项（~0.5 天，挂在 P1 上）

**目标**：把「深夜≠下午」变成一个**可度量指标**，以后每次改动都能看见有没有退化。这是治本的前提。

**动作**：
- 在 `eval/golden.json`（P1 产物）加场景相关性子集：
  - `{"query":"推荐几首适合深夜的歌","scene":"深夜","must_exclude_title_patterns":["午后","下午","早晨","清晨","白天"],"must_exclude_vibe":"明快/高能"}`
  - 同理加 下午/早晨/跑步/学习 各一两条。
- 在 `eval/metrics.py` 加 `scene_relevance_rate`：结果中**命中 must_exclude_title_patterns** 的占比（越低越好）+ （可选）人工 vibe 标注命中率。
- vibe 标注先人工（10-20 条结果），等 M2 情绪建模成熟再自动化。

**验收**：
- P1 报告里能看到 `scene_relevance_rate`；改 M0 规则前后该指标有 Δ。
- 深夜 query 的 title-pattern 命中率 = 0（M0 兜住标题层）。

**注意**：M1 只能量化「标题层」和「人工 vibe 层」；真正自动化 vibe 判别要等 M2。

---

## M2 — 治本：场景目标情绪画像 + 候选情绪对齐（~3–5 天）

**目标**：让系统真的懂「深夜 = 低 valence / 中低 arousal / 内省 / 慢」，按这个画像给候选打分，把「下午 vibe 的 R&B」从深夜推荐里压下去。

**前提**：候选有情绪标签。现状 `_ensure_track_tags` / `extract_mood`（`app/graph/tag_rules.py`）**已经能从标题+歌手推断 mood**——所以信号有，只是推荐路径没用它对齐场景。

**动作（文件级）**：
1. **场景→目标情绪画像**（`app/recommend/scene_profile.py` 新建，或扩 `app/profile/`）：
   - 固定查表（确定性、可解释、防幻觉，对齐 profile builder 风格）：
     - 深夜 → 低 valence（-0.6~-0.2）、低 arousal（-0.5~-0.1）、偏好 内省/迷幻/氛围/慢板。
     - 下午 → 中 valence（0~0.4）、低 arousal、偏好 轻松/慵懒/明亮。
     - 跑步 → 高 arousal（0.4~0.8）、偏好 律动/电子/快板。
   - 与 `app/profile/models.py::MoodPoint`（已有 valence/arousal）对齐坐标系。
2. **候选情绪对齐打分**（接 `_rerank_tracks`）：
   - 候选推断 mood（`extract_mood`）→ 映射到 valence/arousal（查表）→ 算与场景画像的距离（如 1 - 归一化欧氏距离）→ 作为 rerank 的一个**场景锚**（与现有三锚并列，可选权重）。
   - vibe 严重跑偏（距离超阈值）→ 直接降权（不一定要硬剔，避免过激）。
3. **歌单源收紧**（可选，Route B）：
   - 深夜 query 的歌单搜索词别只用「深夜」（太水），加更具体的「深夜 伤感 / 迷幻 / 氛围」变体，减少「下午 chill」杂烩召回。

**验收**：
- M1 的 `scene_relevance_rate`（含 vibe）显著改善（人工标注 baseline → 目标值）。
- 单测：场景画像查表确定；候选 mood→valence/arousal 映射可单测；场景锚打分能区分「深夜向 R&B」vs「下午向 R&B」（构造两条 mock 候选，深夜画像下前者分高）。
- 不回归：跑 P1 全量 golden，precision@k 不掉。

**风险**：
- `extract_mood` 关键词推断本身有噪声（标题没线索的歌推不出 mood）→ 场景锚对这类候选无信号，靠其它锚兜底，别让场景锚权重过大盖过语义/口味锚。
- valence/arousal 查表主观 → 先小规模（深夜/下午/早晨/跑步/学习 5 个场景），跑稳再扩。
- 终极瓶颈：网易云无音频特征，标题推断是天花板。真要质的飞跃得接音频特征或 LLM 曲目情绪推断（成本/复杂度上一档），M2 先把「标题推断 + 场景画像」这条确定性链路榨干。

---

## 顺序与依赖

```
M0（治标，0.5d）── 立刻止血，深夜不再出 Sunny Afternoon
   └→ M1（度量，0.5d，挂 P1）── 能看见退化
        └→ M2（治本，3–5d）── 真正懂场景情绪
```
- M0/M1 可立刻做，不依赖 P1 落地（M1 的 golden 结构可以先写，等 P1 runner 接管）。
- M2 依赖 M1（要度量才知道画像调对了），且建议在 **P2 拆出 RecommendationService 之后**做（场景锚属于推荐内部逻辑，拆完更好加）。

---

## Definition of Done

- [ ] M0：深夜/早晨 query 不返回标题明写反时段的歌（单测 + 线上肉眼验）。
- [ ] M1：`scene_relevance_rate` 进 P1 报告，有 baseline。
- [ ] M2：场景画像查表 + 候选情绪对齐打分上线；深夜推荐的 vibe 相关性（人工标）显著优于 baseline；P1 全量 precision@k 不退化。
- [ ] 诚实文档：在用户能看见的地方（如推荐解释）说明「场景情绪匹配是启发式，网易云无音频特征是天花板」。

---

## 建议起步

**先做 M0 + M1**（合计 1 天）：M0 让线上立刻不再出 *Sunny Afternoon* 这种笑话；M1 把「深夜 vibe 跑偏」变成数字，给 M2 兜底。M2 等 P1/P2 就位再动——没有度量就调画像，等于又回到拍脑袋。
