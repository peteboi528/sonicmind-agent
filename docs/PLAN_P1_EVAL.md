# 下一轮计划：P1 — 推荐质量离线 eval

> 前置：P0（治 flaky）已完成，`pytest -q` 现在离线、确定、可重复。P1 是关键路径下一步。
> 目标：让每一次推荐相关改动（rerank 权重、hygiene 规则、画像接线、local 比例）都能**量化对比**，不再靠直觉。
> 一句话：把「推荐好不好」从主观判断变成可 diff 的数字。

---

## 为什么是 P1 而不是先拆对象（P2）

- P2 拆 `agent.py` 现在已经**安全**（有可信测试网），但「安全」只保证不崩，不保证**质量不退化**。没有 P1，拆完不知道推荐有没有变差。
- P1 是面试官复盘的第 2 大红线（"推荐质量零度量"），价值最高。
- P1 工作量更小（2–3 天）、风险更低，且产出的 baseline 是后续所有调参的参照系。

---

## 交付物（文件级）

```
eval/
├── __init__.py
├── golden.json          # 固定 query 集 + 期望（核心资产）
├── metrics.py           # precision@k / junk_rate / intent_accuracy / local_ratio / coverage / empty_rate
├── runner.py            # 跑 recommend_for_query → 算指标 → 出报告
└── baseline.json        # 入库的基线（改动对着它 diff）
```
外加：`pyproject.toml`/`Makefile` 里 `make eval` = `python -m eval.runner`。

---

## 1. golden.json —— 固定评测集（~30 条，可长）

每条形如：
```json
{
  "id": "scene-running",
  "query": "推荐几首适合跑步的歌",
  "category": "scene",
  "top_k": 5,
  "expect": {
    "intent": "recommend",
    "must_exclude_patterns": ["氛围", "男声", "钢琴版", "教程", "合集", "伴奏"],
    "min_count": 3,
    "local_ratio_max": 0.6
  }
}
```

**必覆盖的类别（用你真实踩过的坑当样本）：**

| 类别 | 示例 query | 关键期望 |
|---|---|---|
| 实体搜索 | "推荐 Taylor Swift 的歌" | intent=search/recommend；命中 Taylor Swift |
| 场景/情绪 | "适合跑步的歌" / "深夜一个人的" | intent=recommend；must_exclude 假歌 |
| 否定/排除 | "不要中文歌曲" | language=en；中文结果为 0 |
| 本地控制 | "推荐几首，不要 local" | local_ratio≈0 |
| 本地控制 | "减少 local 推荐几首" | local_ratio≤0.2 |
| 假歌陷阱 | （构造候选含「雨爱 - R&B氛围男声」）| must_exclude 命中→junk_rate=0 |
| 假歌陷阱 | （候选含「夜曲(钢琴曲)(原唱：周杰伦)」）| must_exclude 命中 |
| 续续指令 | 上轮"跑步"→本轮"再来几首" | 去重、延续场景 |
| 冷启动 | 空 memory 跑推荐 | empty_rate 低、不报错 |
| 语言混合 | "来点英文 R&B" | language=en；非空 |

> 假歌陷阱类要在 golden 里**注入**对应 mock 候选（复用 conftest 的 `fake_online_music_search` 机制），才能测 hygiene 是否挡住。

---

## 2. metrics.py —— 指标定义

| 指标 | 公式 | 含义 |
|---|---|---|
| `precision@k` | 命中 must_include 的结果数 / k（无 must_include 时记 N/A） | 相关性 |
| `junk_rate` | 命中 must_exclude 的结果数 / k | **越低越好**，直接量化 hygiene |
| `intent_accuracy` | 路由 intent == expect.intent 的 query 占比 | 路由正确性 |
| `local_ratio` | 结果中 local 占比 | 对照 expect.local_ratio_max |
| `coverage` | 全 golden 集结果里**不同艺人**数 / 总结果数 | 多样性代理 |
| `empty_rate` | 返回空结果的 query 占比 | 别空手 |

聚合：每条 query 一行 + 全集汇总（均值）。junk_rate 和 empty_rate 是**越低越好**，其余越高越好。

---

## 3. runner.py —— 跑法

- 复用 **P0 的确定性基建**：mock 掉外部源（`fake_online_music_search` 模式）、`random.seed` 固定 → eval 结果**逐次可重复**。
- 对每条 golden query：构造独立 `AudioVisualAgent(JsonStore(tmp))`（顺便享受 P0 的资源库隔离修复）→ 调 `recommend_for_query` / recommend 工具 → 收 top_k → 算指标。
- 输出：
  - stdout：人类可读报告（每类别的指标 + 全集汇总）。
  - `eval/baseline.json`：机器可 diff 的结构化结果。
- `--diff baseline.json`：对比上次基线，打印每个指标的 Δ，**任一「越高越好」指标退化超阈值或 junk_rate 上升 → 退出码非 0**（供 CI 门禁）。

---

## 4. baseline + CI

1. 跑一次产 `eval/baseline.json` 入库。
2. CI：`make eval --diff` 先**非门禁**（贴 Δ 评论），跑稳后转门禁。
3. 之后任何动 rerank/hygiene/profile/local 的 PR，eval 报告里能看到精确数字变化（例：「avoid 减分 -0.12→-0.08：precision@k 0.64→0.61，junk_rate 0.03→0.05」）。

---

## 步骤拆解（工时）

| 步骤 | 内容 | 工时 |
|---|---|---|
| 1 | `eval/` 骨架 + runner 调通 `recommend_for_query`（mock 源、seed） | 0.5 天 |
| 2 | golden.json v1（~20 条，覆盖 6 类） | 1 天（最慢，要推敲期望） |
| 3 | metrics.py 六个指标 + 报告输出 | 0.5 天 |
| 4 | baseline.json + `--diff` + 退出码 | 0.5 天 |
| 5 | （可选）CI 接线、非门禁报告 | 0.5 天 |

**合计 ~2.5–3 天。** 步骤 2 是瓶颈，可以先上 10 条跑通管线再扩。

---

## 验收（Definition of Done）

- [ ] `python -m eval.runner` 离线、确定，连续两次输出**逐字节一致**。
- [ ] 至少有 `precision@k` + `junk_rate` + `intent_accuracy` 三个指标。
- [ ] `eval/baseline.json` 入库；`--diff` 能输出每个指标的 Δ。
- [ ] 能回答：「把 `local_ratio` 默认从 0.4 改 0.3，`coverage` 和 `local_ratio` 指标怎么变？」
- [ ] 假歌陷阱类 query 的 `junk_rate = 0`（验证 hygiene 真挡住）。

---

## 风险与边界

- **golden 集主观**：先小（10–20 条）跑通，再凭真实日志扩。期望写错会比没指标更误导——所以 baseline 要人工 sanity check 一遍。
- **mock 源 ≠ 真实网易云质量**：eval 测的是「给定固定输入，你的逻辑产出什么」，不是「网易云返的歌好不好」。后者属于 P1 之外的线上观测（埋点/反馈），先不做。
- **冷启动 query**：用空 `UserMemory` 构造，验证不崩 + empty_rate 合理。
- eval 不替代单测：它是**质量回归门禁**，单测仍管逻辑正确性。

---

## 之后

P1 baseline 立住后，**P2（拆 agent.py）** 就能放心动——每次抽 service 跑一次 eval，质量退化立刻可见。P3（LLM provider 抽象）、P5（配置归一 + ablation）可在 P1 之后插队；P4（reflect 加深 + 延迟预算）必须等 P1（否则没法度量 reflect 改进效果）。

---

## 建议起步

先做**步骤 1+3 的最小闭环**：`eval/runner.py` 跑通一条 query、算出一个 precision@k、打印出来。管线通了，再回头填 golden 集——避免一上来就陷入「写 30 条期望」的体力活却不知道管线对不对。
