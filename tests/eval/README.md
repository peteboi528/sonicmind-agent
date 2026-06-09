# LLM-as-judge Eval

固定一组真实用户场景，用强 LLM 作为评委对 Agent 的回复打分。

## 为什么需要

`pytest` 测的是 **代码不报错**，eval 测的是 **agent 答得好**。
每次改 prompt / 模型 / ReAct 逻辑前后跑一遍，能客观看到质量变化。

## 使用方法

需要真实 LLM_API_KEY（mock 模式跑没意义）。

```bash
# 默认：跑全部 case
python -m tests.eval.run

# 只跑某一个
python -m tests.eval.run --case recommend_basic

# 用更强的模型当评委
JUDGE_MODEL=gpt-4o python -m tests.eval.run
```

## Case 集（10 个）

| case_id | 覆盖能力 |
|---|---|
| recommend_basic | 新用户冷启动推荐 |
| recommend_with_taste | 基于品味档案的推荐 |
| playlist_specific | 歌单生成 |
| multi_turn_context | 多轮上下文指代 |
| search_specific | 具体歌曲搜索 |
| taste_query | 品味分析 |
| **anti_hallucination** | 反幻觉：虚构查询不得编造歌名 |
| **scenario_diversity** | MMR 多样性：场景推荐不堆砌同质曲目 |
| **journey_multi_phase** | 多阶段音乐旅程编排 |
| **multi_intent_goal** | 复合目标跟踪（导入→推荐） |

> 加粗的是升级后新增 case，针对反幻觉、三锚精排+MMR、目标跟踪等新能力。

## 结构化冒烟测试（无需 LLM key）

`tests/test_eval_cases.py` 把全部 case 在 mock 模式跑一遍，验证不崩、答案非空、
`must_not_mention` 禁词不泄漏、反幻觉 case 输出诚实。这部分进 CI 防退化；
完整主观打分仍走 `python -m tests.eval.run`。

## 输出示例

```
[1/10] recommend_basic: 新用户首次请求推荐
  query: '给我推荐几首适合工作时听的歌'
  answer (3.2s, 4 steps):
    根据你目前的情况，我推荐这几首适合专注工作的轻音乐...
  ✅ overall=4.20  mention_hit=1.0  violations=0
  judge: 给出具体歌曲且匹配场景，理由清晰

============================================================
汇总:
  平均分: 3.85/5.0
  通过率: 8/10
```

## 评分逻辑

- 每个 case 定义若干维度（criteria），由 judge LLM 0-5 打分
- `must_mention` 关键词漏 → overall × 命中率
- `must_not_mention` 违规 → overall 上限 1.5
- `overall >= 3.0 && 无违规` 才算 passed

## 添加新 case

编辑 `tests/eval/cases.py`，追加 `EvalCase(...)`。建议：
- 评分维度 3-4 个，描述具体
- `must_mention` 只放绝对必要的关键词（不要太苛刻）
- `must_not_mention` 用来抓兜底回复（如 "LLM 请求失败"）

## CI 集成（未来）

把通过率当作 KPI：每次 PR 合并前跑一次，阈值 < 80% 阻断合并。
