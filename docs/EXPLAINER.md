# SONICMIND Agent 原理讲解

这份文档描述当前真实实现：一个 Vue + FastAPI 的音乐推荐 Agent。它不是只靠一次 LLM 回复的聊天壳，而是把意图规划、工具执行、反幻觉、记忆和播放体验串成一条可回归的产品链路。

## 1. 系统定位

SonicMind 面向“我现在想听什么”的连续音乐场景工作：

```text
用户输入
  -> Graph 编排：load_context -> plan_intent -> execute_tools -> evaluate/reflect -> finalize
  -> Answer Guard：只保留可追溯歌名/专辑名
  -> SSE：plan / tool_start / candidates / album_card / eval / final
  -> Vue 前端：聊天、探索工作台、播放器、曲库、歌单、偏好
```

真实能力和降级能力分层：

- 真实源：网易云歌曲/专辑/歌单、B站/YouTube 视频、Tavily/DuckDuckGo 歌手信息、Last.fm 标签发现。
- 降级源：MockLLM、MockSource、metadata tag enrichment，保证无 key 也能跑通。
- 占位边界：当前不声称已完成 Whisper/CLIP 级别的真实音视频理解；未识别字段保持空值或未分类，不随机伪造。

## 2. Agent 编排

主路径是 `app/graph/`：

- `load_context` 读取记忆、历史、资源库，并通过 GSSC 控制上下文预算。
- `plan_intent` 让 LLM 或 MockLLM 输出结构化 `AgentPlan`，关键词 registry 做兜底和安全升级。
- `execute_tools` 按 intent 调用搜索、推荐、歌单、专辑、视频、歌手信息、品味等工具。
- `evaluate` / `reflect` 在候选不足或约束不满足时触发补救。
- `finalize` 统一生成 grounded answer、运行 Answer Guard、写回记忆和对话状态。

独立 ReActLoop 已删除。图异常会返回明确错误，不再把同一请求切换到第二套编排。

## 3. 记忆与推荐

记忆分四类进入推荐：

- 显式偏好：用户直接表达的口味、排除项。
- 行为记忆：收听时长、完成率、评分、不喜欢。
- 语义/情景记忆：跨会话召回“之前说过”的偏好片段。
- 巩固画像：每隔若干轮把零散偏好压缩成一句稳定画像。

推荐排序使用三锚/四锚框架：

```text
final = semantic + personalize + behavior (+ collaborative)
```

无 embedding 或无协同过滤数据时，权重自动回分到可用锚点；MMR 再做列表内多样性控制。

## 4. 反幻觉与可解释性

系统的反幻觉策略不是“让 LLM 小心一点”，而是把候选来源和答案生成分离：

- 工具先返回真实候选或明确 fallback。
- 回答里的《歌名》/《专辑名》必须在候选白名单内。
- `guard_answer` 会移除无法追溯的书名号内容。
- final SSE payload 带 `trace_summary`，前端能展示 intent、工具、来源、fallback、guard、最终卡片数。

这让项目可以用自动化 smoke 检查“有没有走对工具、有没有发出 album_card、有没有 final”。

## 5. 前端产品形态

当前主前端是 Vue 3 SPA：

- 对话页：SSE 流式、候选卡片、专辑卡、决策摘要、底部播放器避让。
- 探索工作台：歌手搜索、歌手信息、代表专辑、完整专辑曲目、热门歌曲。
- 曲库/歌单/偏好：本地资产、保存专辑、生成歌单、排除规则、网易云扫码导入。

## 6. 质量闭环

项目有三层回归：

- `pytest`：单元和集成测试，覆盖 intent、graph、answer guard、API、播放、存储并发等。
- `scripts/long_dialogue_smoke.py`：无需真实 key 的长对话结构化 smoke，输出 Markdown + JSON 报告。
- `tests/eval/`：真实 LLM key 下的 LLM-as-judge / regress / A/B ranking，用于质量趋势而不是基础正确性。

CI 同时跑 pytest 和 long dialogue smoke，保证“无 key 可跑”这个项目特性持续成立。

## 7. 接下来最值得做

下一步不应先堆更多 intent，而是做三件事：

1. 把真实源 golden cases 做成小型发布前检查，覆盖网易云搜索、专辑顺序、MV、歌手信息、导入歌单。
2. 把 per-user auth、CORS、health 状态和部署文档打磨成可上线基本盘。
3. 继续收束前端工作流，让“探索歌手 -> 播放专辑 -> 收藏/入库/反馈”成为一条顺滑路径。
