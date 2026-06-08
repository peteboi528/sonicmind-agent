# SONICMIND Agent 原理讲解

这份文档面向面试讲解。当前版本不再把项目表述为“单一影音问答 demo”，而是一个带记忆、RAG 和工具调用的推荐 agent。

## 1. 系统定位

项目的核心不是单次 LLM 对话，而是一个围绕用户音乐品味持续工作的 agent：

```text
入库 -> 分段分析 -> 建立证据库 -> 记录偏好/收听/评分 -> 更新品味模型
    -> Agent 根据问题选择工具 -> 检索证据 / 生成推荐 / 搜索 / 生成歌单 / 总结品味
```

它更接近“面向音乐推荐场景的 agent 工程项目”，而不是泛化聊天机器人。

## 2. 记忆系统

记忆分为三层：

- 显式记忆：用户直接说出的偏好，例如“我喜欢电子音乐”。
- 行为记忆：收听历史、完成度、评分反馈。
- 派生记忆：由历史和评分计算出的 `taste_profile`，包含风格、情绪、能量和速度偏好。

`MemoryManager.weighted_query()` 会把长期记忆转成检索增强查询，这样记忆不只是存起来，而是真正进入推荐和搜索主链路。

## 3. RAG 设计

每个素材会被拆成多个 segment，每个 segment 再拆成四类证据：

- `text`: transcript
- `vision`: visual tags
- `audio`: audio tags
- `summary`: scene summary

`HybridRetriever` 用轻量混合检索把 query 映射到这些证据块。虽然当前不是向量数据库版本，但接口已经是 RAG 形态：

```text
query -> 检索 evidence -> 带时间戳和模态返回 -> 进入 agent 回答
```

这让推荐解释、搜索解释和片段问答都有“证据来源”，而不是只靠模型口头解释。

## 4. Agent 编排

主入口是纯 Python 的 `ReActLoop`，不是 LangGraph。原因是这一版更强调可讲清楚和稳定演示。

当前 agent 会根据用户问题在这些动作之间路由：

- `recommend_music`
- `search_music`
- `generate_playlist`
- `summarize_taste`
- `find_similar_assets`
- `find_similar_segments`
- `retrieve_evidence`
- `update_user_memory`
- `generate_report`
- `search_web_music`
- `fetch_track_metadata`
- `import_netease_playlist`

因此主入口 `/agent/run` 不再只是把上下文拼成 prompt，而是让 LLM 在 ReAct 循环里选择工具、观察结果、决定下一步，最后把 trace 和证据整理成回答。`/chat` 保留为兼容别名。

关键词规则仍然存在，但只作为 LLM 连续失败后的兜底路径；默认主链路是 native tool calling。

## 5. 目标状态

多步任务会进入 `AgentGoal`：

```text
goal -> steps_done -> steps_pending -> status
```

例如用户说“导入网易云歌单，然后挑适合跑步的歌，再生成歌单”，agent 会把这个目标持久化到记忆 store，并在每轮工具调用后更新进度。API 返回中可以看到：

- `pending_goal`
- `goal_progress`
- `agent_trace`

这能体现 agent 不只是单轮问答，而是在推进一个可追踪的任务。

## 6. 离线优先

默认模式是离线优先：

- `ingest` 不阻塞联网识别标题。
- `enrich` 是显式可选步骤。
- mock LLM、mock source、demo analyzer 保证没有外网也能完整演示。
- 测试环境强制使用 mock LLM，避免本机 `.env` 或不可用的模型服务影响稳定性。

这样做的目的不是偷懒，而是为了让面试现场稳定，并把“真实能力接入”留成下一阶段的明确扩展点。

## 7. 面试讲法

可以这样介绍：

> 我把这个项目从普通推荐 demo 收束成了一个 recommendation agent。它会持续学习用户的偏好和行为，用记忆增强检索查询，再通过 RAG 和工具调用给出推荐、搜索解释、品味分析和歌单生成。主入口是 `/agent/run`，LLM 负责决定是否要联网、是否要检索证据、是否要写记忆；目标进度会持久化并返回。默认离线优先，保证稳定演示；真实平台搜索、元数据抓取和网易云歌单导入作为可选工具接入。

继续扩展时，优先做三件事：

1. 用真实音视频处理替换 demo analyzer。
2. 把轻量检索替换成 embedding + vector DB + rerank。
3. 把纯 Python ReActLoop 升级成显式状态图编排。
