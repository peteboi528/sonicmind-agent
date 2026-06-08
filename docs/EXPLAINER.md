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

- `retrieve`
- `recommend`
- `search`
- `playlist`
- `taste`
- `similar_cross`
- `similar_intra`
- `memory_update`
- `report`

因此 `chat` 不再只是把上下文拼成 prompt，而是先做意图判断，再调用具体工具，最后把 trace 和证据整理成回答。

## 5. 离线优先

默认模式是离线优先：

- `ingest` 不阻塞联网识别标题。
- `enrich` 是显式可选步骤。
- mock LLM、mock source、demo analyzer 保证没有外网也能完整演示。

这样做的目的不是偷懒，而是为了让面试现场稳定，并把“真实能力接入”留成下一阶段的明确扩展点。

## 6. 面试讲法

可以这样介绍：

> 我把这个项目从普通推荐 demo 收束成了一个 recommendation agent。它会持续学习用户的偏好和行为，用记忆增强检索查询，再通过 RAG 和工具调用给出推荐、搜索解释、品味分析和歌单生成。默认是离线优先，保证稳定演示；如果需要，可以再接入真实媒体处理和在线元数据补全。

继续扩展时，优先做三件事：

1. 用真实音视频处理替换 demo analyzer。
2. 把轻量检索替换成 embedding + vector DB + rerank。
3. 把纯 Python ReActLoop 升级成显式状态图编排。
