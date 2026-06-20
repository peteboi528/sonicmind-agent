# ADR: LangGraph + Tool Runtime V2

状态：已采用并完成迁移（2026-06-19）

## 决策

- LangGraph 是唯一业务编排层；动态 tool-calling 只负责生成 `ToolCall`。
- `app.tools.registry` 是工具名称、参数模型、风险、超时和 handler 的唯一来源。
- Graph 的所有工具调用都必须经过 `ToolRuntime.execute/execute_sync`，不得新增工具分支。
- 工具的多轮增强（延续去重、翻页、search_variants、语言加权）由 handler 从 `ctx.plan` 防御性读取，不在 graph 节点里复制。
- 外部账号写操作必须返回 `confirmation_required`，经 `/agent/resume` 和同一 action id 恢复。
- SSE 由 LangGraph custom stream 输出；API 不再手动推进同步生成器。
- SSE 规划、恢复、自省、工具和最终 LLM token 均走原生 async；不存在同步 Graph/ReAct 线程池桥接。
- Checkpoint、确认账本和 trace 分库存储，不写入音乐资源库。

## 兼容与开关

- 异步 LangGraph 与原生 stream 已成为唯一执行方式，不再提供同步编排开关。
- `AGENT_CHECKPOINTS=true`、`LOCAL_TRACING=true`：默认启用，数据保留 30 天。

## 禁止事项

- 不在 Graph node 或 LLM schema 中复制工具实现。
- 不手写与 Pydantic 参数模型平行的第二份工具 schema。
- 不在 checkpoint/trace 中保存 Cookie、API Key、完整歌词或音频。
- 不自动修改、迁移或合并用户音乐库。

## 迁移完成（2026-06-19）

旧工具执行分支、独立 `react_loop.py` 与网络源同步灰度路径均已删除。生产入口、复合任务和失败恢复都由同一异步 LangGraph + Tool Runtime 执行。
