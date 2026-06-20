# SonicMind (MusicAgent) 架构全景报告

> 更新日期：2026-06-17。本文记录当前实现，不把规划中的能力写成已完成能力。

## 项目定位

SonicMind 是一个可解释、记忆驱动、反幻觉的音乐推荐 Agent。它的核心价值不是“问一句答一句”，而是围绕用户品味持续工作：识别意图、调用真实音乐工具、维护记忆和反馈、生成可追溯推荐，并把决策过程通过 SSE 交给前端展示。

当前技术栈：

- 后端：FastAPI、Pydantic v2、SQLite 资源库、JSON store、可选 LangGraph。
- Agent：`app/graph/` 是生产环境唯一编排层；复合任务复用同一异步图。
- 前端：Vue 3 SPA，包含聊天、探索工作台、播放器、曲库、歌单和偏好页。
- 数据源：网易云、B站、YouTube、Last.fm、Tavily/DuckDuckGo；无 key 时使用 MockLLM/MockSource 降级。

## 核心链路

```text
HTTP /agent/stream
  -> AudioVisualAgent.stream_chat()
  -> Graph.stream()
  -> load_context
  -> plan_intent
  -> execute_tools
  -> web_fallback? / evaluate / reflect
  -> finalize
  -> SSE final payload: AgentAnswer + cards + trace_summary
```

关键约束：

- LLM 负责意图和实体，曲风/心情/场景标签尽量走确定性规则。
- 推荐、搜索、专辑、视频、歌手信息等能力通过工具结果落地。
- Answer Guard 只允许答案保留可追溯的歌名/专辑名。
- 前端依赖稳定 SSE 事件：`plan`、`tool_start`、`candidates`、`album_card`、`eval`、`final`。

## 子系统

| 子系统 | 作用 | 当前状态 |
|---|---|---|
| `app/intents.py` | intent registry、关键词优先级、continuation 判断 | 已集中化 |
| `app/graph/` | 主 Agent 编排、条件 fallback、reflection、SSE | 主路径 |
| `app/tools/` | Tool Runtime V2、registry、handler、checkpoint/trace | 统一工具执行层 |
| `app/answer.py` | 候选收集、Answer Guard、卡片/进度辅助 | Graph 共用 |
| `app/recommend/` | 三锚/四锚精排、MMR、协同过滤辅助 | 可配置降级 |
| `app/memory.py` | 偏好、行为、语义记忆、巩固画像、排除规则 | 已接入主链路 |
| `app/sources/` | 网易云/B站/YouTube/Last.fm/web search | 真实源 + mock/fallback |
| `frontend/src` | Vue 产品界面、播放器、发现页、曲库和设置 | 主前端 |
| `scripts/long_dialogue_smoke.py` | 长对话结构化回归，输出 MD/JSON | CI smoke |

## 真实能力与降级边界

真实能力：

- 网易云歌曲搜索、专辑搜索、专辑曲目按原始顺序加载、扫码登录与歌单导入。
- B站/YouTube 视频搜索与播放代理。
- 歌手信息搜索、代表专辑、热门歌曲。
- 行为反馈、评分、不喜欢、排除规则进入推荐排序。
- 长对话 continuation、去重、专辑卡 SSE、trace summary。

降级/占位：

- 无 LLM key 时使用 MockLLM，保证结构链路可跑，但不代表真实模型质量。
- 无外部 key 或接口限流时，部分源会返回 fallback 或空结果，系统应诚实说明。
- 媒体分析当前以元数据富化和确定性占位为主，不声称已有 Whisper/CLIP 级真实音视频理解。

## API 与部署基本盘

主要公开入口：

- `POST /agent/stream`：主流式 Agent。
- `POST /chat` / `POST /agent/run`：同步 Agent。
- `POST /search`、`POST /artist/info`、`POST /artist/album_tracks`。
- `POST /album/save`、`GET /albums/saved/{user_id}`、`DELETE /album/saved/{user_id}/{album_id}`。
- `POST /playlist/generate`、`GET /playlists/{user_id}`、`DELETE /playlist/{user_id}/{playlist_id}`。

部署配置：

- `AUTH_ENABLED=false`：本地 demo，不校验。
- `AUTH_ENABLED=true` + `USER_API_KEYS=user:key`：服务端从 key 解析 user_id，并覆盖客户端传入的 user_id。
- `AUTH_ENABLED=true` + `API_KEY`：共享 key 兼容模式，只做访问门禁。
- `ALLOWED_ORIGINS`：CORS 白名单，本地可设 `*`，部署应收紧。
- `/health` 返回 LLM mode、auth mode、store path、frontend build hash、last smoke report 等状态。

## 质量闭环

```bash
python3 -m compileall app tests scripts -q
python3 -m pytest
python3 scripts/long_dialogue_smoke.py
npm run build
```

CI 当前跑：

- ruff
- pytest + coverage
- long dialogue smoke

`tests/eval/` 作为真实 LLM 质量趋势工具，不替代 deterministic smoke。推荐把真实源 golden cases 保持小而稳，用于发布前人工/半自动检查。
