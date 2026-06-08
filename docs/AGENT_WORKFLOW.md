# SONICMIND Agent — 工作流与功能全景

> 版本：0.3.0 · 最后更新：2026-06-08

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 项目结构](#2-项目结构)
- [3. 核心工作流：ReAct Loop](#3-核心工作流react-loop)
- [4. 13 个注册工具](#4-13-个注册工具)
- [5. 三层记忆系统](#5-三层记忆系统)
- [6. 推荐引擎](#6-推荐引擎)
- [7. 检索系统（Hybrid RAG）](#7-检索系统hybrid-rag)
- [8. 外部集成](#8-外部集成)
- [9. API 端点（FastAPI）](#9-api-端点fastapi)
- [10. UI 层（Streamlit）](#10-ui-层streamlit)
- [11. LLM 集成](#11-llm-集成)
- [12. 测试覆盖](#12-测试覆盖)
- [13. 配置说明](#13-配置说明)

---

## 1. 项目概览

SONICMIND（原 CineSonic）是一个**可解释、离线优先的音乐推荐智能体**，采用纯 Python 构建。核心能力包括：

- **ReAct 循环 + 原生 Function Calling**：LLM 自主决策调用工具，迭代推理
- **三层记忆系统**：显式偏好 + 行为记忆 + 派生品味画像
- **Hybrid RAG 检索**：TF 余弦 + 关键词 + 可选稠密嵌入
- **反幻觉机制**：Answer Guard 自动剥离未验证的歌名
- **完全离线运行**：无 API Key 时自动切换 MockLLM，保证 demo 可用

---

## 2. 项目结构

```
MusicAgent/
├── pyproject.toml                  # 项目配置与依赖
├── README.md
├── app/
│   ├── agent.py                    # 核心 Agent 类（AudioVisualAgent）
│   ├── react_loop.py               # ReAct 循环 + 原生工具调用
│   ├── config.py                   # 环境变量配置
│   ├── models.py                   # Pydantic 数据模型
│   ├── memory.py                   # 记忆管理 + 行为评分
│   ├── storage.py                  # JSON 文件存储
│   ├── similarity.py               # 资产/片段相似度
│   ├── netease_auth.py             # 网易云音乐 QR 码登录
│   ├── api/
│   │   └── main.py                 # FastAPI 端点
│   ├── llm/
│   │   ├── client.py               # OpenAI 兼容 LLM 客户端
│   │   ├── protocol.py             # LLMProvider 协议 + 数据类
│   │   ├── tools.py                # 13 个工具定义（OpenAI function calling schema）
│   │   ├── mock.py                 # MockLLM 离线演示
│   │   └── structured.py           # JSON 提取与验证
│   ├── recommend/
│   │   ├── engine.py               # 打分排序 + 品味画像计算
│   │   └── daily.py                # 每日推荐（Explore/Exploit）
│   ├── retrieval/
│   │   ├── vector_store.py         # HybridRetriever（TF + 关键词 + 可选稠密）
│   │   └── embeddings.py           # sentence-transformers 后端
│   ├── media/
│   │   ├── pipeline.py             # 媒体管道（导入/分析/URL 规范化）
│   │   └── analyzer.py             # DemoAnalyzer（确定性合成片段）
│   ├── prompts/
│   │   ├── __init__.py             # Prompt 注册中心
│   │   ├── agent_system.py         # Agent 系统 Prompt
│   │   ├── intent.py               # 意图分类（降级路径）
│   │   ├── recommend.py            # 每日推荐 Prompt
│   │   ├── search.py               # 搜索 Prompt
│   │   ├── playlist.py             # 歌单生成 Prompt
│   │   ├── identify.py             # URL 歌曲识别 Prompt
│   │   └── reflect.py              # ReAct 反思 Prompt
│   ├── sources/
│   │   ├── protocol.py             # ExternalSource 协议
│   │   └── mock_source.py          # 47 首中英文 Mock 曲库
│   └── ui/
│       └── streamlit_app.py        # Spotify 主题 Streamlit UI（~1135 行）
├── tests/
│   ├── conftest.py                 # 强制 Mock 模式
│   ├── test_agent_flow.py
│   ├── test_react.py
│   ├── test_react_tool_calling.py
│   ├── test_answer_guard.py
│   ├── test_daily_recommend.py
│   ├── test_memory_enhanced.py
│   ├── test_similarity.py
│   ├── test_embeddings.py
│   ├── test_api.py
│   ├── test_behavior_scoring.py
│   └── eval/                      # LLM-as-Judge 评估框架
├── docs/
│   ├── EXPLAINER.md                # 架构面试说明
│   └── ROADMAP_ALIGNMENT.md        # 学习路线覆盖
├── scripts/
│   └── demo_flow.py                # CLI 演示脚本
└── data/store/                     # 持久化 JSON 数据
```

---

## 3. 核心工作流：ReAct Loop

### 3.1 主路径：原生工具调用

```
用户输入
  ↓
构建 Prompt（system + 历史 + 查询 + 目标状态）
  ↓
LLM.chat_with_tools(messages, AGENT_TOOLS, temperature=0.3)
  ↓
┌─ LLM 返回 tool_calls？──→ 是 ──→ 执行 tool → 结果作为 tool role 喂回 ──→ 循环
│                                                         （最多 5 轮）
└─ LLM 无 tool_calls？ ──→ 提取最终文本回答
                              ↓
                     Answer Guard（反幻觉检查）
                              ↓
                         最终回复
```

**核心流程（`_tool_calling_loop`）：**

1. **构建消息**：系统 Prompt + 对话历史 + 用户查询（注入资产上下文和目标状态）
2. **LLM 决策**：通过 OpenAI 兼容的 function calling 机制选择工具
3. **执行工具**：每个 `tool_call` 经 `_execute_tool()` 执行，结果作为 `tool` 角色消息返回
4. **循环判断**：无工具调用则提取最终答案并退出；最多 5 轮迭代
5. **早退出**：歌单目标数量已满足时提前终止

### 3.2 降级路径

如果主路径抛出异常两次，自动回退到：

```
_keyword_think() ──→ 关键词匹配意图规则 ──→ ActionType 枚举
                                                ↓
                                    _execute_tool() 顺序执行
```

### 3.3 Answer Guard（反幻觉）

```
收集工具结果中的所有"已知歌名" → 白名单
  ↓
扫描最终回答中 《》 包裹的歌名
  ↓
不在白名单中 → 剥离（歌单/专辑名豁免）
```

---

## 4. 13 个注册工具

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `recommend_music` | 个性化音乐推荐 | genre, mood, energy, tempo |
| `search_music` | 本地 + 外部曲库搜索 | query, source, limit |
| `generate_playlist` | 创建主题歌单 | instruction, track_count |
| `summarize_taste` | 用户品味画像摘要 | user_id |
| `find_similar_assets` | 跨资产相似度（Jaccard 标签） | asset_id, top_k |
| `find_similar_segments` | 资产内片段相似度（RAG） | asset_id, query, top_k |
| `retrieve_evidence` | RAG 证据检索 | query, top_k |
| `analyze_media` | 触发媒体分析 | asset_id |
| `generate_report` | 生成资产报告 | asset_id |
| `update_user_memory` | 存储用户偏好 | user_id, content |
| `search_web_music` | 真实在线搜索（网易云 + B站） | query, source |
| `fetch_track_metadata` | 抓取/丰富曲目元数据 | url |
| `import_netease_playlist` | 批量导入网易云歌单 | playlist_id |

---

## 5. 三层记忆系统

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: 显式偏好（Explicit Preferences）                │
│  ─ 正则提取用户语句中的偏好 ("我喜欢周杰伦", "I like jazz") │
│  ─ 存储为 MemoryEntry：frequency + last_used 时间戳       │
├──────────────────────────────────────────────────────────┤
│  Layer 2: 行为记忆（Behavioral Memory）                   │
│  ─ 播放事件（asset_id, duration, completed, context）     │
│  ─ 评分记录（0-10 分）                                    │
│  ─ 上限 200 条播放事件                                     │
├──────────────────────────────────────────────────────────┤
│  Layer 3: 派生记忆（Derived Memory）                      │
│  ─ TasteProfile：从资产 + 评分 + 播放行为自动计算         │
│  ─ 包含：genres, moods, energy, tempo, discovery_openness │
└──────────────────────────────────────────────────────────┘
```

### 5.1 加权查询（`weighted_query`）

将结构化偏好转化为搜索查询：

```
score(entry) = frequency × exp(-0.05 × age_days)
```

- 权重最高的偏好重复出现次数更多
- 结合近期目标状态
- 增强所有搜索和推荐操作

### 5.2 BaRT 行为评分（Spotify 论文启发）

```
reward(event) = {
    +1.0    完整播放
    -1.0    跳过（<15s 或 <10% 时长）
    线性插值  部分播放
}

score(asset) = Σ reward × exp(-λ × age)
```

### 5.3 探索开放度（`discovery_openness`）

| 行为 | 效果 |
|------|------|
| 完整播放非偏好风格曲目 | ↑ 开放度（最高 0.6） |
| 跳过探索曲目 | ↓ 开放度（最低 0.1） |
| 默认值 | 0.3 |

### 5.4 目标状态（`AgentGoal`）

- 用户查询含目标关键词时自动创建
- 相关性门控防止过期目标绑定
- `update_goal_progress()` 追踪完成步骤
- 技术动作（finalize, max_steps_reached, fallback, plan）不计入进度

---

## 6. 推荐引擎

### 6.1 打分公式

```
score(track) = 0.30 × genre_match
             + 0.25 × mood_match
             + 0.20 × energy_proximity
             + 0.15 × tempo_fit（高斯）
             + 0.10 × novelty
             + 0.25 × behavior_reward
```

| 因子 | 计算方式 |
|------|----------|
| **流派匹配** | 候选流派 ∩ 品味流派 |
| **情绪匹配** | 候选情绪 ∩ 时段情绪标签 |
| **能量接近度** | 与偏好能量的距离 |
| **节拍适配** | 以偏好 tempo 为中心的高斯分布 |
| **新颖度** | 未被最近播放的曲目加分 |
| **行为奖励** | BaRT 时间衰减累积分数 |

### 6.2 每日推荐（`DailyRecommender`）

```
                ┌─────────────────────┐
                │  时段感知            │
                │  早晨 / 专注 / 下午  │
                │  傍晚 / 夜间         │
                └────────┬────────────┘
                         ↓
              ┌──── 主路径 ────┐
              │  LLM 生成候选   │
              └────┬───────────┘
                   ↓ 失败
              ┌──── 降级路径 ───┐
              │  引擎排序兜底    │
              └────┬───────────┘
                   ↓
         Explore / Exploit 分配
         （由 discovery_openness 控制）
                   ↓
         每首歌附带推荐理由
```

**推荐理由来源：**
- 用户显式偏好匹配
- 品味画像匹配
- 曲库相似度

---

## 7. 检索系统（Hybrid RAG）

### 7.1 文档生成

```
资产 → 分 6 段 → 每段生成 4 类 SearchDocument
                   ├── TEXT（转录文本）
                   ├── VISION（视觉标签）
                   ├── AUDIO（音频标签）
                   └── SUMMARY（场景摘要）
```

### 7.2 混合检索

```
查询
  ↓
┌──────────────────┬───────────────────────┐
│   稀疏路径        │   稠密路径（可选）      │
│   TF 余弦相似度   │   sentence-transformers│
│   + 关键词 Jaccard│   MiniLM-L12 嵌入      │
└────────┬─────────┴──────────┬────────────┘
         ↓                    ↓
    final_score = 0.72 × dense + 0.28 × keyword
```

- 稠密嵌入模型：`paraphrase-multilingual-MiniLM-L12-v2`
- 无 `sentence-transformers` 依赖时自动降级为 TF 余弦
- 线程安全懒加载单例模式

---

## 8. 外部集成

| 平台 | 能力 | 实现方式 |
|------|------|----------|
| **网易云音乐** | QR 码登录 | `netease_auth.py` 自动轮询 |
| | 歌单导入 | 批量导入 + 元数据映射 |
| | 歌曲详情 | API 精确匹配标题/歌手/专辑/封面 |
| | 音频播放 | MP3 URL 获取（含 VIP 支持） |
| **Bilibili** | MV 搜索 | API 关键词搜索 |
| | 嵌入播放 | iframe 播放器 |
| **YouTube** | 元数据获取 | oEmbed API |
| | 降级获取 | `yt-dlp` 兜底 |

### URL 规范化

自动清洗各平台 URL 的追踪参数，实现稳定去重：

```python
# 示例：剥离 utm_source、t 等参数
normalize_url("https://music.163.com/song?id=123&tm=abc")
→ "https://music.163.com/song?id=123"
```

---

## 9. API 端点（FastAPI）

### 9.1 资产管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/assets` | 列出所有资产 |
| `POST` | `/assets/ingest` | 导入 URL 为新资产 |
| `POST` | `/assets/{id}/enrich` | 丰富资产元数据 |
| `POST` | `/assets/{id}/analyze` | 分析媒体生成片段 |
| `DELETE` | `/assets/{id}` | 删除资产及引用 |
| `DELETE` | `/cache` | 清除资产/片段缓存 |

### 9.2 评分与品味

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rate` | 评分（0-10） |
| `GET` | `/ratings/{user_id}` | 获取用户评分 |
| `GET` | `/taste/{user_id}` | 获取品味画像 |

### 9.3 推荐与搜索

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/recommend/daily` | 生成每日推荐 |
| `GET` | `/recommend/daily/{user_id}` | 获取每日推荐 |
| `GET` | `/assets/{id}/similar` | 查找相似资产 |
| `POST` | `/search` | 本地 + 外部搜索 |

### 9.4 记忆与对话

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/listen` | 记录播放事件 |
| `POST` | `/chat` | 与 Agent 对话（ReAct Loop） |
| `POST` | `/agent/run` | Agent 运行（`/chat` 别名） |
| `POST` | `/memory/update` | 更新用户记忆 |
| `POST` | `/memory/feedback` | 片段反馈 |
| `GET` | `/memory/{user_id}` | 获取完整用户记忆 |

### 9.5 歌单

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/playlist/generate` | 生成主题歌单 |
| `POST` | `/playlist/auto/{user_id}` | 自动分类生成歌单 |
| `GET` | `/playlists/{user_id}` | 列出用户歌单 |
| `DELETE` | `/playlist/{user_id}/{id}` | 删除歌单 |

### 9.6 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 重定向至 `/docs` |
| `GET` | `/health` | 健康检查 |

---

## 10. UI 层（Streamlit）

Spotify 深色主题 UI，约 1135 行，文件位于 `app/ui/streamlit_app.py`。

### 10.1 整体布局

```
┌─────────────────────────────────────────────────────┐
│  侧边栏                                              │
│  ├─ 用户切换                                          │
│  ├─ 音乐 URL 导入（可选仅离线模式）                     │
│  ├─ 网易云歌单导入                                     │
│  ├─ 偏好训练                                          │
│  ├─ 网易云 QR 码登录（自动轮询）                        │
│  └─ 已学习偏好展示                                     │
├─────────────────────────────────────────────────────┤
│  Hero 区：品味流派标签 | 播放/评分/资产统计 | Agent 状态  │
├─────────────────────────────────────────────────────┤
│  5 个 Tab                                            │
│  ├─ 🎵 每日推荐：时段选择 | SIMILAR/DISCOVER/MOOD 标签  │
│  ├─ 🔍 发现：搜索 + 证据展示                           │
│  ├─ 📚 我的库：资产列表 + 圆环评分滑块                   │
│  ├─ 📋 歌单：手动/自动分类生成                          │
│  └─ 💬 聊天：多轮对话 + 可折叠 Trace/Evidence/Goal      │
├─────────────────────────────────────────────────────┤
│  底部播放器：旋转唱片 + 封面 + HTML5 音频控制             │
│  浮动 MV 窗：B站/YouTube 嵌入播放器                     │
└─────────────────────────────────────────────────────┘
```

### 10.2 核心交互

| 交互 | 说明 |
|------|------|
| **播放** | 双模式：纯音频（网易云 MP3）+ MV/视频（B站嵌入优先，YouTube 兜底） |
| **评分** | Pitchfork 风格圆环滑块（0-10） |
| **导入** | 支持 YouTube / Bilibili / 网易云 URL |
| **对话** | 多轮对话，每条回复可展开查看 Trace / Evidence / Goal |

---

## 11. LLM 集成

### 11.1 架构

```
LLMProvider（Protocol）
├── OpenAICompatibleLLM    ← urllib.request 原生 HTTP 调用
└── MockLLM                ← 关键词工具选择 + 模板回复
```

### 11.2 OpenAI 兼容客户端

- 通过 `urllib.request` 调用 `/chat/completions` 端点
- 支持三种方法：`generate()` / `chat()` / `chat_with_tools()`
- 工具调用支持 JSON 参数解析
- 网络错误优雅降级

### 11.3 MockLLM（离线模式）

```
第一轮：关键词匹配 → 选择工具
第二轮：模板回复 → 最终答案
```

- 按流派/情绪返回推荐理由
- 跟踪工具调用历史避免重复

### 11.4 工厂方法

```python
build_llm():
    if LLM_API_KEY 为空 or 本地端点不可达（0.2s 探测）:
        return MockLLM()
    else:
        return OpenAICompatibleLLM()
```

### 11.5 Prompt 体系

| Prompt | 用途 |
|--------|------|
| `agent_system` | Agent 系统指令：中文音乐推荐、主动调用工具、不编造、适度更新记忆 |
| `intent` | 降级路径意图分类（13 种 ActionType） |
| `recommend` | 每日推荐模板（偏好 + 品味 + 曲库 + 时段 → JSON 曲目列表） |
| `search` | 搜索模板（查询 → JSON 曲目列表） |
| `playlist` | 歌单生成模板（指令 + 曲库 + 候选 → JSON 歌单） |
| `identify` | URL 歌曲识别（URL + 标题 → 歌名/歌手/流派/情绪） |
| `reflect` | ReAct 反思（决定是否继续循环） |

---

## 12. 测试覆盖

### 12.1 单元/集成测试

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_agent_flow` | 完整 Agent 生命周期（导入→分析→记忆→播放→品味→对话→删除） |
| `test_react` | ReAct 意图分类（similar/recommend/retrieve/taste） |
| `test_react_tool_calling` | 原生工具调用循环、最大步数、未知工具拒绝、目标进度 |
| `test_answer_guard` | 反幻觉白名单、歌单名豁免 |
| `test_daily_recommend` | 每日推荐 + 理由、搜索、品味、歌单生成（LLM + 降级） |
| `test_memory_enhanced` | 结构化偏好、频率递增、时间衰减 |
| `test_similarity` | 跨资产/资产内相似度 |
| `test_embeddings` | 稠密嵌入注入 + TF 余弦降级 |
| `test_api` | 全 FastAPI 端点生命周期 |
| `test_behavior_scoring` | BaRT 行为评分 + discovery_openness 升降 |

### 12.2 LLM-as-Judge 评估框架

位于 `tests/eval/`，包含：

| 组件 | 说明 |
|------|------|
| `cases.py` | 6 个测试用例（基础推荐、品味推荐、歌单、多轮对话、精确搜索、品味查询） |
| `judge.py` | 独立 Judge LLM 对回复打分（0-5 分） |
| `run.py` | 评估运行器 |
| `README.md` | 评估使用说明 |

### 12.3 运行方式

```bash
# 全部测试（Mock 模式）
python3 -m pytest

# LLM-as-Judge 评估（需要真实 API Key）
python -m tests.eval.run
```

---

## 13. 配置说明

### 13.1 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API 地址（默认 Ollama） |
| `LLM_API_KEY` | — | API 密钥（为空时自动切换 MockLLM） |
| `LLM_MODEL` | `qwen2.5` | 模型名称 |
| `LLM_TIMEOUT_SECONDS` | `45` | 请求超时 |
| `LLM_MAX_TOKENS` | `1024` | 最大输出 token |
| `ENABLE_ONLINE_ENRICH` | `false` | 启用在线元数据丰富 |
| `ENABLE_EMBEDDINGS` | `false` | 启用稠密嵌入检索 |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 嵌入模型 |
| `STORE_ROOT` | `data/store` | 数据存储路径 |
| `DAILY_REC_COUNT` | `25` | 每日推荐数量 |

### 13.2 启动方式

```bash
# API 服务
uvicorn app.api.main:app --reload --port 8000

# Streamlit UI
streamlit run app/ui/streamlit_app.py --server.port 8501

# CLI 演示
python scripts/demo_flow.py

# 运行测试
python3 -m pytest
```

---

> 📄 本文档由 SONICMIND 项目自动生成，涵盖 Agent 完整工作流与功能架构。
