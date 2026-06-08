# 路线图对齐说明

本项目把 `Agent工程师学习路线图.md` 中的学习任务改造成一个可展示项目。

## 已覆盖能力

- RAG 系统：本地 hybrid retrieval、证据分块、时间戳引用。
- Agent 架构：工具化能力、执行轨迹、任务分解。
- 记忆系统：短期上下文、长期偏好、项目事件。
- Context Engineering：动态组装用户问题、记忆、证据和结构化元数据。
- 工程能力：FastAPI、Streamlit、Pydantic、测试和文档。

## 任务拆分

### 第 1 阶段：跑通 MVP

- 建立项目结构。
- 实现媒体导入和模拟分段。
- 实现 RAG 证据检索。
- 实现用户记忆写入和读取。
- 实现 FastAPI 与 Streamlit Demo。

### 第 2 阶段：替换真实模型

- 用 `yt-dlp` 下载公开视频。
- 用 `ffmpeg` 抽音频和关键帧。
- 用 Whisper 做 ASR。
- 用 CLIP 或视觉语言模型生成关键帧描述。
- 用 Chroma 或 FAISS 存储 embedding。

### 第 3 阶段：面试加分

- 增加 rerank。
- 增加 RAG 评测集。
- 用 LangGraph 改写 agent 编排。
- 增加 memory policy，对写入、压缩、遗忘做策略化管理。

