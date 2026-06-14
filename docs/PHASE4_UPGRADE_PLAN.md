# MusicAgent (SonicMind) 升级计划 · Phase 4
## 从「聪明的路由器 + 精排器」到「成熟的 Agent」

> **目标**:在守住现有核心优势(零依赖降级 / 反幻觉 / 单图编排)的前提下,堵住两个硬伤、补齐工程债,并完成一次能力跃迁——让系统能处理它没被显式编程过的复合任务。
>
> **当前基线**:核心链路完整、323 测试通过、总覆盖 64%。本计划将其推向「可投递、可演示、可量化」。
>
> **现状论断来源**:以下"现状"来自上一轮代码评审,每条都在对应 P0 项的"上盘核对"步骤里用 grep/读码当场确认,确认不过不往下走。
>
> **不可违背的编排原则**:任何改动都必须保持 ① 无 key 可跑(MockLLM)② 无 langgraph 可降级到同步执行 ③ 答案仍过 Answer Guard ④ 单图 + 条件路由不分裂为双编排。

---

## 总览:优先级与时间线

| 优先级 | 工作项 | 工时 | 价值 | 依赖 |
|---|---|---|---|---|
| **P0** | A. 工程卫生:CI / Docker / .env.example / 删死代码 / lint | 0.5d | 立刻"专业",面试第一印象 | 无 |
| **P0** | B. 多租户安全:user_id 鉴权 + JsonStore 并发锁 | 1d | 堵硬伤二 | 无 |
| **P0** | C. 媒体分析诚实化 | 0.5d | 堵硬伤一(文档↔实现裂缝) | 无 |
| **P1** | D. Eval 回归护栏:golden set + 4 指标 + diff + A/B | 1d | 后续改 ranking 的安全网;portfolio 杀手锏 | 无(越早越好) |
| **P1** | E. Deep/Agentic 模式:复合任务走真 ReAct | 2–3d | **能力跃迁主菜** | D(eval 回归) |
| **P1** | F. Reflection 自省节点 | 1d | 质量:事后清理 → 交付前自查 | 无 |
| **P1** | G. 记忆升级:语义召回 + 巩固 | 1–2d | 长期个性化深度 | embeddings 已有 |
| **P2** | H. 工具并发 + 协同过滤(可选 4th anchor) | 1–1.5d | 延迟↓ / 召回↑ | 无 |

**两周节奏**
- Week 1 前三天:扫清三个 P0(A/B/C)+ 搭好 eval 护栏(D)。
- Week 1 后两天 ~ Week 2:Deep 模式(E)+ Reflection(F)+ 记忆升级(G)。
- P2(H)视余量插入。

---

## 落地顺序与依赖图

```
P0-A 工程卫生 ─┐
P0-B 安全    ─┼──→  Week1 前三天:硬伤清零
P0-C 媒体诚实 ─┘
P1-D eval 护栏 ──┬──→  P1-E Deep 模式(用 D 做回归)
                ├──→  P1-F Reflection
                └──→  P1-G 记忆升级
P2-H 并发 + CF(余量插入)
```

> 关键:**D(eval 护栏)要尽早**,它是 E/F/G 所有改动的回归安全网。先有度量,再谈优化——"能量化的 agent 才是工程,不能量化的只是 demo"。

---

# P0-A · 工程卫生 & DevOps(0.5 天)

**现状(待上盘核验)**:无 `.github/workflows`、无 `Dockerfile`、无 `.env.example`、无 ruff/mypy 配置;`app/graph/routing.py` 是死代码(0 引用、0% 覆盖)。

### 上盘核对(本项第一步,确认后继续)
```bash
ls .github/workflows/ Dockerfile .env.example pyproject.toml 2>/dev/null   # 预期全 MISSING
grep -rn "routing" app/ tests/ | grep -v "\.pyc"                            # 预期 0 命中 → routing.py 确为死代码
grep -rhoEn "(os\.environ|os\.getenv)\[['\"][A-Z_]+['\"]\]" app/ | sort -u  # 得到真实 env 变量全集,喂给 .env.example
```

### 交付物
1. **CI** — `.github/workflows/ci.yml`:`pip install -e .` → `ruff check .` → `pytest -q --cov=app --cov-fail-under=60`。Python 矩阵 3.10/3.11。
2. **Dockerfile** — 多阶段,`CMD` 跑 API;**关键:无 key 时仍能启动**(零依赖降级路径必须活)。
3. **`.env.example`** — 枚举代码真实读取的 env 变量(上面 grep 的结果),不臆造。预期含 LLM key、各数据源 key(netease/bilibili/youtube/tavily/lastfm)、端口等。这是工程纪律的体现,必须精确。
4. **lint 配置** — `pyproject.toml` 加 `[tool.ruff]`(line-length、ignore 适配现有风格)。mypy 仅作 `--ignore-missing-imports` 的 warning,不阻塞。
5. **删死代码** — `app/graph/routing.py`(grep 证据留 commit message);顺手清其它 0 引用模块。

### 验收
- PR 上 CI 绿;`docker build` 成功且无 key 启动正常;`ruff check .` 干净;`routing.py` 及其它死代码已删。

### 风险
- 覆盖率门槛设 60%(当前 64%)留余量,别一上来设太高挡住日常提交。

---

# P0-B · 多租户安全 & 并发(1 天)

**现状(待上盘核验)**:`ChatRequest.user_id` 由客户端传入、API 无鉴权 → 任何人传别人 user_id 即可读写他人记忆;`JsonStore` 读-改-写无文件锁,`os.replace` 只保证单次写原子,并发 RMW 会丢更新。

### 上盘核对
```bash
grep -rn "user_id" app/                          # 枚举所有信任 user_id 的端点/模型
grep -rn "class JsonStore\|os.replace" app/      # 定位 RMW 调用点
```

### 交付物
1. **鉴权层** — 最小可用:共享密钥 / per-user API key(header `X-API-Key`),FastAPI dependency 注入。本地单用户用 env `AUTH_ENABLED=false` 关闭,保持 demo 便利。所有带 user_id 的路由统一过 dependency。
2. **并发安全** — `JsonStore` 的 RMW 加 POSIX 文件锁(**推荐 `fcntl` flock**,零依赖、POSIX 自带;或 `filelock` 库跨平台但加依赖)。锁粒度 = per-user 文件。明确标出所有 RMW 调用点。
3. (长期,不在本期)`JsonStore` → sqlite(WAL 天然并发安全),避免范围蔓延。

### 验收
- 无 token 访问被拒;用别人 user_id 被拒;并发写同一 user 的 pytest(threading 模拟)不丢更新。

### 风险
- 鉴权不能破坏现有 bot 接入(微信/飞书)回调路径——给 bot 一个 server-side 信任的内部 token 旁路。

---

# P0-C · 媒体分析诚实化(0.5 天)

**现状(待上盘核验)**:`app/media/analyzer.py` 的 `DemoAnalyzer` 用 `random.sample(GENRE_POOL)` + 静态 tag 池(假分析);README 流水线图写"入库→分段分析→证据库"、路线图 Phase 2 规划 Whisper/CLIP/ffmpeg 但从未落地;`ingest_video` 只存 URL 壳。

### 上盘核对
```bash
grep -n "random\|GENRE_POOL\|sample" app/media/analyzer.py
grep -rn "Whisper\|CLIP\|ffmpeg\|分段分析" README.md docs/
```

### 方案(A 必做,B 可选)
- **A(必做 · 诚实化文档+去随机)**:README/路线图里"分段分析 / Whisper/CLIP"改为诚实表述——"基于平台元数据的标签富化(metadata-based tag enrichment)"。`DemoAnalyzer` 重命名/标注为 `MetadataTagEnricher`,**移除 `random` 行为**:要么用真实拿到的 netease/lastfm 元数据(BPM/genre/tag),要么明确标 `unanalyzed`,**绝不随机猜**。
- **B(可选 · 一个真实信号)**:dep check 后接可选的 ffmpeg-based BPM/energy 抽取器(有 ffmpeg 才跑,没有标 `unanalyzed`),零依赖路径不受影响。

> 推荐:**先做 A**(低成本、消除最大裂缝),B 进路线图后续项,写明"需 ffmpeg,默认关闭",**不画饼**。

### 验收
- 代码无 `random.sample(GENRE_POOL)` 之类伪造;README 流水线图与实现一致;`ingest` 行为文档诚实。

---

# P1-D · Eval 回归护栏(1 天)— 尽早做

**现状(待上盘核验)**:`tests/eval/` 有 LLM-as-judge 框架,但需真实 key、一次性、无 golden set、无指标 diff、无 A/B。

### 上盘核对
```bash
ls tests/eval/ && sed -n '1,80p' tests/eval/*.py     # 看现有 judge 框架可复用多少
grep -n "weight\|alpha\|anchor" app/recommend/rerank.py  # 三锚权重是否已可配置
```

### 交付物
1. **Golden set** — `tests/eval/golden/*.yaml`,每条 = 用户多轮 + 预期意图 + must-include/must-exclude 曲目 + 约束检查。先种 5–8 条覆盖 recommend / search / discuss / playlist / artist_info / 复合任务。
2. **4 指标**:
   - **意图命中率**(intent-hit)
   - **反幻觉通过率**(答案曲目都在真实候选里 / 来源可追溯)
   - **列内多样性**(MMR 已有,直接算)
   - **推荐相关性**(LLM-as-judge;无 key 时用确定性子集:must-include/exclude 命中)
3. **回归命令** — `python -m tests.eval.regress` 跑当前 vs `tests/eval/baseline.json`,打印 before/after diff 表(四指标 ±)。
4. **A/B ranking** — `recommend/rerank.py` 三锚权重做成可配置 profile(`three_anchor` / `pure_semantic` / `pure_behavior`),脚本跑三套出对比表。**portfolio 直接能讲半小时的硬货。**

### 验收
- 无 key 能跑确定性子集并出表;改 rerank 权重后 diff 表反映变化;有 key 时四指标齐全。

---

# P1-E · Deep / Agentic 模式(2–3 天)— 能力跃迁主菜

**现状(待上盘核验)**:`app/react_loop.py` 有真 think→act→observe 循环,但被降级为 fallback;`app/agent.py` 的 `chat()` 主走图、异常才落 react;单轮意图路由处理不了复合任务(如"导入网易云歌单→分析华英比例→基于英文部分做夜跑步歌单")。

### 上盘核对
```bash
grep -n "def chat\|react_loop\|fallback" app/agent.py
sed -n '1,60p' app/react_loop.py
grep -n "add_node\|add_edge\|conditional" app/graph/builder.py
```

### 方案
1. **复合任务检测器** — 启发式信号(多动词、多约束、"然后/之后/and then"链)+ 可选轻量 LLM 分类器。挂到 `chat()` 分发前。
2. **Deep 模式升格为一级分支** — 不是 crash-fallback,而是正经 intent/路由分支。**复用**现有工具,且**必须复用** evaluate/finalize/guard_answer,保证 deep 模式同样反幻觉安全。
3. **边界** — max iterations、token budget;deep 模式 stall 时优雅降级回单轮。

### 验收
- 复合任务案例走 deep 模式并产出正确多步结果;普通任务不受影响仍走图;deep 模式答案过 guard。

### 依赖
- **D(eval 护栏)**先行——E 的每次迭代都用 eval diff 防回归。

---

# P1-F · Reflection 自省节点(1 天)

**现状(待上盘核验)**:Answer Guard 是反应式(出答案后删未核实歌名);无主动 reflection 在交付前核对候选 vs 用户完整约束("放松的、英文的、不要抖音热歌")。

### 上盘核对
```bash
grep -n "guard_answer\|def finalize\|def evaluate" app/
grep -n "add_node\|finalize" app/graph/builder.py
```

### 方案
1. 在图里 evaluate/candidate-build 与 finalize 之间插 `reflect` 节点。
2. 结构化打分 prompt:逐条核对候选 vs 抽取出的约束,返回 per-constraint pass/fail + 简短理由。**一次调用批所有候选**(省 token)。
3. 失败动作:re-rank / 换下一个候选 / regenerate。
4. 零依赖安全:无 key 时跳过 reflection,回退当前行为。

### 验收
- 注入"不要 X"约束时,reflect 能拦截含 X 的候选并替换;无 key 时行为不变。

---

# P1-G · 记忆升级(1–2 天)

**现状(待上盘核验)**:偏好抽取靠正则(`PREFERENCE_PATTERNS`),绕一点就漏;记忆是频率+衰减,无语义召回、无情景/语义区分、无巩固画像。

### 上盘核对
```bash
grep -rn "PREFERENCE_PATTERNS\|class.*Memory\|sentence_transformers\|embed" app/
```

### 方案
1. **LLM 结构化偏好抽取兜底** — 正则未命中时调 LLM 抽结构化偏好。
2. **语义召回** — 每轮交互 embed 进向量库(per-user `.npy`/jsonl,**复用已有 embeddings helper**),"你三周前说过想要慵懒爵士"能召回。
3. **情景 vs 语义记忆** — 具体某次听歌(情景)/ 稳定口味(语义)分开。
4. **巩固** — 每 N 轮让 LLM 把零散偏好巩固成一句话画像。

### 验收
- 绕口表述能抽到偏好;跨会词语义召回命中;巩固画像可读;无 key 时正则+TF 路径不变。

---

# P2-H · 工具并发 + 协同过滤(可选,1–1.5 天)

- **工具并发**:`execute_tools` 里独立工具(多源搜索 IO-bound)用 `asyncio.gather`/线程池并发,per-tool timeout,结果排序保持确定性(测试不破)。
- **协同过滤**:行为日志做 item-item/user-user,作为可选 4th anchor 归一到 0–1 尺度,冷启动回退现有三锚。

---

# 量化与交付

- **能量化的 agent 才是工程**:eval 护栏(D)产出四指标 + A/B 表,是本期能拿出去讲的核心证据。
- 每个 P1 项落地后跑一次 eval diff,贴进 PR 描述。
- **终态自检**:
  - 测试 323 → 更多;覆盖 64% → 70%+
  - CI 绿;Docker 可跑
  - 文档 ↔ 实现零裂缝
  - 复合任务可演示(Deep 模式)
  - 无 key 全链路仍可跑

---

## 建议的第一刀

如果只想立刻动一处,按投入产出比:**P0-A(半天,CI + Docker + .env.example + 删死代码)**——立刻让项目"专业"起来,且不碰任何核心逻辑,零风险。其次 **P1-D(eval 护栏)**,它是后面所有能力跃迁的度量基石。
