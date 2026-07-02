# 多意图并行手动测试指南

前置条件：
- 后端启动：`ENABLE_MULTI_INTENT=true` 环境变量
- 前端启动：`npm run dev`（Vue 3 + 热更新）
- 网络通：能访问网易云、MusicBrainz、last.fm

## 测试场景

### 场景 1：推荐 + 艺人介绍（最常见双意图）

**输入**：
```
推几首 The Weeknd，顺便讲讲他的风格
```

**期望输出**（同一条 message）：
- 📋 上半段：编号曲目清单
  ```
  为你推荐了 5-8 首 The Weeknd 的歌：
  1. 《Blinding Lights》 - The Weeknd（netease）
  2. 《Save Your Tears》 - The Weeknd（netease）
  ...
  ```
- 📚 下半段：艺人风格讲解（知识叙述）
  ```
  The Weeknd 以暗黑另类 R&B 起家，2010 年代早期通过 YouTube 免费发歌积累粉丝。
  他的音乐风格融合电子、说唱、灵魂乐，标志是虚伪、压抑的嗓音处理。
  代表作《Blinding Lights》（2019）打破多项流媒体纪录，是世纪最流行歌曲之一。
  ...
  ```
- 🎵 底部：song cards（从推荐的曲目生成）
- 📖 右侧或底部：dossier card（艺人档案卡片）

**检验点**：
- [ ] 同一条 message 里看到曲目清单 + 艺人讲解
- [ ] 两段都有真实内容，不是"正在加载..."
- [ ] song cards 数量与上面列表一致
- [ ] dossier 卡片能展开查看完整档案
- [ ] 后端 trace 显示 `[plan] recommend + artist_deep_dive` 或类似双意图信号

---

### 场景 2：搜索 + 专辑解读

**输入**：
```
找几首 Blonde 专辑的歌，再讲讲这张专辑为什么经典
```

**期望输出**：
- 📋 上半段：Blonde 专辑内的歌曲清单
  ```
  为你找到 4 首来自《Blonde》的歌：
  1. 《Nights》 - Frank Ocean（netease）
  2. 《Pink + White》 - Frank Ocean（netease）
  ...
  ```
- 📚 下半段：专辑解读
  ```
  《Blonde》是 Frank Ocean 2016 年的第二张录音室专辑，标志他从 R&B 转向实验性流行。
  这张专辑因其...被评为 2010 年代最伟大专辑之一。
  听法建议：先听《Nights》体验编排，再按顺序完整听整张。
  ...
  ```
- 🎵 底部：song cards + dossier card（专辑档案）

**检验点**：
- [ ] 曲目都来自 Blonde（不是 Frank Ocean 其他专辑）
- [ ] dossier 讲的是专辑本身，不是混淆成歌手传记
- [ ] 两段都有真实乐评信息（不重复、互补）

---

### 场景 3：搜索 + 歌手背景（西方歌手百科）

**输入**：
```
找找 Adele 的歌，顺便介绍她的背景
```

**期望输出**：
- 📋 上半段：搜索到的 Adele 歌曲
  ```
  为你找到 6 首 Adele 的歌：
  1. 《Hello》 - Adele（netease）
  2. 《Someone Like You》 - Adele（netease）
  ...
  ```
- 📚 下半段：歌手百科
  ```
  Adele Laurie Blue Adkins（1988 年生于伦敦）是英国流行歌手兼词曲作家。
  她因其深沉的女中音和情感化的歌词获得过 15 次格莱美奖。
  代表作《Hello》（2015）创造单周流媒体播放纪录。
  ...
  ```
- 参考来源：底部可能显示 Wikipedia / 新闻源 URL

**检验点**：
- [ ] 上半段是歌曲搜索结果
- [ ] 下半段是百科信息（不是乐评、不是音乐风格讲解）
- [ ] 两段各自完整，有清晰分隔（通常 `\n\n`）

---

### 场景 4：单意图控制（验证 flag 不破坏今天的行为）

**输入**（纯单意图）：
```
推荐几首适合学习的歌
```

**期望输出**：
- 单独的推荐答案 + song cards
- 不出现双意图、不出现 dossier 卡片
- 日志 `[plan] recommend`（无第二意图）

**检验点**：
- [ ] 与 `ENABLE_MULTI_INTENT=false` 时的输出完全一样
- [ ] 后端 trace 显示 `sub_plans: []`

---

### 场景 5：非白名单组合（应该回退单意图）

**输入**：
```
推个歌单 The Weeknd，再推荐几首
```
*（recommend + playlist，不在白名单）*

**期望输出**：
- 单意图（歌单生成），不分两段
- 或后端 trace 显示 `secondary=None`（被丢弃了）

**检验点**：
- [ ] 不出现双意图现象
- [ ] 单纯生成一个歌单

---

### 场景 6：知识 primary + 知识 secondary（不支持，应回退）

**输入**：
```
讲讲 Blonde 这张专辑，再比较一下 Channel Orange
```
*（album_deep_dive + music_compare，都是知识类，不在白名单）*

**期望输出**：
- 单意图（可能只讲 Blonde，或只做对比）
- 不出现双段落

**检验点**：
- [ ] 后端自动舍弃 secondary，单意图处理

---

## 调试技巧

### 1. 查看后端 trace

打开浏览器 DevTools，看 SSE 流：
```
[plan] recommend + artist_deep_dive（预算 15s）
[load_context] 载入记忆、目标和资源库摘要。
...
[final] 输出 grounded answer。
```

搜索关键词：
- `sub_plans` — 多意图标志
- `secondary` — 第二意图名
- `_merge_multi_intent_stages` — stage 合并日志（可选）

### 2. 检查前端 render

Vue DevTools → ChatTab.vue → `messages` 数组，看单条 message 的 `payload`：
```json
{
  "cards": [...],           // song cards
  "dossier": {...},         // 知识档案
  "trace_summary": "..."
}
```

两个都非空 = 双意图成功。

### 3. 测试网络情况

- 知识链路依赖 MusicBrainz/last.fm（偶尔慢或超时）
- 若 dossier 出现"资料不完整"或"降级"，可能是网络延迟 > knowledge_turn_budget_seconds（默认 50s）
- 检查 trace 里 `deadline_at`

### 4. 本地离线测试（无网络）

```bash
pytest tests/test_multi_intent.py -v
```

所有 17 例无网络依赖，验证核心逻辑。

---

## 常见问题排查

| 现象 | 可能原因 | 解决 |
|------|--------|-----|
| 后端不出现 `secondary` | flag 关闭或 pair 不在白名单 | 检查 env 或 intents.py 白名单 |
| 出现两段但是重复 | composer 逻辑错误 | 看 trace 里是否调了 _compose_multi_intent_stream |
| dossier 正文被 guard 删掉了 | guard 逻辑错 | 看 _skip_guard 判断，应该含知识时 true |
| 只出现 cards，没有 dossier | 知识工具失败 | 检查 build_music_dossier 工具日志 |
| SSE 卡死（长时间无新内容） | 知识链路超时 | 检查 deadline_at 距离当前时间 |

---

## 性能检查

**期望的 wall-clock 时间**：
- 纯 recommend（单意图）：2–4s
- recommend + 知识（双意图）：6–12s（两链并行，知识链较长）

若实际超过 15s，check：
- knowledge_turn_budget_seconds 是否过小
- 网络延迟是否过高
- 本地库大小是否导致 embedding/rerank 慢

在后端 trace 里看 `[meta]` 行记录的 metrics，对比 wall-clock 推理。

---

## 提交反馈

若遇到预期外行为，收集：
1. 用户输入 query
2. 后端完整 trace（包含所有 `[...]` 行）
3. 前端 final payload（JSON）
4. 期望 vs 实际

Grep logs 查 `enable_multi_intent`，确认 flag 状态。
