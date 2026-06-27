# MusicAgent 修复计划

> 基于面试官视角复盘产出。核心判断：**agent 设计思路不差，缺的是工程纪律与可度量性**。
> 原则：先让测试可信 → 再度量 → 再重构 → 再提智能。**顺序不可乱**（否则重构没安全网）。
> 每个阶段独立可交付、可回滚、有明确验收。

---

## 现状基线（一句话）

`agent.py` 4647 行 / 128 方法（上帝对象）｜17 个 flaky 测试（隔离跑全过，依赖真网络+embedding）｜**0 推荐质量度量**｜DeepSeek 强耦合（8 处 reasoning_content 特判）｜启发式层厚且中文硬编码。

---

## 阶段总览

| 阶段 | 主题 | 目标 | 工时 | 依赖 | 价值 |
|------|------|------|------|------|------|
| **P0** | 止血：测试可信 | `pytest` 纯离线、确定、可重复 | 1–2 天 | 无 | 解锁后面一切 |
| **P1** | 度量：推荐 eval | 离线 golden set + precision@k/假歌率 | 2–3 天 | P0 | 改动可量化 |
| **P2** | 结构：拆上帝对象 | agent.py < 1000 行，服务化 | ~1 周 | P0 | 可维护/可测 |
| **P3** | 解耦：LLM provider | reasoning_content 不出 provider | 2 天 | P0 | 换模型/降级 |
| **P4** | 智能：reflect+预算 | 多步恢复 + 全局延迟预算 | ~1 周 | P1/P2 | 体验/健壮 |
| **P5** | 可调：配置归一 | 魔法数进 config + ablation | 1–2 天 | P1 | 调参有据 |

**关键路径**：P0 → P1 → P2（P3/P5 可并行插队）。P4 必须在 P1 之后（否则没法度量 reflect 改进的效果）。

---

## P0 — 让测试套件可信（止血，最先做）✅ 已完成 2026-06-27

**实测根因（三类 flake，全部已修）**：
1. **全局 `random` 未 seed**（总根因）：`library.sample_ts_scores` 的 `random.betavariate`（TS 探索，`enable_explore` 默认开）+ `MockLLM` 5 处 `random.choice` 让推荐排序/mock 输出随用例顺序漂移。→ `conftest._seed_random` 每用例 `random.seed(固定)`。
2. **资源库隔离路径写错**（`agent.py:104`）：`Path(store.root).parent/resource_library.sqlite` 让所有 `JsonStore(tempfile.mkdtemp())` 测试共享同一个 `/tmp/resource_library.sqlite`，别家入库的「夜曲(钢琴曲)」被 `_dense_library_fallback` 召回。→ 改成 `Path(store.root)/resource_library.sqlite`，真按测试隔离。
3. **知识 agent 测试打真网络**：`test_sample_*` 不 mock 就调 `web_search_info`/musicbrainz 查「Bound 2 采样溯源」。→ `@pytest.mark.network` + `--run-network` 开关 + conftest 兜底 mock `web_search_info`。

**已排除的误判**：网易云搜索早已在 conftest mock（非 flake 源）；mock `_query_plan` 本身确定；HF embedding 只慢不 flake。

**验收（已达成）**：`pytest -q` 离线、确定、连续两次结果完全一致（731 passed / 2 skipped / 0 failed），耗时从 ~157s 降到 ~55s。两个真联网集成测试用 `pytest --run-network` 按需跑。

**原计划动作（归档）**：

**问题**：`test_phase0_orchestration` / `test_recommended_tracks` / `test_search_modules` 隔离跑全过、合跑随机失败 → 依赖真 netease 网络 + HF embedding 加载 + 用例顺序。CI 不可信 = 没有回归门禁 = 每次改动都是盲改。

**动作（文件级）**：
1. `tests/conftest.py`：
   - 注册 `@pytest.mark.network` marker；默认 `--run-network` 关闭，标记的用例默认 skip。
   - 把 `fake_online_music_search`（已有的 autouse fixture）推广成「所有 netease 源都被 monkeypatch」的总闸，确保主套件零真实网络。
   - 加 `embeddings_available` 的 autouse monkeypatch → 主套件恒走「规则+source」降级路径（确定）。
2. 把打真网络的用例迁出：`test_recommended_tracks.py`、`test_search_modules.py::TestRecommendForQueryRoutes`、`test_phase0_orchestration` 里真联网的 case → 要么改用录制 fixture（`tests/fixtures/netease_*.json`），要么标 `network` 移出主套件。
3. 消除用例间隐式依赖：每个用例用独立 `tmp` JsonStore（部分已是），禁止共享全局 `_default_cookie` / 进程内缓存污染 → 每个用例 setUp 显式 reset。
4. embedding 模型只加载一次（session fixture），避免重复 load 导致耗时/竞态。
5. CI：`pytest -q`（主，必须 0 失败 5 连跑一致）+ `pytest -m network`（可选，允许网络）。

**验收**：
- `pytest -q` 完全离线、断网也能跑、连续 5 次结果完全一致、0 failed。
- `pytest -m network` 单独跑那批联网用例。
- 任何一条用例 `pytest -p no:randomly` 与默认顺序结果相同。

**风险/回滚**：把用例标 network 可能让覆盖率「看起来」下降——记录迁出清单，不算回归。回滚 = 去掉 marker。

---

## P1 — 推荐质量离线 eval（让改动可量化）

**问题**：核心价值是「推荐好」，但拿不出数据。所有权重/delta（local_ratio、rerank ±、MMR λ）靠直觉调，无 ablation。

**进展（2026-06-27）**：
1. `tests/eval/regress.py` 已改为默认离线确定：复用 `tests/offline_fakes.py`，默认关闭真实 LLM、真实外部源、embedding 加载；只有显式 `--online` / `--with-llm-judge` 才放开。
2. `tests/conftest.py` 与 eval 共用同一套 agent 在线入口 fake，但保留搜索模块专项测试对真实 pipeline 的覆盖，避免 fake 过度遮蔽。
3. `tests/eval/cases.py` 新增 `similar_artists_continue_more`、`recommend_no_local_preference` 与 `recommend_english_no_chinese`，分别覆盖“同类歌手翻页去重”“不要本地库”和“英文歌且不要中文时本地中文曲目漏出”三类长期使用场景。
4. 已用 eval 定位并修复四个质量缺口：`playlist_specific` 因 `target_count=None` 导致 playlist 工具参数校验失败，已改为未指定数量时不传该字段；`journey_multi_phase` 的确定性回答已显式输出“阶段 1/2/3”；`recommend` 工具现已区分“用户原句 query”和“用于召回的 search_query”，避免 query rewrite 吃掉“不要本地库里”的约束；同一出口现在也会重新应用原始 query 的语言否定约束，防止“推荐几首适合放松的英文歌，不要中文”仍漏出本地中文歌。
5. `tests/eval/metrics.py` 已新增 `junk_rate` 与 `local_ratio`，报告和 baseline 可同时观察 `intent_hit_rate`、`anti_halluc_rate`、`avg_junk_rate`、`avg_local_ratio`、`local_ratio_pass_rate`、`avg_must_mention_hit`、`avg_diversity`。
6. `tests/eval/layers.py` 现已把 `content_negation_accuracy` 纳入层级 eval，与 `hygiene_pass_rate`、`local_ratio_accuracy` 一起做独立回归，避免“不要中文/越南/日语”这类硬排除项以后悄悄退化。
7. 单轮 `query_plan` 现在也会把 `extract_content_negations(query)` 写入 `retrieval_plan.excluded_terms`，不再只有延续对话才保留硬排除项，约束传播在单轮/多轮路径上已对齐。
8. `tests/eval/baseline.json` 已刷新为 13 条 case 的当前离线基线；当前 `avg_must_mention_hit=1.0`、`avg_junk_rate=0.0`、`local_ratio_pass_rate=1.0`。`tests/eval/layers_baseline.json` 也已同步到当前离线模式，并新增 `content_negation_accuracy=1.0`。

**已验收**：
- `python -m tests.eval.regress` 离线、无 HuggingFace/网易云真实调用、连续输出稳定。
- `pytest -q`：`757 passed, 2 skipped`。

**动作**：
1. 新建 `eval/` 包：
   - `eval/golden.json`：30–50 条固定 query，每条带期望——`must_include_artists`、`must_exclude`（假歌/氛围翻唱/教程）、`intent`、`expect_local_ratio_band`、`language`。覆盖：实体搜索 / 场景 / 续续 / 否定 / 冷启动 / 假歌陷阱。
   - `eval/metrics.py`：precision@k、junk_rate（must_exclude 命中率）、intent_accuracy、novelty、diversity、coverage、cold_start_hit。
   - `eval/run_recommend.py`：用 P0 的 fixture 跑 `recommend_for_query`（离线、确定），输出报告 + 写 `eval/baseline.json`。
2. 落一条基线 `eval/baseline.json` 入库；之后每次改 rerank/hygiene/profile 都跑一次，diff 基线。
3. CI：先「非门禁报告」（贴 delta），稳定后「门禁回归」（指标退化则 fail）。

**验收**：
- `python -m eval.run_recommend` 产出确定报告（离线）。
- 至少有 precision@k + junk_rate 两个指标。
- 你能回答「把 avoid 减分从 -0.12 改成 -0.08，precision@k 变多少」。

**风险**：golden set 主观——先小（30 条）再长。冷启动 case 用空 memory 构造。

---

## P2 — 拆解上帝对象（结构）

**问题**：`AudioVisualAgent` 4647 行/128 方法，ingest+recommend+rerank+lyrics+knowledge+profile+playlist+search+memory+cookie 全在一个类。`nodes.py` 2564 行、`handlers.py` 904 行。

**起步进展（2026-06-27）**：
1. 已先把推荐来源平衡逻辑抽到 `app/recommend/source_balance.py`，`agent.py` 仅保留薄包装；这是推荐链路里第一块成功外移、且经全量测试验证的纯函数模块。
2. 已新增 `app/recommend/service.py`，把“本地候选挑选 / 推荐历史落盘 / 搜索词是否含实体”三块逻辑从 `AudioVisualAgent` 中迁出；`agent.py` 对应方法现为轻量委托。
3. 为兼容历史测试里大量 `AudioVisualAgent.__new__` 的轻量构造方式，agent 侧增加了 `RecommendationService` 懒初始化包装，避免把构造期副作用硬编码成迁移前提。
4. 推荐主链路的后处理流水线也已进一步迁入 `RecommendationService`：候选验证、跨轮去重、线上 fallback、资源池补位、fresh/repeated 优先级都不再直接写在 `agent.py` 的长函数体里，而是由 service 负责。
5. 推荐主链路前半段也已开始拆出：`exact/anchor route` 与 `playlist/discovery route` 的候选收集逻辑已迁入 `RecommendationService`，`agent.py` 只保留上下文准备与最终编排。
6. rerank 相关逻辑现也已迁入 `RecommendationService`：`_rerank_tracks`、`_collaborative_scores`、`_enrich_candidate_tags` 在 agent 侧都只剩兼容薄包装，三锚精排与协同锚计算不再直接耦合在 `agent.py`。
7. 迁移过程中顺手修复了一条“全局 FastAPI app + 资源库副作用”相关测试的隐性不稳定点：`tests/test_graph_resource_library.py` 现在显式 seed 资源库，不再依赖前序测试或真实网络副作用才能通过。
8. 推荐上下文准备现也已迁入 `RecommendationService`：`memory_query / search_goal / anchors / scene_queries / has_entity / taste_summary / library_artists / local_ratio / seed_supply` 已由 service 统一构建，`agent.py` 不再手动拼装这些前置变量。
9. 到这一步，`recommend_for_query` 已从“全链路细节都写在一个长函数里”收缩到“service context + service 调用 + 最终组装返回”，推荐链路的 P2 拆分目标已基本达成；后续若继续拆，更像是横向整理其它能力（playlist / journey / search），而不是继续在推荐链里硬拆。
10. `app/playlist_service.py` 已落地并接入：`generate_playlist`、`auto_playlists`、`save/list/delete_playlist`、`_playlist_candidates`、`_fallback_playlist`、`_fallback_auto_playlists` 都已从 `AudioVisualAgent` 抽离，agent 侧只保留兼容薄包装与懒初始化入口。
11. 这次 playlist 拆分后已完成针对性和全量回归：`tests/test_daily_recommend.py`、`tests/test_playlist_count_consistency.py`、`tests/test_api.py`、`tests/test_graph_resource_library.py` 全绿，随后 `pytest -q` 维持 `757 passed, 2 skipped`，离线 `python -m tests.eval.regress` 13/13 case 继续通过。
12. `app/search_service.py` 也已接入：`search_web_music`、`search_web_music_async`、`_dense_library_fallback`、`_lexical_resource_fallback` 已从 agent 中外移，在线搜歌与资源库候选回退不再塞在 `AudioVisualAgent` 的单个长方法里。
13. search 拆分过程中额外补齐了两类兼容约束：一是避免 `self.search` 属性名覆盖原有 `search()` 方法；二是保留 async 回退路径对 `agent.search_web_music` monkeypatch 的可观测行为，确保老测试和外部调用约定不被 service 化悄悄改掉。
14. `app/taste_experiment_service.py` 已落地并接入：`generate/list/get/delete/report/regenerate_taste_experiment` 与候选汇总逻辑都已脱离 `AudioVisualAgent`，agent 侧主要剩兼容 helper、评分分桶与反馈回流的薄包装。
15. `app/journey_service.py` 也已接入：`generate_music_journey` 和 `_record_journey_history` 现已外移，旅程阶段召回、轮换去重、重排与序列化不再写在 agent 长函数体里。
16. 这两次拆分后补跑了 `tests/test_taste_experiment.py`、`tests/test_stream.py`、`tests/test_graph_resource_library.py`、`tests/test_phase0_orchestration.py`、`tests/test_api.py` 等专项，再跑全量 `pytest -q` 维持 `757 passed, 2 skipped`，离线 `python -m tests.eval.regress` 13/13 case 继续通过。
17. `app/discover_service.py` 现已接入：`search`、`search_videos`、`search_videos_async`、`search_artist_info`、`search_artist_info_async`、`classify_discover_query` 都已从 agent 抽离，Discover 页面的本地检索、线上补召回、视频搜索与歌手档案分类不再混在一个大对象里。
18. discover/search 拆分时继续保留了两类兼容契约：一是 async 视频搜索失败时仍会通过 `agent.search_videos` 的同步回退口降级，二是 `search()` 仍保留原始方法签名与返回结构，所以上层 handler、streaming graph 和 monkeypatch 型测试都无需改写。
19. 本轮又补跑了 `tests/test_search_modules.py`、`tests/test_async_music_source.py`、`tests/test_daily_recommend.py`、`tests/test_api.py`、`tests/test_stream.py` 等专项，再跑全量 `pytest -q` 维持 `757 passed, 2 skipped`，离线 `python -m tests.eval.regress` 13/13 case 继续通过。
20. `app/playback_service.py` 现已接入：`get_playback_url`、`get_audio_url`、`get_mv_url`、`get_lyrics` 与平台级 helper（YouTube/B 站/网易云解析与取流）都已脱离 agent，播放链路与歌词链路不再散落在 `AudioVisualAgent` 中部。
21. playback/lyrics 拆分时额外保留了实例级 monkeypatch 兼容：对 `agent._search_netease`、`agent._get_netease_audio_url` 的测试覆写仍可生效，因此 `SimpleNamespace` 鸭子类型取流和老的 Web 播放测试不用跟着重写。
22. 本轮补跑了 `tests/test_source_and_playback.py`、`tests/test_web_routes.py`、`tests/test_api.py` 以及更大范围的 `tests/test_async_music_source.py`、`tests/test_search_modules.py`、`tests/test_daily_recommend.py`、`tests/test_stream.py`，再跑全量 `pytest -q` 维持 `757 passed, 2 skipped`，离线 `python -m tests.eval.regress` 13/13 case 继续通过。
23. `app/catalog_service.py` 现已接入：`fetch_track_metadata`、`recommend_artist_albums`、`recommend_artist_albums_async` 已从 agent 中移出，专辑卡片推荐与元数据抓取不再夹在 taste/profile/recommend 逻辑之间。
24. catalog 拆分时保留了 async 专辑推荐的同步回退口，因此 `recommend_artist_albums_async` 在源异常时仍会退回 `agent.recommend_artist_albums(...)` 的既有契约；同时 `fetch_track_metadata(url=...)` 仍保留对泛化标题的过滤，不会因为 service 化悄悄放宽 `found` 判定。
25. 本轮补跑了 `tests/test_agent_flow.py`、`tests/test_api.py`、`tests/test_visual_result_contract.py`、`tests/test_source_and_playback.py` 等专项，再跑全量 `pytest -q` 维持 `757 passed, 2 skipped`，离线 `python -m tests.eval.regress` 13/13 case 继续通过。
26. 结构收益继续扩大：`app/agent.py` 当前已从 4600+ 行进一步收缩到约 3360 行；但离 `<1000` 的终态还有明显距离，下一步应优先继续处理 `phase/add_many` 相关图编排块，或者把知识链路入口与剩余 profile/memory 交叉逻辑进一步服务化，而不是回头把已拆出的 service 再做微观重写。
27. `app/discover_rules.py` 已正式落地并接管 Discover/playlist/journey 的一组纯规则 helper：搜索词抽取、场景歌单 query 改写、线上歌单扩展 query、内容变体放行、本地占比覆盖、旅程 phase 生成、搜索摘要格式化，以及 artist/query match 的规范化与模糊匹配逻辑都已从 `agent.py` 收口到独立模块。
28. 这次规则层抽离后，`app/agent.py` 已进一步缩到 `2864` 行，`app/discover_rules.py` 为 `449` 行；并已完成 `python -m compileall app/agent.py app/discover_rules.py app/playback_service.py`、`pytest -q tests/test_search_modules.py tests/test_async_music_source.py tests/test_daily_recommend.py tests/test_stream.py`（89 passed）、全量 `pytest -q`（`757 passed, 2 skipped`）以及 `python -m tests.eval.regress`（13/13 case 通过）的整套回归验证。
29. `app/track_rules.py` 已新增并接管通用曲目/候选 helper：metadata 可靠性判断、asset-context query 判断、playlist match score、`track_key`、推荐/线上验证、fallback 判定、候选类型分类、外部候选有效性过滤、线上候选 reason、去重、query 变体合并、延续去重过滤和补位填充都已从 `agent.py` 尾部迁出，供 recommendation/search/discover/catalog 等 service 统一复用。
30. 这次通用曲目工具抽离后，`app/agent.py` 已进一步缩到 `2624` 行，新增 `app/track_rules.py` 为 `236` 行；并已完成 `python -m compileall app/agent.py app/track_rules.py app/search_service.py app/catalog_service.py app/discover_service.py app/recommend/service.py`、`pytest -q tests/test_search_modules.py tests/test_candidate_quality.py tests/test_daily_recommend.py tests/test_async_music_source.py tests/test_stream.py`（120 passed）、全量 `pytest -q`（`757 passed, 2 skipped`）以及 `python -m tests.eval.regress`（13/13 case 通过）的整套回归验证。
31. `app/recommend_rules.py` 已新增并接管推荐链路专用的纯规则 helper：网易云 song id 提取、歌单数量推断、时间段文案、`RecommendationAnchors`、anchor/seeds 构造、推荐质量门禁、场景推荐识别、functional-audio 识别，以及 playlist scene 兼容性判断都已从 `agent.py` 迁出，推荐规则层现在与 agent 编排层分离得更清楚。
32. 这次推荐规则抽离后，`app/agent.py` 已进一步缩到 `2258` 行，新增 `app/recommend_rules.py` 为 `364` 行；并已完成 `python -m compileall app/agent.py app/recommend_rules.py`、`pytest -q tests/test_search_modules.py tests/test_candidate_quality.py tests/test_daily_recommend.py tests/test_stream.py tests/test_async_music_source.py`（120 passed）、全量 `pytest -q`（`757 passed, 2 skipped`）以及 `python -m tests.eval.regress`（13/13 case 通过）的整套回归验证。
33. `app/taste_experiment_rules.py` 已新增并接管 Taste Experiment 里的规则/helper 层：候选过滤、熟悉度计算、safe/stretch/bold 分桶、bucket 切片、候选 key、track key、实验内查找、TS 反馈回流、listening_history 回写、反馈计数、bucket 统计与 bucket 文案都已从 `agent.py` 的长方法体中抽离。
34. 这次品味实验规则抽离后，`app/agent.py` 已进一步缩到 `2137` 行，新增 `app/taste_experiment_rules.py` 为 `218` 行；并已完成 `python -m compileall app/agent.py app/taste_experiment_rules.py app/taste_experiment_service.py`、`pytest -q tests/test_taste_experiment.py tests/test_daily_recommend.py tests/test_stream.py tests/test_async_music_source.py tests/test_search_modules.py`（97 passed）、全量 `pytest -q`（`757 passed, 2 skipped`）以及 `python -m tests.eval.regress`（13/13 case 通过）的整套回归验证。
35. `app/library_service.py` 已落地并接管音乐库内容与资产生命周期：`ingest_video`/`enrich_asset`/`_fetch_video_title`/`_enrich_from_netease`/`_apply_title_artist_hint`/`_identify_from_url`/`analyze_media`/`_playlist_tags_to_genres`/`_batch_classify_tracks`/`_classify_once`/`_ensure_track_tags`/`import_netease_playlist`/`list_assets`/`_invalidate_assets_cache`/`delete_asset`/`clear_cache`/`cleanup_resource_library`/`list_resource_tracks` 以及类常量（`_VALID_GENRES`/`_NETEASE_TAG_TO_GENRE`/`_FALLBACK_*`）和进程内缓存状态（`_assets_cache`/`_caching_enabled`/`_assets_synced_dirty`）都已从 `AudioVisualAgent` 抽离，agent 侧只保留同名薄委托 + 构造末尾 `enable_cache()` 两阶段开关。`list_assets` 缓存是核心共享基础设施（被 recommendation/playlists/discover/journeys 等 6 个 service 通过 `list_assets=self.list_assets` 回调注入），故随库操作整体下沉、agent 保留薄委托，所有回调引用与外部 `agent.list_assets` 调用点零改动；`_apply_netease_cookie` 是横切（daily/search/recommend 三处都调），留在 agent 不进 LibraryService。
36. 兼容关键：`LibraryService.llm` 用动态 property（agent 注入 `llm_provider=lambda: self.llm`）而非构造期快照——测试 `monkeypatch agent.llm` 后 `_batch_classify_tracks`/`_classify_once`/`_enrich_from_netease`/`_identify_from_url` 委托链立即读到新 llm，保持搬家前行为，治住 `test_invalid_llm_genre_filtered` 的回归。这是"实现里用 `self.llm` 的 service 化"通用坑：必须动态取，不能构造期快照。
37. 这次库链路拆分后，`app/agent.py` 已从 `2137` 行缩到 `1739` 行，新增 `app/library_service.py` 为 `533` 行；并已完成 `import` 检查、`python -m compileall app/agent.py app/library_service.py`、库操作专项测试（`test_agent_flow`/`test_daily_recommend`/`test_similarity`/`test_memory_enhanced`/`test_api`/`test_auth_routes`/`test_visual_result_contract`/`test_search_modules`/`test_tool_registry`，131 passed）、全量 `pytest -q`（`757 passed, 2 skipped`）以及 `python -m tests.eval.regress`（13/13 case 通过）的整套回归验证。
38. 目录归一：散落在 app/ 顶层的 8 个 `*_service.py` + 子包内 `recommend/profile/tools/service.py`（共 11 个）统一收进 `app/services/`（短名 catalog/discover/journey/library/playback/playlist/search/taste_experiment/recommend/profile/tools），4 个 `*_rules.py` 收进 `app/rules/`（discover/recommend/taste_experiment/track）。service 内部都用绝对 import，迁址后内部 import 路径不变；外部调用点用 sed 全局改 `from-import`（15 条精确规则，覆盖 app/ tests/）。踩两坑：① `app/profile/__init__.py` 曾 re-export `UserProfileService`，service 迁出后形成 `__init__ → services.profile → profile.builder → __init__` 的循环（`from app.services.profile` 先触发时炸）→ 删掉该 re-export（无人用包级 `from app.profile import`，且 service 已属 services 层、不该由 profile 包 re-export）；② sed 只覆盖 `from … import`，漏了 `tests/test_profile_context.py` 的裸 `import app.profile.service as …`，手动改。app/ 顶层从 27 个 `.py` 降到 15 个，service/rules 各归一层；领域包 `recommend/profile/tools` 保留各自算法/契约模块（engine/rerank/builder/handlers 等），只把 service 抽走。全量 `pytest -q` 维持 `757 passed, 2 skipped`，eval 13/13。

**动作（增量、靠 P0 测试网兜底）**：
1. 抽 `RecommendationService`（`app/recommend/service.py`）：搬 `recommend_for_query`、`_rerank_tracks`、`_balance_recommendation_sources`、`daily_recommend`、路由 A–E、`_local_ratio_from_query`、`_dense_library_fallback`。agent 持有其引用。
2. 抽 `LibraryService`：`ingest_video`/`enrich_asset`/`analyze_media`/`list_assets`/`save_album`/`upsert_external` 编排。
3. `KnowledgeService`（knowledge.py 已较独立，理清依赖）、`LyricsService`、`PlaylistService`、`SearchService`。
4. `app/profile/` 已是独立模块——agent 只调 `UserProfileService`（这一步基本已到位）。
5. 把 `agent.py` 里 48 个模块级函数归位到对应 service 模块或 `recommend/pipeline.py`。
6. agent 类最终只留：构造各 service、`chat_async` 编排入口、cookie/用户态切换。

**验收**：
- `agent.py` < 1000 行，且不再含推荐/入库/歌词的具体实现。
- 每个 service 可独立 import、独立测。
- P0 套件全程绿（这是安全网）。

**风险/回滚**：纯搬家易引入微妙行为变化——**必须 P0 先绿**，每抽一个 service 跑一次全量。回滚 = 一个 service 一个 commit，可逐个 revert。

---

## P3 — LLM Provider 抽象

**问题**：`llm/client.py` 8 处 deepseek/reasoning_content 特判，provider 怪癖泄漏进客户端；换模型要改核心；agent 健壮性部分依赖「记得处理 content 为空」。

**动作**：
1. 定义 `LLMProvider` Protocol：`generate / agenerate / generate_stream / 结构化输出 / thinking 开关 / 内容字段归一化`。
2. `DeepSeekProvider` 实现，封装 reasoning_content 兜底 + thinking 默认关（现状逻辑搬进去）。
3. 其余代码只依赖 Protocol；配置选 provider。
4. （可选）加一个 stub/deterministic provider，让 P1 eval 完全不依赖真 LLM。

**验收**：`reasoning_content`/`deepseek` 字样不出现在 provider 实现之外；eval 可换 provider 跑。

**风险**：低，纯封装。注意流式语义差异。

---

## P4 — Agent 智能 & 健壮性

**问题**：reflect 硬上限「最多重试一次」（`nodes.py:1382`）；无 orchestrator 全局延迟预算（超时是逐工具打补丁）；启发式关键词换说法就漏。

**动作**：
1. **reflect 策略阶梯**：retry 上限 1→2~3，按「重搜正向词 → 补变体 → 切本地召回 → 诚实空」阶梯，每步带预算。
2. **全局延迟预算**：orchestrator 每轮墙钟上限 + 降级表（关 CF 锚 → 关 LLM 候选生成 → dense 召回 → 诚实空）。替换散落的 `_SEARCH_DEADLINE`。
3. **启发式可度量**：关键词表（intent / hygiene / local-ratio）做成数据驱动 + 双语 + **纳入 P1 golden set 的 paraphrase case**（「别用我库里的」「纯发现新的」）——换说法漏了会被 eval 抓到。

**验收**：
- p95 单轮延迟 ≤ 预算；超预算走降级而非卡死。
- eval 的 paraphrase 子集 intent_accuracy 达标。
- reflect 能从一个「被强制清空的首轮」恢复出候选。

**风险**：retry 上调会增加 LLM 调用/成本——必须和延迟预算一起上。

---

## P5 — 配置归一 & 可调

**问题**：魔法数散落（local_ratio 0.4/0.3、rerank +0.06/-0.12、MMR λ、各类 timeout/deadline）。

**动作**：
1. 全部归到 `app/config.py` 的带命名空间分组（`rerank.*` / `recommend.*` / `latency.*`）。
2. 在 P1 eval 里支持 ablation：跑多组参数 → 报告指标曲线。让 `-0.12` 这种数有证据支撑。

**验收**：调参只改 config；eval 能产出 ablation 表。

---

## Definition of Done（整体）

- [ ] `pytest -q` 离线、确定、0 failed、5 连一致。
- [ ] `eval/baseline.json` 入库，precision@k + junk_rate 可 diff。
- [ ] `agent.py` < 1000 行，服务化拆分完成。
- [ ] `reasoning_content` 不出 provider 实现。
- [ ] 单轮延迟有预算 + 降级表；超时不再卡死。
- [ ] 魔法数全进 config，关键参数有 ablation 依据。

---

## 建议起步

**从 P0 开始**（治 flaky）——它是后面所有阶段的安全网，且 1–2 天就能拿到「可以信的 `pytest`」这个质变。完成后再决定 P1（度量）还是 P2（拆对象）优先。
