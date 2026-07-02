"""固定的端到端 eval 测试用例。

每个 case 描述一个真实用户场景，定义评分维度。
不要硬编码"正确答案"——LLM judge 根据 criteria 主观打分。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    case_id: str
    description: str
    user_id: str
    query: str
    # 可选：模拟的对话历史
    history: list[dict[str, str]] = field(default_factory=list)
    # 可选：预先准备的状态（在 setup 里跑）
    setup_actions: list[dict] = field(default_factory=list)
    # 评分维度（0-5 分），LLM judge 根据这些维度评估 agent 回复
    criteria: list[str] = field(default_factory=list)
    # 必须出现的关键词（如果 agent 漏了，扣分明显）
    must_mention: list[str] = field(default_factory=list)
    # 不应该出现的（幻觉/兜底回答检测）
    must_not_mention: list[str] = field(default_factory=list)
    # 预期路由意图（intent_hit 回归指标用，确定性，不依赖 LLM judge）。None = 不检查。
    expected_intent: str | None = None
    # 若设置，则要求最终推荐里 local 来源占比不得超过该值。
    max_local_ratio: float | None = None


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="recommend_basic",
        description="新用户首次请求推荐（无历史、无品味档案）",
        user_id="eval_new_user",
        query="给我推荐几首适合工作时听的歌",
        criteria=[
            "是否给出了具体的歌曲推荐（不是抽象描述）",
            "是否说明了推荐理由",
            "推荐是否符合'工作时听'的场景（轻松/专注/低人声）",
        ],
        must_not_mention=["LLM 请求失败", "暂无推荐", "无法识别"],
    ),
    EvalCase(
        case_id="recommend_with_taste",
        description="有听歌历史的用户请求推荐",
        user_id="eval_taste_user",
        query="今天心情有点低落，给我推荐些歌",
        setup_actions=[
            {"type": "listen", "asset_id": "a_seed1", "duration": 200, "completed": True},
            {"type": "rate", "asset_id": "a_seed1", "score": 5.0},
        ],
        criteria=[
            "是否表达了对'低落心情'的回应/共情",
            "推荐是否与情绪相关（治愈/温暖/舒缓）",
            "是否引用了用户的偏好或品味",
        ],
    ),
    EvalCase(
        case_id="playlist_specific",
        description="明确的歌单生成指令",
        user_id="eval_pl_user",
        query="帮我做一个跑步用的高能量歌单",
        criteria=[
            "是否生成了歌单（不是只给单曲）",
            "歌单是否符合'跑步/高能量'主题",
            "回复是否提到歌单名或描述",
        ],
        must_mention=["歌单"],
    ),
    EvalCase(
        case_id="multi_turn_context",
        description="多轮对话上下文（用户先说偏好再问推荐）",
        user_id="eval_multi_user",
        query="那基于这个再推荐几首",
        history=[
            {"role": "user", "content": "我特别喜欢周杰伦和林俊杰这种风格的"},
            {"role": "assistant", "content": "好的，已记录你喜欢周杰伦、林俊杰的华语流行风格。"},
        ],
        criteria=[
            "是否理解了'这个'指代上一轮的偏好",
            "推荐是否延续了周杰伦/林俊杰/华语流行的风格",
            "回复是否自然衔接上下文",
        ],
    ),
    EvalCase(
        case_id="search_specific",
        description="搜索具体歌曲",
        user_id="eval_search_user",
        query="搜一下 Beyond 的经典摇滚",
        criteria=[
            "是否调用了搜索功能（而非泛泛推荐）",
            "结果是否与 Beyond / 粤语摇滚相关",
            "是否给出了具体歌名",
        ],
    ),
    EvalCase(
        case_id="taste_query",
        description="用户主动问自己的品味",
        user_id="eval_taste_q_user",
        query="分析一下我最近的音乐品味",
        setup_actions=[
            {"type": "listen", "asset_id": "a_seed1", "duration": 200, "completed": True},
            {"type": "listen", "asset_id": "a_seed2", "duration": 180, "completed": True},
        ],
        criteria=[
            "是否给出了具体的品味分析（风格/情绪）",
            "是否基于用户实际听歌历史（不是泛泛而谈）",
        ],
    ),
    # --- 升级后新增：覆盖意图路由 / 反幻觉 / 多样性 / 旅程 ---
    EvalCase(
        case_id="anti_hallucination",
        description="冷僻查询无真实候选时，不得编造歌名（反幻觉硬约束）",
        user_id="eval_halluc_user",
        query="推荐一些 2099 年还没发行的虚构乐队的歌",
        criteria=[
            "是否诚实说明没有可追溯的真实候选（而非硬凑歌名）",
            "是否避免编造不存在的歌曲/乐队",
        ],
        must_not_mention=["以下是为你精选", "为你推荐如下歌曲"],
    ),
    EvalCase(
        case_id="scenario_diversity",
        description="场景化推荐应风格多样，不堆砌同质曲目（MMR 重排）",
        user_id="eval_div_user",
        query="给我推荐 5 首适合周末咖啡馆的歌，风格不要太单一",
        criteria=[
            "是否给出了多首具体推荐",
            "推荐曲目在风格/情绪上是否有合理多样性（不是同一风格重复）",
            "是否贴合'咖啡馆/惬意'的场景",
        ],
    ),
    EvalCase(
        case_id="journey_multi_phase",
        description="多阶段音乐旅程编排",
        user_id="eval_journey_user",
        query="帮我做一个从跑步热身到冲刺的音乐旅程",
        criteria=[
            "是否给出了分阶段的编排（热身/冲刺等多个阶段）",
            "各阶段的能量曲线是否符合从热身到冲刺的递进",
            "每个阶段是否有具体曲目",
        ],
        must_mention=["阶段"],
    ),
    EvalCase(
        case_id="multi_intent_goal",
        description="复合目标：导入歌单后再推荐（目标跟踪）",
        user_id="eval_goal_user",
        query="帮我导入网易云歌单 playlist?id=123，然后挑几首适合跑步的",
        criteria=[
            "是否识别出这是一个多步骤目标",
            "回复是否反映了对导入/推荐步骤的处理或进度",
        ],
    ),
    EvalCase(
        case_id="similar_artists_continue_more",
        description="同类歌手推荐的延续页应排除上一轮已展示歌手",
        user_id="eval_similar_artist_user",
        query="类似歌手再来一点",
        setup_actions=[
            {"type": "seed_similar_library"},
            {"type": "chat", "query": "推荐和 Drake、Future、Molly Santana 类似的歌手"},
        ],
        criteria=[
            "是否理解这是上一轮 similar_artists 的延续请求",
            "第二轮是否给出新的相似歌手，而不是重复上一轮前排结果",
            "推荐依据是否来自曲库标签而非凭空列名单",
        ],
        must_not_mention=["A$AP Rocky", "BROCKHAMPTON", "Don Toliver", "Fetty Wap"],
    ),
    EvalCase(
        case_id="recommend_no_local_preference",
        description="用户明确不要本地库时，推荐应保留约束而不是在 query rewrite 后失效",
        user_id="eval_no_local_user",
        query="推荐几首适合放松的歌，不要本地库里的",
        setup_actions=[
            {"type": "seed_local_library"},
        ],
        criteria=[
            "是否保留了'不要本地库'这一约束，而不是只保留场景词",
            "最终推荐是否主要来自线上结果而非本地库",
            "若线上不足，是否诚实变少而不是拿本地补齐",
        ],
        must_not_mention=["Local Seed 1", "Local Seed 2", "Local Seed 3"],
        max_local_ratio=0.0,
    ),
    EvalCase(
        case_id="recommend_english_no_chinese",
        description="用户要求英文歌且明确不要中文时，推荐不应漏出本地中文曲目",
        user_id="eval_en_no_zh_user",
        query="推荐几首适合放松的英文歌，不要中文",
        setup_actions=[
            {"type": "seed_mixed_language_local_library"},
        ],
        criteria=[
            "是否保留了英文歌/不要中文这一语言约束",
            "最终推荐是否没有中文歌曲漏出，尤其是不应从本地库带出中文歌",
            "即便候选变少，也应优先诚实满足语言约束",
        ],
        must_not_mention=["七里香", "晴天"],
    ),
    EvalCase(
        case_id="recommend_pure_discovery_paraphrase",
        description="'纯发现新的'同义改写——应路由到 recommend 且不带本地偏好（paraphrase：换说法不丢意图）",
        user_id="eval_no_local_user",
        query="别用我库里的，给我纯发现的新歌",
        setup_actions=[
            {"type": "seed_local_library"},
        ],
        criteria=[
            "是否识别为'不要本地库/纯发现'意图，而非把'别用'误解为排除关键词",
            "最终推荐是否主要来自线上结果",
        ],
        must_not_mention=["Local Seed 1", "Local Seed 2"],
        max_local_ratio=0.0,
    ),
    EvalCase(
        case_id="recommend_no_library_paraphrase",
        description="'别推荐我听过的'同义改写——应路由到 recommend 且尊重去重约束（paraphrase）",
        user_id="eval_no_local_user",
        query="别推荐我库里已经有的，来点新鲜的",
        setup_actions=[
            {"type": "seed_local_library"},
        ],
        criteria=[
            "是否保留'不要库里已有'的约束",
            "最终推荐是否避开本地种子曲目",
        ],
        must_not_mention=["Local Seed 1"],
        max_local_ratio=0.0,
    ),
    EvalCase(
        case_id="fact_check_non_hallucination",
        description="音乐事实核验：证据不足时应标不确定，不得强行断言",
        user_id="eval_fact_check_user",
        query="Frank Ocean 的 Blonde 是 2016 年的专辑吗，帮我核验一下",
        criteria=[
            "是否明确这是核验型回答，而不是泛泛介绍",
            "是否在证据不足时保守表达，而不是强行肯定或否定",
            "是否给出了可追溯来源或核验理由",
        ],
        expected_intent="music_fact_check",
    ),
    EvalCase(
        case_id="recommend_explainer_grounded",
        description="解释上一轮推荐依据，应 grounded 到真实推荐与用户口味",
        user_id="eval_rec_explain_user",
        query="为什么推荐这些歌",
        setup_actions=[
            {"type": "seed_local_library"},
            {"type": "chat", "query": "给我推荐几首适合放松的歌"},
        ],
        criteria=[
            "是否明确解释上一轮推荐依据",
            "是否引用用户口味、场景或推荐理由，而不是空泛夸赞",
            "是否保持 grounded，不编造不存在的推荐来源",
        ],
        expected_intent="recommend_explainer",
    ),
    EvalCase(
        case_id="multi_intent_recommend_plus_deep_dive",
        description="双意图并行：一句话既要推荐又要讲解艺人（需 ENABLE_MULTI_INTENT=true）",
        user_id="eval_multi_intent_user",
        query="推几首 The Weeknd，顺便讲讲他的风格",
        criteria=[
            "是否同时给出了具体的歌曲推荐（编号曲目清单）",
            "是否同时给出了对 The Weeknd 风格/生涯的讲解",
            "两段内容是否都真实 grounded，不编造",
        ],
        must_mention=["The Weeknd"],
        must_not_mention=["LLM 请求失败", "无法识别"],
    ),
]


# 给每个 case 标注预期路由意图（intent_hit 回归指标，确定性，不依赖 LLM judge）。
# 命中判据：expected_intent 作为子串出现在 agent_trace 里。
_CASE_EXPECTED_INTENT = {
    "recommend_basic": "recommend",
    "recommend_with_taste": "recommend",
    "playlist_specific": "playlist",
    "multi_turn_context": "recommend",
    "search_specific": "search",
    "taste_query": "taste",
    "anti_hallucination": "recommend",
    "scenario_diversity": "recommend",
    "journey_multi_phase": "journey",
    "multi_intent_goal": "import",
    "similar_artists_continue_more": "similar_artists",
    "recommend_no_local_preference": "recommend",
    "recommend_english_no_chinese": "recommend",
    "recommend_pure_discovery_paraphrase": "recommend",
    "recommend_no_library_paraphrase": "recommend",
    "fact_check_non_hallucination": "music_fact_check",
    "recommend_explainer_grounded": "recommend_explainer",
    # 多意图：primary 恒为 track 类（recommend），trace 里 [plan] 行会带 primary 意图。
    "multi_intent_recommend_plus_deep_dive": "recommend",
}
for _c in EVAL_CASES:
    _c.expected_intent = _CASE_EXPECTED_INTENT.get(_c.case_id)
