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
]
