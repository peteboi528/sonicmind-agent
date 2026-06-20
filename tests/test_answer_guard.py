"""Answer Guard 测试：锁住反幻觉的核心行为。

guard_answer 必须剔除答案里追溯不到任何工具结果的歌名（《》包裹），
保留白名单内的真实曲目，且不误伤普通文本。
"""

from app.answer import guard_answer


def test_removes_hallucinated_titles():
    known = {"成都", "晚安"}
    answer = "为你推荐《成都》《撒野》《晚安》《千禧》，都很适合深夜听。"
    cleaned, removed = guard_answer(answer, known)
    assert "《撒野》" not in cleaned
    assert "《千禧》" not in cleaned
    assert "《成都》" in cleaned
    assert "《晚安》" in cleaned
    assert set(removed) == {"撒野", "千禧"}


def test_keeps_all_known_titles():
    known = {"成都", "晚安"}
    answer = "本地命中《成都》和《晚安》。"
    cleaned, removed = guard_answer(answer, known)
    assert removed == []
    assert "《成都》" in cleaned and "《晚安》" in cleaned


def test_empty_whitelist_strips_every_song():
    answer = "推荐《撒野》《千禧》《想你》。"
    cleaned, removed = guard_answer(answer, set())
    assert set(removed) == {"撒野", "千禧", "想你"}
    assert "《" not in cleaned


def test_no_false_positive_on_plain_text():
    answer = "这是一段没有任何歌名的普通回复。"
    cleaned, removed = guard_answer(answer, {"成都"})
    assert removed == []
    assert cleaned == answer


def test_tolerates_subtitle_variants():
    """真实标题带副标题/译名时，包含匹配应判为已知。"""
    known = {"晴天（Sunny Day）"}
    answer = "推荐《晴天》给你。"
    cleaned, removed = guard_answer(answer, known)
    assert removed == []
    assert "《晴天》" in cleaned


def test_keeps_playlist_names_in_book_marks():
    known = {"成都"}
    answer = "已生成歌单《深夜 Chill》：前几首包括《成都》和《不存在》。"
    cleaned, removed = guard_answer(answer, known)
    assert "歌单《深夜 Chill》" in cleaned
    assert "《成都》" in cleaned
    assert "《不存在》" not in cleaned
    assert removed == ["不存在"]


def test_removes_quoted_hallucination_without_bookmarks():
    """纵深防御：LLM 不用书名号、改用引号提编造歌名，也要拦下。"""
    known = {"Blinding Lights"}
    answer = '为你挑了 "Blinding Lights" 和 "Fake Song" 两首。'
    cleaned, removed = guard_answer(answer, known)
    assert "Fake Song" in removed
    assert "Blinding Lights" in cleaned
    assert "Fake Song" not in cleaned


def test_quoted_known_title_kept():
    known = {"Save Your Tears"}
    answer = 'The Weeknd 的 "Save Your Tears" 很适合。'
    cleaned, removed = guard_answer(answer, known)
    assert removed == []
    assert "Save Your Tears" in cleaned


def test_quoted_long_sentence_not_treated_as_song():
    """引号里是长句/带句末标点 → 不当歌名，避免误伤。"""
    known = set()
    answer = '他说"今天天气真好，我们出去走走吧。"然后就走了。'
    cleaned, removed = guard_answer(answer, known)
    assert removed == []
