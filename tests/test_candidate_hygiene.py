"""候选质量闸门 Candidate Quality Gate 单测。

对应线上脏/低质数据：教程、DJ串烧(南宁Dj阿聪)、车载DJ、车祸新闻、抖音高潮版。
验证 classify_candidate 的三态判定 + query-aware mix 放行 + filter_music_tracks 报告。
"""
from __future__ import annotations

from app.models import ExternalTrack
from app.recommend.hygiene import classify_candidate, filter_music_tracks, query_allows_mix


def _t(title: str, artist: str = "Demo", source: str = "netease", kind: str = "track") -> ExternalTrack:
    return ExternalTrack(external_id="x", title=title, artist=artist, source=source, candidate_kind=kind)


def test_tutorial_and_compilation_rejected():
    q = classify_candidate(_t("编曲技巧:怎么做一首流行R&B风格的音乐？", "音乐人小树", "bilibili"), "推荐几首流行歌")
    assert q.status == "reject"
    assert q.entity_type in {"tutorial", "program", "playlist"}

    q2 = classify_candidate(_t("2024热门歌曲合集", "某账号", "netease"), "推荐几首流行歌")
    assert q2.status == "reject"


def test_dj_mix_default_rejected_but_allowed_when_query_asks():
    track = _t("2024全网最火车载DJ串烧", "DJ", "netease")
    # 普通 query（不要 mix）→ 拒
    assert classify_candidate(track, "推荐几首适合深夜的歌").status == "reject"
    # query 明确要车载DJ → 放行（accept 或 maybe）
    q = classify_candidate(track, "给我来点车载DJ串烧")
    assert q.status in {"accept", "maybe"}
    assert query_allows_mix("给我来点车载DJ串烧") is True


def test_dj_in_artist_name_rejected_for_normal_query():
    """「南宁Dj阿聪」这类艺人名带 DJ 的低质候选，普通 query 下拒。"""
    q = classify_candidate(_t("全旋律说唱", "南宁Dj阿聪", "netease"), "推荐几首适合深夜的歌")
    assert q.status == "reject"
    assert q.entity_type == "dj_mix"


def test_normal_songs_accepted():
    for title, artist in [("Ditto", "NewJeans"), ("Firework", "Katy Perry"), ("Classic", "Drake")]:
        q = classify_candidate(_t(title, artist), "推荐几首歌")
        assert q.status == "accept", f"应 accept: {title} (got {q.status} {q.reasons})"


def test_mood_descriptor_suffix_rejected():
    """网易云用户上传的氛围/翻唱/助眠假歌：标题「曲名 - <氛围|男声|女声…>」后缀 → 拒。

    这类脏候选艺人栏是网名（小仓鼠要早睡），原本会靠 source 兜底（netease+artist=accept）放行；
    新规则挡在兜底之前。覆盖中文描述词与英文 slowed/8d 用户改版签名。
    """
    cases = [
        ("雨爱 - R&B氛围男声", "小仓鼠要早睡"),
        ("晴天 - 钢琴版", "翻唱账号"),
        ("某歌 - 助眠版", "网名用户"),
        ("Love - Slowed", "user"),
        ("Beat - 8D Audio", "user"),
    ]
    for title, artist in cases:
        q = classify_candidate(_t(title, artist), "推荐几首适合深夜的歌")
        assert q.status == "reject", f"应拒: {title} (got {q.status} {q.reasons})"
        assert "mood_descriptor_title" in q.reasons


def test_legit_dashed_titles_not_false_positive():
    """官方曲目的合法后缀（Live/Remix）与主标题含「氛围/深夜」的真歌不被误杀。"""
    legit = [
        ("Vampire - Live", "Olivia Rodrigo"),
        ("Blinding Lights - Remix", "The Weeknd"),
        ("深夜食堂", "铃木常吉"),     # 标题含「深夜」但无破折号后缀
        ("氛围感单曲", "某歌手"),     # 「氛围」在主标题、不在破折号描述段
    ]
    for title, artist in legit:
        q = classify_candidate(_t(title, artist), "推荐几首歌")
        assert q.status != "reject", f"误杀真歌: {title} (got {q.status} {q.reasons})"


def test_sentence_news_title_rejected():
    """句子/新闻型标题（。！？【）一律拒——车祸新闻、vlog。"""
    for title in ["女子深夜开车撞羊！【1901期】", "没想到深夜的抖音酒吧区这么炸裂，三男一女"]:
        # 后者无强标点但是 bilibili 长句
        src = "bilibili"
        q = classify_candidate(_t(title, "UP主", src), "推荐几首适合深夜的歌")
        assert q.status == "reject", f"应 reject: {title}"


def test_filter_music_tracks_report_counts():
    tracks = [
        _t("Ditto", "NewJeans"),
        _t("Firework", "Katy Perry"),
        _t("全旋律说唱", "南宁Dj阿聪"),
        _t("编曲技巧:怎么做R&B", "UP主", "bilibili"),
        _t("2024全网最火车载DJ串烧", "DJ"),
    ]
    accepted, report = filter_music_tracks(tracks, query="推荐几首适合深夜的歌", target_count=10)
    assert report.raw_count == 5
    assert report.accepted_count == 2  # Ditto, Firework
    assert report.rejected_count == 3
    assert "mix_not_allowed_by_query" in report.reasons or "hard_reject_pattern" in report.reasons
    assert len(report.rejected_examples) == 3
    # accepted 里不含脏数据
    assert {t.title for t in accepted} == {"Ditto", "Firework"}


def test_missing_title_or_video_without_artist_rejected():
    assert classify_candidate(_t("", "Artist"), "歌").status == "reject"
    assert classify_candidate(_t("某视频", "", "bilibili"), "歌").status == "reject"
