"""fetch_netease_lyrics_timed 解析契约：保留 LRC 时间戳、丢署名/元数据标签、多时间戳展开、升序。

网络层 monkeypatch 掉（离线、确定），只验纯解析逻辑。plain 版（fetch_netease_lyrics）
由 timed 版派生，一并覆盖。
"""
from __future__ import annotations

import json

import pytest

from app.sources.netease import fetch_netease_lyrics, fetch_netease_lyrics_timed


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _patch_lyrics(monkeypatch, lyric: str) -> None:
    """让 urlopen 返回给定 lyric 文本的 /api/song/lyric 响应。"""

    def fake_urlopen(req, timeout=8):  # noqa: ARG001
        return _FakeResp({"lrc": {"lyric": lyric}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


@pytest.fixture(autouse=True)
def _clear_lyrics_cache():
    # lru_cache 跨测试会串：清掉，避免上一个用例的 song_id 命中缓存。
    fetch_netease_lyrics_timed.cache_clear()
    yield
    fetch_netease_lyrics_timed.cache_clear()


def test_non_digit_song_id_returns_empty(monkeypatch):
    _patch_lyrics(monkeypatch, "[00:01.0]词")
    assert fetch_netease_lyrics_timed("not-a-number") == []
    assert fetch_netease_lyrics_timed("") == []


def test_missing_lrc_returns_empty(monkeypatch):
    _patch_lyrics(monkeypatch, "")
    assert fetch_netease_lyrics_timed("12345") == []
    # 顶层缺 lrc 键也应是空，不崩
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=8: _FakeResp({"code": 200}))
    assert fetch_netease_lyrics_timed("12346") == []


def test_timestamps_parsed_and_sorted(monkeypatch):
    _patch_lyrics(
        monkeypatch,
        "[00:11.320]告诉你恋爱的秘方\n"
        "[00:03.0]前奏后第一句\n"      # 乱序：应排到最前
        "[00:20.500]第三句\n",
    )
    lines = fetch_netease_lyrics_timed("100")
    assert [l["time"] for l in lines] == [3.0, 11.32, 20.5]
    assert lines[0]["text"] == "前奏后第一句"
    assert lines[1]["text"] == "告诉你恋爱的秘方"
    # 每项只含 time/text 两个键
    assert all(set(l.keys()) == {"time", "text"} for l in lines)


def test_ms_padded_correctly(monkeypatch):
    # LRC 小数部分是「秒的小数」，不是原始毫秒数。ljust(3,"0") 把它归一到毫秒：
    #   .5  -> "500" -> 500ms -> 0.5s
    #   .50 -> "500" -> 500ms -> 0.5s   （与 .5 等价，证明是十进制小数而非裸 ms）
    #   .05 -> "050" -> 50ms  -> 0.05s  （前导 0 有效，区分 .5）
    #   .250-> "250" -> 250ms -> 0.25s  （网易云常见的三位毫秒）
    #   .25 -> "250" -> 250ms -> 0.25s
    _patch_lyrics(
        monkeypatch,
        "[00:05.5]一字\n"
        "[00:06.50]两字\n"
        "[00:07.05]前导零\n"
        "[00:08.250]三位\n"
        "[00:09.25]两位等效\n",
    )
    times = {l["text"]: l["time"] for l in fetch_netease_lyrics_timed("101")}
    assert times["一字"] == 5.5
    assert times["两字"] == 6.5      # .50 == .5
    assert times["前导零"] == 7.05   # .05 == 50ms，区别于 .5
    assert times["三位"] == 8.25     # .250 == 250ms
    assert times["两位等效"] == 9.25 # .25 == .250


def test_multiple_timestamps_expanded(monkeypatch):
    # 一行多时间戳（重复副歌）展开成多条，共享同一 text
    _patch_lyrics(monkeypatch, "[01:00.250][02:00.0]副歌重复\n")
    lines = fetch_netease_lyrics_timed("102")
    assert [l["time"] for l in lines] == [60.25, 120.0]
    assert [l["text"] for l in lines] == ["副歌重复", "副歌重复"]


def test_signature_lines_dropped(monkeypatch):
    _patch_lyrics(
        monkeypatch,
        "[00:10.0]正文\n"
        "作词：张三\n"
        "作曲：李四\n"
        "编曲：王五\n"
        "录音 : 艾志恒Asen\n"     # 半角冒号 + 空格也要丢
        "母带：Provoke\n"
        "发行管理：沈畅\n"        # 多字角色
        "发行：Believe\n"
        "出品：某公司\n"
        "和声：某和声\n",
    )
    texts = [l["text"] for l in fetch_netease_lyrics_timed("103")]
    assert texts == ["正文"]
    # 制作人员角色一个都不该作为歌词正文出现
    for role in ("作词", "作曲", "编曲", "录音", "母带", "发行管理", "发行", "出品", "和声"):
        assert not any(t.startswith(role) for t in texts)


def test_lrc_metadata_tags_dropped(monkeypatch):
    # 真实 API 常带 [ti]/[ar]/[al]/[by]/[offset] 元数据标签，不能当歌词正文显示
    _patch_lyrics(
        monkeypatch,
        "[ti:恋爱]\n"
        "[ar:某艺人]\n"
        "[al:某专辑]\n"
        "[by:上传者]\n"
        "[offset:500]\n"
        "[00:11.320]告诉你恋爱的秘方\n",
    )
    texts = [l["text"] for l in fetch_netease_lyrics_timed("104")]
    assert "告诉你恋爱的秘方" in texts
    for meta in ("[ti:恋爱]", "[ar:某艺人]", "[al:某专辑]", "[by:上传者]", "[offset:500]"):
        assert meta not in texts


def test_untimed_lines_sink_to_bottom(monkeypatch):
    _patch_lyrics(
        monkeypatch,
        "[00:10.0]第一句\n"
        "没有时间戳的间奏说明\n"
        "[00:20.0]第二句\n",
    )
    lines = fetch_netease_lyrics_timed("105")
    # 有时间戳的升序在前，无时间戳的沉到底
    assert lines[-1] == {"time": None, "text": "没有时间戳的间奏说明"}
    assert [l["time"] for l in lines] == [10.0, 20.0, None]


def test_real_world_front_matter_stripped(monkeypatch):
    # 用户实测：网易云把 [ti]/[ar]/[al] 元数据标签 + 录音/母带/发行/出品 人员行，
    # 都带时间戳塞在最前（intro 间奏处）。这些都不该作为歌词正文/字幕行出现。
    _patch_lyrics(
        monkeypatch,
        "[ti:说唱钱]\n"
        "[ar:艾志恒Asen]\n"
        "[al:某专辑]\n"
        "[by:上传者]\n"
        "[00:03.0]录音 : 艾志恒Asen\n"
        "[00:05.0]母带 : Provoke\n"
        "[00:06.0]发行管理：沈畅\n"
        "[00:07.0]发行：Believe Artist Service\n"
        "[00:08.0]出品：四川斯摩堂文化传播有限公司\n"
        "[00:27.07]我们赚的钱叫说唱钱\n"
        "[00:30.0]几百万的合同放桌上面\n",
    )
    lines = fetch_netease_lyrics_timed("108")
    texts = [l["text"] for l in lines]
    # 只剩真实歌词两句，时间升序
    assert texts == ["我们赚的钱叫说唱钱", "几百万的合同放桌上面"]
    assert [l["time"] for l in lines] == [27.07, 30.0]


def test_plain_lyrics_derived_from_timed(monkeypatch):
    _patch_lyrics(
        monkeypatch,
        "[00:03.0]前奏句\n"
        "作词：应被丢\n"
        "[00:11.320]告诉你恋爱的秘方\n",
    )
    # plain 版 = timed 版去掉 time，且署名行同样被过滤
    assert fetch_netease_lyrics("106") == ["前奏句", "告诉你恋爱的秘方"]


def test_network_failure_returns_empty(monkeypatch):
    def boom(req, timeout=8):  # noqa: ARG001
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert fetch_netease_lyrics_timed("107") == []
    assert fetch_netease_lyrics("107") == []
