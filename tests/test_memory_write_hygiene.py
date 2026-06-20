"""记忆写入卫生：一次性约束/句内修饰不入长期排除，纠正指令删冲突正偏好。

对应 realistic_memory_eval 的 Turn 5/6/8/9 四个真 bug。
"""
from __future__ import annotations

import asyncio

from app.memory import MemoryManager
from app.storage import JsonStore


def _mgr(tmp_path) -> MemoryManager:
    return MemoryManager(JsonStore(tmp_path / "store"))


def test_ephemeral_constraint_not_long_term_exclusion():
    # "这轮/上一批/今天" 等一次性标记 → 不升级为长期排除。
    assert MemoryManager._extract_negative_preference("这轮不要中文歌，英文或日文都行") is None
    assert MemoryManager._extract_negative_preference("再来几首，不要上一批") is None
    assert MemoryManager._extract_negative_preference("今晚不要太吵的") is None


def test_in_sentence_constraint_not_exclusion():
    # 句内修饰（前面还有实质内容）→ 非主意图，不抓为长期排除。
    assert MemoryManager._extract_negative_preference("最近喜欢慵懒爵士，尤其有铜管但鼓不要太炸") is None


def test_standalone_negation_still_extracted():
    # 句首主意图的否定仍正常抓为长期排除（不能误伤真排除）。
    assert MemoryManager._extract_negative_preference("不要抖音神曲") == "抖音神曲"
    assert MemoryManager._extract_negative_preference("别推孟菲斯说唱") == "孟菲斯说唱"


def test_correction_detected():
    assert MemoryManager._detect_preference_correction("其实我现在不喜欢 city pop 了") is not None
    assert "city pop" in MemoryManager._detect_preference_correction("其实我现在不喜欢 city pop 了").lower()
    assert MemoryManager._detect_preference_correction("把 city pop 那条偏好改掉") is not None


def test_correction_removes_conflicting_positive_and_adds_exclusion(tmp_path):
    mgr = _mgr(tmp_path)
    uid = "u-correct"

    # 先建立"喜欢 city pop"的正偏好。
    asyncio.run(mgr.auto_learn_from_turn_async(uid, "我喜欢 city pop，工作时少一点人声", results=[]))
    mem = mgr.get_memory(uid)
    assert any("city pop" in e.text.lower() for e in mem.structured_preferences)

    # 纠正：不喜欢了，改掉。
    asyncio.run(mgr.auto_learn_from_turn_async(uid, "其实我现在不喜欢 city pop 了，之前那条偏好请改掉", results=[]))
    mem = mgr.get_memory(uid)
    # 正偏好应被删除，且不与"喜欢"并存。
    assert not any("city pop" in e.text.lower() for e in mem.structured_preferences)
    assert not any("city pop" in p.lower() for p in mem.preferences)
    # 落一条排除，后续不再推。
    assert any("city pop" in r.lower() for r in mem.exclusion_rules)


def test_ephemeral_turn_does_not_pollute_exclusions_via_auto_learn(tmp_path):
    mgr = _mgr(tmp_path)
    uid = "u-eph"
    asyncio.run(mgr.auto_learn_from_turn_async(uid, "这轮不要中文歌，英文或日文都行", results=[]))
    asyncio.run(mgr.auto_learn_from_turn_async(uid, "再来几首，不要上一批", results=[]))
    mem = mgr.get_memory(uid)
    joined = " ".join(mem.exclusion_rules)
    assert "中文" not in joined
    assert "上一批" not in joined
