"""JsonStore 并发安全：验证 per-key 锁防止 read-modify-write 丢更新。

回归守卫：若有人移除 MemoryManager 各 RMW 方法上的 store.lock(...)，这些测试在
多线程下会间歇性失败（history 条数 < 预期、评分重复写入）。
"""
from __future__ import annotations

import threading

from app.memory import MemoryManager
from app.models import Asset
from app.storage import JsonStore


def test_record_listen_concurrent_no_lost_updates(tmp_path):
    store = JsonStore(tmp_path / "store")
    mgr = MemoryManager(store)
    user_id = "concurrent-user"
    n = 50

    def worker(i: int):
        mgr.record_listen(user_id, f"asset-{i}", duration=60, completed=True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    memory = mgr.get_memory(user_id)
    assert len(memory.listening_history) == n, (
        f"lost update: expected {n} events, got {len(memory.listening_history)}"
    )


def test_record_rating_concurrent_no_duplicate(tmp_path):
    store = JsonStore(tmp_path / "store")
    mgr = MemoryManager(store)
    user_id = "rate-user"
    asset = Asset(asset_id="a1", source_url="http://x", title="t", artist="x", duration_seconds=180)
    n = 40

    def worker(_i: int):
        mgr.record_rating(user_id, asset, 8.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    memory = mgr.get_memory(user_id)
    assert len(memory.ratings) == 1, f"expected 1 rating, got {len(memory.ratings)}"
