from __future__ import annotations

import time

from app.concurrency import run_parallel


def test_results_in_submission_order_regardless_of_finish():
    # 第一个任务慢、第二个快——结果仍按提交顺序返回
    def slow():
        time.sleep(0.05)
        return "slow"

    def fast():
        return "fast"

    out = run_parallel([("slow", slow), ("fast", fast)], timeout=2.0)
    assert out == ["slow", "fast"]


def test_failing_task_falls_back_to_default():
    def boom():
        raise RuntimeError("nope")

    def ok():
        return 42

    out = run_parallel([("boom", boom), ("ok", ok)], timeout=2.0, default=None)
    assert out == [None, 42]


def test_timeout_task_takes_default():
    started = time.monotonic()
    def hang():
        time.sleep(5.0)
        return "never"

    def quick():
        return "done"

    out = run_parallel([("hang", hang), ("quick", quick)], timeout=0.2, default="DEF")
    elapsed = time.monotonic() - started
    assert out[0] == "DEF"
    assert out[1] == "done"
    assert elapsed < 0.6  # timeout 不能被 executor shutdown(wait=True) 重新拖回 5 秒


def test_single_task_runs_synchronously():
    out = run_parallel([("one", lambda: "v")], timeout=1.0)
    assert out == ["v"]


def test_empty_tasks_returns_empty():
    assert run_parallel([]) == []


def test_concurrency_actually_parallel():
    # 两个各 sleep 0.1s 的任务并发应明显快于串行 0.2s
    def work():
        time.sleep(0.1)
        return 1

    start = time.monotonic()
    out = run_parallel([("a", work), ("b", work)], timeout=2.0)
    elapsed = time.monotonic() - start
    assert out == [1, 1]
    assert elapsed < 0.18  # 并发，远小于串行 0.2s
