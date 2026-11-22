"""确定性并发小工具——IO-bound 多源检索并行化。

动机：search_videos / 多源搜索里 B站、YouTube 等是独立 IO，串行等待白白叠加延迟。
用线程池并行发起，但**结果按调用方给定的固定顺序合并**，保证输出确定性（测试不破）。

设计要点：
- per-task 超时：单源卡住不拖垮整体，超时/异常的源安静降级为空结果。
- 顺序确定：返回值严格按 tasks 的传入顺序，与各源完成先后无关。
- 零依赖：仅用标准库 concurrent.futures，未引入任何外部包。

线程治理（P1 长期运行稳定性）：
- 进程级**共享有界** ThreadPoolExecutor + BoundedSemaphore(W)，W=concurrency_max_workers。
- 旧实现每次 run_parallel 新建 len(tasks) 个非守护线程，超时后 shutdown(wait=False) 仅返回、
  已进入阻塞 syscall 的线程无法取消，长跑会逐渐积压线程与连接。改为共享池后总量被 W 封顶。
- 顶层调用的排队时间计入批次预算。只有池内 worker 的子任务可以在饱和时内联执行。
- 已开始的同步 I/O 直到自身超时才退出。在它退出前保留对应槽位。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SHARED_POOL: ThreadPoolExecutor | None = None
_SHARED_SEM: threading.BoundedSemaphore | None = None
_INIT_LOCK = threading.Lock()
_WORKER = threading.local()


def _shared_executor() -> tuple[ThreadPoolExecutor, threading.BoundedSemaphore]:
    """懒加载进程级共享有界线程池 + 槽位信号量（双检锁）。"""
    global _SHARED_POOL, _SHARED_SEM
    if _SHARED_POOL is None or _SHARED_SEM is None:
        with _INIT_LOCK:
            if _SHARED_POOL is None or _SHARED_SEM is None:
                from app.config import settings

                workers = max(1, settings.concurrency_max_workers)
                _SHARED_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ma-worker")
                _SHARED_SEM = threading.BoundedSemaphore(workers)
    assert _SHARED_POOL is not None and _SHARED_SEM is not None
    return _SHARED_POOL, _SHARED_SEM


def shutdown_shared_executor(wait: bool = False) -> None:
    """关停共享池（lifespan 关停 / 测试隔离用）。

    wait=False：cancel 未启动的 future 后立即返回；已在阻塞 syscall 的线程仍会跑到自身
    socket 超时才退出（Python 无法取消已开始的阻塞线程）。
    """
    global _SHARED_POOL, _SHARED_SEM
    with _INIT_LOCK:
        pool = _SHARED_POOL
        _SHARED_POOL = None
        _SHARED_SEM = None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=True)


def run_parallel(
    tasks: list[tuple[str, Callable[[], T]]],
    timeout: float = 8.0,
    default: Any = None,
) -> list[T]:
    """并行执行带标签的任务，按传入顺序返回结果。

    tasks：[(label, thunk), ...]，每个 thunk 无参、返回该任务结果。
    timeout：整体墙钟超时（秒）；超时未完成的任务取 default。
    单个 thunk 抛异常 → 该位置取 default 并记 debug 日志，不影响其余任务。

    返回与 tasks 等长、同序的结果列表。向共享有界池提交；池饱和时该 thunk 内联执行
    （caller-runs，防嵌套死锁）。
    """
    if not tasks:
        return []
    results: list[Any] = [default] * len(tasks)
    deadline = time.monotonic() + max(0.0, timeout)
    inherited_deadline = getattr(_WORKER, "deadline", None)
    nested = inherited_deadline is not None
    if nested:
        deadline = min(deadline, inherited_deadline)
    pool, sem = _shared_executor()
    future_to_idx: dict[Future, tuple[int, str]] = {}

    def run(thunk: Callable[[], T]) -> T | Any:
        previous = getattr(_WORKER, "deadline", None)
        _WORKER.deadline = deadline
        try:
            return thunk() if time.monotonic() < deadline else default
        finally:
            _WORKER.deadline = previous

    def release(_future: Future) -> None:
        sem.release()

    for idx, (label, thunk) in enumerate(tasks):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        acquired = sem.acquire(blocking=False) if nested else sem.acquire(timeout=remaining)
        if acquired:
            try:
                future = pool.submit(run, thunk)
            except RuntimeError:
                sem.release()
                logger.debug("并发池已关闭，任务 %s 取默认值", label)
                break
            future.add_done_callback(release)
            future_to_idx[future] = (idx, label)
        elif nested:
            # Pool workers may run child work inline to avoid waiting for themselves.
            try:
                results[idx] = run(thunk)
            except Exception:
                logger.debug("内联任务 %s 失败，取默认值", label, exc_info=True)
        else:
            logger.debug("并发池没有在预算内释放槽位，任务 %s 取默认值", label)
            break
    if future_to_idx:
        _, not_done = wait(future_to_idx, timeout=max(0.0, deadline - time.monotonic()), return_when=ALL_COMPLETED)
        for future, (idx, label) in future_to_idx.items():
            if future in not_done:
                future.cancel()
                logger.debug("并发任务 %s 超时（%.1fs），取默认值", label, timeout)
                continue
            try:
                results[idx] = future.result()
            except Exception:
                logger.debug("并发任务 %s 失败，取默认值", label, exc_info=True)
    return results
