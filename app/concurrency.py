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
- **caller-runs 防嵌套死锁**：run_parallel 存在嵌套调用（变体并行 → search_web_music →
  netease 多端点并行）。共享池饱和时若仍排队，内层任务会因外层 worker 占满而饿死（退化为
  超时空结果）。故 sem.acquire(blocking=False) 失败时，该 thunk 在调用方线程内联执行——
  外层 worker 自行完成工作，向前推进，永不死锁。
- 超时未完成的 thunk 仍占用槽位直到其自身 socket 超时（netease _SEARCH_TIMEOUT / httpx
  per-request Timeout / AsyncSourceTransport 8s 均已存在且有界）——「请求返回 ≠ 后台停止」
  的诚实边界，但被 W 封顶，积压有上界。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SHARED_POOL: ThreadPoolExecutor | None = None
_SHARED_SEM: threading.BoundedSemaphore | None = None
_INIT_LOCK = threading.Lock()


def _shared_executor() -> tuple[ThreadPoolExecutor, threading.BoundedSemaphore]:
    """懒加载进程级共享有界线程池 + 槽位信号量（双检锁）。"""
    global _SHARED_POOL, _SHARED_SEM
    if _SHARED_POOL is None or _SHARED_SEM is None:
        with _INIT_LOCK:
            if _SHARED_POOL is None or _SHARED_SEM is None:
                from app.config import settings

                workers = max(1, settings.concurrency_max_workers)
                _SHARED_POOL = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="ma-worker"
                )
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
    pool, sem = _shared_executor()
    future_to_idx: dict[Future, tuple[int, str]] = {}
    for idx, (label, thunk) in enumerate(tasks):
        if sem.acquire(blocking=False):
            # 拿到槽位 → 提交到池；完成时释放槽位（含超时/异常路径，_runner 的 finally 兜底）。
            def _runner(t: Callable[[], T] = thunk) -> T:
                try:
                    return t()
                finally:
                    sem.release()

            fut = pool.submit(_runner)
            future_to_idx[fut] = (idx, label)
        else:
            # 池饱和 → 调用方内联执行。嵌套场景下外层 worker 自行完成该 thunk，
            # 不再排队等空闲 worker，向前推进，避免内层任务饿死。
            logger.debug("并发池饱和，任务 %s 改内联执行", label)
            try:
                results[idx] = thunk()
            except (FuturesTimeout, Exception):
                logger.debug("内联任务 %s 失败，取默认值", label, exc_info=True)
    if future_to_idx:
        done, not_done = wait(future_to_idx, timeout=timeout, return_when=ALL_COMPLETED)
        for fut, (idx, label) in future_to_idx.items():
            if fut in not_done:
                fut.cancel()
                logger.debug("并发任务 %s 超时（%.1fs），取默认值", label, timeout)
                continue
            try:
                results[idx] = fut.result(timeout=0)
            except (FuturesTimeout, Exception):
                logger.debug("并发任务 %s 失败，取默认值", label, exc_info=True)
    return results
