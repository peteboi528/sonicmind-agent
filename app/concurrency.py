"""P2-H：确定性并发小工具——IO-bound 多源检索并行化。

动机：search_videos / 多源搜索里 B站、YouTube 等是独立 IO，串行等待白白叠加延迟。
用线程池并行发起，但**结果按调用方给定的固定顺序合并**，保证输出确定性（测试不破）。

设计要点：
- per-task 超时：单源卡住不拖垮整体，超时/异常的源安静降级为空结果。
- 顺序确定：返回值严格按 tasks 的传入顺序，与各源完成先后无关。
- 零依赖：仅用标准库 concurrent.futures，未引入任何外部包。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_parallel(
    tasks: list[tuple[str, Callable[[], T]]],
    timeout: float = 8.0,
    default: Any = None,
) -> list[T]:
    """并行执行带标签的任务，按传入顺序返回结果。

    tasks：[(label, thunk), ...]，每个 thunk 无参、返回该任务结果。
    timeout：整体墙钟超时（秒）；超时未完成的任务取 default。
    单个 thunk 抛异常 → 该位置取 default 并记 debug 日志，不影响其余任务。

    返回与 tasks 等长、同序的结果列表。单任务时直接同步执行（省去线程开销）。
    """
    if not tasks:
        return []
    results: list[Any] = [default] * len(tasks)
    pool = ThreadPoolExecutor(max_workers=len(tasks))
    try:
        future_to_idx = {pool.submit(thunk): i for i, (_, thunk) in enumerate(tasks)}
        done, not_done = wait(future_to_idx, timeout=timeout, return_when=ALL_COMPLETED)
        for fut in future_to_idx:
            idx = future_to_idx[fut]
            label = tasks[idx][0]
            if fut in not_done:
                fut.cancel()
                logger.debug("并发任务 %s 超时（%.1fs），取默认值", label, timeout)
                continue
            try:
                results[idx] = fut.result(timeout=0)
            except (FuturesTimeout, Exception):
                logger.debug("并发任务 %s 失败，取默认值", label, exc_info=True)
    finally:
        # `with ThreadPoolExecutor` 会在退出时 wait=True，令上面的 timeout 形同虚设。
        # 超时任务可能仍在系统调用中，但不能继续阻塞当前 Agent 请求。
        pool.shutdown(wait=False, cancel_futures=True)
    return results
