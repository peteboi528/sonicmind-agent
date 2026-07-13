"""限流中间件核心：自研令牌桶（按 user_id|IP 维度分档限流）。

动机：默认 ``AUTH_ENABLED=false`` 公网部署时，``/api/playback/audio`` 会沦为免费
网易云代理（消耗绑定用户的 VIP 权益），``/chat`` 可被刷 LLM 成本。用令牌桶给
聊天 / 播放代理两类端点分档限流，超限返回 429 + Retry-After。

设计要点：
- per-key 令牌桶：每个 (档位, user_id|IP) 独立桶，互不影响。
- 懒 refill：基于 ``time.monotonic()`` 时间差补令牌，无需后台线程。
- 线程安全：``threading.Lock`` 保护桶 dict（与 JsonStore 同款）。
- LRU 上限：桶数超 ``_MAX_BUCKETS`` 时整体重置，防多用户实例无界增长。
- 零依赖：仅标准库，未引入 slowapi/redis。
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_MAX_BUCKETS = 4096  # 桶数上限：超过则整体重置（粗粒度防泄漏，与长期运行取舍）


class _Bucket:
    """单桶状态。"""

    __slots__ = ("tokens", "last")

    def __init__(self, capacity: float):
        self.tokens = capacity
        self.last = time.monotonic()


class TokenBucket:
    """令牌桶：每分钟 ``rpm`` 个令牌，容量默认等于 rpm（允许短时突发到整分钟预算）。"""

    __slots__ = ("capacity", "rpm", "_lock", "_state")

    def __init__(self, rpm: int, capacity: int | None = None):
        self.rpm = max(1, rpm)
        self.capacity = float(max(1, capacity if capacity is not None else rpm))
        self._lock = threading.Lock()
        self._state = _Bucket(self.capacity)

    def allow(self, cost: float = 1.0) -> tuple[bool, float]:
        """尝试取 ``cost`` 个令牌。返回 (是否放行, 建议重试等待秒数)。"""
        refill_per_sec = self.rpm / 60.0
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._state.last
            self._state.tokens = min(self.capacity, self._state.tokens + elapsed * refill_per_sec)
            self._state.last = now
            if self._state.tokens >= cost:
                self._state.tokens -= cost
                return True, 0.0
            deficit = cost - self._state.tokens
            retry = deficit / refill_per_sec if refill_per_sec > 0 else 1.0
            return False, retry


class RateLimiter:
    """分档限流器：按 (档位, key) 维护独立令牌桶。

    ``tier_rpm``：``{档位名: 每分钟配额}``，如 ``{"chat": 30, "playback": 60}``。
    未在 dict 里的档位直接放行（便于按需开关某档）。
    """

    __slots__ = ("tier_rpm", "_buckets", "_lock")

    def __init__(self, tier_rpm: dict[str, int]):
        self.tier_rpm = dict(tier_rpm)
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._lock = threading.Lock()

    def acquire(self, tier: str, key: str) -> tuple[bool, float]:
        """尝试为 ``(tier, key)`` 取一个令牌。未配置 tier 时直接放行。"""
        rpm = self.tier_rpm.get(tier)
        if not rpm:
            return True, 0.0
        bkey = (tier, key)
        with self._lock:
            bucket = self._buckets.get(bkey)
            if bucket is None:
                # 防 dict 无界增长（多用户实例长期运行）
                if len(self._buckets) >= _MAX_BUCKETS:
                    self._buckets.clear()
                    logger.warning("限流桶数达上限 %d，已整体重置（粗粒度防泄漏）。", _MAX_BUCKETS)
                bucket = TokenBucket(rpm=rpm)
                self._buckets[bkey] = bucket
        return bucket.allow()

    def clear(self) -> None:
        """清空所有桶（测试 / 配置热更新用）。"""
        with self._lock:
            self._buckets.clear()
