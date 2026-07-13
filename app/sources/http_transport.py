from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float = 0.0


class AsyncSourceTransport:
    """Process-shared pooled HTTP transport with source isolation and safe read retries."""

    def __init__(self, *, timeout_seconds: float = 8.0, max_connections: int = 40) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 4.0)),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections // 2),
            follow_redirects=True,
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._circuits: dict[str, CircuitState] = {}

    async def request(
        self,
        source: str,
        method: str,
        url: str,
        *,
        retries: int = 1,
        concurrency: int = 4,
        **kwargs: Any,
    ) -> httpx.Response:
        method = method.upper()
        idempotent = method in {"GET", "HEAD", "OPTIONS"}
        circuit = self._circuits.setdefault(source, CircuitState())
        if circuit.opened_at and time.monotonic() - circuit.opened_at < 30:
            raise ConnectionError(f"source circuit open: {source}")
        if circuit.opened_at:
            circuit.opened_at = 0
            circuit.failures = 0
        semaphore = self._semaphores.setdefault(source, asyncio.Semaphore(concurrency))
        attempts = retries + 1 if idempotent else 1
        async with semaphore:
            for attempt in range(attempts):
                try:
                    response = await self.client.request(method, url, **kwargs)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            "retryable source response", request=response.request, response=response
                        )
                    circuit.failures = 0
                    return response
                except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                    circuit.failures += 1
                    if circuit.failures >= 5:
                        circuit.opened_at = time.monotonic()
                    if attempt + 1 >= attempts:
                        raise
                    await asyncio.sleep(min(0.2 * (2**attempt), 1.0))
        raise AssertionError("unreachable")

    async def close(self) -> None:
        await self.client.aclose()


source_transport = AsyncSourceTransport()
