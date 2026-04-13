"""
Abstract base class shared by all platform adapters.

Every concrete adapter must implement:
- fetch_campaign_metrics()
- set_bid_adjustment()
- set_budget()
- get_audience_segments()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.getenv("DATA_PIPELINE_RETRY_MAX_ATTEMPTS", "3"))
BACKOFF_BASE_MS = int(os.getenv("DATA_PIPELINE_RETRY_BACKOFF_MS", "1000"))


class BasePlatformAdapter(ABC):
    """Uniform interface every platform adapter exposes."""

    platform_name: str = "unknown"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._request_count = 0
        self._last_request_ts = 0.0

    # ── abstract methods ──────────────────────────────────────────

    @abstractmethod
    async def fetch_campaign_metrics(
        self, account_id: str, campaign_id: str,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def set_bid_adjustment(
        self, campaign_id: str, adjustment: float,
    ) -> bool:
        ...

    @abstractmethod
    async def set_budget(
        self, campaign_id: str, daily_budget: float,
    ) -> bool:
        ...

    @abstractmethod
    async def get_audience_segments(
        self, campaign_id: str,
    ) -> List[Dict[str, Any]]:
        ...

    # ── retry helper ──────────────────────────────────────────────

    async def _retry(self, coro_factory, label: str = "api_call"):
        """Execute an async callable with exponential backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await coro_factory()
                self._request_count += 1
                self._last_request_ts = time.time()
                if os.getenv("LOG_PLATFORM_API_CALLS", "TRUE").upper() == "TRUE":
                    logger.info(
                        "%s.%s succeeded (attempt %d)",
                        self.platform_name, label, attempt,
                    )
                return result
            except Exception as exc:
                last_exc = exc
                wait = (BACKOFF_BASE_MS / 1000) * (2 ** (attempt - 1))
                logger.warning(
                    "%s.%s attempt %d failed (%s) — retrying in %.1fs",
                    self.platform_name, label, attempt, exc, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(
            f"{self.platform_name}.{label} failed after {MAX_RETRIES} attempts: {last_exc}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(requests={self._request_count})"
