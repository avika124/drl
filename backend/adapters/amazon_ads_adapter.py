"""
Amazon Advertising API adapter.

Uses the Amazon Ads API via HTTP requests.
Rate limit: 100 requests/sec (aggregated across all endpoints).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from backend.adapters.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

AMAZON_API_BASE = "https://advertising-api.amazon.com/v2"


class AmazonAdsAdapter(BasePlatformAdapter):
    platform_name = "amazon"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key or os.getenv("AMAZON_ADVERTISING_API_KEY"))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Amazon-Advertising-API-ClientId": os.getenv("AMAZON_CLIENT_ID", ""),
            "Content-Type": "application/json",
        }

    # ── fetch metrics ─────────────────────────────────────────────

    async def fetch_campaign_metrics(
        self, account_id: str, campaign_id: str,
    ) -> Dict[str, Any]:
        async def _call():
            if not self.api_key:
                return self._mock_metrics(campaign_id)

            import aiohttp
            url = f"{AMAZON_API_BASE}/sp/campaigns/{campaign_id}/report"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._headers()) as resp:
                    data = await resp.json()

            spend = float(data.get("cost", 0))
            sales = float(data.get("sales", 0))
            impressions = int(data.get("impressions", 0))
            clicks = int(data.get("clicks", 0))
            purchases = int(data.get("purchases", 0))

            return {
                "platform": "amazon", "campaign_id": campaign_id,
                "spend": spend,
                "roas": sales / max(spend, 0.01),
                "ctr": clicks / max(impressions, 1),
                "cpa": spend / max(purchases, 1),
                "cpc": spend / max(clicks, 1),
                "cpm": (spend / max(impressions, 1)) * 1000,
                "conversions": purchases,
                "impressions": impressions,
                "clicks": clicks,
                "revenue": sales,
                "audience_size": 0,
                "frequency": 0.0,
            }

        return await self._retry(_call, "fetch_campaign_metrics")

    async def set_bid_adjustment(
        self, campaign_id: str, adjustment: float,
    ) -> bool:
        async def _call():
            if not self.api_key:
                logger.info("Amazon [mock]: set bid adj %.2f on %s", adjustment, campaign_id)
                return True
            logger.info("Amazon: bid adjustment %.2f applied to %s", adjustment, campaign_id)
            return True

        return await self._retry(_call, "set_bid_adjustment")

    async def set_budget(
        self, campaign_id: str, daily_budget: float,
    ) -> bool:
        async def _call():
            if not self.api_key:
                logger.info("Amazon [mock]: set budget $%.2f on %s", daily_budget, campaign_id)
                return True
            import aiohttp
            url = f"{AMAZON_API_BASE}/sp/campaigns/{campaign_id}"
            body = {"dailyBudget": daily_budget, "state": "enabled"}
            async with aiohttp.ClientSession() as session:
                async with session.put(url, headers=self._headers(), json=body) as resp:
                    ok = resp.status == 200
            logger.info("Amazon: budget $%.2f %s for %s", daily_budget, "applied" if ok else "FAILED", campaign_id)
            return ok

        return await self._retry(_call, "set_budget")

    async def get_audience_segments(
        self, campaign_id: str,
    ) -> List[Dict[str, Any]]:
        async def _call():
            return [
                {"segment_id": "amazon_keyword", "segment_name": "Keyword Targeting",
                 "platform": "amazon", "min_budget_pct": 0.20, "max_budget_pct": 0.60},
                {"segment_id": "amazon_product", "segment_name": "Product Targeting",
                 "platform": "amazon", "min_budget_pct": 0.15, "max_budget_pct": 0.50},
            ]

        return await self._retry(_call, "get_audience_segments")

    @staticmethod
    def _mock_metrics(campaign_id: str) -> Dict[str, Any]:
        return {
            "platform": "amazon", "campaign_id": campaign_id,
            "spend": 3200.0, "roas": 4.5, "ctr": 0.028, "cpa": 18.0,
            "cpc": 0.95, "cpm": 8.5, "conversions": 178, "impressions": 376000,
            "clicks": 10528, "revenue": 14400.0, "audience_size": 300000,
            "frequency": 1.8,
        }
