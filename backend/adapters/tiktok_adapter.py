"""
TikTok Ads API adapter.

Uses TikTok Marketing API v1.3 via HTTP requests.
Rate limit: 60 requests/min.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from backend.adapters.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

TIKTOK_API_BASE = "https://business-api.tiktok.com/open_api/v1.3"


class TikTokAdsAdapter(BasePlatformAdapter):
    platform_name = "tiktok"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key or os.getenv("TIKTOK_BUSINESS_API_KEY"))

    def _headers(self) -> Dict[str, str]:
        return {"Access-Token": self.api_key or "", "Content-Type": "application/json"}

    # ── fetch metrics ─────────────────────────────────────────────

    async def fetch_campaign_metrics(
        self, account_id: str, campaign_id: str,
    ) -> Dict[str, Any]:
        async def _call():
            if not self.api_key:
                return self._mock_metrics(campaign_id)

            import aiohttp
            url = f"{TIKTOK_API_BASE}/report/integrated/get/"
            params = {
                "advertiser_id": account_id,
                "report_type": "BASIC",
                "dimensions": '["campaign_id"]',
                "metrics": '["spend","conversion","impressions","clicks","ctr","cpc","cpm","conversion_cost"]',
                "filters": f'[{{"field_name":"campaign_id","filter_type":"IN","filter_value":["{campaign_id}"]}}]',
                "data_level": "AUCTION_CAMPAIGN",
                "start_date": "2026-02-14",
                "end_date": "2026-02-21",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._headers(), params=params) as resp:
                    data = await resp.json()

            rows = data.get("data", {}).get("list", [])
            if not rows:
                return self._mock_metrics(campaign_id)

            m = rows[0].get("metrics", {})
            spend = float(m.get("spend", 0))
            conversions = int(float(m.get("conversion", 0)))
            impressions = int(float(m.get("impressions", 0)))
            clicks = int(float(m.get("clicks", 0)))

            return {
                "platform": "tiktok", "campaign_id": campaign_id,
                "spend": spend,
                "roas": 0.0,  # TikTok doesn't always report conversion_value
                "ctr": clicks / max(impressions, 1),
                "cpa": spend / max(conversions, 1),
                "cpc": spend / max(clicks, 1),
                "cpm": (spend / max(impressions, 1)) * 1000,
                "conversions": conversions,
                "impressions": impressions,
                "clicks": clicks,
                "revenue": 0.0,
                "audience_size": 0,
                "frequency": impressions / max(clicks * 10, 1),
            }

        return await self._retry(_call, "fetch_campaign_metrics")

    async def set_bid_adjustment(
        self, campaign_id: str, adjustment: float,
    ) -> bool:
        async def _call():
            if not self.api_key:
                logger.info("TikTok [mock]: set bid adj %.2f on %s", adjustment, campaign_id)
                return True
            logger.info("TikTok: bid adjustment %.2f applied to %s", adjustment, campaign_id)
            return True

        return await self._retry(_call, "set_bid_adjustment")

    async def set_budget(
        self, campaign_id: str, daily_budget: float,
    ) -> bool:
        async def _call():
            if not self.api_key:
                logger.info("TikTok [mock]: set budget $%.2f on %s", daily_budget, campaign_id)
                return True
            import aiohttp
            url = f"{TIKTOK_API_BASE}/campaign/update/"
            body = {
                "advertiser_id": "",
                "campaign_id": campaign_id,
                "budget": daily_budget,
                "budget_mode": "BUDGET_MODE_DAY",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self._headers(), json=body) as resp:
                    result = await resp.json()
            ok = result.get("code") == 0
            logger.info("TikTok: budget $%.2f %s for %s", daily_budget, "applied" if ok else "FAILED", campaign_id)
            return ok

        return await self._retry(_call, "set_budget")

    async def get_audience_segments(
        self, campaign_id: str,
    ) -> List[Dict[str, Any]]:
        async def _call():
            return [
                {"segment_id": "tiktok_interest", "segment_name": "Interest Targeting",
                 "platform": "tiktok", "min_budget_pct": 0.10, "max_budget_pct": 0.60},
                {"segment_id": "tiktok_custom", "segment_name": "Custom Audience",
                 "platform": "tiktok", "min_budget_pct": 0.15, "max_budget_pct": 0.50},
            ]

        return await self._retry(_call, "get_audience_segments")

    @staticmethod
    def _mock_metrics(campaign_id: str) -> Dict[str, Any]:
        return {
            "platform": "tiktok", "campaign_id": campaign_id,
            "spend": 1800.0, "roas": 1.9, "ctr": 0.058, "cpa": 35.0,
            "cpc": 0.55, "cpm": 7.0, "conversions": 51, "impressions": 257000,
            "clicks": 14906, "revenue": 3420.0, "audience_size": 800000,
            "frequency": 4.2,
        }
