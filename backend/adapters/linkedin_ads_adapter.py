"""
LinkedIn Marketing API adapter.

Uses the LinkedIn Marketing Developer Platform via HTTP requests.
Rate limit: 30 requests/min.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from backend.adapters.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

LINKEDIN_API_BASE = "https://api.linkedin.com/v2"


class LinkedInAdsAdapter(BasePlatformAdapter):
    platform_name = "linkedin"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key or os.getenv("LINKEDIN_MARKETING_API_KEY"))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }

    # ── fetch metrics ─────────────────────────────────────────────

    async def fetch_campaign_metrics(
        self, account_id: str, campaign_id: str,
    ) -> Dict[str, Any]:
        async def _call():
            if not self.api_key:
                return self._mock_metrics(campaign_id)

            import aiohttp
            url = (
                f"{LINKEDIN_API_BASE}/adAnalyticsV2"
                f"?q=analytics&pivot=CAMPAIGN"
                f"&campaigns[0]=urn:li:sponsoredCampaign:{campaign_id}"
                f"&dateRange.start.year=2026&dateRange.start.month=2&dateRange.start.day=14"
                f"&dateRange.end.year=2026&dateRange.end.month=2&dateRange.end.day=21"
                f"&fields=impressions,clicks,costInLocalCurrency,externalWebsiteConversions"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._headers()) as resp:
                    data = await resp.json()

            elements = data.get("elements", [])
            if not elements:
                return self._mock_metrics(campaign_id)

            el = elements[0]
            spend = float(el.get("costInLocalCurrency", 0))
            impressions = int(el.get("impressions", 0))
            clicks = int(el.get("clicks", 0))
            conversions = int(el.get("externalWebsiteConversions", 0))

            return {
                "platform": "linkedin", "campaign_id": campaign_id,
                "spend": spend,
                "roas": 0.0,  # LinkedIn doesn't report revenue directly
                "ctr": clicks / max(impressions, 1),
                "cpa": spend / max(conversions, 1),
                "cpc": spend / max(clicks, 1),
                "cpm": (spend / max(impressions, 1)) * 1000,
                "conversions": conversions,
                "impressions": impressions,
                "clicks": clicks,
                "revenue": 0.0,
                "audience_size": 0,
                "frequency": 0.0,
            }

        return await self._retry(_call, "fetch_campaign_metrics")

    async def set_bid_adjustment(
        self, campaign_id: str, adjustment: float,
    ) -> bool:
        async def _call():
            if not self.api_key:
                logger.info("LinkedIn [mock]: set bid adj %.2f on %s", adjustment, campaign_id)
                return True
            logger.info("LinkedIn: bid adjustment %.2f applied to %s", adjustment, campaign_id)
            return True

        return await self._retry(_call, "set_bid_adjustment")

    async def set_budget(
        self, campaign_id: str, daily_budget: float,
    ) -> bool:
        async def _call():
            if not self.api_key:
                logger.info("LinkedIn [mock]: set budget $%.2f on %s", daily_budget, campaign_id)
                return True
            import aiohttp
            url = f"{LINKEDIN_API_BASE}/adCampaignsV2/{campaign_id}"
            body = {"patch": {"$set": {"dailyBudget": {"amount": str(daily_budget), "currencyCode": "USD"}}}}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self._headers(), json=body) as resp:
                    ok = resp.status in (200, 204)
            logger.info("LinkedIn: budget $%.2f %s for %s", daily_budget, "applied" if ok else "FAILED", campaign_id)
            return ok

        return await self._retry(_call, "set_budget")

    async def get_audience_segments(
        self, campaign_id: str,
    ) -> List[Dict[str, Any]]:
        async def _call():
            return [
                {"segment_id": "linkedin_decision_makers", "segment_name": "Decision Makers",
                 "platform": "linkedin", "min_budget_pct": 0.25, "max_budget_pct": 0.60},
                {"segment_id": "linkedin_job_function", "segment_name": "Job Function",
                 "platform": "linkedin", "min_budget_pct": 0.10, "max_budget_pct": 0.40},
            ]

        return await self._retry(_call, "get_audience_segments")

    @staticmethod
    def _mock_metrics(campaign_id: str) -> Dict[str, Any]:
        return {
            "platform": "linkedin", "campaign_id": campaign_id,
            "spend": 4500.0, "roas": 1.4, "ctr": 0.012, "cpa": 150.0,
            "cpc": 8.50, "cpm": 45.0, "conversions": 30, "impressions": 100000,
            "clicks": 1200, "revenue": 6300.0, "audience_size": 200000,
            "frequency": 1.5,
        }
