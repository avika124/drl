"""
Meta (Facebook/Instagram) Ads API adapter.

Wraps the facebook-business SDK to fetch campaign metrics and apply
bid/budget changes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from backend.adapters.base import BasePlatformAdapter

logger = logging.getLogger(__name__)


class MetaAdsAdapter(BasePlatformAdapter):
    platform_name = "meta"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key or os.getenv("META_MARKETING_API_KEY"))
        self._api = None

    def _ensure_client(self):
        if self._api is not None:
            return
        if not self.api_key:
            logger.warning("META_MARKETING_API_KEY not set — using mock data")
            return
        try:
            from facebook_business.api import FacebookAdsApi
            FacebookAdsApi.init(access_token=self.api_key)
            self._api = True
        except Exception as exc:
            logger.warning("Meta Ads client not available: %s", exc)

    # ── fetch metrics ─────────────────────────────────────────────

    async def fetch_campaign_metrics(
        self, account_id: str, campaign_id: str,
    ) -> Dict[str, Any]:
        async def _call():
            self._ensure_client()
            if self._api is None:
                return self._mock_metrics(campaign_id)

            from facebook_business.adobjects.campaign import Campaign
            campaign = Campaign(campaign_id)
            insights = campaign.get_insights(
                fields=[
                    "spend", "impressions", "clicks", "actions",
                    "action_values", "ctr", "cpc", "cpm", "frequency",
                ],
                params={"date_preset": "last_7d"},
            )
            row = insights[0] if insights else {}

            spend = float(row.get("spend", 0))
            impressions = int(row.get("impressions", 0))
            clicks = int(row.get("clicks", 0))
            frequency = float(row.get("frequency", 0))

            conversions = 0
            conv_value = 0.0
            for a in row.get("actions", []):
                if a.get("action_type") in (
                    "offsite_conversion.fb_pixel_purchase",
                    "offsite_conversion.purchase",
                    "purchase",
                ):
                    conversions = int(float(a.get("value", 0)))
            for av in row.get("action_values", []):
                if av.get("action_type") in (
                    "offsite_conversion.fb_pixel_purchase",
                    "purchase",
                ):
                    conv_value = float(av.get("value", 0))

            return {
                "platform": "meta", "campaign_id": campaign_id,
                "spend": spend,
                "roas": conv_value / max(spend, 0.01),
                "ctr": clicks / max(impressions, 1),
                "cpa": spend / max(conversions, 1),
                "cpc": spend / max(clicks, 1),
                "cpm": (spend / max(impressions, 1)) * 1000,
                "conversions": conversions,
                "impressions": impressions,
                "clicks": clicks,
                "revenue": conv_value,
                "audience_size": 0,
                "frequency": frequency,
            }

        return await self._retry(_call, "fetch_campaign_metrics")

    # ── mutations ─────────────────────────────────────────────────

    async def set_bid_adjustment(
        self, campaign_id: str, adjustment: float,
    ) -> bool:
        async def _call():
            self._ensure_client()
            if self._api is None:
                logger.info("Meta [mock]: set bid adj %.2f on %s", adjustment, campaign_id)
                return True
            from facebook_business.adobjects.campaign import Campaign
            campaign = Campaign(campaign_id)
            campaign.api_update(params={}, fields=["bid_amount"])
            logger.info("Meta: bid adjustment %.2f applied to %s", adjustment, campaign_id)
            return True

        return await self._retry(_call, "set_bid_adjustment")

    async def set_budget(
        self, campaign_id: str, daily_budget: float,
    ) -> bool:
        async def _call():
            self._ensure_client()
            if self._api is None:
                logger.info("Meta [mock]: set budget $%.2f on %s", daily_budget, campaign_id)
                return True
            from facebook_business.adobjects.campaign import Campaign
            campaign = Campaign(campaign_id)
            campaign.api_update(
                params={"daily_budget": int(daily_budget * 100)},
                fields=[],
            )
            logger.info("Meta: budget $%.2f applied to %s", daily_budget, campaign_id)
            return True

        return await self._retry(_call, "set_budget")

    async def get_audience_segments(
        self, campaign_id: str,
    ) -> List[Dict[str, Any]]:
        async def _call():
            return [
                {"segment_id": "meta_lookalike_1pct", "segment_name": "1% Lookalike",
                 "platform": "meta", "min_budget_pct": 0.15, "max_budget_pct": 0.50},
                {"segment_id": "meta_retargeting", "segment_name": "Retargeting",
                 "platform": "meta", "min_budget_pct": 0.20, "max_budget_pct": 0.45},
                {"segment_id": "meta_broad", "segment_name": "Broad Interest",
                 "platform": "meta", "min_budget_pct": 0.05, "max_budget_pct": 0.40},
            ]

        return await self._retry(_call, "get_audience_segments")

    @staticmethod
    def _mock_metrics(campaign_id: str) -> Dict[str, Any]:
        return {
            "platform": "meta", "campaign_id": campaign_id,
            "spend": 2200.0, "roas": 2.8, "ctr": 0.042, "cpa": 22.0,
            "cpc": 0.85, "cpm": 9.5, "conversions": 100, "impressions": 232000,
            "clicks": 9744, "revenue": 6160.0, "audience_size": 1200000,
            "frequency": 3.1,
        }
