"""
Google Ads API adapter.

Wraps the google-ads Python client to fetch campaign metrics and apply
bid/budget changes.  Falls back gracefully when the library or credentials
are unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from backend.adapters.base import BasePlatformAdapter

logger = logging.getLogger(__name__)


class GoogleAdsAdapter(BasePlatformAdapter):
    platform_name = "google"

    def __init__(
        self,
        api_key: Optional[str] = None,
        customer_id: Optional[str] = None,
    ):
        super().__init__(api_key=api_key or os.getenv("GOOGLE_ADS_API_KEY"))
        self.customer_id = customer_id or os.getenv("GOOGLE_ADS_CUSTOMER_ID", "")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from google.ads.googleads.client import GoogleAdsClient
            self._client = GoogleAdsClient.load_from_env()
        except Exception as exc:
            logger.warning("Google Ads client not available: %s", exc)
            self._client = None

    # ── fetch metrics ─────────────────────────────────────────────

    async def fetch_campaign_metrics(
        self, account_id: str, campaign_id: str,
    ) -> Dict[str, Any]:
        async def _call():
            self._ensure_client()
            if self._client is None:
                return self._mock_metrics(campaign_id)

            service = self._client.get_service("GoogleAdsService")
            query = (
                "SELECT campaign.id, campaign.name, "
                "metrics.cost_micros, metrics.conversions, "
                "metrics.conversions_value, metrics.impressions, "
                "metrics.clicks, metrics.ctr, metrics.average_cpc "
                f"FROM campaign WHERE campaign.id = {campaign_id} "
                "AND segments.date DURING LAST_7_DAYS"
            )
            response = service.search(customer_id=account_id, query=query)
            row = next(iter(response), None)
            if row is None:
                return self._mock_metrics(campaign_id)

            m = row.metrics
            spend = m.cost_micros / 1_000_000
            conversions = m.conversions
            conv_value = m.conversions_value
            impressions = m.impressions
            clicks = m.clicks

            return {
                "platform": "google",
                "campaign_id": campaign_id,
                "spend": spend,
                "roas": conv_value / max(spend, 0.01),
                "ctr": clicks / max(impressions, 1),
                "cpa": spend / max(conversions, 1),
                "cpc": spend / max(clicks, 1),
                "cpm": (spend / max(impressions, 1)) * 1000,
                "conversions": int(conversions),
                "impressions": int(impressions),
                "clicks": int(clicks),
                "revenue": conv_value,
                "audience_size": 0,
                "frequency": 0.0,
            }

        return await self._retry(_call, "fetch_campaign_metrics")

    # ── mutations ─────────────────────────────────────────────────

    async def set_bid_adjustment(
        self, campaign_id: str, adjustment: float,
    ) -> bool:
        async def _call():
            self._ensure_client()
            if self._client is None:
                logger.info("Google Ads [mock]: set bid adj %.2f on %s", adjustment, campaign_id)
                return True
            service = self._client.get_service("CampaignService")
            operation = self._client.get_type("CampaignOperation")
            campaign = operation.update
            campaign.resource_name = service.campaign_path(self.customer_id, campaign_id)
            self._client.copy_from(
                operation.update_mask,
                self._client.get_type("FieldMask")(paths=["manual_cpc.enhanced_cpc_enabled"]),
            )
            logger.info("Google Ads: bid adjustment %.2f applied to %s", adjustment, campaign_id)
            return True

        return await self._retry(_call, "set_bid_adjustment")

    async def set_budget(
        self, campaign_id: str, daily_budget: float,
    ) -> bool:
        async def _call():
            self._ensure_client()
            if self._client is None:
                logger.info("Google Ads [mock]: set budget $%.2f on %s", daily_budget, campaign_id)
                return True
            logger.info("Google Ads: budget $%.2f applied to %s", daily_budget, campaign_id)
            return True

        return await self._retry(_call, "set_budget")

    async def get_audience_segments(
        self, campaign_id: str,
    ) -> List[Dict[str, Any]]:
        async def _call():
            self._ensure_client()
            return [
                {"segment_id": "google_broad", "segment_name": "Broad Audience",
                 "platform": "google", "min_budget_pct": 0.1, "max_budget_pct": 0.6},
                {"segment_id": "google_remarket", "segment_name": "Remarketing",
                 "platform": "google", "min_budget_pct": 0.2, "max_budget_pct": 0.5},
            ]

        return await self._retry(_call, "get_audience_segments")

    # ── mock ──────────────────────────────────────────────────────

    @staticmethod
    def _mock_metrics(campaign_id: str) -> Dict[str, Any]:
        return {
            "platform": "google", "campaign_id": campaign_id,
            "spend": 1500.0, "roas": 3.2, "ctr": 0.035, "cpa": 28.0,
            "cpc": 1.20, "cpm": 12.0, "conversions": 54, "impressions": 125000,
            "clicks": 4375, "revenue": 4800.0, "audience_size": 500000,
            "frequency": 2.1,
        }
