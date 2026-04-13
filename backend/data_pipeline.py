"""
Data Pipeline — orchestrates cross-platform data fetching, normalisation,
and action execution.

Responsibilities:
- Fetch metrics from all 5 platforms in parallel
- Normalise raw metrics to 39-dim CampaignState
- Execute DRL actions back to platforms via adapters
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = int(os.getenv("DATA_PIPELINE_FETCH_TIMEOUT_SECONDS", "30"))

_ADAPTER_REGISTRY: Dict[str, type] = {}


def _get_adapter(platform: str):
    """Lazy-import and cache adapter classes."""
    if not _ADAPTER_REGISTRY:
        from backend.adapters.google_ads_adapter import GoogleAdsAdapter
        from backend.adapters.meta_ads_adapter import MetaAdsAdapter
        from backend.adapters.tiktok_adapter import TikTokAdsAdapter
        from backend.adapters.amazon_ads_adapter import AmazonAdsAdapter
        from backend.adapters.linkedin_ads_adapter import LinkedInAdsAdapter
        _ADAPTER_REGISTRY.update({
            "google": GoogleAdsAdapter,
            "meta": MetaAdsAdapter,
            "tiktok": TikTokAdsAdapter,
            "amazon": AmazonAdsAdapter,
            "linkedin": LinkedInAdsAdapter,
        })
    cls = _ADAPTER_REGISTRY.get(platform.lower())
    if cls is None:
        raise ValueError(f"Unknown platform: {platform}")
    return cls()


class DataPipeline:
    """Orchestrates data collection, normalisation, and action execution."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes

    # ── fetch all platforms ────────────────────────────────────────

    async def fetch_all_platforms(
        self,
        campaign_mapping: Dict[str, List[str]],
        account_ids: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Fetch metrics from all platforms in parallel.

        Args:
            campaign_mapping: {platform: [campaign_ids]}
            account_ids: {platform: account_id} (optional)

        Returns:
            {platform: {campaign_id: metrics_dict}}
        """
        account_ids = account_ids or {}
        tasks = []
        keys = []

        for platform, campaign_ids in campaign_mapping.items():
            adapter = _get_adapter(platform)
            acct = account_ids.get(platform, "default")
            for cid in campaign_ids:
                keys.append((platform, cid))
                tasks.append(
                    self._fetch_with_fallback(adapter, acct, cid)
                )

        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - t0

        output: Dict[str, Dict[str, Dict[str, Any]]] = {}
        success = 0
        for (platform, cid), result in zip(keys, results):
            if isinstance(result, Exception):
                logger.error(
                    "Failed to fetch %s/%s: %s", platform, cid, result,
                )
                cached = self._get_cached(platform, cid)
                if cached:
                    result = cached
                    logger.info("Using cached metrics for %s/%s", platform, cid)
                else:
                    continue
            else:
                self._set_cached(platform, cid, result)
                success += 1

            output.setdefault(platform, {})[cid] = result

        logger.info(
            "DataPipeline.fetch_all_platforms: %d/%d succeeded in %.1fs",
            success, len(tasks), elapsed,
        )
        return output

    async def _fetch_with_fallback(self, adapter, account_id: str, campaign_id: str):
        """Fetch with a timeout."""
        return await asyncio.wait_for(
            adapter.fetch_campaign_metrics(account_id, campaign_id),
            timeout=FETCH_TIMEOUT,
        )

    # ── normalise to CampaignState ────────────────────────────────

    @staticmethod
    def normalize_to_campaign_state(
        metrics: Dict[str, Any],
        history: Optional[List[float]] = None,
    ):
        """Convert raw adapter metrics to a 39-dim CampaignState.

        Args:
            metrics: dict from any platform adapter
            history: optional 7-day metric history for trend computation

        Returns:
            drl.state_action.CampaignState
        """
        from drl.state_action import CampaignState, MAX_DAILY_SPEND, MAX_TOTAL_SPEND, MAX_DAILY_BUDGET

        platform = metrics.get("platform", "google")
        goal_map = {"roas": 0.0, "cpa": 0.25, "conversions": 0.5, "ctr": 0.75, "revenue": 1.0}
        platform_map = {"google": 0.0, "meta": 0.25, "tiktok": 0.5, "amazon": 0.75, "linkedin": 1.0}

        spend = metrics.get("spend", 0.0)
        daily_budget = metrics.get("daily_budget", max(spend * 1.2, 100))
        total_spend = metrics.get("total_spend", spend * 7)
        impressions = metrics.get("impressions", 0)
        clicks = metrics.get("clicks", 0)
        conversions = metrics.get("conversions", 0)
        frequency = metrics.get("frequency", 0.0)

        ctr_trend = cvr_trend = roas_trend = cpa_trend = spend_trend = 0.0
        if history and len(history) >= 14:
            recent = history[-7:]
            prev = history[-14:-7]
            avg_r = sum(recent) / max(len(recent), 1)
            avg_p = sum(prev) / max(len(prev), 1)
            trend = (avg_r - avg_p) / max(abs(avg_p), 1e-6)
            roas_trend = max(-1, min(1, trend))

        segments = metrics.get("audience_segments", [])
        seg_count = max(len(segments), 1)
        top_seg_roas = 0.0
        if segments:
            top_seg_roas = max(s.get("roas", 0.0) for s in segments) if isinstance(segments[0], dict) else 0.0

        return CampaignState(
            campaign_id=metrics.get("campaign_id", ""),
            platform=platform,
            ctr=metrics.get("ctr", 0.0),
            cvr=conversions / max(clicks, 1),
            roas=min(metrics.get("roas", 0.0) / 10.0, 1.0),
            cpa=1.0 / (1.0 + metrics.get("cpa", 0.0) / 100.0),
            cpc=min(metrics.get("cpc", 0.0) / 10.0, 1.0),
            cpm=min(metrics.get("cpm", 0.0) / 50.0, 1.0),
            spend_velocity=min(spend / max(daily_budget, 1.0), 1.0),
            impression_volume=min(impressions / 1_000_000, 1.0),
            click_volume=min(clicks / 100_000, 1.0),
            conversion_volume=min(conversions / 10_000, 1.0),
            hour_of_day=0.5,
            day_of_week=0.3,
            day_of_month=0.5,
            is_weekend=0.0,
            is_holiday=0.0,
            days_remaining=metrics.get("days_remaining", 30) / 365.0,
            ctr_trend_7d=ctr_trend,
            cvr_trend_7d=cvr_trend,
            roas_trend_7d=roas_trend,
            cpa_trend_7d=cpa_trend,
            spend_trend_7d=spend_trend,
            impression_share=metrics.get("impression_share", 0.5),
            auction_pressure=0.5,
            competitive_position=0.4,
            audience_quality_score=metrics.get("audience_quality_score", 0.5),
            creative_fatigue_score=metrics.get("creative_fatigue_score", 0.0),
            predicted_cvr=conversions / max(clicks, 1),
            predicted_ltv=0.5,
            propensity_score=0.5,
            optimization_goal_encoding=goal_map.get(
                metrics.get("optimization_goal", "roas"), 0.0,
            ),
            platform_encoding=platform_map.get(platform, 0.0),
            campaign_maturity=metrics.get("campaign_maturity", 0.1),
            budget_utilization=min(spend / max(daily_budget, 1.0), 1.0),
            log_daily_spend=float(np.log1p(spend) / np.log1p(MAX_DAILY_SPEND)),
            log_total_campaign_spend=float(np.log1p(total_spend) / np.log1p(MAX_TOTAL_SPEND)),
            log_daily_budget=float(np.log1p(daily_budget) / np.log1p(MAX_DAILY_BUDGET)),
            segment_count=seg_count,
            top_segment_roas=min(top_seg_roas / 10.0, 1.0),
            avg_frequency=frequency,
        )

    # ── execute action on platform ────────────────────────────────

    async def map_drl_action_to_platforms(
        self,
        action: Dict[str, Any],
        platform: str,
        campaign_id: str,
        current_bid: float = 1.0,
        current_budget: float = 100.0,
    ) -> Dict[str, Any]:
        """Execute a DRL action on the appropriate platform.

        Args:
            action: dict with bid_adjustment, budget_adjustment, etc.
            platform: platform name
            campaign_id: campaign to update

        Returns:
            dict with execution status per action type
        """
        adapter = _get_adapter(platform)
        results: Dict[str, Any] = {"platform": platform, "campaign_id": campaign_id}

        bid_adj = action.get("bid_adjustment", 0.0)
        if abs(bid_adj) > 0.01:
            try:
                ok = await adapter.set_bid_adjustment(campaign_id, bid_adj)
                results["bid_applied"] = ok
                results["new_bid"] = current_bid * (1 + bid_adj)
            except Exception as exc:
                results["bid_applied"] = False
                results["bid_error"] = str(exc)

        budget_adj = action.get("budget_adjustment", 0.0)
        if abs(budget_adj) > 0.01:
            new_budget = current_budget * (1 + budget_adj)
            try:
                ok = await adapter.set_budget(campaign_id, new_budget)
                results["budget_applied"] = ok
                results["new_budget"] = new_budget
            except Exception as exc:
                results["budget_applied"] = False
                results["budget_error"] = str(exc)

        logger.info(
            "DataPipeline executed action on %s/%s: %s",
            platform, campaign_id, results,
        )
        return results

    # ── caching ───────────────────────────────────────────────────

    def _cache_key(self, platform: str, campaign_id: str) -> str:
        return f"{platform}:{campaign_id}"

    def _get_cached(self, platform: str, campaign_id: str) -> Optional[Dict[str, Any]]:
        key = self._cache_key(platform, campaign_id)
        ts = self._cache_ts.get(key, 0)
        if time.time() - ts > self._cache_ttl:
            return None
        return self._cache.get(key)

    def _set_cached(self, platform: str, campaign_id: str, data: Dict[str, Any]):
        key = self._cache_key(platform, campaign_id)
        self._cache[key] = data
        self._cache_ts[key] = time.time()

    def __repr__(self) -> str:
        return f"DataPipeline(cached={len(self._cache)})"
