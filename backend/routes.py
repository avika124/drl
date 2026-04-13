"""
DRL API routes — 5 endpoints under ``/api/drl``.

Endpoints:
  POST /api/drl/optimize              – run DRL optimisation
  POST /api/drl/record-outcome         – record action outcome
  GET  /api/drl/status                 – model health & metrics
  POST /api/drl/reload-checkpoint      – hot-swap checkpoint
  POST /api/drl/budget-recommendation  – recommend starting budget for new campaigns
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from backend.engine import get_engine
from backend.models import (
    BudgetRecommendationRequest,
    CampaignOutcomeSchema,
    CampaignStateSchema,
    DRLOptimizationActionSchema,
    OptimizeRequest,
    RecordOutcomeRequest,
    ReloadCheckpointRequest,
    StatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/drl", tags=["drl"])


# ── Dependency placeholder ────────────────────────────────────────

async def get_current_user():
    """Replace with real auth in production."""
    return {"user_id": "system"}


async def get_db_session():
    """Replace with real async DB session in production."""
    return None


# ── 1. POST /optimize ─────────────────────────────────────────────

@router.post("/optimize", response_model=DRLOptimizationActionSchema)
async def drl_optimize(
    body: OptimizeRequest,
    user=Depends(get_current_user),
    db=Depends(get_db_session),
):
    """Run SAC inference on a campaign state and return recommendations.

    Latency SLA: < 100 ms (excl. platform API calls).
    """
    engine = get_engine()

    if not engine.initialized:
        ok = await engine.initialize()
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DRL model not loaded — checkpoint missing or corrupt",
            )

    campaign_state = _schema_to_campaign_state(body.campaign_state)

    try:
        result = await engine.get_optimization(
            campaign_state,
            raw_context=body.campaign_state,
            campaign_id=body.campaign_id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    if db is not None:
        from backend.shared.models import DRLOptimizationAction
        row = DRLOptimizationAction(
            id=result["action_id"],
            campaign_id=body.campaign_id,
            action_type="drl_optimize",
            action_details={
                "bid_adjustment": result["bid_adjustment"],
                "budget_adjustment": result["budget_adjustment"],
                "audience_action": result["audience_action"],
                "creative_action": result["creative_action"],
            },
            confidence=result["confidence"],
            requires_review=result["requires_review"],
            is_auto_applied=result["auto_apply"],
            reasoning=result["narrative"],
            status="pending",
        )
        db.add(row)
        await db.commit()

    return DRLOptimizationActionSchema(**result)


# ── 2. POST /record-outcome ───────────────────────────────────────

@router.post("/record-outcome")
async def drl_record_outcome(
    body: RecordOutcomeRequest,
    user=Depends(get_current_user),
    db=Depends(get_db_session),
):
    """Record the outcome of a previously applied DRL action.

    Triggers continuous learning when the replay buffer crosses the
    configured threshold.
    """
    engine = get_engine()

    if not engine.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DRL engine not initialized",
        )

    outcome_dict = body.outcome.model_dump()
    result = await engine.record_outcome(
        campaign_id=body.campaign_id,
        action_id=body.action_id,
        outcome=outcome_dict,
    )

    if db is not None:
        pass  # update DRLOptimizationAction row with outcome fields

    response = {
        "recorded": True,
        "model_version": result["model_version"],
        "reward": result["reward"],
        "retrain_triggered": result["retrain_triggered"],
        "buffer_size": result["buffer_size"],
    }
    if result.get("rollback_required"):
        response["rollback_required"] = True
        response["rollback_action"] = result["rollback_action"]
    return response


# ── 3. POST /budget-recommendation ─────────────────────────────────

@router.post("/budget-recommendation")
async def drl_budget_recommendation(
    body: BudgetRecommendationRequest,
    user=Depends(get_current_user),
):
    """Recommend a starting daily budget for a new campaign before launch."""
    engine = get_engine()

    if not engine.initialized:
        ok = await engine.initialize()
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DRL engine not initialized",
            )

    from drl.cross_platform_optimizer import BudgetRecommendationConfig

    config = BudgetRecommendationConfig(
        campaign_duration_days=body.campaign_duration_days,
    )
    if body.candidate_budgets:
        config.candidate_budgets = body.candidate_budgets

    recommendation = await engine._xp_optimizer.recommend_budget(
        organization_id=user.get("organization_id", "default"),
        new_campaign_info={
            "platform": body.platform,
            "goal": body.goal,
            "duration_days": body.campaign_duration_days,
        },
        total_portfolio_budget=body.total_portfolio_budget,
        config=config,
    )

    return recommendation.to_dict()


# ── 4. GET /status ────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
async def drl_status(user=Depends(get_current_user)):
    """Return DRL system health — for monitoring dashboards."""
    engine = get_engine()
    info = engine.get_model_version()
    return StatusResponse(**info)


# ── 5. POST /reload-checkpoint ─────────────────────────────────────

@router.post("/reload-checkpoint")
async def drl_reload_checkpoint(
    body: ReloadCheckpointRequest,
    user=Depends(get_current_user),
):
    """Hot-swap to a different checkpoint without downtime."""
    engine = get_engine()

    try:
        result = await engine.reload_checkpoint(version=body.version)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return {
        "reloaded": True,
        "new_version": result["new_version"],
        "old_version": result["old_version"],
    }


# ── Helpers ───────────────────────────────────────────────────────

def _schema_to_campaign_state(schema: CampaignStateSchema):
    """Convert Pydantic CampaignStateSchema to ``drl.state_action.CampaignState``."""
    import numpy as np
    from drl.state_action import CampaignState, MAX_DAILY_SPEND, MAX_TOTAL_SPEND, MAX_DAILY_BUDGET

    goal_map = {"roas": 0.0, "cpa": 0.25, "conversions": 0.5, "ctr": 0.75, "revenue": 1.0}
    platform_map = {"google": 0.0, "meta": 0.25, "tiktok": 0.5, "amazon": 0.75, "linkedin": 1.0}

    seg_count = max(len(schema.audience_segments), 1)
    top_seg_roas = 0.0
    avg_freq = schema.frequency
    if schema.audience_segments:
        top_seg_roas = max(s.roas for s in schema.audience_segments)
        freqs = [s.frequency for s in schema.audience_segments if s.frequency > 0]
        if freqs:
            avg_freq = sum(freqs) / len(freqs)

    return CampaignState(
        campaign_id=schema.campaign_id,
        platform=schema.platform,
        ctr=schema.ctr,
        cvr=schema.cvr,
        roas=min(schema.roas / 10.0, 1.0),
        cpa=1.0 / (1.0 + schema.cpa / 100.0),
        cpc=min(schema.cpc / 10.0, 1.0),
        cpm=min(schema.cpm / 50.0, 1.0),
        spend_velocity=min(schema.spend / max(schema.daily_budget, 1.0), 1.0),
        impression_volume=min(schema.impressions / 1_000_000, 1.0),
        click_volume=min(schema.clicks / 100_000, 1.0),
        conversion_volume=min(schema.conversions / 10_000, 1.0),
        days_remaining=schema.days_remaining / 365.0,
        impression_share=schema.impression_share,
        audience_quality_score=schema.audience_quality_score,
        creative_fatigue_score=schema.creative_fatigue_score,
        budget_utilization=schema.budget_utilization,
        optimization_goal_encoding=goal_map.get(schema.optimization_goal, 0.0),
        platform_encoding=platform_map.get(schema.platform, 0.0),
        log_daily_spend=float(np.log1p(schema.spend) / np.log1p(MAX_DAILY_SPEND)),
        log_total_campaign_spend=float(np.log1p(schema.total_budget * schema.budget_utilization) / np.log1p(MAX_TOTAL_SPEND)),
        log_daily_budget=float(np.log1p(schema.daily_budget) / np.log1p(MAX_DAILY_BUDGET)),
        segment_count=seg_count,
        top_segment_roas=min(top_seg_roas / 10.0, 1.0),
        avg_frequency=avg_freq,
    )
