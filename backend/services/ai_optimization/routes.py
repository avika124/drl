"""
DRL optimisation API routes.

Provides 4 endpoints under /api/ai-optimization/drl:
  GET  /status                              – DRL system health
  POST /campaigns/{campaign_id}/optimize    – run DRL optimisation
  POST /campaigns/{campaign_id}/record-outcome – record action outcome
  POST /budget-recommendation              – recommend starting budget
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.shared.models import DRLOptimizationAction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-optimization/drl", tags=["drl"])


# ── Pydantic schemas ─────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    optimization_goal: str = "roas"
    multi_objective: bool = False
    objectives: List[str] = Field(default_factory=list)
    generate_tactical: bool = False


class RecordOutcomeRequest(BaseModel):
    tracking_id: str
    current_metrics: Dict[str, Any]
    is_terminal: bool = False


class BudgetRecommendationRequest(BaseModel):
    platform: str
    goal: str = "roas"
    total_portfolio_budget: float
    candidate_budgets: Optional[List[float]] = None
    campaign_duration_days: int = 30


# ── Dependency placeholder ────────────────────────────────────────
# In production these would be real auth + DB session dependencies.

async def get_current_user():
    """Placeholder auth dependency — replace with real implementation."""
    return {"user_id": "system", "organization_id": str(uuid4())}


async def get_db_session():
    """Placeholder DB session dependency — replace with real implementation."""
    return None


# ── Helpers ───────────────────────────────────────────────────────

def _get_drl_layer():
    """Import singleton lazily to avoid circular imports at module load."""
    from backend.services.ai_optimization.engine import drl_integration_layer
    return drl_integration_layer


def _get_cross_platform_optimizer():
    """Import cross-platform optimizer lazily."""
    from backend.services.ai_optimization.engine import cross_platform_optimizer
    return cross_platform_optimizer


# ── 1.  GET /status ───────────────────────────────────────────────

@router.get("/status")
async def drl_status(user=Depends(get_current_user)):
    """Return DRL system health information."""
    layer = _get_drl_layer()
    return {
        "initialized": layer.initialized,
        "model_loaded": layer.initialized and layer.hybrid_optimizer is not None,
        "device": layer.device,
        "state_dim": 36,
        "version": "1.2.0",
    }


# ── 2.  POST /campaigns/{campaign_id}/optimize ───────────────────

@router.post("/campaigns/{campaign_id}/optimize")
async def drl_optimize(
    campaign_id: str,
    body: OptimizeRequest,
    user=Depends(get_current_user),
    db=Depends(get_db_session),
):
    """Run DRL optimisation for a campaign and persist the action."""
    layer = _get_drl_layer()

    if not layer.initialized:
        ok = await layer.initialize()
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DRL engine failed to initialize",
            )

    # In a real system, fetch campaign + metrics from DB.
    campaign_data: Dict[str, Any] = {"campaign_id": campaign_id, "optimization_goal": body.optimization_goal}
    metrics: Dict[str, Any] = {}

    result = await layer.get_optimization(
        campaign_id=campaign_id,
        campaign_data=campaign_data,
        metrics=metrics,
        optimization_goal=body.optimization_goal,
        generate_tactical=body.generate_tactical,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    action_id = str(uuid4())

    # Persist audit row (placeholder — requires real DB session)
    if db is not None:
        action_row = DRLOptimizationAction(
            id=action_id,
            campaign_id=campaign_id,
            organization_id=user.get("organization_id", ""),
            action_type="drl_optimize",
            action_details=result.get("action", {}) if isinstance(result.get("action"), dict) else {},
            state_before=result.get("state", {}) if isinstance(result.get("state"), dict) else {},
            metrics_before=metrics,
            confidence=result.get("confidence", 0.0),
            requires_review=result.get("requires_review", False),
            is_auto_applied=result.get("auto_apply", False),
            status="pending",
        )
        db.add(action_row)
        await db.commit()

    return {
        "action_id": action_id,
        "campaign_id": campaign_id,
        "recommendations": result.get("recommendations", []),
        "confidence": result.get("confidence", 0.0),
        "requires_review": result.get("requires_review", True),
        "auto_apply": result.get("auto_apply", False),
        "optimization_goal": body.optimization_goal,
    }


# ── 3.  POST /campaigns/{campaign_id}/record-outcome ─────────────

@router.post("/campaigns/{campaign_id}/record-outcome")
async def drl_record_outcome(
    campaign_id: str,
    body: RecordOutcomeRequest,
    user=Depends(get_current_user),
    db=Depends(get_db_session),
):
    """Record the outcome of a previously applied DRL action."""
    layer = _get_drl_layer()

    if not layer.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DRL engine not initialized",
        )

    from drl.state_action import CampaignState, ActionSpace

    # In production, load DRLOptimizationAction row by tracking_id from DB.
    state = CampaignState()
    action = ActionSpace()

    await layer.record_outcome(
        campaign_id=campaign_id,
        action=action,
        state=state,
        outcome_metrics=body.current_metrics,
        is_terminal=body.is_terminal,
    )

    # Update persisted row (placeholder)
    if db is not None:
        pass  # db.query(DRLOptimizationAction).filter_by(id=body.tracking_id).update(...)

    return {"status": "recorded", "tracking_id": body.tracking_id}


# ── 4.  POST /budget-recommendation ──────────────────────────────

@router.post("/budget-recommendation")
async def drl_budget_recommendation(
    body: BudgetRecommendationRequest,
    user=Depends(get_current_user),
):
    """Recommend a starting daily budget for a new campaign."""
    optimizer = _get_cross_platform_optimizer()
    if optimizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cross-platform optimizer not available",
        )

    from drl.cross_platform_optimizer import BudgetRecommendationConfig

    config = BudgetRecommendationConfig(
        campaign_duration_days=body.campaign_duration_days,
    )
    if body.candidate_budgets:
        config.candidate_budgets = body.candidate_budgets

    recommendation = await optimizer.recommend_budget(
        organization_id=user.get("organization_id", ""),
        new_campaign_info={
            "platform": body.platform,
            "goal": body.goal,
            "duration_days": body.campaign_duration_days,
        },
        total_portfolio_budget=body.total_portfolio_budget,
        config=config,
    )

    return recommendation.to_dict()
