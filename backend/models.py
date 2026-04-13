"""
Backend data models for the DRL advertising optimisation platform.

Two layers:
- Pydantic schemas: request/response validation for the API
- SQLAlchemy ORM models: database persistence (re-exported from shared/models.py)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional  # noqa: F401
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════
#  Pydantic API Schemas
# ═══════════════════════════════════════════════════════════════════


class AudienceSegmentSchema(BaseModel):
    """Single audience segment attached to a campaign."""
    segment_id: str
    segment_name: str = ""
    platform: str = ""
    roas: float = 0.0
    cvr: float = 0.0
    ctr: float = 0.0
    spend: float = 0.0
    frequency: float = 0.0


class CampaignStateSchema(BaseModel):
    """39-dim campaign state submitted by the caller."""
    platform: str = "google"
    campaign_id: str = ""
    spend: float = Field(0.0, ge=0)
    roas: float = Field(0.0, ge=0)
    ctr: float = Field(0.0, ge=0, le=1)
    cvr: float = Field(0.0, ge=0, le=1)
    cpa: float = Field(0.0, ge=0)
    cpc: float = Field(0.0, ge=0)
    cpm: float = Field(0.0, ge=0)
    conversions: int = Field(0, ge=0)
    impressions: int = Field(0, ge=0)
    clicks: int = Field(0, ge=0)
    audience_size: int = Field(0, ge=0)
    frequency: float = Field(0.0, ge=0)
    creative_count: int = Field(1, ge=0)
    daily_budget: float = Field(0.0, ge=0)
    total_budget: float = Field(0.0, ge=0)
    days_remaining: int = Field(30, ge=0)
    budget_utilization: float = Field(0.0, ge=0, le=1)
    creative_fatigue_score: float = Field(0.0, ge=0, le=1)
    audience_quality_score: float = Field(0.5, ge=0, le=1)
    impression_share: float = Field(0.5, ge=0, le=1)
    optimization_goal: str = "roas"
    history: List[float] = Field(default_factory=list)
    audience_segments: List[AudienceSegmentSchema] = Field(default_factory=list)

    @field_validator("platform")
    @classmethod
    def _validate_platform(cls, v: str) -> str:
        allowed = {"google", "meta", "tiktok", "amazon", "linkedin"}
        if v.lower() not in allowed:
            raise ValueError(f"platform must be one of {allowed}")
        return v.lower()

    @field_validator("optimization_goal")
    @classmethod
    def _validate_goal(cls, v: str) -> str:
        allowed = {"roas", "cpa", "conversions", "ctr", "revenue"}
        if v.lower() not in allowed:
            raise ValueError(f"optimization_goal must be one of {allowed}")
        return v.lower()


class DRLOptimizationActionSchema(BaseModel):
    """Output returned by the /optimize endpoint."""
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    bid_adjustment: float = Field(0.0, ge=-0.5, le=0.5)
    budget_adjustment: float = Field(0.0, ge=-0.3, le=0.3)
    audience_action: str = "hold"
    creative_action: str = "hold"
    confidence: float = Field(0.0, ge=0, le=1)
    reasoning: List[str] = Field(default_factory=list)
    narrative: str = ""
    requires_review: bool = False
    auto_apply: bool = False
    model_version: Optional[str] = None
    latency_ms: float = 0.0

    @field_validator("audience_action")
    @classmethod
    def _validate_audience(cls, v: str) -> str:
        allowed = {"hold", "expand", "refine", "exclude"}
        if v.lower() not in allowed:
            raise ValueError(f"audience_action must be one of {allowed}")
        return v.lower()

    @field_validator("creative_action")
    @classmethod
    def _validate_creative(cls, v: str) -> str:
        allowed = {"hold", "rotate", "pause_underperforming", "test_new"}
        if v.lower() not in allowed:
            raise ValueError(f"creative_action must be one of {allowed}")
        return v.lower()


class CampaignOutcomeSchema(BaseModel):
    """Recorded after an action has run for 1+ days."""
    action_id: str
    campaign_id: str = ""
    conversions: int = Field(0, ge=0)
    revenue: float = Field(0.0, ge=0)
    spend: float = Field(0.0, ge=0)
    roas: float = Field(0.0, ge=0)
    ctr: float = Field(0.0, ge=0, le=1)
    outcome_timestamp: Optional[datetime] = None


class DRLDirectiveSchema(BaseModel):
    """LLM integration directive."""
    tone: str = "balanced"
    urgency: int = Field(3, ge=1, le=5)
    discount_limit: float = Field(0.0, ge=0, le=1)
    audience_segments: List[str] = Field(default_factory=list)

    @field_validator("tone")
    @classmethod
    def _validate_tone(cls, v: str) -> str:
        allowed = {"conservative", "aggressive", "balanced",
                   "aggressive_growth", "efficiency_focused",
                   "fresh_angle", "urgency", "consistent"}
        if v.lower() not in allowed:
            raise ValueError(f"tone must be one of {allowed}")
        return v.lower()


# ── Request / Response helpers ────────────────────────────────────

class OptimizeRequest(BaseModel):
    campaign_id: str
    campaign_state: CampaignStateSchema


class RecordOutcomeRequest(BaseModel):
    action_id: str
    campaign_id: str
    outcome: CampaignOutcomeSchema


class ReloadCheckpointRequest(BaseModel):
    version: Optional[str] = None


class BudgetRecommendationRequest(BaseModel):
    """Request for budget recommendation (new campaigns)."""
    platform: str = "google"
    goal: str = "roas"
    campaign_duration_days: int = 30
    total_portfolio_budget: Optional[float] = None
    candidate_budgets: Optional[List[float]] = None


class StatusResponse(BaseModel):
    model_version: Optional[str] = None
    checkpoint_date: Optional[str] = None
    state_dim: int = 39
    initialized: bool = False
    device: str = "cpu"
    inference_count: int = 0
    inference_latency_p95_ms: float = 0.0
    next_retrain_date: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
#  SQLAlchemy ORM models (re-export for convenience)
# ═══════════════════════════════════════════════════════════════════

from backend.shared.models import Base, Campaign, DRLOptimizationAction  # noqa: E402, F401
