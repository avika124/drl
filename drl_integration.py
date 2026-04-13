"""
DRL integration layer for the AI Optimization Engine.

Provides:
- DRLIntegrationLayer: full async integration between platform campaign data
  and the DRL module.  Maps raw metrics to 39-dim CampaignState, converts
  DRL actions to actionable recommendations, and coordinates the hybrid
  DRL+LLM optimisation pipeline with continuous learning.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import DRLConfig, GuardrailConfig, RewardConfig, TrainingConfig, OptimizationGoal
from .state_action import (
    AudienceAction,
    CampaignState,
    ActionSpace,
    CreativeAction,
    DRLDirective,
    MAX_DAILY_BUDGET,
    MAX_DAILY_SPEND,
    MAX_TOTAL_SPEND,
)
from .sac_agent import load_sac_for_inference
from .reward_functions import RewardComputer

logger = logging.getLogger(__name__)

# ── Encoding tables ──────────────────────────────────────────────────

PLATFORM_ENCODING: Dict[str, float] = {
    "google": 0.0,
    "meta": 0.25,
    "tiktok": 0.5,
    "amazon": 0.75,
    "linkedin": 1.0,
}

GOAL_ENCODING: Dict[str, float] = {
    "roas": 0.0,
    "cpa": 0.25,
    "conversions": 0.5,
    "ctr": 0.75,
    "revenue": 1.0,
    "profit": 1.0,
}

STATE_FIELDS: List[str] = [
    "ctr", "cvr", "roas", "cpa", "cpc", "cpm",
    "spend_velocity", "impression_volume", "click_volume", "conversion_volume",
    "hour_of_day", "day_of_week", "day_of_month",
    "is_weekend", "is_holiday", "days_remaining",
    "ctr_trend_7d", "cvr_trend_7d", "roas_trend_7d", "cpa_trend_7d", "spend_trend_7d",
    "impression_share", "auction_pressure", "competitive_position",
    "audience_quality_score", "creative_fatigue_score",
    "predicted_cvr", "predicted_ltv", "propensity_score",
    "optimization_goal_encoding", "platform_encoding",
    "campaign_maturity", "budget_utilization",
    "log_daily_spend", "log_total_campaign_spend", "log_daily_budget",
    "segment_count", "top_segment_roas", "avg_frequency",
    "target_cpa_norm", "min_roas_norm", "daily_budget_limit_norm",
]


class DRLIntegrationLayer:
    """Full async integration between the AI Optimization Engine and the DRL module.

    Handles:
    - Mapping platform campaign data to 39-dim CampaignState
    - Converting DRL actions to platform-level recommendations
    - Managing the DRL agent lifecycle (load, init, checkpoint)
    - Coordinating hybrid DRL+LLM optimisation
    - Recording outcomes for continuous learning
    """

    def __init__(
        self,
        model_dir: str = "models/drl",
        device: str = "cpu",
        min_confidence: float = 0.7,
        auto_apply_threshold: float = 0.85,
    ):
        self.model_dir = model_dir
        self.device = device
        self.min_confidence = min_confidence
        self.auto_apply_threshold = auto_apply_threshold

        self.hybrid_optimizer = None
        self.continuous_learning = None
        self.initialized = False

        self._reward_computer = RewardComputer(RewardConfig())

        logger.info("DRLIntegrationLayer created (model_dir=%s, device=%s)", model_dir, device)

    # ── Lifecycle ─────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Load SAC model, set up HybridDRLLLMOptimizer and ContinuousLearningEngine.

        Returns True on success, False on failure (logged, never raises).
        """
        try:
            from .safe_agent import SafeDRLAgent, CampaignContext
            from .hybrid_optimizer import HybridDRLLLMOptimizer
            from .continuous_learning import ContinuousLearningEngine
            from .replay_buffer import create_replay_buffer

            state_dim = int(os.environ.get("DRL_STATE_DIM", "42"))
            drl_config = DRLConfig(
                state_dim=state_dim,
                continuous_action_dim=2,
                discrete_action_dims=[4, 4],
                hidden_dim=256,
                model_dir=self.model_dir,
            )
            guardrail_config = GuardrailConfig()

            # drl-2: state_dim=42 (39 + 3 constraint features). Use 39 for legacy checkpoints.
            state_dim = int(os.environ.get("DRL_STATE_DIM", "42"))
            agent, _features = load_sac_for_inference(
                model_dir=self.model_dir,
                device=self.device,
                state_dim=state_dim,
            )

            safe_agent = SafeDRLAgent(
                agent=agent,
                guardrails=guardrail_config,
            )

            self.hybrid_optimizer = HybridDRLLLMOptimizer(
                drl_agent=safe_agent,
            )

            replay_buffer = create_replay_buffer(capacity=100_000, use_prioritized=True)
            training_config = TrainingConfig(batch_size=256, min_buffer_size=1000, use_per=True)
            from .continuous_learning import LearningMode
            self.continuous_learning = ContinuousLearningEngine(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=training_config,
                learning_mode=LearningMode.HYBRID,
                update_frequency=int(os.environ.get("CONTINUOUS_LEARNING_THRESHOLD", "1000")),
                batch_interval_minutes=60,
            )

            self.initialized = True
            logger.info("DRL components initialised successfully")
            return True

        except Exception as exc:
            logger.error("Failed to initialise DRL components: %s", exc, exc_info=True)
            self.initialized = False
            return False

    # ── State mapping ─────────────────────────────────────────────

    def map_campaign_to_state(
        self,
        campaign_data: dict,
        metrics: dict,
        historical_metrics: list | None = None,
        ml_features: dict | None = None,
    ) -> CampaignState:
        """Build a full 39-dim CampaignState from raw platform data."""

        impressions = metrics.get("impressions", 0)
        clicks = metrics.get("clicks", 0)
        conversions = metrics.get("conversions", 0)
        spend = metrics.get("spend", 0.0)
        revenue = metrics.get("revenue", 0.0)

        ctr = clicks / max(impressions, 1)
        cvr = conversions / max(clicks, 1)
        roas = revenue / max(spend, 1.0)
        cpa = spend / max(conversions, 1)
        cpc = spend / max(clicks, 1.0)
        cpm = (spend / max(impressions, 1)) * 1000.0

        budget = campaign_data.get("budget", 1.0) or 1.0
        daily_budget = campaign_data.get("daily_budget", budget) or budget
        daily_spend = spend
        total_campaign_spend = campaign_data.get("total_spend", spend)

        # Trends
        ctr_trend = cvr_trend = roas_trend = cpa_trend = spend_trend = 0.0
        if historical_metrics and len(historical_metrics) > 1:
            recent = historical_metrics[-7:] if len(historical_metrics) >= 7 else historical_metrics
            older = historical_metrics[-14:-7] if len(historical_metrics) >= 14 else historical_metrics[:len(recent)]
            if older:
                def _safe_div(a, b): return a / b if b else 0.0

                old_imp = max(sum(m.get("impressions", 0) for m in older), 1)
                new_imp = max(sum(m.get("impressions", 0) for m in recent), 1)
                old_clk = sum(m.get("clicks", 0) for m in older)
                new_clk = sum(m.get("clicks", 0) for m in recent)
                old_conv = sum(m.get("conversions", 0) for m in older)
                new_conv = sum(m.get("conversions", 0) for m in recent)
                old_spend = max(sum(m.get("spend", 0) for m in older), 0.1)
                new_spend = max(sum(m.get("spend", 0) for m in recent), 0.1)
                old_rev = sum(m.get("revenue", 0) for m in older)
                new_rev = sum(m.get("revenue", 0) for m in recent)

                old_ctr = old_clk / old_imp
                new_ctr = new_clk / new_imp
                ctr_trend = np.clip(_safe_div(new_ctr - old_ctr, max(old_ctr, 1e-6)), -1, 1)

                old_cvr = old_conv / max(old_clk, 1)
                new_cvr = new_conv / max(new_clk, 1)
                cvr_trend = np.clip(_safe_div(new_cvr - old_cvr, max(old_cvr, 1e-6)), -1, 1)

                old_roas = old_rev / old_spend
                new_roas = new_rev / new_spend
                roas_trend = np.clip(_safe_div(new_roas - old_roas, max(old_roas, 0.1)), -1, 1)

                old_cpa = old_spend / max(old_conv, 1)
                new_cpa = new_spend / max(new_conv, 1)
                cpa_trend = np.clip(-_safe_div(new_cpa - old_cpa, max(old_cpa, 0.1)), -1, 1)

                spend_trend = np.clip(_safe_div(new_spend - old_spend, old_spend), -1, 1)

        # Temporal
        now = datetime.now(timezone.utc)
        hour_of_day = now.hour / 24.0
        day_of_week = now.weekday() / 7.0
        day_of_month = now.day / 31.0
        is_weekend = 1.0 if now.weekday() >= 5 else 0.0
        is_holiday = 0.0

        start_date = campaign_data.get("start_date")
        end_date = campaign_data.get("end_date")
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        days_remaining = 0.0
        if end_date:
            days_remaining = max((end_date - now).days, 0) / 365.0

        campaign_maturity = 0.0
        if start_date:
            campaign_maturity = min((now - start_date).days / 365.0, 1.0)

        budget_utilization = min(total_campaign_spend / max(budget, 1.0), 1.0)

        ml = ml_features or {}
        platform = str(campaign_data.get("platform", "google")).lower()
        goal = str(campaign_data.get("optimization_goal", "roas")).lower()

        return CampaignState(
            campaign_id=str(campaign_data.get("campaign_id", "")),
            organization_id=str(campaign_data.get("organization_id", "")),
            platform=platform,
            ctr=min(ctr / 0.10, 1.0),
            cvr=min(cvr / 0.20, 1.0),
            roas=min(roas / 10.0, 1.0),
            cpa=1.0 / (1.0 + cpa / 100.0),
            cpc=min(cpc / 10.0, 1.0),
            cpm=min(cpm / 50.0, 1.0),
            spend_velocity=min(daily_spend / max(daily_budget, 1.0), 1.0),
            impression_volume=min(impressions / 1_000_000, 1.0),
            click_volume=min(clicks / 10_000, 1.0),
            conversion_volume=min(conversions / 1_000, 1.0),
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            days_remaining=days_remaining,
            ctr_trend_7d=float(ctr_trend),
            cvr_trend_7d=float(cvr_trend),
            roas_trend_7d=float(roas_trend),
            cpa_trend_7d=float(cpa_trend),
            spend_trend_7d=float(spend_trend),
            impression_share=metrics.get("impression_share", 0.5),
            auction_pressure=metrics.get("auction_pressure", 0.5),
            competitive_position=metrics.get("competitive_position", 0.5),
            audience_quality_score=ml.get("audience_quality_score", 0.5),
            creative_fatigue_score=ml.get("creative_fatigue_score", 0.0),
            predicted_cvr=ml.get("predicted_cvr", cvr),
            predicted_ltv=ml.get("predicted_ltv", 0.5),
            propensity_score=ml.get("propensity_score", 0.5),
            optimization_goal_encoding=GOAL_ENCODING.get(goal, 0.0),
            platform_encoding=PLATFORM_ENCODING.get(platform, 0.0),
            campaign_maturity=campaign_maturity,
            budget_utilization=budget_utilization,
            log_daily_spend=float(np.log1p(daily_spend) / np.log1p(MAX_DAILY_SPEND)),
            log_total_campaign_spend=float(np.log1p(total_campaign_spend) / np.log1p(MAX_TOTAL_SPEND)),
            log_daily_budget=float(np.log1p(daily_budget) / np.log1p(MAX_DAILY_BUDGET)),
            segment_count=float(campaign_data.get("segment_count", 1)) / 10.0,
            top_segment_roas=min(metrics.get("top_segment_roas", roas) / 10.0, 1.0),
            avg_frequency=min(metrics.get("avg_frequency", 0.0) / 10.0, 1.0),
            target_cpa_norm=float(np.log1p(campaign_data.get("target_cpa", 50.0))) / np.log1p(1000.0),
            min_roas_norm=min(campaign_data.get("min_roas", 2.0) / 10.0, 1.0),
            daily_budget_limit_norm=float(np.log1p(daily_budget) / np.log1p(MAX_DAILY_BUDGET)),
        )

    # ── build_state (legacy convenience) ──────────────────────────

    def build_state(self, raw_metrics: Dict[str, float]) -> torch.Tensor:
        """Convert a raw_metrics dict into a [39] float tensor.

        Kept for backward compatibility with code that expects a plain tensor.
        """
        impressions = raw_metrics.get("impressions", 0.0)
        clicks = raw_metrics.get("clicks", 0.0)
        conversions = raw_metrics.get("conversions", 0.0)
        spend = raw_metrics.get("spend", 0.0)
        revenue = raw_metrics.get("revenue", 0.0)

        ctr = clicks / max(impressions, 1)
        cvr = conversions / max(clicks, 1)
        roas = revenue / max(spend, 1)
        cpa = spend / max(conversions, 1)
        cpc = spend / max(clicks, 1)
        cpm = (spend / max(impressions, 1)) * 1000

        platform_str = str(raw_metrics.get("platform", "google")).lower()
        goal_str = str(raw_metrics.get("optimization_goal", "roas")).lower()

        values: Dict[str, float] = {
            "ctr": ctr,
            "cvr": cvr,
            "roas": roas,
            "cpa": cpa,
            "cpc": cpc,
            "cpm": cpm,
            "spend_velocity": raw_metrics.get("spend_velocity", 0.0),
            "impression_volume": raw_metrics.get("impression_volume", impressions),
            "click_volume": raw_metrics.get("click_volume", clicks),
            "conversion_volume": raw_metrics.get("conversion_volume", conversions),
            "hour_of_day": raw_metrics.get("hour_of_day", 12.0),
            "day_of_week": raw_metrics.get("day_of_week", 3.0),
            "day_of_month": raw_metrics.get("day_of_month", 15.0),
            "is_weekend": raw_metrics.get("is_weekend", 0.0),
            "is_holiday": raw_metrics.get("is_holiday", 0.0),
            "days_remaining": raw_metrics.get("days_remaining", 30.0),
            "ctr_trend_7d": raw_metrics.get("ctr_trend_7d", 0.0),
            "cvr_trend_7d": raw_metrics.get("cvr_trend_7d", 0.0),
            "roas_trend_7d": raw_metrics.get("roas_trend_7d", 0.0),
            "cpa_trend_7d": raw_metrics.get("cpa_trend_7d", 0.0),
            "spend_trend_7d": raw_metrics.get("spend_trend_7d", 0.0),
            "impression_share": raw_metrics.get("impression_share", 0.5),
            "auction_pressure": raw_metrics.get("auction_pressure", 0.5),
            "competitive_position": raw_metrics.get("competitive_position", 0.5),
            "audience_quality_score": raw_metrics.get("audience_quality_score", 0.5),
            "creative_fatigue_score": raw_metrics.get("creative_fatigue_score", 0.0),
            "predicted_cvr": raw_metrics.get("predicted_cvr", cvr),
            "predicted_ltv": raw_metrics.get("predicted_ltv", 0.5),
            "propensity_score": raw_metrics.get("propensity_score", 0.5),
            "optimization_goal_encoding": GOAL_ENCODING.get(goal_str, 0.0),
            "platform_encoding": PLATFORM_ENCODING.get(platform_str, 0.0),
            "campaign_maturity": raw_metrics.get("campaign_maturity", 0.5),
            "budget_utilization": raw_metrics.get("budget_utilization", 0.5),
            "log_daily_spend": float(np.log1p(raw_metrics.get("daily_spend", 0.0)) / np.log1p(MAX_DAILY_SPEND)),
            "log_total_campaign_spend": float(np.log1p(raw_metrics.get("total_campaign_spend", 0.0)) / np.log1p(MAX_TOTAL_SPEND)),
            "log_daily_budget": float(np.log1p(raw_metrics.get("daily_budget", 0.0)) / np.log1p(MAX_DAILY_BUDGET)),
            "segment_count": raw_metrics.get("segment_count", 1) / 10.0,
            "top_segment_roas": raw_metrics.get("top_segment_roas", 0.0),
            "avg_frequency": raw_metrics.get("avg_frequency", 0.0) / 10.0,
            "target_cpa_norm": float(np.log1p(raw_metrics.get("target_cpa", 50.0))) / np.log1p(1000.0),
            "min_roas_norm": min(raw_metrics.get("min_roas", 2.0) / 10.0, 1.0),
            "daily_budget_limit_norm": float(np.log1p(raw_metrics.get("daily_budget", 100.0)) / np.log1p(MAX_DAILY_BUDGET)),
        }

        vec = np.array([values[f] for f in STATE_FIELDS], dtype=np.float32)
        return torch.tensor(vec, dtype=torch.float32, device=self.device)

    # ── Action → recommendations ──────────────────────────────────

    def map_action_to_recommendations(
        self,
        action: ActionSpace,
        state: CampaignState,
        campaign_data: dict,
    ) -> List[Dict[str, Any]]:
        """Convert ActionSpace to a list of recommendation dicts."""
        recs: List[Dict[str, Any]] = []
        current_bid = campaign_data.get("current_bid", 1.0)
        current_budget = campaign_data.get("daily_budget", 100.0)
        conf = action.confidence

        # Bid: scale [-1,1] → [-30%, +50%]
        if action.bid_adjustment >= 0:
            scaled_bid = action.bid_adjustment * 0.50
        else:
            scaled_bid = action.bid_adjustment * 0.30
        if abs(scaled_bid) > 0.005:
            recs.append({
                "type": "bid_optimization",
                "action": "increase_bid" if scaled_bid > 0 else "decrease_bid",
                "current_value": current_bid,
                "recommended_value": round(current_bid * (1 + scaled_bid), 4),
                "change_percent": round(scaled_bid * 100, 2),
                "rationale": f"{'Increase' if scaled_bid > 0 else 'Decrease'} bid by {abs(scaled_bid)*100:.1f}% to optimise performance",
                "confidence": conf,
                "priority": 1,
            })

        # Budget: scale [-1,1] → [-30%, +30%]
        scaled_budget = action.budget_adjustment * 0.30
        if abs(scaled_budget) > 0.005:
            recs.append({
                "type": "budget_optimization",
                "action": "increase_budget" if scaled_budget > 0 else "decrease_budget",
                "current_value": current_budget,
                "recommended_value": round(current_budget * (1 + scaled_budget), 2),
                "change_percent": round(scaled_budget * 100, 2),
                "rationale": f"{'Increase' if scaled_budget > 0 else 'Decrease'} budget by {abs(scaled_budget)*100:.1f}%",
                "confidence": conf,
                "priority": 1,
            })

        _audience_labels = {
            1: ("expand_audience", "Expand audience to increase reach"),
            2: ("refine_audience", "Refine audience targeting for efficiency"),
            3: ("exclude_segments", "Exclude underperforming segments"),
        }
        if action.audience_action in _audience_labels:
            label, rationale = _audience_labels[action.audience_action]
            recs.append({
                "type": "audience_optimization",
                "action": label,
                "current_value": None,
                "recommended_value": label,
                "change_percent": None,
                "rationale": rationale,
                "confidence": conf,
                "priority": 2,
            })

        _creative_labels = {
            1: ("rotate_creatives", "Rotate creatives to combat fatigue"),
            2: ("pause_underperforming", "Pause underperforming creatives"),
            3: ("test_new", "Test new creative variants"),
        }
        if action.creative_action in _creative_labels:
            label, rationale = _creative_labels[action.creative_action]
            recs.append({
                "type": "creative_optimization",
                "action": label,
                "current_value": None,
                "recommended_value": label,
                "change_percent": None,
                "rationale": rationale,
                "confidence": conf,
                "priority": 2,
            })

        return recs

    # ── Full optimisation pipeline ────────────────────────────────

    async def get_optimization(
        self,
        campaign_id: str,
        campaign_data: dict,
        metrics: dict,
        optimization_goal: str = "roas",
        historical_metrics: list | None = None,
        ml_features: dict | None = None,
        generate_tactical: bool = False,
    ) -> Dict[str, Any]:
        """Run the full DRL optimisation pipeline for a single campaign."""
        if not self.initialized:
            await self.initialize()

        if not self.initialized or not self.hybrid_optimizer:
            return {"error": "DRL not initialized", "recommendations": []}

        try:
            from .safe_agent import CampaignContext

            state = self.map_campaign_to_state(
                campaign_data=campaign_data,
                metrics=metrics,
                historical_metrics=historical_metrics,
                ml_features=ml_features,
            )

            context = CampaignContext(
                campaign_id=campaign_id,
                current_bid=campaign_data.get("current_bid", 1.0),
                current_budget=campaign_data.get("daily_budget", 100.0),
                last_action_at=None,
                actions_today=0,
                current_roas=metrics.get("roas", 0.0),
                current_cpa=metrics.get("cpa", 0.0),
                target_cpa=campaign_data.get("target_cpa"),
                min_roas=campaign_data.get("min_roas"),
                total_spend=metrics.get("spend", 0.0),
            )

            result = await self.hybrid_optimizer.optimize(
                state=state,
                context=context,
                campaign_info=campaign_data,
                generate_tactical=generate_tactical,
            )

            recs = self.map_action_to_recommendations(
                action=result.action,
                state=state,
                campaign_data=campaign_data,
            )

            confidence = result.combined_confidence

            return {
                "action": result.action,
                "state": state,
                "recommendations": recs,
                "confidence": confidence,
                "requires_review": confidence < self.min_confidence,
                "auto_apply": confidence >= self.auto_apply_threshold,
                "optimization_goal": optimization_goal,
                "campaign_id": campaign_id,
            }

        except Exception as exc:
            logger.error("DRL optimisation error: %s", exc, exc_info=True)
            return {"error": str(exc), "recommendations": []}

    # ── Outcome recording ─────────────────────────────────────────

    async def record_outcome(
        self,
        campaign_id: str,
        action: ActionSpace,
        state: CampaignState,
        outcome_metrics: dict,
        is_terminal: bool = False,
    ) -> None:
        """Compute reward and feed transition to continuous learning."""
        if not self.initialized or not self.continuous_learning:
            logger.warning("Cannot record outcome — DRL not initialised")
            return

        try:
            from .replay_buffer import Transition

            state_dict = state.to_dict()
            reward_obj = self._reward_computer.compute(
                metrics_before=state_dict,
                metrics_after=outcome_metrics,
                action={"bid_adjustment": action.bid_adjustment, "budget_adjustment": action.budget_adjustment},
                goal=OptimizationGoal(state_dict.get("optimization_goal_encoding", "roas") if isinstance(state_dict.get("optimization_goal_encoding"), str) else "roas"),
                constraints={},
            )

            transition = Transition(
                state=state.to_tensor().numpy(),
                continuous_action=np.array([action.bid_adjustment, action.budget_adjustment]),
                discrete_action=np.array([action.audience_action, action.creative_action]),
                reward=reward_obj.total,
                next_state=None,
                done=is_terminal,
            )

            self.continuous_learning.add_transition(transition)
            logger.info(
                "Recorded outcome for campaign %s — reward=%.4f",
                campaign_id, reward_obj.total,
            )
        except Exception as exc:
            logger.error("Error recording outcome: %s", exc, exc_info=True)

    # ── repr ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DRLIntegrationLayer(model_dir={self.model_dir!r}, "
            f"device={self.device!r}, initialized={self.initialized})"
        )
