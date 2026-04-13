"""
Cross-Platform Traffic Optimizer

Orchestration layer that jointly allocates budget across all platforms
simultaneously, sitting above the per-campaign DRL optimization.

Architecture:
    Portfolio State (all platforms)
            ↓
    Cross-Platform Optimizer (this module)
    ├─ Marginal ROAS estimation per platform
    ├─ Convex budget allocation via constrained optimization
    ├─ Platform shift recommendations
    └─ Portfolio-level constraint enforcement
            ↓
    Per-Platform DRL Optimization (existing HybridDRLLLMOptimizer)
    ├─ Campaign-level bid/budget/audience/creative
    └─ Tactical LLM execution
            ↓
    Combined Cross-Platform + Campaign Recommendations
"""

import asyncio
import numpy as np
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict

from .config import DRLConfig, GuardrailConfig, RewardConfig, OptimizationGoal
from .state_action import CampaignState, ActionSpace, MAX_DAILY_SPEND, MAX_TOTAL_SPEND, MAX_DAILY_BUDGET
from .safe_agent import SafeDRLAgent, CampaignContext, ActionValidationResult
from .hybrid_optimizer import (
    HybridDRLLLMOptimizer,
    OptimizationResult,
    BatchOptimizer,
)
from .audience_constraints import (
    AudienceConstraintManager,
    AudienceConstraintResult,
    AudienceSegment,
)
from .platform_model_registry import PlatformModelRegistry
from .x_model import XModelAgent, XModelAction, build_x_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform definitions
# ---------------------------------------------------------------------------

class Platform(Enum):
    """Supported advertising platforms"""
    META = "meta"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    AMAZON = "amazon"
    WALMART = "walmart"


PLATFORM_ENCODING: Dict[str, float] = {
    "meta": 0.0,
    "google": 0.2,
    "tiktok": 0.4,
    "amazon": 0.6,
    "walmart": 0.8,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CrossPlatformConfig:
    """Configuration for cross-platform budget allocation"""

    # Rebalance constraints
    max_single_shift_pct: float = 0.20       # Max budget shift per platform per cycle
    min_platform_budget_pct: float = 0.05    # Floor: every active platform keeps >= 5%
    max_platform_budget_pct: float = 0.80    # Ceiling: no platform gets > 80%
    rebalance_cooldown_hours: float = 24.0   # Min hours between rebalances
    min_campaigns_for_signal: int = 1        # Min campaigns per platform for valid signal

    # Marginal return estimation
    lookback_days: int = 14                  # Days of data for response curve fitting
    smoothing_alpha: float = 0.3             # EMA smoothing for marginal ROAS

    # Optimization objective weights
    roas_weight: float = 0.50
    volume_weight: float = 0.25             # Reward for total conversions
    diversification_weight: float = 0.10    # Penalty for extreme concentration
    momentum_weight: float = 0.15           # Reward for allocating to improving platforms

    # Safety
    min_confidence_for_shift: float = 0.60
    emergency_roas_floor: float = 0.5       # Pull budget if platform ROAS < this


@dataclass
class BudgetRecommendationConfig:
    """Configuration for budget recommendation mode (new campaigns)"""

    # Candidate daily budgets to evaluate via Q-function sweep
    candidate_budgets: List[float] = field(default_factory=lambda: [
        50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000,
    ])
    min_q_value_threshold: float = -10.0  # Reject budgets with Q below this
    use_portfolio_context: bool = True     # Consider existing portfolio constraints
    campaign_duration_days: int = 30      # Default campaign duration for total budget


@dataclass
class BudgetRecommendation:
    """Recommended budget for a new campaign"""
    campaign_id: str
    platform: str
    recommended_daily_budget: float
    recommended_total_budget: float
    confidence: float
    q_value: float
    budget_candidates_evaluated: int
    marginal_roas_at_recommendation: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "platform": self.platform,
            "recommended_daily_budget": round(self.recommended_daily_budget, 2),
            "recommended_total_budget": round(self.recommended_total_budget, 2),
            "confidence": round(self.confidence, 3),
            "q_value": round(self.q_value, 4),
            "budget_candidates_evaluated": self.budget_candidates_evaluated,
            "marginal_roas_at_recommendation": round(self.marginal_roas_at_recommendation, 3),
            "rationale": self.rationale,
        }


@dataclass
class PlatformMetrics:
    """Aggregated performance metrics for a single platform"""
    platform: str
    num_campaigns: int = 0

    # Financials
    total_spend: float = 0.0
    total_revenue: float = 0.0
    total_conversions: int = 0
    total_clicks: int = 0
    total_impressions: int = 0

    # Derived
    roas: float = 0.0
    cpa: float = 0.0
    ctr: float = 0.0
    cvr: float = 0.0

    # Estimated marginal return (revenue per incremental dollar)
    marginal_roas: float = 0.0
    marginal_roas_confidence: float = 0.0

    # Trends
    roas_trend_7d: float = 0.0
    spend_trend_7d: float = 0.0
    conversion_trend_7d: float = 0.0

    # Audience segment features (used by X-Model state builder)
    segment_count: int = 0                  # number of distinct audience segments
    top_segment_roas: float = 0.0           # best segment ROAS
    avg_frequency: float = 0.0              # average impression frequency
    max_frequency: float = 1.0              # max impression frequency (floor=1)

    # Budget
    current_budget_share: float = 0.0       # fraction of total portfolio budget

    def compute_derived(self):
        """Compute derived metrics from totals"""
        self.roas = self.total_revenue / max(self.total_spend, 1.0)
        self.cpa = self.total_spend / max(self.total_conversions, 1)
        self.ctr = self.total_clicks / max(self.total_impressions, 1)
        self.cvr = self.total_conversions / max(self.total_clicks, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "num_campaigns": self.num_campaigns,
            "total_spend": self.total_spend,
            "total_revenue": self.total_revenue,
            "total_conversions": self.total_conversions,
            "roas": self.roas,
            "cpa": self.cpa,
            "ctr": self.ctr,
            "cvr": self.cvr,
            "marginal_roas": self.marginal_roas,
            "marginal_roas_confidence": self.marginal_roas_confidence,
            "roas_trend_7d": self.roas_trend_7d,
            "spend_trend_7d": self.spend_trend_7d,
            "conversion_trend_7d": self.conversion_trend_7d,
            "segment_count": self.segment_count,
            "top_segment_roas": self.top_segment_roas,
            "avg_frequency": self.avg_frequency,
            "max_frequency": self.max_frequency,
            "current_budget_share": self.current_budget_share,
        }


@dataclass
class PlatformPortfolio:
    """Complete portfolio state across all platforms"""
    organization_id: str
    total_budget: float = 0.0
    platforms: Dict[str, PlatformMetrics] = field(default_factory=dict)
    timestamp: str = ""

    # Per-platform campaign lists for downstream optimisation
    campaign_states: Dict[str, List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def active_platforms(self) -> List[str]:
        """Platforms with at least one campaign"""
        return [p for p, m in self.platforms.items() if m.num_campaigns > 0]

    def total_spend(self) -> float:
        return sum(m.total_spend for m in self.platforms.values())

    def total_revenue(self) -> float:
        return sum(m.total_revenue for m in self.platforms.values())

    def portfolio_roas(self) -> float:
        spend = self.total_spend()
        return self.total_revenue() / max(spend, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organization_id": self.organization_id,
            "total_budget": self.total_budget,
            "portfolio_roas": self.portfolio_roas(),
            "active_platforms": self.active_platforms(),
            "platforms": {p: m.to_dict() for p, m in self.platforms.items()},
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Budget allocation result
# ---------------------------------------------------------------------------

@dataclass
class AllocationRecommendation:
    """Recommended budget shift for a single platform"""
    platform: str
    current_share: float               # current fraction of total
    recommended_share: float            # target fraction of total
    shift_pct: float                    # recommended_share - current_share
    current_budget: float
    recommended_budget: float
    rationale: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "current_share": round(self.current_share, 4),
            "recommended_share": round(self.recommended_share, 4),
            "shift_pct": round(self.shift_pct, 4),
            "current_budget": round(self.current_budget, 2),
            "recommended_budget": round(self.recommended_budget, 2),
            "rationale": self.rationale,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class CrossPlatformResult:
    """Complete cross-platform optimization result"""
    organization_id: str
    timestamp: str

    # Portfolio snapshot
    portfolio_roas: float = 0.0
    total_budget: float = 0.0
    total_spend: float = 0.0

    # Allocation recommendations
    allocations: List[AllocationRecommendation] = field(default_factory=list)
    allocation_confidence: float = 0.0

    # Per-platform campaign results (from downstream DRL)
    platform_campaign_results: Dict[str, List[OptimizationResult]] = field(
        default_factory=dict
    )

    # Projected impact
    projected_portfolio_roas: float = 0.0
    projected_incremental_revenue: float = 0.0

    # Metadata
    rebalance_triggered: bool = False
    blocked_reason: Optional[str] = None

    # Audience constraint results per platform
    audience_constraints: Optional[Dict[str, Any]] = None

    # Portfolio-level narrative from X-Model (xAI narrator output)
    portfolio_narrative: Optional[Any] = None

    # Full portfolio snapshot dict for downstream use (training data, diagnostics)
    portfolio_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "organization_id": self.organization_id,
            "timestamp": self.timestamp,
            "portfolio_roas": round(self.portfolio_roas, 3),
            "total_budget": round(self.total_budget, 2),
            "total_spend": round(self.total_spend, 2),
            "allocations": [a.to_dict() for a in self.allocations],
            "allocation_confidence": round(self.allocation_confidence, 3),
            "projected_portfolio_roas": round(self.projected_portfolio_roas, 3),
            "projected_incremental_revenue": round(self.projected_incremental_revenue, 2),
            "rebalance_triggered": self.rebalance_triggered,
            "blocked_reason": self.blocked_reason,
            "num_campaign_results": {
                p: len(r) for p, r in self.platform_campaign_results.items()
            },
        }
        if self.audience_constraints is not None:
            result["audience_constraints"] = {
                platform: repr(ac) for platform, ac in self.audience_constraints.items()
            }
        if self.portfolio_narrative is not None:
            result["portfolio_narrative"] = (
                self.portfolio_narrative.to_dict()
                if hasattr(self.portfolio_narrative, "to_dict")
                else str(self.portfolio_narrative)
            )
        return result


# ---------------------------------------------------------------------------
# Marginal ROAS estimator
# ---------------------------------------------------------------------------

class MarginalReturnEstimator:
    """
    Estimates the marginal ROAS for each platform.

    Uses a simple log-linear response model:
        Revenue(spend) ≈ a * ln(1 + spend/b)

    Marginal ROAS = dRevenue/dSpend = a / (b + spend)

    This captures diminishing returns as spend increases on a platform.
    """

    def __init__(self, smoothing_alpha: float = 0.3):
        self.smoothing_alpha = smoothing_alpha
        # History: platform -> list of (spend, revenue) daily observations
        self._history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        # EMA of marginal ROAS per platform
        self._ema_marginal: Dict[str, float] = {}

    def record_observation(self, platform: str, daily_spend: float, daily_revenue: float):
        """Record a daily observation for a platform"""
        self._history[platform].append((daily_spend, daily_revenue))
        # Keep last 90 days
        if len(self._history[platform]) > 90:
            self._history[platform] = self._history[platform][-90:]

    def estimate(
        self,
        platform: str,
        current_daily_spend: float,
        lookback_days: int = 14,
    ) -> Tuple[float, float]:
        """
        Estimate marginal ROAS at current spend level.

        Returns:
            (marginal_roas, confidence)
            confidence is in [0, 1] based on data sufficiency
        """
        history = self._history.get(platform, [])
        recent = history[-lookback_days:] if len(history) >= lookback_days else history

        if len(recent) < 3:
            # Not enough data: return average ROAS as proxy
            if recent:
                avg_roas = sum(r for _, r in recent) / max(sum(s for s, _ in recent), 1.0)
                return avg_roas, 0.3
            return 1.0, 0.1

        spends = np.array([s for s, _ in recent])
        revenues = np.array([r for _, r in recent])

        # Fit log-linear: revenue = a * ln(1 + spend / b)
        # Simplified: use linear regression on (spend, revenue) to get average ROAS,
        # then estimate diminishing returns via variance in ROAS across spend levels
        total_spend = np.sum(spends)
        total_revenue = np.sum(revenues)
        avg_roas = total_revenue / max(total_spend, 1.0)

        # Estimate marginal by comparing high-spend vs low-spend days
        median_spend = np.median(spends)
        low_mask = spends <= median_spend
        high_mask = spends > median_spend

        if np.sum(low_mask) > 0 and np.sum(high_mask) > 0:
            low_roas = np.sum(revenues[low_mask]) / max(np.sum(spends[low_mask]), 1.0)
            high_roas = np.sum(revenues[high_mask]) / max(np.sum(spends[high_mask]), 1.0)

            # Diminishing returns: marginal at current spend is closer to high_roas
            # since we're already at current_daily_spend level
            if current_daily_spend > median_spend:
                marginal = high_roas * 0.9  # Slight discount for further diminishing
            else:
                marginal = (low_roas + high_roas) / 2.0
        else:
            marginal = avg_roas

        # EMA smoothing
        if platform in self._ema_marginal:
            marginal = (
                self.smoothing_alpha * marginal
                + (1 - self.smoothing_alpha) * self._ema_marginal[platform]
            )
        self._ema_marginal[platform] = marginal

        # Confidence based on data points and variance
        n = len(recent)
        roas_per_day = revenues / np.maximum(spends, 1.0)
        cv = np.std(roas_per_day) / max(np.mean(roas_per_day), 0.01)
        confidence = min(1.0, n / 14.0) * max(0.3, 1.0 - cv)

        return float(marginal), float(np.clip(confidence, 0.1, 1.0))

    def get_all_estimates(
        self,
        portfolio: PlatformPortfolio,
        lookback_days: int = 14,
    ) -> Dict[str, Tuple[float, float]]:
        """Get marginal ROAS estimates for all active platforms"""
        estimates = {}
        for platform in portfolio.active_platforms():
            pm = portfolio.platforms[platform]
            avg_daily_spend = pm.total_spend / max(lookback_days, 1)
            marginal, conf = self.estimate(platform, avg_daily_spend, lookback_days)
            estimates[platform] = (marginal, conf)
        return estimates


# ---------------------------------------------------------------------------
# Cross-platform budget allocator (core optimisation)
# ---------------------------------------------------------------------------

class BudgetAllocator:
    """
    Solves the constrained budget allocation problem across platforms.

    Objective:
        max  Σ_p  w_roas * marginal_roas_p * budget_p
           + w_vol * log(1 + conversions_p) * budget_p / current_spend_p
           + w_div * entropy(allocation)
           + w_mom * trend_p * budget_p

    Subject to:
        Σ budget_p = total_budget
        min_pct * total <= budget_p <= max_pct * total   for all p
        |budget_p - current_p| <= max_shift * total      for all p
    """

    def __init__(self, config: CrossPlatformConfig):
        self.config = config

    def allocate(
        self,
        portfolio: PlatformPortfolio,
        marginal_estimates: Dict[str, Tuple[float, float]],
    ) -> List[AllocationRecommendation]:
        """
        Compute optimal budget allocation.

        Uses projected gradient ascent on the allocation simplex.
        """
        active = portfolio.active_platforms()
        n = len(active)

        if n == 0:
            return []
        if n == 1:
            p = active[0]
            pm = portfolio.platforms[p]
            return [AllocationRecommendation(
                platform=p,
                current_share=1.0,
                recommended_share=1.0,
                shift_pct=0.0,
                current_budget=portfolio.total_budget,
                recommended_budget=portfolio.total_budget,
                rationale="Only one active platform",
                confidence=1.0,
            )]

        total = portfolio.total_budget

        # Current shares
        current_shares = np.array([
            portfolio.platforms[p].current_budget_share for p in active
        ])
        # If shares don't sum to ~1, normalize
        share_sum = current_shares.sum()
        if share_sum < 0.01:
            current_shares = np.ones(n) / n
        else:
            current_shares = current_shares / share_sum

        # Marginal ROAS scores
        marginal_scores = np.array([
            marginal_estimates.get(p, (1.0, 0.5))[0] for p in active
        ])
        confidences = np.array([
            marginal_estimates.get(p, (1.0, 0.5))[1] for p in active
        ])

        # Trend scores (positive = improving)
        trend_scores = np.array([
            portfolio.platforms[p].roas_trend_7d for p in active
        ])

        # Volume scores (log conversions)
        volume_scores = np.array([
            np.log1p(portfolio.platforms[p].total_conversions) for p in active
        ])
        # Normalize volume scores
        vol_max = volume_scores.max() if volume_scores.max() > 0 else 1.0
        volume_scores = volume_scores / vol_max

        # Build composite score per platform
        cfg = self.config
        composite = (
            cfg.roas_weight * self._normalize(marginal_scores)
            + cfg.volume_weight * volume_scores
            + cfg.momentum_weight * self._normalize(trend_scores)
        )

        # Scale composite by confidence
        composite = composite * confidences

        # Convert scores to target allocation via softmax
        # Temperature controls how aggressively we follow scores
        temperature = 0.5
        exp_scores = np.exp(composite / temperature)
        target_shares = exp_scores / exp_scores.sum()

        # Apply constraints
        target_shares = self._apply_constraints(
            current_shares, target_shares, n, total
        )

        # Diversification bonus: penalize extreme concentration
        entropy = -np.sum(target_shares * np.log(target_shares + 1e-10))
        max_entropy = np.log(n)
        diversification_score = entropy / max_entropy if max_entropy > 0 else 1.0

        # Overall confidence
        avg_confidence = float(np.mean(confidences))
        allocation_confidence = avg_confidence * diversification_score

        # Build recommendations
        recommendations = []
        for i, p in enumerate(active):
            pm = portfolio.platforms[p]
            shift = target_shares[i] - current_shares[i]
            recommended_budget = target_shares[i] * total

            rationale = self._build_rationale(
                p, pm, marginal_estimates.get(p, (1.0, 0.5)),
                current_shares[i], target_shares[i], shift
            )

            recommendations.append(AllocationRecommendation(
                platform=p,
                current_share=float(current_shares[i]),
                recommended_share=float(target_shares[i]),
                shift_pct=float(shift),
                current_budget=float(current_shares[i] * total),
                recommended_budget=float(recommended_budget),
                rationale=rationale,
                confidence=float(confidences[i]),
            ))

        return recommendations

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range"""
        r = arr.max() - arr.min()
        if r < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / r

    def _apply_constraints(
        self,
        current: np.ndarray,
        target: np.ndarray,
        n: int,
        total: float,
    ) -> np.ndarray:
        """Apply allocation constraints with iterative projection"""
        cfg = self.config

        constrained = target.copy()

        # 1. Clamp per-platform shift
        for i in range(n):
            max_shift = cfg.max_single_shift_pct
            shift = constrained[i] - current[i]
            if shift > max_shift:
                constrained[i] = current[i] + max_shift
            elif shift < -max_shift:
                constrained[i] = current[i] - max_shift

        # 2. Clamp floor/ceiling
        constrained = np.clip(
            constrained,
            cfg.min_platform_budget_pct,
            cfg.max_platform_budget_pct,
        )

        # 3. Re-normalize to sum to 1 (simplex projection)
        constrained = self._project_simplex(constrained)

        # 4. Final floor/ceiling after projection
        constrained = np.clip(
            constrained,
            cfg.min_platform_budget_pct,
            cfg.max_platform_budget_pct,
        )
        constrained = constrained / constrained.sum()

        return constrained

    def _project_simplex(self, v: np.ndarray) -> np.ndarray:
        """Project onto the probability simplex (sum to 1, all >= 0)"""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, n + 1)
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        return np.maximum(v - theta, 0)

    def _build_rationale(
        self,
        platform: str,
        metrics: PlatformMetrics,
        marginal_est: Tuple[float, float],
        current_share: float,
        target_share: float,
        shift: float,
    ) -> str:
        """Build human-readable rationale for allocation recommendation"""
        marginal_roas, confidence = marginal_est
        direction = "increase" if shift > 0 else "decrease" if shift < 0 else "maintain"

        parts = [f"{direction.capitalize()} {platform} allocation by {abs(shift):.1%}"]

        if shift > 0.01:
            reasons = []
            if marginal_roas > metrics.roas * 0.8:
                reasons.append(f"strong marginal ROAS ({marginal_roas:.2f}x)")
            if metrics.roas_trend_7d > 0.05:
                reasons.append(f"improving ROAS trend (+{metrics.roas_trend_7d:.1%})")
            if metrics.total_conversions > 0:
                reasons.append(f"{metrics.total_conversions} conversions")
            parts.append("because: " + ", ".join(reasons) if reasons else "based on portfolio optimization")
        elif shift < -0.01:
            reasons = []
            if marginal_roas < 1.0:
                reasons.append(f"low marginal ROAS ({marginal_roas:.2f}x)")
            if metrics.roas_trend_7d < -0.05:
                reasons.append(f"declining ROAS trend ({metrics.roas_trend_7d:.1%})")
            if metrics.roas < 1.0:
                reasons.append(f"below breakeven (ROAS {metrics.roas:.2f}x)")
            parts.append("because: " + ", ".join(reasons) if reasons else "to reallocate to higher-performing platforms")
        else:
            parts.append("current allocation is near-optimal")

        return ". ".join(parts)


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------

class PlatformPerformanceTracker:
    """
    Tracks platform performance over time and builds portfolio snapshots.
    """

    def __init__(self):
        # Historical daily metrics per platform
        self._daily_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._last_rebalance: Dict[str, datetime] = {}

    def record_daily_metrics(
        self,
        organization_id: str,
        platform: str,
        metrics: Dict[str, float],
    ):
        """Record daily platform-level metrics"""
        entry = {
            "date": datetime.now(timezone.utc).isoformat()[:10],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        key = f"{organization_id}:{platform}"
        self._daily_history[key].append(entry)

        # Keep last 90 days
        if len(self._daily_history[key]) > 90:
            self._daily_history[key] = self._daily_history[key][-90:]

    def build_portfolio(
        self,
        organization_id: str,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]],
        total_budget: float,
    ) -> PlatformPortfolio:
        """
        Build a PlatformPortfolio from a list of campaigns.

        Args:
            organization_id: Organization identifier
            campaigns: List of (state, context, info) for each campaign
            total_budget: Total budget across all platforms

        Returns:
            PlatformPortfolio snapshot
        """
        portfolio = PlatformPortfolio(
            organization_id=organization_id,
            total_budget=total_budget,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Aggregate by platform
        platform_agg: Dict[str, PlatformMetrics] = {}
        # Track per-platform segment data for audience feature aggregation
        platform_segments: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        platform_frequencies: Dict[str, List[float]] = defaultdict(list)

        for state, context, info in campaigns:
            platform = info.get("platform", state.platform or "unknown").lower()

            if platform not in platform_agg:
                platform_agg[platform] = PlatformMetrics(platform=platform)

            pm = platform_agg[platform]
            pm.num_campaigns += 1
            pm.total_spend += info.get("spend", context.total_spend)
            pm.total_revenue += info.get("revenue", 0.0)
            pm.total_conversions += info.get("conversions", 0)
            pm.total_clicks += info.get("clicks", 0)
            pm.total_impressions += info.get("impressions", 0)

            # Collect audience segment data for X-Model state features
            for _seg_id, seg_perf in info.get("segment_performance", {}).items():
                platform_segments[platform].append(seg_perf)
            if "frequency" in info:
                platform_frequencies[platform].append(info["frequency"])

            # Collect campaign for downstream optimization
            portfolio.campaign_states[platform].append((state, context, info))

        # Compute derived metrics and budget shares
        total_spend = sum(pm.total_spend for pm in platform_agg.values())
        for platform, pm in platform_agg.items():
            pm.compute_derived()
            pm.current_budget_share = pm.total_spend / max(total_spend, 1.0)

            # Aggregate audience segment features
            segs = platform_segments.get(platform, [])
            if segs:
                pm.segment_count = len(segs)
                seg_roas_vals = [s.get("roas", 0.0) for s in segs]
                pm.top_segment_roas = max(seg_roas_vals) if seg_roas_vals else 0.0
            freqs = platform_frequencies.get(platform, [])
            if freqs:
                pm.avg_frequency = sum(freqs) / len(freqs)
                pm.max_frequency = max(freqs)

            # Add trend data from history
            key = f"{organization_id}:{platform}"
            history = self._daily_history.get(key, [])
            if len(history) >= 2:
                recent = history[-7:]
                older = history[-14:-7] if len(history) >= 14 else history[:max(len(history) - 7, 1)]

                recent_roas = sum(d.get("revenue", 0) for d in recent) / max(
                    sum(d.get("spend", 0) for d in recent), 1.0
                )
                older_roas = sum(d.get("revenue", 0) for d in older) / max(
                    sum(d.get("spend", 0) for d in older), 1.0
                )
                pm.roas_trend_7d = (recent_roas - older_roas) / max(older_roas, 0.01)

                recent_conv = sum(d.get("conversions", 0) for d in recent)
                older_conv = sum(d.get("conversions", 0) for d in older)
                pm.conversion_trend_7d = (recent_conv - older_conv) / max(older_conv, 1)

        portfolio.platforms = platform_agg
        return portfolio

    def can_rebalance(
        self,
        organization_id: str,
        cooldown_hours: float = 24.0,
    ) -> Tuple[bool, Optional[str]]:
        """Check if rebalance cooldown has passed"""
        last = self._last_rebalance.get(organization_id)
        if last is None:
            return True, None

        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        if elapsed < cooldown_hours:
            return False, f"Cooldown: {elapsed:.1f}h since last rebalance, {cooldown_hours}h required"
        return True, None

    def mark_rebalance(self, organization_id: str):
        """Mark that a rebalance occurred"""
        self._last_rebalance[organization_id] = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class CrossPlatformOptimizer:
    """
    Top-level orchestrator that jointly allocates budget across platforms
    and coordinates per-campaign DRL optimization.

    Usage:
        optimizer = CrossPlatformOptimizer(hybrid_optimizer, config)
        result = await optimizer.optimize_portfolio(portfolio)
    """

    def __init__(
        self,
        hybrid_optimizer: HybridDRLLLMOptimizer,
        config: Optional[CrossPlatformConfig] = None,
        max_concurrent_campaigns: int = 10,
        audience_manager: Optional[AudienceConstraintManager] = None,
        platform_registry: Optional[PlatformModelRegistry] = None,
        x_model_agent: Optional[XModelAgent] = None,
    ):
        self.hybrid = hybrid_optimizer
        self.config = config or CrossPlatformConfig()
        self.batch_optimizer = BatchOptimizer(
            hybrid_optimizer=hybrid_optimizer,
            max_concurrent=max_concurrent_campaigns,
        )

        # Heuristic allocator (fallback when X-Model is not available)
        self.allocator = BudgetAllocator(self.config)
        self.estimator = MarginalReturnEstimator(
            smoothing_alpha=self.config.smoothing_alpha
        )
        self.tracker = PlatformPerformanceTracker()
        self.audience_manager = audience_manager

        # P & X Model infrastructure
        self.platform_registry = platform_registry  # M2: per-platform P-Models
        self.x_model_agent = x_model_agent          # M5: cross-platform X-Model

        # Allocation history for audit
        self._allocation_history: List[Dict[str, Any]] = []

        mode = "X-Model" if x_model_agent else "heuristic"
        p_mode = "per-platform P-Models" if platform_registry else "global agent"
        logger.info(
            f"CrossPlatformOptimizer initialized "
            f"(allocation={mode}, inference={p_mode})"
        )

    async def optimize_portfolio(
        self,
        organization_id: str,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]],
        total_budget: float,
        force_rebalance: bool = False,
    ) -> CrossPlatformResult:
        """
        Run the full cross-platform optimization pipeline.

        1. Build portfolio snapshot
        2. Estimate marginal returns per platform
        3. Solve budget allocation
        4. Run per-campaign DRL optimization with adjusted budgets
        5. Return combined results

        Args:
            organization_id: Organization identifier
            campaigns: All campaigns across all platforms
            total_budget: Total portfolio budget
            force_rebalance: Skip cooldown check

        Returns:
            CrossPlatformResult with allocations and campaign results
        """
        start_time = datetime.now(timezone.utc)

        # Phase 1: Build portfolio snapshot
        portfolio = self.tracker.build_portfolio(
            organization_id=organization_id,
            campaigns=campaigns,
            total_budget=total_budget,
        )

        result = CrossPlatformResult(
            organization_id=organization_id,
            timestamp=start_time.isoformat(),
            portfolio_roas=portfolio.portfolio_roas(),
            total_budget=total_budget,
            total_spend=portfolio.total_spend(),
            portfolio_snapshot=portfolio.to_dict(),
        )

        # Phase 2: Check cooldown
        if not force_rebalance:
            can_rebalance, reason = self.tracker.can_rebalance(
                organization_id,
                self.config.rebalance_cooldown_hours,
            )
            if not can_rebalance:
                result.blocked_reason = reason
                # Still run per-campaign optimization without rebalance
                result.platform_campaign_results = await self._run_campaign_optimization(
                    portfolio
                )
                return result

        # Phase 3: Estimate marginal returns (always, used by both paths)
        self._update_estimator(portfolio)
        marginal_estimates = self.estimator.get_all_estimates(
            portfolio, self.config.lookback_days
        )

        # Enrich platform metrics with marginal estimates
        for platform, (marginal, conf) in marginal_estimates.items():
            if platform in portfolio.platforms:
                portfolio.platforms[platform].marginal_roas = marginal
                portfolio.platforms[platform].marginal_roas_confidence = conf

        # Phase 4: Emergency check - pull budget from platforms below ROAS floor
        self._apply_emergency_overrides(portfolio, marginal_estimates)

        # Phase 5: Solve allocation
        #   M5 path — use learned X-Model when available
        #   Heuristic path — fall back to BudgetAllocator
        if self.x_model_agent is not None:
            allocations = self._x_model_allocate(portfolio, total_budget)
        else:
            allocations = self.allocator.allocate(portfolio, marginal_estimates)
        result.allocations = allocations
        result.rebalance_triggered = True

        # Compute allocation confidence
        if allocations:
            result.allocation_confidence = float(np.mean([
                a.confidence for a in allocations
            ]))

        # Phase 6: Compute projected impact
        result.projected_portfolio_roas = self._project_roas(
            portfolio, allocations, marginal_estimates
        )
        result.projected_incremental_revenue = (
            (result.projected_portfolio_roas - portfolio.portfolio_roas())
            * portfolio.total_spend()
        )

        # Phase 7: Adjust campaign budgets and run per-campaign DRL
        adjusted_portfolio = self._adjust_campaign_budgets(portfolio, allocations)

        # Phase 7b: Apply audience constraints at portfolio level (if configured)
        audience_results: Optional[Dict[str, AudienceConstraintResult]] = None
        if self.audience_manager is not None:
            audience_results = {}
            for platform, campaigns in adjusted_portfolio.campaign_states.items():
                if not campaigns:
                    continue
                # Compute platform-level budget for audience allocation
                platform_budget = sum(
                    ctx.current_budget for _, ctx, _ in campaigns
                )
                # Gather per-segment performance from campaign info
                perf_signals: Dict[str, Dict[str, float]] = {}
                for _, _, info in campaigns:
                    for seg_id, seg_perf in info.get("segment_performance", {}).items():
                        perf_signals[seg_id] = seg_perf
                # Use the first campaign's action as the DRL audience signal
                first_action = ActionSpace()
                if campaigns:
                    first_action_info = campaigns[0][2]
                    if "last_action" in first_action_info:
                        first_action = first_action_info["last_action"]
                try:
                    aud_result = self.audience_manager.allocate_budget(
                        platform_budget=platform_budget,
                        action=first_action,
                        performance_signals=perf_signals,
                    )
                    audience_results[platform] = aud_result
                except Exception as e:
                    logger.warning(f"Audience constraint failed for {platform}: {e}")

            if audience_results:
                result.audience_constraints = audience_results

        result.platform_campaign_results = await self._run_campaign_optimization(
            adjusted_portfolio
        )

        # Record
        self.tracker.mark_rebalance(organization_id)
        self._log_allocation(result)

        return result

    def _update_estimator(self, portfolio: PlatformPortfolio):
        """Feed portfolio data into marginal return estimator"""
        for platform, pm in portfolio.platforms.items():
            if pm.num_campaigns > 0:
                self.estimator.record_observation(
                    platform=platform,
                    daily_spend=pm.total_spend,
                    daily_revenue=pm.total_revenue,
                )

    def _x_model_allocate(
        self,
        portfolio: PlatformPortfolio,
        total_budget: float,
    ) -> List[AllocationRecommendation]:
        """
        M5: Use the learned X-Model to determine budget allocation.

        Builds an XModelState from the portfolio, runs the X-Model actor
        to get allocation weights, and converts to AllocationRecommendation.
        """
        # Build X-State from portfolio snapshot
        x_state = build_x_state(portfolio.to_dict(), total_budget)

        # Run X-Model inference
        x_action = self.x_model_agent.select_allocation(
            x_state,
            deterministic=True,
            min_share=self.config.min_platform_budget_pct,
            max_share=self.config.max_platform_budget_pct,
        )

        # Convert X-Model output to AllocationRecommendation list
        active = portfolio.active_platforms()
        recommendations = []

        for platform in active:
            pm = portfolio.platforms[platform]
            target_share = x_action.allocation_weights.get(platform, 0.0)
            current_share = pm.current_budget_share

            # Respect max single shift constraint
            shift = target_share - current_share
            max_shift = self.config.max_single_shift_pct
            if shift > max_shift:
                target_share = current_share + max_shift
            elif shift < -max_shift:
                target_share = current_share - max_shift

            shift = target_share - current_share
            recommended_budget = target_share * total_budget

            recommendations.append(AllocationRecommendation(
                platform=platform,
                current_share=current_share,
                recommended_share=target_share,
                shift_pct=shift,
                current_budget=current_share * total_budget,
                recommended_budget=recommended_budget,
                rationale=f"X-Model allocation (confidence={x_action.confidence:.2f})",
                confidence=x_action.confidence,
            ))

        # Re-normalize target shares to sum to 1 across active platforms
        total_target = sum(r.recommended_share for r in recommendations)
        if total_target > 0 and abs(total_target - 1.0) > 0.01:
            for r in recommendations:
                r.recommended_share /= total_target
                r.shift_pct = r.recommended_share - r.current_share
                r.recommended_budget = r.recommended_share * total_budget

        logger.info(
            f"X-Model allocation: {x_action.allocation_weights} "
            f"(Q={x_action.q_value:.3f}, confidence={x_action.confidence:.2f})"
        )
        return recommendations

    def _apply_emergency_overrides(
        self,
        portfolio: PlatformPortfolio,
        marginal_estimates: Dict[str, Tuple[float, float]],
    ):
        """Override marginal estimates for platforms in emergency state"""
        for platform, pm in portfolio.platforms.items():
            if pm.roas < self.config.emergency_roas_floor and pm.total_spend > 0:
                logger.warning(
                    f"Emergency: {platform} ROAS {pm.roas:.2f} below floor "
                    f"{self.config.emergency_roas_floor}"
                )
                # Force very low marginal ROAS estimate to trigger budget pull
                marginal_estimates[platform] = (0.1, 0.9)

    def _project_roas(
        self,
        portfolio: PlatformPortfolio,
        allocations: List[AllocationRecommendation],
        marginal_estimates: Dict[str, Tuple[float, float]],
    ) -> float:
        """Project portfolio ROAS after reallocation"""
        projected_revenue = 0.0
        projected_spend = 0.0

        for alloc in allocations:
            marginal, _ = marginal_estimates.get(alloc.platform, (1.0, 0.5))
            pm = portfolio.platforms.get(alloc.platform)
            if pm is None:
                continue

            # Current revenue from existing spend
            base_revenue = pm.total_revenue
            budget_change = alloc.recommended_budget - alloc.current_budget

            # Incremental revenue at marginal rate
            incremental_revenue = budget_change * marginal
            projected_revenue += base_revenue + incremental_revenue
            projected_spend += alloc.recommended_budget

        return projected_revenue / max(projected_spend, 1.0)

    def _adjust_campaign_budgets(
        self,
        portfolio: PlatformPortfolio,
        allocations: List[AllocationRecommendation],
    ) -> PlatformPortfolio:
        """
        Distribute platform-level budget changes proportionally
        across campaigns within each platform.
        """
        alloc_map = {a.platform: a for a in allocations}

        for platform, campaigns in portfolio.campaign_states.items():
            alloc = alloc_map.get(platform)
            if alloc is None or len(campaigns) == 0:
                continue

            budget_ratio = alloc.recommended_budget / max(alloc.current_budget, 1.0)

            # Adjust each campaign's budget proportionally
            for i, (state, context, info) in enumerate(campaigns):
                new_budget = context.current_budget * budget_ratio
                # Clamp to guardrail bounds
                new_budget = max(10.0, min(new_budget, 100_000.0))

                # Create updated context with adjusted budget
                adjusted_context = CampaignContext(
                    campaign_id=context.campaign_id,
                    current_bid=context.current_bid,
                    current_budget=new_budget,
                    last_action_at=context.last_action_at,
                    actions_today=context.actions_today,
                    current_roas=context.current_roas,
                    current_cpa=context.current_cpa,
                    target_cpa=context.target_cpa,
                    min_roas=context.min_roas,
                    is_new_campaign=context.is_new_campaign,
                    total_spend=context.total_spend,
                )

                # Re-derive budget features in CampaignState so the P-model
                # sees the updated budget when making its action decision.
                from dataclasses import replace
                adjusted_state = replace(
                    state,
                    log_daily_budget=float(
                        np.log1p(new_budget) / np.log1p(MAX_DAILY_BUDGET)
                    ),
                    budget_utilization=min(
                        state.budget_utilization * context.current_budget / max(new_budget, 1.0),
                        1.0,
                    ),
                    daily_budget_limit_norm=float(
                        np.log1p(new_budget) / np.log1p(MAX_DAILY_BUDGET)
                    ),
                )

                # Update the info dict with cross-platform context
                adjusted_info = {
                    **info,
                    "cross_platform_budget_ratio": budget_ratio,
                    "cross_platform_recommended_budget": new_budget,
                    "original_budget": context.current_budget,
                }

                campaigns[i] = (adjusted_state, adjusted_context, adjusted_info)

        return portfolio

    async def _run_campaign_optimization(
        self,
        portfolio: PlatformPortfolio,
    ) -> Dict[str, List[OptimizationResult]]:
        """
        M2: Run per-campaign DRL optimization for all platforms.

        When a PlatformModelRegistry is configured, the per-platform
        P-Model is injected into the HybridDRLLLMOptimizer before running
        each platform's campaigns.  This ensures each platform uses its
        own trained policy.

        Falls back to the single global agent when no registry is present.
        """
        results: Dict[str, List[OptimizationResult]] = {}

        for platform, campaigns in portfolio.campaign_states.items():
            if not campaigns:
                continue

            try:
                # M2: swap in the per-platform P-Model if available
                if self.platform_registry is not None and self.platform_registry.has_platform(platform):
                    p_agent = self.platform_registry.get(platform)
                    original_agent = self.hybrid.drl_agent.agent
                    self.hybrid.drl_agent.agent = p_agent
                    try:
                        platform_results = await self.batch_optimizer.optimize_batch(campaigns)
                    finally:
                        # Restore the original agent
                        self.hybrid.drl_agent.agent = original_agent
                else:
                    platform_results = await self.batch_optimizer.optimize_batch(campaigns)

                results[platform] = platform_results
            except Exception as e:
                logger.error(f"Campaign optimization failed for {platform}: {e}")
                results[platform] = []

        return results

    def _log_allocation(self, result: CrossPlatformResult):
        """Log allocation for audit trail"""
        entry = {
            "timestamp": result.timestamp,
            "organization_id": result.organization_id,
            "portfolio_roas": result.portfolio_roas,
            "projected_roas": result.projected_portfolio_roas,
            "allocations": [a.to_dict() for a in result.allocations],
        }
        self._allocation_history.append(entry)
        if len(self._allocation_history) > 1000:
            self._allocation_history = self._allocation_history[-1000:]

    def get_allocation_history(
        self, organization_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get allocation history, optionally filtered by org"""
        if organization_id is None:
            return self._allocation_history
        return [
            e for e in self._allocation_history
            if e["organization_id"] == organization_id
        ]

    # ------------------------------------------------------------------
    # Public helpers for CrossPlatformDRLEngine
    # ------------------------------------------------------------------

    def heuristic_allocate(
        self,
        portfolio: PlatformPortfolio,
        marginal_estimates: Dict[str, Tuple[float, float]],
    ) -> List[AllocationRecommendation]:
        """
        Public entry point for the heuristic BudgetAllocator.

        Exposed so that CrossPlatformDRLEngine can call it independently
        for dual-run benchmarking without going through optimize_portfolio.
        """
        return self.allocator.allocate(portfolio, marginal_estimates)

    def build_portfolio_snapshot(
        self,
        organization_id: str,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]],
        total_budget: float,
    ) -> Tuple[PlatformPortfolio, Dict[str, Tuple[float, float]]]:
        """
        Build a portfolio snapshot and compute marginal estimates.

        Returns (portfolio, marginal_estimates) for external use by
        CrossPlatformDRLEngine during dual-run benchmarking.
        """
        portfolio = self.tracker.build_portfolio(
            organization_id=organization_id,
            campaigns=campaigns,
            total_budget=total_budget,
        )
        self._update_estimator(portfolio)
        marginal_estimates = self.estimator.get_all_estimates(
            portfolio, self.config.lookback_days
        )
        for platform, (marginal, conf) in marginal_estimates.items():
            if platform in portfolio.platforms:
                portfolio.platforms[platform].marginal_roas = marginal
                portfolio.platforms[platform].marginal_roas_confidence = conf
        return portfolio, marginal_estimates

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get optimizer diagnostics"""
        diag: Dict[str, Any] = {
            "allocation_history_size": len(self._allocation_history),
            "estimator_platforms": list(self.estimator._history.keys()),
            "allocation_mode": "x_model" if self.x_model_agent else "heuristic",
            "inference_mode": "per_platform" if self.platform_registry else "global",
            "config": {
                "max_single_shift_pct": self.config.max_single_shift_pct,
                "min_platform_budget_pct": self.config.min_platform_budget_pct,
                "rebalance_cooldown_hours": self.config.rebalance_cooldown_hours,
            },
        }
        if self.platform_registry:
            diag["p_models"] = self.platform_registry.get_diagnostics()
        if self.x_model_agent:
            diag["x_model"] = self.x_model_agent.get_diagnostics()
        return diag

    # ------------------------------------------------------------------
    # Budget recommendation for new campaigns
    # ------------------------------------------------------------------

    async def recommend_budget(
        self,
        organization_id: str,
        new_campaign_info: Dict[str, Any],
        existing_campaigns: Optional[List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]]] = None,
        total_portfolio_budget: Optional[float] = None,
        config: Optional[BudgetRecommendationConfig] = None,
    ) -> BudgetRecommendation:
        """
        Recommend a starting daily budget for a new campaign.

        Sweeps candidate budgets, builds a synthetic CampaignState for each,
        evaluates via the DRL critic Q-function, and selects the budget that
        maximises a composite score of Q-value, marginal ROAS, and confidence.

        Args:
            organization_id: Organisation identifier.
            new_campaign_info: Dict with keys like ``platform``, ``goal``,
                ``audience_quality``, ``predicted_ltv``, ``duration_days``,
                ``campaign_id``.
            existing_campaigns: Optional list of running campaigns for
                portfolio-constraint enforcement.
            total_portfolio_budget: Total budget across all platforms.
            config: Budget recommendation configuration.

        Returns:
            BudgetRecommendation with the recommended daily budget, projected
            total, Q-value, confidence, and human-readable rationale.
        """
        config = config or BudgetRecommendationConfig()
        platform = new_campaign_info.get("platform", "unknown").lower()
        campaign_id = new_campaign_info.get("campaign_id", "new")

        # Access the platform-specific agent (M2) or fall back to global
        if self.platform_registry and self.platform_registry.has_platform(platform):
            agent = self.platform_registry.get(platform)
        else:
            agent = self.hybrid.drl_agent.agent  # SACAgent

        # Evaluate each candidate budget
        candidate_results: List[Dict[str, Any]] = []

        for budget in config.candidate_budgets:
            synthetic_state = self._build_synthetic_state(
                new_campaign_info, budget, platform
            )

            # Q-function evaluation
            with torch.no_grad():
                state_tensor = synthetic_state.to_tensor(agent.device).unsqueeze(0)
                continuous, discrete_soft, _, _log_prob, entropy = agent.actor.sample(
                    state_tensor
                )
                q1, q2 = agent.critic(state_tensor, continuous, discrete_soft)
                q_value = torch.min(q1, q2).item()

            # Marginal ROAS estimate from historical data
            marginal_roas, marg_conf = self.estimator.estimate(
                platform, budget, self.config.lookback_days,
            )

            candidate_results.append({
                "budget": budget,
                "q_value": q_value,
                "marginal_roas": marginal_roas,
                "marginal_conf": marg_conf,
                "entropy": entropy.item() if hasattr(entropy, "item") else float(entropy),
            })

        # Score candidates
        q_values = [c["q_value"] for c in candidate_results]
        q_min, q_max = min(q_values), max(q_values)
        q_range = q_max - q_min if q_max > q_min else 1.0

        best: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        for c in candidate_results:
            if c["q_value"] < config.min_q_value_threshold:
                continue
            q_norm = (c["q_value"] - q_min) / q_range
            score = 0.5 * q_norm + 0.3 * c["marginal_roas"] + 0.2 * c["marginal_conf"]
            if score > best_score:
                best_score = score
                best = c

        # Fallback to median candidate if nothing passes threshold
        if best is None:
            best = candidate_results[len(candidate_results) // 2]

        recommended_budget = best["budget"]

        # Apply portfolio constraints
        if (
            config.use_portfolio_context
            and existing_campaigns
            and total_portfolio_budget
        ):
            portfolio = self.tracker.build_portfolio(
                organization_id, existing_campaigns, total_portfolio_budget
            )
            platform_metrics = portfolio.platforms.get(platform)
            current_platform_spend = platform_metrics.total_spend if platform_metrics else 0.0
            max_allowed = total_portfolio_budget * self.config.max_platform_budget_pct
            headroom = max_allowed - current_platform_spend
            min_floor = total_portfolio_budget * self.config.min_platform_budget_pct
            recommended_budget = min(recommended_budget, max(headroom, min_floor))
            recommended_budget = max(recommended_budget, 0.0)

        duration = new_campaign_info.get("duration_days", config.campaign_duration_days)

        return BudgetRecommendation(
            campaign_id=campaign_id,
            platform=platform,
            recommended_daily_budget=recommended_budget,
            recommended_total_budget=recommended_budget * duration,
            confidence=best["marginal_conf"],
            q_value=best["q_value"],
            budget_candidates_evaluated=len(candidate_results),
            marginal_roas_at_recommendation=best["marginal_roas"],
            rationale=self._build_budget_rationale(best, candidate_results, platform),
        )

    def _build_synthetic_state(
        self,
        campaign_info: Dict[str, Any],
        daily_budget: float,
        platform: str,
    ) -> CampaignState:
        """
        Build a synthetic CampaignState for a new campaign at a given budget.

        Uses platform-average metrics as priors where available; otherwise
        falls back to reasonable defaults.
        """
        avg = self._get_platform_averages(platform)

        return CampaignState(
            campaign_id=campaign_info.get("campaign_id", "new"),
            platform=platform,
            # Core metrics from platform averages
            ctr=avg.get("ctr", 0.02),
            cvr=avg.get("cvr", 0.03),
            roas=min(avg.get("roas", 2.0) / 10.0, 1.0),
            cpa=1.0 - min(avg.get("cpa", 30) / 200.0, 1.0),
            cpc=min(avg.get("cpc", 1.0) / 10.0, 1.0),
            cpm=min(avg.get("cpm", 10.0) / 50.0, 1.0),
            # Volume: estimate proportional to budget
            spend_velocity=0.0,
            impression_volume=min(daily_budget * 1000 / 1_000_000, 1.0),
            click_volume=min(daily_budget * 1000 * 0.02 / 10_000, 1.0),
            conversion_volume=min(daily_budget * 1000 * 0.02 * 0.03 / 1000, 1.0),
            # Temporal defaults
            hour_of_day=0.5,
            day_of_week=0.3,
            day_of_month=0.5,
            is_weekend=0.0,
            is_holiday=0.0,
            days_remaining=campaign_info.get("duration_days", 30) / 90.0,
            # Trends: zero (new campaign)
            ctr_trend_7d=0.0,
            cvr_trend_7d=0.0,
            roas_trend_7d=0.0,
            cpa_trend_7d=0.0,
            spend_trend_7d=0.0,
            # Competitive: conservative defaults
            impression_share=0.3,
            auction_pressure=0.5,
            competitive_position=0.4,
            # ML features
            audience_quality_score=campaign_info.get("audience_quality", 0.5),
            creative_fatigue_score=0.0,
            predicted_cvr=avg.get("cvr", 0.03),
            predicted_ltv=campaign_info.get("predicted_ltv", 0.5),
            propensity_score=0.5,
            # Context
            optimization_goal_encoding=self._encode_goal(
                campaign_info.get("goal", "roas")
            ),
            platform_encoding=PLATFORM_ENCODING.get(platform, 1.0),
            campaign_maturity=0.0,
            budget_utilization=0.0,
            # Absolute spend features — key differentiator across candidates
            log_daily_spend=0.0,  # No spend yet
            log_total_campaign_spend=0.0,
            log_daily_budget=np.log1p(daily_budget) / np.log1p(MAX_DAILY_BUDGET),
        )

    def _get_platform_averages(self, platform: str) -> Dict[str, float]:
        """
        Return average performance metrics for a platform from historical
        estimator data.  Falls back to sensible defaults.
        """
        history = self.estimator._history.get(platform, [])
        if not history:
            return {
                "ctr": 0.02,
                "cvr": 0.03,
                "roas": 2.0,
                "cpa": 30.0,
                "cpc": 1.0,
                "cpm": 10.0,
            }

        total_spend = sum(s for s, _ in history)
        total_revenue = sum(r for _, r in history)
        avg_roas = total_revenue / max(total_spend, 1.0)

        return {
            "ctr": 0.02,
            "cvr": 0.03,
            "roas": avg_roas,
            "cpa": total_spend / max(len(history), 1),
            "cpc": 1.0,
            "cpm": 10.0,
        }

    @staticmethod
    def _encode_goal(goal: str) -> float:
        """Encode optimization goal as float for state vector."""
        goal_map = {
            "roas": 0.0,
            "cpa": 0.25,
            "conversions": 0.5,
            "ctr": 0.75,
            "revenue": 1.0,
        }
        return goal_map.get(goal.lower(), 0.0)

    @staticmethod
    def _build_budget_rationale(
        best: Dict[str, Any],
        all_candidates: List[Dict[str, Any]],
        platform: str,
    ) -> str:
        """Build a human-readable rationale for the budget recommendation."""
        budget = best["budget"]
        q = best["q_value"]
        mroas = best["marginal_roas"]
        n = len(all_candidates)

        parts = [
            f"Evaluated {n} candidate budgets for {platform}.",
            f"Recommended ${budget:,.0f}/day based on Q-value {q:.3f} "
            f"and estimated marginal ROAS {mroas:.2f}x.",
        ]

        # Find runner-up for context
        sorted_cands = sorted(
            all_candidates,
            key=lambda c: 0.5 * c["q_value"] + 0.3 * c["marginal_roas"],
            reverse=True,
        )
        if len(sorted_cands) > 1 and sorted_cands[0]["budget"] == budget:
            runner = sorted_cands[1]
            parts.append(
                f"Next best: ${runner['budget']:,.0f}/day "
                f"(Q={runner['q_value']:.3f}, mROAS={runner['marginal_roas']:.2f}x)."
            )

        return " ".join(parts)
