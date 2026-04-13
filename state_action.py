"""
State and Action Space Definitions for Campaign Optimization DRL

Defines the Markov Decision Process (MDP) components:
- State: What the agent observes (campaign metrics, temporal, competitive features)
- Action: What the agent can do (bid/budget adjustments, audience/creative actions)
- Directive: High-level strategic output from DRL to constrain LLM layer
"""

import uuid
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum
from datetime import datetime


# Log-normalization ceilings for absolute spend features
MAX_DAILY_SPEND = 100_000.0
MAX_TOTAL_SPEND = 10_000_000.0
MAX_DAILY_BUDGET = 100_000.0


class AudienceAction(IntEnum):
    """Discrete audience optimization actions"""
    HOLD = 0
    EXPAND = 1
    REFINE = 2
    EXCLUDE = 3


class CreativeAction(IntEnum):
    """Discrete creative optimization actions"""
    HOLD = 0
    ROTATE = 1
    PAUSE_UNDERPERFORMING = 2
    TEST_NEW = 3


class MessagingTone(IntEnum):
    """Messaging tone derived from DRL decisions"""
    CONSISTENT = 0
    AGGRESSIVE_GROWTH = 1
    EFFICIENCY_FOCUSED = 2
    URGENCY = 3
    FRESH_ANGLE = 4


@dataclass
class CampaignState:
    """
    Complete state representation for DRL agent
    
    Combines:
    - Current campaign performance metrics
    - Temporal/seasonal features
    - Competitive context
    - Historical trends
    - ML-derived predictions
    """
    
    # === Campaign Identifiers (not part of state vector) ===
    campaign_id: str = ""
    organization_id: str = ""
    platform: str = ""
    
    # === Core Performance Metrics (normalized 0-1) ===
    ctr: float = 0.0                    # Click-through rate
    cvr: float = 0.0                    # Conversion rate
    roas: float = 0.0                   # Return on ad spend (normalized)
    cpa: float = 0.0                    # Cost per acquisition (inverse normalized)
    cpc: float = 0.0                    # Cost per click
    cpm: float = 0.0                    # Cost per mille
    
    # === Spend/Volume Metrics ===
    spend_velocity: float = 0.0         # Budget burn rate (0-1)
    impression_volume: float = 0.0      # Normalized impression count
    click_volume: float = 0.0           # Normalized click count
    conversion_volume: float = 0.0      # Normalized conversion count
    
    # === Temporal Features ===
    hour_of_day: float = 0.0            # Normalized 0-1 (0=midnight, 0.5=noon)
    day_of_week: float = 0.0            # Normalized 0-1 (0=Monday, 1=Sunday)
    day_of_month: float = 0.0           # Normalized 0-1
    is_weekend: float = 0.0             # Binary
    is_holiday: float = 0.0             # Binary
    days_remaining: float = 0.0         # Campaign days left (normalized)
    
    # === Historical Trends (7-day) ===
    ctr_trend_7d: float = 0.0           # Positive = improving
    cvr_trend_7d: float = 0.0
    roas_trend_7d: float = 0.0
    cpa_trend_7d: float = 0.0           # Negative = improving (lower CPA)
    spend_trend_7d: float = 0.0
    
    # === Competitive Context ===
    impression_share: float = 0.0       # Share of voice
    auction_pressure: float = 0.0       # Bid landscape intensity
    competitive_position: float = 0.0   # Relative performance vs competitors
    
    # === ML-Derived Features ===
    audience_quality_score: float = 0.0     # Predicted audience LTV
    creative_fatigue_score: float = 0.0     # Predicted creative decay
    predicted_cvr: float = 0.0              # ML-predicted conversion rate
    predicted_ltv: float = 0.0              # Predicted customer lifetime value
    propensity_score: float = 0.0           # Purchase propensity
    
    # === Campaign Context ===
    optimization_goal_encoding: float = 0.0  # Encoded goal type
    platform_encoding: float = 0.0           # Encoded platform
    campaign_maturity: float = 0.0           # Days since launch (normalized)
    budget_utilization: float = 0.0          # Percent of budget used

    # === Absolute Spend Features (log-normalized) ===
    # These capture absolute spend levels so the agent can learn behavior
    # at different budget scales (e.g., $100/day vs $10,000/day)
    log_daily_spend: float = 0.0            # log1p(daily_spend) / log1p(MAX_DAILY_SPEND)
    log_total_campaign_spend: float = 0.0   # log1p(total_spend) / log1p(MAX_TOTAL_SPEND)
    log_daily_budget: float = 0.0           # log1p(daily_budget) / log1p(MAX_DAILY_BUDGET)

    # === Audience Segmentation Features (indices 36-38) ===
    segment_count: float = 0.0              # Number of active audience segments (normalized /10)
    top_segment_roas: float = 0.0           # ROAS of the best-performing segment (0-1)
    avg_frequency: float = 0.0              # Average ad frequency across segments (normalized /10)

    # === Constraint Features (indices 39-41) ===
    # Let the policy see the advertiser's targets so it can learn
    # constraint-aware behavior instead of relying only on post-hoc guardrails.
    target_cpa_norm: float = 0.0            # log1p(target_cpa) / log1p(1000)
    min_roas_norm: float = 0.0              # min_roas / 10.0 (cap at 10x)
    daily_budget_limit_norm: float = 0.0    # log1p(daily_budget_limit) / log1p(MAX_DAILY_BUDGET)

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """
        Convert state to PyTorch tensor for neural network input
        
        Returns:
            Tensor of shape (state_dim,)
        """
        state_vector = [
            # Core metrics (0-5)
            self.ctr, self.cvr, self.roas, self.cpa, self.cpc, self.cpm,
            # Volume (6-9)
            self.spend_velocity, self.impression_volume, self.click_volume, self.conversion_volume,
            # Temporal (10-15)
            self.hour_of_day, self.day_of_week, self.day_of_month,
            self.is_weekend, self.is_holiday, self.days_remaining,
            # Trends (16-20)
            self.ctr_trend_7d, self.cvr_trend_7d, self.roas_trend_7d,
            self.cpa_trend_7d, self.spend_trend_7d,
            # Competitive (21-23)
            self.impression_share, self.auction_pressure, self.competitive_position,
            # ML features (24-28)
            self.audience_quality_score, self.creative_fatigue_score,
            self.predicted_cvr, self.predicted_ltv, self.propensity_score,
            # Context (29-32)
            self.optimization_goal_encoding, self.platform_encoding,
            self.campaign_maturity, self.budget_utilization,
            # Absolute spend features (33-35)
            self.log_daily_spend, self.log_total_campaign_spend, self.log_daily_budget,
            # Audience segmentation (36-38)
            self.segment_count, self.top_segment_roas, self.avg_frequency,
            # Constraint features (39-41)
            self.target_cpa_norm, self.min_roas_norm, self.daily_budget_limit_norm,
        ]
        return torch.tensor(state_vector, dtype=torch.float32, device=device)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "CampaignState":
        """Reconstruct state from tensor"""
        values = tensor.cpu().numpy().tolist()
        # Support loading both 36-dim (legacy) and 42-dim tensors
        v = lambda i, default=0.0: values[i] if i < len(values) else default
        return cls(
            ctr=v(0), cvr=v(1), roas=v(2), cpa=v(3),
            cpc=v(4), cpm=v(5), spend_velocity=v(6),
            impression_volume=v(7), click_volume=v(8),
            conversion_volume=v(9), hour_of_day=v(10),
            day_of_week=v(11), day_of_month=v(12),
            is_weekend=v(13), is_holiday=v(14),
            days_remaining=v(15), ctr_trend_7d=v(16),
            cvr_trend_7d=v(17), roas_trend_7d=v(18),
            cpa_trend_7d=v(19), spend_trend_7d=v(20),
            impression_share=v(21), auction_pressure=v(22),
            competitive_position=v(23), audience_quality_score=v(24),
            creative_fatigue_score=v(25), predicted_cvr=v(26),
            predicted_ltv=v(27), propensity_score=v(28),
            optimization_goal_encoding=v(29), platform_encoding=v(30),
            campaign_maturity=v(31), budget_utilization=v(32),
            log_daily_spend=v(33), log_total_campaign_spend=v(34),
            log_daily_budget=v(35),
            segment_count=v(36), top_segment_roas=v(37), avg_frequency=v(38),
            target_cpa_norm=v(39), min_roas_norm=v(40), daily_budget_limit_norm=v(41),
        )
    
    @classmethod
    def state_dim(cls) -> int:
        """Return dimensionality of state vector (42 = 36 base + 3 audience + 3 constraint)"""
        return 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return asdict(self)
    
    @classmethod
    def from_campaign_metrics(
        cls,
        campaign_id: str,
        metrics: Dict[str, Any],
        temporal: Dict[str, Any],
        ml_features: Dict[str, Any],
        normalization_params: Optional[Dict[str, Tuple[float, float]]] = None,
        constraints: Optional[Dict[str, float]] = None,
        audience_info: Optional[Dict[str, float]] = None,
    ) -> "CampaignState":
        """
        Factory method to create state from raw campaign data
        
        Args:
            campaign_id: Campaign identifier
            metrics: Raw performance metrics
            temporal: Temporal/seasonal information
            ml_features: Pre-computed ML predictions
            normalization_params: Optional (mean, std) for each feature
        """
        def normalize(value: float, feature_name: str) -> float:
            if normalization_params and feature_name in normalization_params:
                mean, std = normalization_params[feature_name]
                return (value - mean) / (std + 1e-8)
            return value
        
        # Compute derived metrics
        impressions = metrics.get("impressions", 0)
        clicks = metrics.get("clicks", 0)
        conversions = metrics.get("conversions", 0)
        spend = metrics.get("spend", 0)
        revenue = metrics.get("revenue", 0)
        
        ctr = clicks / max(impressions, 1)
        cvr = conversions / max(clicks, 1)
        roas = revenue / max(spend, 1)
        cpa = spend / max(conversions, 1)
        cpc = spend / max(clicks, 1)
        cpm = (spend / max(impressions, 1)) * 1000
        
        # Compute log-normalized absolute spend features
        daily_spend = metrics.get("spend", 0)
        daily_budget = metrics.get("daily_budget", 0)
        total_campaign_spend = metrics.get("total_campaign_spend", 0)

        log_daily_spend = np.log1p(daily_spend) / np.log1p(MAX_DAILY_SPEND)
        log_total_campaign_spend = np.log1p(total_campaign_spend) / np.log1p(MAX_TOTAL_SPEND)
        log_daily_budget = np.log1p(daily_budget) / np.log1p(MAX_DAILY_BUDGET)

        # Audience segmentation features
        aud = audience_info or {}
        seg_count = aud.get("segment_count", 0) / 10.0
        top_seg_roas = aud.get("top_segment_roas", 0.0)
        avg_freq = aud.get("avg_frequency", 0.0) / 10.0

        # Constraint features — let the policy see the advertiser's targets
        con = constraints or {}
        target_cpa_val = con.get("target_cpa", 0.0)
        min_roas_val = con.get("min_roas", 0.0)
        budget_limit_val = con.get("daily_budget_limit", 0.0)
        target_cpa_n = np.log1p(target_cpa_val) / np.log1p(1000.0) if target_cpa_val > 0 else 0.0
        min_roas_n = min(min_roas_val / 10.0, 1.0)
        budget_limit_n = np.log1p(budget_limit_val) / np.log1p(MAX_DAILY_BUDGET) if budget_limit_val > 0 else 0.0

        return cls(
            campaign_id=campaign_id,
            ctr=normalize(ctr, "ctr"),
            cvr=normalize(cvr, "cvr"),
            roas=normalize(roas, "roas"),
            cpa=normalize(1.0 / max(cpa, 0.01), "cpa_inverse"),  # Inverse for "higher is better"
            cpc=normalize(cpc, "cpc"),
            cpm=normalize(cpm, "cpm"),
            spend_velocity=metrics.get("spend_velocity", 0),
            impression_volume=normalize(impressions, "impressions"),
            click_volume=normalize(clicks, "clicks"),
            conversion_volume=normalize(conversions, "conversions"),
            hour_of_day=temporal.get("hour", 0) / 24.0,
            day_of_week=temporal.get("day_of_week", 0) / 7.0,
            day_of_month=temporal.get("day_of_month", 1) / 31.0,
            is_weekend=float(temporal.get("is_weekend", False)),
            is_holiday=float(temporal.get("is_holiday", False)),
            days_remaining=temporal.get("days_remaining", 30) / 90.0,
            ctr_trend_7d=metrics.get("ctr_trend_7d", 0),
            cvr_trend_7d=metrics.get("cvr_trend_7d", 0),
            roas_trend_7d=metrics.get("roas_trend_7d", 0),
            cpa_trend_7d=-metrics.get("cpa_trend_7d", 0),  # Negate so positive = improving
            spend_trend_7d=metrics.get("spend_trend_7d", 0),
            impression_share=metrics.get("impression_share", 0.5),
            auction_pressure=metrics.get("auction_pressure", 0.5),
            competitive_position=metrics.get("competitive_position", 0.5),
            audience_quality_score=ml_features.get("audience_quality", 0.5),
            creative_fatigue_score=ml_features.get("creative_fatigue", 0),
            predicted_cvr=ml_features.get("predicted_cvr", cvr),
            predicted_ltv=ml_features.get("predicted_ltv", 0.5),
            propensity_score=ml_features.get("propensity", 0.5),
            optimization_goal_encoding=metrics.get("goal_encoding", 0),
            platform_encoding=metrics.get("platform_encoding", 0),
            campaign_maturity=temporal.get("campaign_age_days", 0) / 365.0,
            budget_utilization=metrics.get("budget_utilization", 0),
            log_daily_spend=log_daily_spend,
            log_total_campaign_spend=log_total_campaign_spend,
            log_daily_budget=log_daily_budget,
            segment_count=seg_count,
            top_segment_roas=top_seg_roas,
            avg_frequency=avg_freq,
            target_cpa_norm=float(target_cpa_n),
            min_roas_norm=float(min_roas_n),
            daily_budget_limit_norm=float(budget_limit_n),
        )


@dataclass
class ActionSpace:
    """
    Complete action space for DRL agent
    
    Combines:
    - Continuous actions: bid/budget adjustments (-1 to 1, scaled to bounds)
    - Discrete actions: audience/creative decisions (categorical)
    """
    
    # === Continuous Actions (normalized -1 to 1) ===
    bid_adjustment: float = 0.0         # Scaled to [-max_decrease, +max_increase]
    budget_adjustment: float = 0.0      # Scaled to [-max_decrease, +max_increase]

    # === Discrete Actions ===
    audience_action: int = 0            # AudienceAction enum value
    creative_action: int = 0            # CreativeAction enum value

    # === Action Metadata ===
    confidence: float = 0.0             # Agent's confidence in this action
    entropy: float = 0.0                # Policy entropy at this state
    q_value: float = 0.0                # Estimated Q-value

    # === Tracking ===
    action_id: str = ""                 # Unique ID for tracing this action
    timestamp: str = ""                 # ISO timestamp when action was produced

    def __post_init__(self):
        if not self.action_id:
            self.action_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    @classmethod
    def continuous_dim(cls) -> int:
        """Dimension of continuous action space"""
        return 2
    
    @classmethod
    def discrete_dims(cls) -> List[int]:
        """Dimensions of discrete action spaces"""
        return [4, 4]  # 4 audience actions, 4 creative actions
    
    def to_tensor(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to tensors for neural network
        
        Returns:
            Tuple of (continuous_tensor, discrete_tensor)
        """
        continuous = torch.tensor(
            [self.bid_adjustment, self.budget_adjustment],
            dtype=torch.float32, device=device
        )
        discrete = torch.tensor(
            [self.audience_action, self.creative_action],
            dtype=torch.long, device=device
        )
        return continuous, discrete
    
    @classmethod
    def from_tensors(
        cls, 
        continuous: torch.Tensor, 
        discrete: torch.Tensor,
        confidence: float = 0.0,
        entropy: float = 0.0,
        q_value: float = 0.0
    ) -> "ActionSpace":
        """Reconstruct action from tensors"""
        cont = continuous.cpu().numpy()
        disc = discrete.cpu().numpy()
        return cls(
            bid_adjustment=float(cont[0]),
            budget_adjustment=float(cont[1]),
            audience_action=int(disc[0]),
            creative_action=int(disc[1]),
            confidence=confidence,
            entropy=entropy,
            q_value=q_value,
        )
    
    def scale_to_bounds(
        self,
        max_bid_increase: float = 0.5,
        max_bid_decrease: float = 0.3,
        max_budget_increase: float = 0.3,
        max_budget_decrease: float = 0.3,
    ) -> "ActionSpace":
        """
        Scale normalized actions to actual percentage bounds
        
        The neural network outputs in [-1, 1], this scales to actual bounds
        """
        # Asymmetric scaling for bid (can increase more than decrease)
        if self.bid_adjustment >= 0:
            scaled_bid = self.bid_adjustment * max_bid_increase
        else:
            scaled_bid = self.bid_adjustment * max_bid_decrease
        
        # Symmetric scaling for budget
        if self.budget_adjustment >= 0:
            scaled_budget = self.budget_adjustment * max_budget_increase
        else:
            scaled_budget = self.budget_adjustment * max_budget_decrease
        
        return ActionSpace(
            bid_adjustment=scaled_bid,
            budget_adjustment=scaled_budget,
            audience_action=self.audience_action,
            creative_action=self.creative_action,
            confidence=self.confidence,
            entropy=self.entropy,
            q_value=self.q_value,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "bid_adjustment": self.bid_adjustment,
            "budget_adjustment": self.budget_adjustment,
            "audience_action": AudienceAction(self.audience_action).name,
            "creative_action": CreativeAction(self.creative_action).name,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "q_value": self.q_value,
        }


@dataclass
class DRLDirective:
    """
    High-level strategic directive from DRL macro layer
    
    This is passed to the LLM micro layer to constrain tactical execution.
    The DRL decides WHAT to do, the LLM decides HOW to communicate it.
    """
    
    # === Strategic Decisions ===
    budget_allocation: float = 0.0      # Budget change percentage
    bid_strategy: float = 0.0           # Bid change percentage
    audience_priority: str = "hold"     # Audience action as string
    creative_direction: str = "hold"    # Creative action as string
    
    # === Derived Messaging Guidance ===
    messaging_tone: str = "consistent"  # Inferred tone for LLM
    urgency_level: float = 0.0          # 0-1 urgency score
    value_emphasis: float = 0.0         # 0-1 value/efficiency emphasis
    
    # === Constraints for LLM ===
    max_offer_discount: float = 0.0     # Maximum discount LLM can offer
    product_focus: str = ""             # Product category to emphasize
    audience_segment: str = ""          # Target segment for personalization
    
    # === Confidence Metrics ===
    strategic_confidence: float = 0.0   # DRL confidence
    recommended_test: bool = False      # Suggest A/B test
    
    @classmethod
    def from_action(
        cls,
        action: ActionSpace,
        state: CampaignState,
        campaign_context: Dict[str, Any]
    ) -> "DRLDirective":
        """
        Create directive from DRL action and campaign context
        
        Args:
            action: Raw action from DRL agent
            state: Current campaign state
            campaign_context: Additional campaign information
        """
        # Determine messaging tone based on action + state
        if action.bid_adjustment > 0.2 and state.roas > 0.6:  # Normalized ROAS
            tone = "aggressive_growth"
            urgency = 0.8
            value_emphasis = 0.2
        elif action.budget_adjustment < -0.1:
            tone = "efficiency_focused"
            urgency = 0.3
            value_emphasis = 0.9
        elif state.creative_fatigue_score > 0.7:
            tone = "fresh_angle"
            urgency = 0.5
            value_emphasis = 0.5
        elif action.bid_adjustment > 0.1:
            tone = "urgency"
            urgency = 0.7
            value_emphasis = 0.4
        else:
            tone = "consistent"
            urgency = 0.4
            value_emphasis = 0.5
        
        # Determine max discount based on performance
        if state.roas > 0.7:  # High ROAS = can afford discounts
            max_discount = 0.25
        elif state.roas > 0.5:
            max_discount = 0.15
        else:
            max_discount = 0.05
        
        return cls(
            budget_allocation=action.budget_adjustment,
            bid_strategy=action.bid_adjustment,
            audience_priority=AudienceAction(action.audience_action).name.lower(),
            creative_direction=CreativeAction(action.creative_action).name.lower(),
            messaging_tone=tone,
            urgency_level=urgency,
            value_emphasis=value_emphasis,
            max_offer_discount=max_discount,
            product_focus=campaign_context.get("product_focus", ""),
            audience_segment=campaign_context.get("target_segment", ""),
            strategic_confidence=action.confidence,
            recommended_test=action.confidence < 0.7,
        )
    
    def to_llm_prompt_context(self) -> str:
        """
        Generate context string for LLM prompt
        """
        return f"""
Strategic Directive from Optimization Engine:
- Budget Direction: {self.budget_allocation:+.1%} adjustment recommended
- Bid Direction: {self.bid_strategy:+.1%} adjustment recommended  
- Audience Strategy: {self.audience_priority}
- Creative Strategy: {self.creative_direction}

Messaging Guidelines:
- Tone: {self.messaging_tone}
- Urgency Level: {self.urgency_level:.0%}
- Value Emphasis: {self.value_emphasis:.0%}

Constraints:
- Maximum offer discount: {self.max_offer_discount:.0%}
- Product focus: {self.product_focus or 'General'}
- Target segment: {self.audience_segment or 'Broad'}

Confidence: {self.strategic_confidence:.0%}
{"⚠️ Consider A/B testing this recommendation" if self.recommended_test else ""}
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# Type aliases for clarity
StateType = CampaignState
ActionType = ActionSpace
DirectiveType = DRLDirective
