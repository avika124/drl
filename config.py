"""
DRL Configuration and Hyperparameters

Central configuration for all DRL components including:
- Model architecture parameters
- Training hyperparameters
- Safety guardrails
- Production deployment settings
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np


class OptimizationGoal(Enum):
    """Supported campaign optimization goals"""
    ROAS = "roas"
    CPA = "cpa"
    CONVERSIONS = "conversions"
    CTR = "ctr"
    REVENUE = "revenue"
    PROFIT = "profit"


class ActionType(Enum):
    """Types of optimization actions"""
    BID_ADJUSTMENT = "bid_adjustment"
    BUDGET_ADJUSTMENT = "budget_adjustment"
    AUDIENCE_EXPANSION = "audience_expansion"
    AUDIENCE_REFINEMENT = "audience_refinement"
    CREATIVE_ROTATION = "creative_rotation"
    PLATFORM_SHIFT = "platform_shift"


@dataclass
class DRLConfig:
    """
    Core DRL model configuration
    """
    # State space dimensions
    # 33 original + 3 spend + 3 audience segmentation + 3 constraint = 42
    state_dim: int = 42
    
    # Action space configuration
    continuous_action_dim: int = 2  # bid_adjustment, budget_adjustment
    discrete_action_dims: List[int] = field(default_factory=lambda: [4, 4])  # audience, creative
    
    # Network architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    activation: str = "relu"
    use_layer_norm: bool = True
    dropout_rate: float = 0.1
    
    # SAC specific
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005   # Target network soft update
    alpha: float = 0.2   # Entropy coefficient (auto-tuned if None)
    auto_entropy_tuning: bool = True
    target_entropy: Optional[float] = None  # Auto-computed if None
    
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    
    # Model paths
    model_dir: str = "models/drl"
    checkpoint_frequency: int = 1000
    
    def __post_init__(self):
        if self.target_entropy is None:
            # OPTIMIZATION: Adjusted for Hybrid Action Space
            # Continuous entropy is often negative (Gaussian).
            # Discrete entropy is always positive (Categorical).
            # Standard heuristic (-dim(A)) fails for hybrid spaces.
            
            # 1. Target for continuous: -dim(A_cont) (Standard SAC)
            target_cont = -float(self.continuous_action_dim)
            
            # 2. Target for discrete: ratio * max_entropy
            # We want to maintain some randomness in discrete choices.
            # Max entropy for 4 options is ln(4) ≈ 1.38
            target_disc = sum([np.log(d) for d in self.discrete_action_dims]) * 0.98
            
            # Sum them up
            self.target_entropy = target_cont + target_disc

@dataclass
class TrainingConfig:
    """
    Training hyperparameters for offline and online learning
    """
    # Batch and buffer
    batch_size: int = 256
    replay_buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000  # Start training after this many samples
    
    # Training schedule
    num_offline_epochs: int = 100
    steps_per_epoch: int = 1000
    gradient_steps_per_update: int = 1
    
    # Online learning
    online_update_frequency: int = 4  # Update every N environment steps
    target_update_frequency: int = 1  # Update target networks every N gradient steps
    
    # Conservative Q-Learning (CQL) for offline training
    use_cql: bool = True
    cql_alpha: float = 1.0  # CQL regularization weight
    cql_num_samples: int = 10  # Number of action samples for CQL
    cql_importance_sample: bool = True
    
    # Prioritized Experience Replay
    use_per: bool = True
    per_alpha: float = 0.6  # Prioritization exponent
    per_beta_start: float = 0.4  # Importance sampling start
    per_beta_end: float = 1.0  # Importance sampling end
    per_beta_anneal_steps: int = 100_000
    
    # Regularization
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Exploration
    initial_exploration_steps: int = 10_000
    exploration_noise_std: float = 0.1
    
    # Validation
    validation_frequency: int = 5000
    validation_episodes: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.01


@dataclass
class GuardrailConfig:
    """
    Safety guardrails for production deployment
    """
    # Bid constraints
    max_bid_increase_pct: float = 0.50  # Max 50% bid increase
    max_bid_decrease_pct: float = 0.30  # Max 30% bid decrease
    min_bid: float = 0.01  # Minimum bid in dollars
    max_bid: float = 100.0  # Maximum bid in dollars
    
    # Budget constraints
    max_budget_increase_pct: float = 0.30  # Max 30% budget increase
    max_budget_decrease_pct: float = 0.30  # Max 30% budget decrease
    min_daily_budget: float = 10.0  # Minimum daily budget
    max_daily_budget: float = 100_000.0  # Maximum daily budget
    
    # Action frequency limits
    min_hours_between_actions: float = 4.0  # Minimum cooldown
    max_actions_per_day: int = 6  # Maximum actions per campaign per day
    
    # Confidence thresholds
    min_confidence_for_action: float = 0.7  # Minimum confidence to act
    min_confidence_for_auto_apply: float = 0.85  # Higher threshold for auto-apply
    
    # Performance safeguards
    max_spend_increase_per_action: float = 1000.0  # Max spend increase per action
    emergency_stop_roas_threshold: float = 0.5  # Stop if ROAS drops below this
    emergency_stop_cpa_multiplier: float = 3.0  # Stop if CPA exceeds target by this factor
    
    # Exploration bounds
    max_exploration_rate: float = 0.2  # Maximum random exploration
    min_exploration_rate: float = 0.01  # Minimum exploration
    exploration_decay_rate: float = 0.995  # Per-step decay
    
    # Rollback settings
    enable_auto_rollback: bool = True
    rollback_observation_hours: int = 24
    rollback_performance_threshold: float = -0.15  # Rollback if performance drops 15%
    
    def validate_bid_adjustment(self, current_bid: float, adjustment: float) -> tuple[float, bool]:
        """
        Validate and constrain bid adjustment
        Returns: (constrained_adjustment, was_constrained)
        """
        constrained = False
        
        # Apply percentage bounds
        if adjustment > self.max_bid_increase_pct:
            adjustment = self.max_bid_increase_pct
            constrained = True
        elif adjustment < -self.max_bid_decrease_pct:
            adjustment = -self.max_bid_decrease_pct
            constrained = True
        
        # Check absolute bounds
        new_bid = current_bid * (1 + adjustment)
        if new_bid < self.min_bid:
            adjustment = (self.min_bid / current_bid) - 1
            constrained = True
        elif new_bid > self.max_bid:
            adjustment = (self.max_bid / current_bid) - 1
            constrained = True
        
        return adjustment, constrained
    
    def validate_budget_adjustment(self, current_budget: float, adjustment: float) -> tuple[float, bool]:
        """
        Validate and constrain budget adjustment
        Returns: (constrained_adjustment, was_constrained)
        """
        constrained = False
        
        # Apply percentage bounds
        if adjustment > self.max_budget_increase_pct:
            adjustment = self.max_budget_increase_pct
            constrained = True
        elif adjustment < -self.max_budget_decrease_pct:
            adjustment = -self.max_budget_decrease_pct
            constrained = True
        
        # Check absolute bounds
        new_budget = current_budget * (1 + adjustment)
        if new_budget < self.min_daily_budget:
            adjustment = (self.min_daily_budget / current_budget) - 1
            constrained = True
        elif new_budget > self.max_daily_budget:
            adjustment = (self.max_daily_budget / current_budget) - 1
            constrained = True
        
        return adjustment, constrained


@dataclass
class RewardConfig:
    """
    Configuration for multi-objective reward function
    """
    # Primary objective weights (should sum to ~1.0)
    roas_weight: float = 0.4
    cpa_weight: float = 0.3
    conversion_weight: float = 0.2
    ctr_weight: float = 0.1
    
    # Bonus/penalty weights
    efficiency_bonus_weight: float = 0.1
    volume_bonus_weight: float = 0.05
    ltv_bonus_weight: float = 0.15
    
    # Constraint violation penalties
    budget_violation_penalty: float = -1.0
    cpa_violation_penalty: float = -0.5
    roas_violation_penalty: float = -0.5
    
    # Action penalties
    action_magnitude_penalty: float = 0.01  # Penalize large changes
    action_frequency_penalty: float = 0.05  # Penalize frequent changes
    
    # Thresholds
    roas_target: float = 2.0  # Target ROAS for bonus
    roas_excellent: float = 4.0  # Excellent ROAS threshold
    cpa_target_multiplier: float = 1.5  # Penalty if CPA exceeds target by this factor
    
    # Spend-efficiency reward (teaches diminishing returns at different spend levels)
    spend_efficiency_weight: float = 0.1
    diminishing_return_threshold: float = 0.7  # Budget utilization above which diminishing returns apply
    spend_efficiency_alpha: float = 0.5        # Concavity parameter for diminishing return curve

    # Temporal discounting
    gamma: float = 0.99  # Discount factor for future rewards
    
    # Normalization
    normalize_rewards: bool = True
    reward_scale: float = 1.0
    reward_clip: Optional[float] = 10.0


@dataclass 
class FeatureConfig:
    """
    Configuration for state feature engineering
    """
    # Temporal features
    include_hour_of_day: bool = True
    include_day_of_week: bool = True
    include_day_of_month: bool = True
    include_is_weekend: bool = True
    include_is_holiday: bool = True
    
    # Rolling window sizes (in days)
    rolling_windows: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    
    # Metric features to include
    metric_features: List[str] = field(default_factory=lambda: [
        "ctr", "cvr", "cpc", "cpm", "roas", "cpa",
        "impressions", "clicks", "conversions", "spend", "revenue"
    ])
    
    # Derived features
    include_trend_features: bool = True
    include_volatility_features: bool = True
    include_competitive_features: bool = True
    include_audience_features: bool = True
    include_creative_features: bool = True
    
    # ML-derived features
    include_predicted_ltv: bool = True
    include_predicted_cvr: bool = True
    include_fatigue_score: bool = True
    include_audience_quality: bool = True
    
    # Normalization
    normalize_features: bool = True
    normalization_method: str = "standard"  # "standard", "minmax", "robust"
    clip_outliers: bool = True
    outlier_std: float = 3.0


# ---------------------------------------------------------------------------
# Cross-Platform DRL Engine Strategy
# ---------------------------------------------------------------------------

class AllocationStrategy(Enum):
    """Strategy for cross-platform budget allocation."""
    DRL_PRIMARY = "drl_primary"              # X-Model primary, heuristic fallback
    HEURISTIC_PRIMARY = "heuristic_primary"  # Heuristic primary (legacy default)
    DUAL_BENCHMARK = "dual_benchmark"        # Run both, compare, apply DRL


@dataclass
class CrossPlatformStrategyConfig:
    """Configuration for the CrossPlatformDRLEngine orchestrator."""

    # Strategy selection
    strategy: AllocationStrategy = AllocationStrategy.DRL_PRIMARY

    # Fallback thresholds
    min_confidence_for_drl: float = 0.40    # Below this, fall back to heuristic
    min_training_steps: int = 500           # X-Model must have >= this many steps

    # Automatic training data collection
    auto_collect_training_data: bool = True

    # Retraining triggers
    retrain_snapshot_threshold: int = 50    # Retrain after N new snapshots
    retrain_min_transitions: int = 30       # Minimum transitions for a training run
    retrain_epochs: int = 10                # Epochs per retrain cycle
    retrain_steps_per_epoch: int = 100      # Steps per epoch

    # Benchmark tracking
    benchmark_history_size: int = 500       # Max dual-run results to retain

    # Model persistence
    checkpoint_dir: str = "models/x_model"


# Default configurations
DEFAULT_DRL_CONFIG = DRLConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_GUARDRAIL_CONFIG = GuardrailConfig()
DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_STRATEGY_CONFIG = CrossPlatformStrategyConfig()