"""
Deep Reinforcement Learning Module for Campaign Optimization

This module implements a production-ready DRL system for advertising
campaign optimization using Soft Actor-Critic (SAC) with:
- Offline pretraining using Conservative Q-Learning (CQL)
- Online fine-tuning with prioritized experience replay
- Safety guardrails for production deployment
- DRL + LLM hybrid architecture integration
- Cross-platform budget allocation and traffic optimization

P & X Model Architecture:
    M5 — X-Execution: XModelAgent selects cross-platform allocation
        ↓
    M2 — P-Execution: Per-platform P-Models (via PlatformModelRegistry)
        ↓
    DRL Macro Layer: Campaign-level strategic decisions
        ↓
    LLM Micro Layer: Tactical execution (creative, messaging, offers)

Training Pipeline:
    M1 — P-Training: Per-platform SAC training (one model per platform)
    M3 — X-Training Data: Portfolio-level transitions from platform outcomes
    M4 — X-Training: XModelAgent training on M3 data
"""

from .config import DRLConfig, TrainingConfig, GuardrailConfig, AllocationStrategy, CrossPlatformStrategyConfig
from .state_action import CampaignState, ActionSpace, DRLDirective
from .networks import ActorNetwork, CriticNetwork, ValueNetwork
from .sac_agent import SACAgent
from .replay_buffer import PrioritizedReplayBuffer, Transition
from .reward_functions import RewardComputer, MultiObjectiveReward
from .offline_trainer import OfflineTrainer
from .safe_agent import SafeDRLAgent, ActionValidator
from .hybrid_optimizer import HybridDRLLLMOptimizer, OptimizationResult
from .continuous_learning import ContinuousLearningEngine, OutcomeTracker
from .ab_testing import DRLABTestManager, ExperimentResult
from .cross_platform_optimizer import (
    CrossPlatformOptimizer,
    CrossPlatformConfig,
    CrossPlatformResult,
    PlatformPortfolio,
    PlatformMetrics,
    PlatformPerformanceTracker,
    MarginalReturnEstimator,
    BudgetAllocator,
    AllocationRecommendation,
    Platform,
    BudgetRecommendation,
    BudgetRecommendationConfig,
)
from .xai_narrator import OptimizationNarrator, RunNarrative, PortfolioNarrative, ParameterGlossary
from .audience_constraints import AudienceConstraintManager, AudienceConstraintResult
from .benchmark_model import CampaignForecaster, CampaignForecast
from .forecast_feedback import ForecastFeedbackLoop, AccuracyMetrics
from .sac_agent import CQLLoss, load_sac_for_inference
from .platform_model_registry import PlatformModelRegistry, PlatformModelMeta
from .x_model import XModelAgent, XModelState, XModelAction, build_x_state
from .x_training_data import XTrainingDataBuilder, XTransition
from .x_training import XModelTrainer
from .cross_platform_drl_engine import (
    CrossPlatformDRLEngine,
    DualRunResult,
    ModelReadinessChecker,
)

__version__ = "2.1.0"
__author__ = "AI Advertising Platform"

__all__ = [
    # Config
    "DRLConfig",
    "TrainingConfig",
    "GuardrailConfig",
    "AllocationStrategy",
    "CrossPlatformStrategyConfig",
    # State/Action
    "CampaignState",
    "ActionSpace",
    "DRLDirective",
    # Networks
    "ActorNetwork",
    "CriticNetwork",
    "ValueNetwork",
    # Agent
    "SACAgent",
    "SafeDRLAgent",
    "ActionValidator",
    # Training
    "PrioritizedReplayBuffer",
    "Transition",
    "OfflineTrainer",
    "CQLLoss",
    # Reward
    "RewardComputer",
    "MultiObjectiveReward",
    # Hybrid
    "HybridDRLLLMOptimizer",
    "OptimizationResult",
    # Cross-Platform
    "CrossPlatformOptimizer",
    "CrossPlatformConfig",
    "CrossPlatformResult",
    "PlatformPortfolio",
    "PlatformMetrics",
    "PlatformPerformanceTracker",
    "MarginalReturnEstimator",
    "BudgetAllocator",
    "AllocationRecommendation",
    "Platform",
    "BudgetRecommendation",
    "BudgetRecommendationConfig",
    # Learning
    "ContinuousLearningEngine",
    "OutcomeTracker",
    # Testing
    "DRLABTestManager",
    "ExperimentResult",
    # xAI Narrative
    "OptimizationNarrator",
    "RunNarrative",
    "PortfolioNarrative",
    "ParameterGlossary",
    # Audience Constraints
    "AudienceConstraintManager",
    "AudienceConstraintResult",
    # Forecasting
    "CampaignForecaster",
    "CampaignForecast",
    # Forecast Feedback Loop
    "ForecastFeedbackLoop",
    "AccuracyMetrics",
    # Inference Helper
    "load_sac_for_inference",
    # P & X Model Architecture
    "PlatformModelRegistry",
    "PlatformModelMeta",
    "XModelAgent",
    "XModelState",
    "XModelAction",
    "build_x_state",
    "XTrainingDataBuilder",
    "XTransition",
    "XModelTrainer",
    # DRL Engine (X-Model primary orchestrator)
    "CrossPlatformDRLEngine",
    "DualRunResult",
    "ModelReadinessChecker",
]
