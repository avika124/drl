"""
Safe DRL Agent for Production Deployment

===== STEP 2: P-MODEL EXECUTION =====
Data Flow: CampaignState + CampaignContext -> SACAgent.select_action -> ActionValidator -> SafeDRLAgent.get_action

Implements:
- Action validation and constraint enforcement
- Safety guardrails (bid/budget bounds, cooldowns)
- Exploration with safety bounds
- Automatic rollback capabilities
- Action logging and audit trail
"""
# QA/Testing: Set True to enable input/output logging for traceability
_QA_IO_LOGGING = True

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import json

from .config import GuardrailConfig, DRLConfig
from .sac_agent import SACAgent
from .state_action import CampaignState, ActionSpace, AudienceAction, CreativeAction

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of action after validation"""
    APPROVED = "approved"
    MODIFIED = "modified"
    BLOCKED = "blocked"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ActionValidationResult:
    """Result of action validation"""
    original_action: ActionSpace
    validated_action: ActionSpace
    status: ActionStatus
    modifications: List[str] = field(default_factory=list)
    blocking_reason: Optional[str] = None
    confidence_adjusted: bool = False
    requires_human_review: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_action": self.original_action.to_dict(),
            "validated_action": self.validated_action.to_dict(),
            "status": self.status.value,
            "modifications": self.modifications,
            "blocking_reason": self.blocking_reason,
            "confidence_adjusted": self.confidence_adjusted,
            "requires_human_review": self.requires_human_review,
        }


@dataclass
class CampaignContext:
    """Context about campaign for validation"""
    campaign_id: str
    current_bid: float
    current_budget: float
    last_action_at: Optional[datetime]
    actions_today: int
    current_roas: float
    current_cpa: float
    target_cpa: Optional[float] = None
    min_roas: Optional[float] = None
    is_new_campaign: bool = False
    total_spend: float = 0.0
    
    def hours_since_last_action(self) -> float:
        if self.last_action_at is None:
            return float("inf")
        return (datetime.now(timezone.utc) - self.last_action_at).total_seconds() / 3600


class ActionValidator:
    """
    [ActionValidator]
    Description: Validates and constrains DRL actions against guardrails (bid/budget bounds, cooldowns, rate limits).
    Input: ActionSpace from SACAgent, CampaignContext from campaign API.
    Output: ActionValidationResult -> consumed by SafeDRLAgent.get_action(), HybridDRLLLMOptimizer.
    """
    
    def __init__(self, guardrails: Optional[GuardrailConfig] = None):
        """
        Args:
            guardrails: Safety guardrail configuration
        """
        self.guardrails = guardrails or GuardrailConfig()
    
    def validate(
        self,
        action: ActionSpace,
        context: CampaignContext
    ) -> ActionValidationResult:
        """
        Validate action against guardrails
        
        Args:
            action: Proposed action from DRL agent
            context: Campaign context
            
        Returns:
            ActionValidationResult with validated action and status
        """
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT validate: action.bid_adj={action.bid_adjustment:.4f}, budget_adj={action.budget_adjustment:.4f}")
            logger.info(f"[IO] INPUT validate: context.campaign_id={context.campaign_id}, current_bid={context.current_bid}, actions_today={context.actions_today}")
        modifications = []
        validated_action = ActionSpace(
            bid_adjustment=action.bid_adjustment,
            budget_adjustment=action.budget_adjustment,
            audience_action=action.audience_action,
            creative_action=action.creative_action,
            confidence=action.confidence,
            entropy=action.entropy,
            q_value=action.q_value,
        )
        
        # Check cooldown
        cooldown_result = self._check_cooldown(context)
        if cooldown_result is not None:
            return cooldown_result
        
        # Check rate limit
        rate_limit_result = self._check_rate_limit(context)
        if rate_limit_result is not None:
            return rate_limit_result
        
        # Check confidence threshold
        if action.confidence < self.guardrails.min_confidence_for_action:
            return ActionValidationResult(
                original_action=action,
                validated_action=validated_action,
                status=ActionStatus.REQUIRES_REVIEW,
                modifications=[],
                blocking_reason=f"Confidence {action.confidence:.2f} below threshold {self.guardrails.min_confidence_for_action}",
                requires_human_review=True,
            )
        
        # Validate bid adjustment
        bid_adj, bid_constrained = self.guardrails.validate_bid_adjustment(
            context.current_bid,
            action.bid_adjustment
        )
        if bid_constrained:
            modifications.append(
                f"Bid adjustment constrained: {action.bid_adjustment:.2%} -> {bid_adj:.2%}"
            )
            validated_action.bid_adjustment = bid_adj
        
        # Validate budget adjustment
        budget_adj, budget_constrained = self.guardrails.validate_budget_adjustment(
            context.current_budget,
            action.budget_adjustment
        )
        if budget_constrained:
            modifications.append(
                f"Budget adjustment constrained: {action.budget_adjustment:.2%} -> {budget_adj:.2%}"
            )
            validated_action.budget_adjustment = budget_adj
        
        # Check emergency conditions
        emergency_result = self._check_emergency_conditions(
            validated_action, context
        )
        if emergency_result is not None:
            return emergency_result
        
        # Check spend increase limit
        spend_increase = context.current_budget * validated_action.budget_adjustment
        if spend_increase > self.guardrails.max_spend_increase_per_action:
            capped_increase = self.guardrails.max_spend_increase_per_action / context.current_budget
            modifications.append(
                f"Budget increase capped by spend limit: {validated_action.budget_adjustment:.2%} -> {capped_increase:.2%}"
            )
            validated_action.budget_adjustment = capped_increase
        
        # Determine status
        if modifications:
            status = ActionStatus.MODIFIED
        else:
            status = ActionStatus.APPROVED
        
        # Check if requires human review for auto-apply
        requires_review = (
            validated_action.confidence < self.guardrails.min_confidence_for_auto_apply
        )
        
        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] OUTPUT validate: status={status.value}, requires_review={requires_review} | Next: SafeDRLAgent, HybridDRLLLMOptimizer")
        return ActionValidationResult(
            original_action=action,
            validated_action=validated_action,
            status=status,
            modifications=modifications,
            requires_human_review=requires_review,
        )
    
    def _check_cooldown(
        self,
        context: CampaignContext
    ) -> Optional[ActionValidationResult]:
        """Check if action is within cooldown period"""
        hours_since = context.hours_since_last_action()
        
        if hours_since < self.guardrails.min_hours_between_actions:
            return ActionValidationResult(
                original_action=ActionSpace(),
                validated_action=ActionSpace(),  # No-op
                status=ActionStatus.BLOCKED,
                blocking_reason=(
                    f"Cooldown: {hours_since:.1f}h since last action, "
                    f"minimum {self.guardrails.min_hours_between_actions}h required"
                ),
            )
        return None
    
    def _check_rate_limit(
        self,
        context: CampaignContext
    ) -> Optional[ActionValidationResult]:
        """Check if action exceeds daily rate limit"""
        if context.actions_today >= self.guardrails.max_actions_per_day:
            return ActionValidationResult(
                original_action=ActionSpace(),
                validated_action=ActionSpace(),
                status=ActionStatus.BLOCKED,
                blocking_reason=(
                    f"Rate limit: {context.actions_today} actions today, "
                    f"maximum {self.guardrails.max_actions_per_day} allowed"
                ),
            )
        return None
    
    def _check_emergency_conditions(
        self,
        action: ActionSpace,
        context: CampaignContext
    ) -> Optional[ActionValidationResult]:
        """Check for emergency stop conditions"""
        # Check ROAS threshold
        if context.current_roas < self.guardrails.emergency_stop_roas_threshold:
            # Only allow budget decreases
            if action.budget_adjustment > 0:
                return ActionValidationResult(
                    original_action=action,
                    validated_action=ActionSpace(
                        bid_adjustment=-0.2,  # Emergency bid decrease
                        budget_adjustment=-0.3,  # Emergency budget decrease
                        audience_action=2,  # REFINE
                        creative_action=1,  # ROTATE
                    ),
                    status=ActionStatus.MODIFIED,
                    modifications=[
                        f"Emergency: ROAS {context.current_roas:.2f} below threshold "
                        f"{self.guardrails.emergency_stop_roas_threshold:.2f}"
                    ],
                )
        
        # Check CPA threshold
        if context.target_cpa and context.current_cpa > (
            context.target_cpa * self.guardrails.emergency_stop_cpa_multiplier
        ):
            if action.bid_adjustment > 0 or action.budget_adjustment > 0:
                return ActionValidationResult(
                    original_action=action,
                    validated_action=ActionSpace(
                        bid_adjustment=-0.2,
                        budget_adjustment=-0.2,
                        audience_action=2,  # REFINE
                        creative_action=0,  # HOLD
                    ),
                    status=ActionStatus.MODIFIED,
                    modifications=[
                        f"Emergency: CPA ${context.current_cpa:.2f} exceeds "
                        f"{self.guardrails.emergency_stop_cpa_multiplier}x target "
                        f"${context.target_cpa:.2f}"
                    ],
                )
        
        return None


class SafeDRLAgent:
    """
    [SafeDRLAgent]
    Description: Production wrapper - gets raw action from SACAgent, validates via ActionValidator, logs for audit.
    Input: CampaignState (from metrics), CampaignContext (from campaign API), exploration flag.
    Output: (ActionSpace, ActionValidationResult) -> consumed by HybridDRLLLMOptimizer.optimize().
    
    Combines:
    - Trained SAC agent for action selection
    - Action validator for safety guardrails
    - Exploration management with safety bounds
    - Logging and audit trail
    """
    
    def __init__(
        self,
        agent: SACAgent,
        guardrails: Optional[GuardrailConfig] = None,
        exploration_rate: float = 0.1
    ):
        """
        Args:
            agent: Trained SAC agent
            guardrails: Safety guardrail configuration
            exploration_rate: Initial exploration rate
        """
        self.agent = agent
        self.guardrails = guardrails or GuardrailConfig()
        self.validator = ActionValidator(self.guardrails)
        
        self.exploration_rate = exploration_rate
        self.min_exploration = self.guardrails.min_exploration_rate
        self.max_exploration = self.guardrails.max_exploration_rate
        self.exploration_decay = self.guardrails.exploration_decay_rate
        
        # Action history for rollback
        self.action_history: List[Dict[str, Any]] = []
        
        logger.info("SafeDRLAgent initialized")
    
    async def get_action(
        self,
        state: CampaignState,
        context: CampaignContext,
        exploration: bool = True
    ) -> Tuple[ActionSpace, ActionValidationResult]:
        """
        Get validated action for campaign
        
        Args:
            state: Current campaign state
            context: Campaign context
            exploration: Whether to apply exploration
            
        Returns:
            Tuple of (validated action, validation result)
        """
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT get_action: campaign_id={context.campaign_id}, exploration={exploration}")
            logger.info(f"[IO] INPUT get_action: state.ctr={state.ctr:.4f}, state.roas={state.roas:.4f}")
        # Get action from trained agent
        with torch.no_grad():
            raw_action = self.agent.select_action(
                state,
                deterministic=not exploration
            )
        
        # Apply exploration noise (bounded)
        if exploration and np.random.random() < self.exploration_rate:
            raw_action = self._apply_bounded_exploration(raw_action)
        
        # Scale action to actual bounds
        scaled_action = raw_action.scale_to_bounds(
            max_bid_increase=self.guardrails.max_bid_increase_pct,
            max_bid_decrease=self.guardrails.max_bid_decrease_pct,
            max_budget_increase=self.guardrails.max_budget_increase_pct,
            max_budget_decrease=self.guardrails.max_budget_decrease_pct,
        )
        
        # Validate action
        validation_result = self.validator.validate(scaled_action, context)
        
        # Log action
        self._log_action(state, context, raw_action, validation_result)
        
        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] OUTPUT get_action: status={validation_result.status.value}, bid_adj={validation_result.validated_action.bid_adjustment:.4f} | Next: HybridDRLLLMOptimizer.optimize()")
        return validation_result.validated_action, validation_result
    
    def _apply_bounded_exploration(self, action: ActionSpace) -> ActionSpace:
        """Apply exploration noise within safety bounds"""
        # Gaussian noise with small std
        noise_std = 0.1
        
        bid_noise = np.clip(
            np.random.normal(0, noise_std),
            -0.2, 0.2
        )
        budget_noise = np.clip(
            np.random.normal(0, noise_std),
            -0.15, 0.15
        )
        
        # Occasionally explore discrete actions
        audience_action = action.audience_action
        creative_action = action.creative_action
        
        if np.random.random() < 0.1:
            audience_action = np.random.randint(0, 4)
        if np.random.random() < 0.1:
            creative_action = np.random.randint(0, 4)
        
        return ActionSpace(
            bid_adjustment=action.bid_adjustment + bid_noise,
            budget_adjustment=action.budget_adjustment + budget_noise,
            audience_action=audience_action,
            creative_action=creative_action,
            confidence=action.confidence * 0.9,  # Reduce confidence for exploration
            entropy=action.entropy,
            q_value=action.q_value,
        )
    
    def _log_action(
        self,
        state: CampaignState,
        context: CampaignContext,
        raw_action: ActionSpace,
        validation_result: ActionValidationResult
    ):
        """Log action for audit trail"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "campaign_id": context.campaign_id,
            "state_summary": {
                "ctr": state.ctr,
                "cvr": state.cvr,
                "roas": state.roas,
                "cpa": state.cpa,
            },
            "context_summary": {
                "current_bid": context.current_bid,
                "current_budget": context.current_budget,
                "current_roas": context.current_roas,
            },
            "raw_action": raw_action.to_dict(),
            "validation_result": validation_result.to_dict(),
            "exploration_rate": self.exploration_rate,
        }
        
        self.action_history.append(log_entry)
        
        # Keep last 10000 actions
        if len(self.action_history) > 10000:
            self.action_history = self.action_history[-10000:]
    
    async def rollback_action(
        self,
        campaign_id: str,
        action_timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """
        Rollback a previously applied action
        
        Args:
            campaign_id: Campaign ID
            action_timestamp: Timestamp of action to rollback
            
        Returns:
            Rollback details or None if not found
        """
        # Find the action
        target_action = None
        for entry in reversed(self.action_history):
            if (
                entry["campaign_id"] == campaign_id and
                entry["timestamp"] == action_timestamp
            ):
                target_action = entry
                break
        
        if target_action is None:
            logger.warning(f"Action not found for rollback: {campaign_id} @ {action_timestamp}")
            return None
        
        # Compute inverse action
        validated = target_action["validation_result"]["validated_action"]
        inverse_action = {
            "bid_adjustment": -validated["bid_adjustment"],
            "budget_adjustment": -validated["budget_adjustment"],
            "audience_action": "hold",  # Safe default
            "creative_action": "hold",
        }
        
        return {
            "original_action": validated,
            "inverse_action": inverse_action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get agent diagnostics"""
        agent_diag = self.agent.get_diagnostics()
        
        # Compute action statistics from history
        recent_actions = self.action_history[-100:] if self.action_history else []
        
        if recent_actions:
            statuses = [a["validation_result"]["status"] for a in recent_actions]
            status_counts = {s: statuses.count(s) for s in set(statuses)}
            
            confidences = [
                a["raw_action"]["confidence"] for a in recent_actions
            ]
            avg_confidence = np.mean(confidences)
        else:
            status_counts = {}
            avg_confidence = 0
        
        return {
            **agent_diag,
            "exploration_rate": self.exploration_rate,
            "action_history_size": len(self.action_history),
            "recent_action_status_counts": status_counts,
            "recent_avg_confidence": avg_confidence,
        }
    
    def save(self, path: str):
        """Save agent and history"""
        self.agent.save(path)
        
        # Save action history
        history_path = f"{path}/action_history.json"
        with open(history_path, "w") as f:
            json.dump(self.action_history[-1000:], f)  # Save last 1000
        
        logger.info(f"SafeDRLAgent saved to {path}")
    
    def load(self, path: str):
        """Load agent and history"""
        self.agent.load(path)
        
        # Load action history
        history_path = f"{path}/action_history.json"
        try:
            with open(history_path, "r") as f:
                self.action_history = json.load(f)
        except FileNotFoundError:
            self.action_history = []
        
        logger.info(f"SafeDRLAgent loaded from {path}")


class RollbackManager:
    """
    [RollbackManager]
    Description: Monitors post-action performance and triggers rollback if degradation exceeds threshold.
    Input: campaign_id, action, pre_metrics (from register_action); current_metrics (from check_rollback).
    Output: Rollback details dict or None -> consumed by rollback API, SafeDRLAgent.rollback_action.
    """
    
    def __init__(
        self,
        observation_hours: int = 24,
        performance_threshold: float = -0.15
    ):
        """
        Args:
            observation_hours: Hours to observe before rollback decision
            performance_threshold: Performance drop threshold for rollback
        """
        self.observation_hours = observation_hours
        self.performance_threshold = performance_threshold
        self.pending_observations: Dict[str, Dict[str, Any]] = {}
    
    def register_action(
        self,
        campaign_id: str,
        action: ActionSpace,
        pre_metrics: Dict[str, float]
    ):
        """Register action for rollback monitoring"""
        self.pending_observations[campaign_id] = {
            "action": action.to_dict(),
            "pre_metrics": pre_metrics,
            "timestamp": datetime.now(timezone.utc),
        }
    
    def check_rollback(
        self,
        campaign_id: str,
        current_metrics: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if rollback is needed
        
        Returns:
            Rollback details if needed, None otherwise
        """
        if campaign_id not in self.pending_observations:
            return None
        
        observation = self.pending_observations[campaign_id]
        hours_elapsed = (
            datetime.now(timezone.utc) - observation["timestamp"]
        ).total_seconds() / 3600
        
        if hours_elapsed < self.observation_hours:
            return None  # Still observing
        
        # Compute performance change
        pre = observation["pre_metrics"]
        roas_change = (
            (current_metrics.get("roas", 0) - pre.get("roas", 1)) /
            max(pre.get("roas", 1), 0.01)
        )
        
        # Check if rollback needed
        if roas_change < self.performance_threshold:
            rollback_info = {
                "campaign_id": campaign_id,
                "reason": f"Performance degradation: ROAS change {roas_change:.2%}",
                "pre_metrics": pre,
                "current_metrics": current_metrics,
                "original_action": observation["action"],
            }
            
            # Remove from pending
            del self.pending_observations[campaign_id]
            
            return rollback_info
        
        # Performance acceptable, remove from pending
        del self.pending_observations[campaign_id]
        return None
