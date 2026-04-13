"""
Multi-Objective Reward Functions for Campaign Optimization

Implements:
- Primary objective rewards (ROAS, CPA, conversions, CTR)
- Bonus rewards (efficiency, volume, LTV)
- Constraint violation penalties
- Action smoothness penalties
- Reward normalization and shaping
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from .config import RewardConfig, OptimizationGoal


class RewardComponent(Enum):
    """Identifiers for reward components (for logging/debugging)"""
    PRIMARY = "primary"
    EFFICIENCY_BONUS = "efficiency_bonus"
    VOLUME_BONUS = "volume_bonus"
    LTV_BONUS = "ltv_bonus"
    BUDGET_VIOLATION = "budget_violation"
    CPA_VIOLATION = "cpa_violation"
    ROAS_VIOLATION = "roas_violation"
    ACTION_MAGNITUDE = "action_magnitude"
    ACTION_FREQUENCY = "action_frequency"
    SPEND_EFFICIENCY = "spend_efficiency"
    CONSTRAINT_ALIGNMENT = "constraint_alignment"


@dataclass
class MultiObjectiveReward:
    """
    Detailed reward breakdown for analysis and debugging
    """
    total: float = 0.0
    
    # Component rewards
    primary: float = 0.0
    efficiency_bonus: float = 0.0
    volume_bonus: float = 0.0
    ltv_bonus: float = 0.0
    
    # Spend efficiency
    spend_efficiency: float = 0.0

    # Constraint alignment bonus
    constraint_alignment: float = 0.0

    # Penalties
    budget_violation: float = 0.0
    cpa_violation: float = 0.0
    roas_violation: float = 0.0
    action_magnitude: float = 0.0
    action_frequency: float = 0.0
    
    # Metadata
    goal: str = ""
    normalized: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "components": {
                "primary": self.primary,
                "efficiency_bonus": self.efficiency_bonus,
                "volume_bonus": self.volume_bonus,
                "ltv_bonus": self.ltv_bonus,
                "spend_efficiency": self.spend_efficiency,
                "constraint_alignment": self.constraint_alignment,
            },
            "penalties": {
                "budget_violation": self.budget_violation,
                "cpa_violation": self.cpa_violation,
                "roas_violation": self.roas_violation,
                "action_magnitude": self.action_magnitude,
                "action_frequency": self.action_frequency,
            },
            "goal": self.goal,
            "normalized": self.normalized,
        }


class RewardComputer:
    """
    Computes multi-objective rewards for campaign optimization
    
    The reward function balances:
    1. Primary objective (ROAS, CPA, conversions, etc.)
    2. Secondary bonuses (efficiency, volume, LTV)
    3. Constraint violations (budget, CPA caps)
    4. Action penalties (smoothness, frequency)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Args:
            config: Reward function configuration
        """
        self.config = config or RewardConfig()
        
        # Running statistics for normalization
        self._reward_mean = 0.0
        self._reward_std = 1.0
        self._reward_count = 0
    
    def compute(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        action: Dict[str, float],
        goal: OptimizationGoal,
        constraints: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        state_constraints: Optional[Dict[str, float]] = None,
    ) -> MultiObjectiveReward:
        """
        Compute reward for a state transition

        Args:
            metrics_before: Campaign metrics before action
            metrics_after: Campaign metrics after action
            action: Action taken (bid_adjustment, budget_adjustment, etc.)
            goal: Primary optimization goal
            constraints: Business constraints (max_cpa, min_roas, etc.)
            context: Optional additional context (time_since_last_action, etc.)
            state_constraints: Optional constraint features from the state vector
                (target_cpa_norm, min_roas_norm, daily_budget_limit_norm).
                When present, a constraint-alignment bonus is added.

        Returns:
            MultiObjectiveReward with detailed breakdown
        """
        reward = MultiObjectiveReward(goal=goal.value)
        
        # 1. Primary objective reward
        reward.primary = self._compute_primary_reward(
            metrics_before, metrics_after, goal
        )
        
        # 2. Efficiency bonus
        reward.efficiency_bonus = self._compute_efficiency_bonus(
            metrics_after
        )
        
        # 3. Volume bonus
        reward.volume_bonus = self._compute_volume_bonus(
            metrics_before, metrics_after
        )
        
        # 4. LTV bonus
        reward.ltv_bonus = self._compute_ltv_bonus(
            metrics_after, context
        )
        
        # 5. Spend efficiency reward (diminishing returns awareness)
        reward.spend_efficiency = self._compute_spend_efficiency_reward(
            metrics_before, metrics_after, constraints
        )

        # 6. Constraint violation penalties
        reward.budget_violation = self._compute_budget_violation(
            metrics_after, constraints
        )
        reward.cpa_violation = self._compute_cpa_violation(
            metrics_after, constraints
        )
        reward.roas_violation = self._compute_roas_violation(
            metrics_after, constraints
        )
        
        # 7. Action penalties
        reward.action_magnitude = self._compute_action_magnitude_penalty(
            action
        )
        reward.action_frequency = self._compute_action_frequency_penalty(
            context
        )

        # 8. Constraint alignment bonus (uses state-embedded constraint features)
        reward.constraint_alignment = self._compute_constraint_alignment(
            metrics_after, constraints, state_constraints
        )

        # Compute total
        reward.total = (
            reward.primary +
            reward.efficiency_bonus +
            reward.volume_bonus +
            reward.ltv_bonus +
            reward.spend_efficiency +
            reward.constraint_alignment +
            reward.budget_violation +
            reward.cpa_violation +
            reward.roas_violation +
            reward.action_magnitude +
            reward.action_frequency
        )
        
        # Normalize if configured
        if self.config.normalize_rewards:
            reward.total = self._normalize_reward(reward.total)
            reward.normalized = True
        
        # Clip if configured
        if self.config.reward_clip:
            reward.total = np.clip(
                reward.total,
                -self.config.reward_clip,
                self.config.reward_clip
            )
        
        # Scale
        reward.total *= self.config.reward_scale
        
        return reward
    
    def _compute_primary_reward(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        goal: OptimizationGoal
    ) -> float:
        """Compute primary objective reward"""
        
        if goal == OptimizationGoal.ROAS:
            # Reward for ROAS improvement
            roas_before = before.get("roas", 1.0)
            roas_after = after.get("roas", 1.0)
            if roas_before > 0:
                improvement = (roas_after - roas_before) / roas_before
            else:
                improvement = roas_after - 1.0
            return improvement * self.config.roas_weight
        
        elif goal == OptimizationGoal.CPA:
            # Reward for CPA reduction (negative improvement is good)
            cpa_before = before.get("cpa", 100.0)
            cpa_after = after.get("cpa", 100.0)
            if cpa_before > 0:
                improvement = (cpa_before - cpa_after) / cpa_before
            else:
                improvement = 0.0
            return improvement * self.config.cpa_weight
        
        elif goal == OptimizationGoal.CONVERSIONS:
            # Reward for conversion increase
            conv_before = before.get("conversions", 0)
            conv_after = after.get("conversions", 0)
            if conv_before > 0:
                improvement = (conv_after - conv_before) / conv_before
            else:
                improvement = conv_after / 10.0  # Normalize new conversions
            return improvement * self.config.conversion_weight
        
        elif goal == OptimizationGoal.CTR:
            # Reward for CTR improvement
            ctr_before = before.get("ctr", 0.01)
            ctr_after = after.get("ctr", 0.01)
            if ctr_before > 0:
                improvement = (ctr_after - ctr_before) / ctr_before
            else:
                improvement = ctr_after * 100  # Normalize
            return improvement * self.config.ctr_weight
        
        elif goal == OptimizationGoal.REVENUE:
            # Reward for revenue increase
            rev_before = before.get("revenue", 0)
            rev_after = after.get("revenue", 0)
            if rev_before > 0:
                improvement = (rev_after - rev_before) / rev_before
            else:
                improvement = rev_after / 1000.0
            return improvement * self.config.roas_weight
        
        elif goal == OptimizationGoal.PROFIT:
            # Reward for profit increase
            profit_before = before.get("revenue", 0) - before.get("spend", 0)
            profit_after = after.get("revenue", 0) - after.get("spend", 0)
            if abs(profit_before) > 0:
                improvement = (profit_after - profit_before) / abs(profit_before)
            else:
                improvement = profit_after / 1000.0
            return improvement * self.config.roas_weight
        
        return 0.0
    
    def _compute_efficiency_bonus(self, metrics: Dict[str, float]) -> float:
        """Bonus for exceeding efficiency thresholds"""
        roas = metrics.get("roas", 0)
        
        if roas > self.config.roas_excellent:
            # Excellent ROAS bonus
            bonus = (roas - self.config.roas_excellent) / self.config.roas_excellent
            return bonus * self.config.efficiency_bonus_weight
        elif roas > self.config.roas_target:
            # Good ROAS bonus (smaller)
            bonus = (roas - self.config.roas_target) / self.config.roas_target * 0.5
            return bonus * self.config.efficiency_bonus_weight
        
        return 0.0
    
    def _compute_volume_bonus(
        self,
        before: Dict[str, float],
        after: Dict[str, float]
    ) -> float:
        """Bonus for conversion volume increase"""
        conv_before = before.get("conversions", 0)
        conv_after = after.get("conversions", 0)
        
        if conv_after > conv_before and conv_after > 0:
            # Log-scaled bonus to prevent explosion
            volume_increase = np.log1p(conv_after - conv_before)
            return volume_increase * self.config.volume_bonus_weight
        
        return 0.0
    
    def _compute_ltv_bonus(
        self,
        metrics: Dict[str, float],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Bonus for acquiring high-LTV customers"""
        if context is None:
            return 0.0
        
        predicted_ltv_delta = context.get("predicted_ltv_delta", 0)
        
        if predicted_ltv_delta > 0:
            return predicted_ltv_delta * self.config.ltv_bonus_weight
        
        return 0.0
    
    def _compute_spend_efficiency_reward(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        constraints: Dict[str, float],
    ) -> float:
        """
        Reward that teaches diminishing returns at different spend levels.

        Two components:
        1. Diminishing returns penalty: quadratic penalty when budget
           utilization exceeds the configured threshold, teaching the agent
           that pushing spend higher yields less incremental value.
        2. Efficiency-at-scale bonus: rewards maintaining high ROAS at higher
           absolute spend levels (harder to do, so more valuable).
        """
        spend_after = metrics_after.get("spend", 0)
        revenue_after = metrics_after.get("revenue", 0)
        max_daily_spend = constraints.get("max_daily_spend", 10000)

        # 1. Diminishing returns penalty above utilization threshold
        utilization = spend_after / max(max_daily_spend, 1.0)
        threshold = self.config.diminishing_return_threshold
        if utilization > threshold:
            overshoot = utilization - threshold
            diminishing_penalty = -(overshoot ** 2) * 0.5
        else:
            diminishing_penalty = 0.0

        # 2. Efficiency-at-scale bonus: ROAS weighted by log-scaled spend level
        if spend_after > 0:
            efficiency = revenue_after / spend_after  # ROAS
            spend_scale = np.log1p(spend_after) / np.log1p(max_daily_spend)
            efficiency_bonus = efficiency * spend_scale * 0.1
        else:
            efficiency_bonus = 0.0

        return (diminishing_penalty + efficiency_bonus) * self.config.spend_efficiency_weight

    def _compute_budget_violation(
        self,
        metrics: Dict[str, float],
        constraints: Dict[str, float]
    ) -> float:
        """Penalty for exceeding budget constraints"""
        spend = metrics.get("spend", 0)
        max_spend = constraints.get("max_daily_spend", float("inf"))
        
        if spend > max_spend:
            # Proportional penalty
            overage = (spend - max_spend) / max_spend
            return self.config.budget_violation_penalty * (1 + overage)
        
        return 0.0
    
    def _compute_cpa_violation(
        self,
        metrics: Dict[str, float],
        constraints: Dict[str, float]
    ) -> float:
        """Penalty for exceeding CPA constraints"""
        cpa = metrics.get("cpa", 0)
        target_cpa = constraints.get("target_cpa", float("inf"))
        
        if cpa > target_cpa * self.config.cpa_target_multiplier:
            # Significant CPA violation
            overage = (cpa - target_cpa) / target_cpa
            return self.config.cpa_violation_penalty * overage
        
        return 0.0
    
    def _compute_roas_violation(
        self,
        metrics: Dict[str, float],
        constraints: Dict[str, float]
    ) -> float:
        """Penalty for falling below ROAS constraints"""
        roas = metrics.get("roas", 0)
        min_roas = constraints.get("min_roas", 0)
        
        if roas < min_roas and min_roas > 0:
            # Proportional penalty
            shortfall = (min_roas - roas) / min_roas
            return self.config.roas_violation_penalty * shortfall
        
        return 0.0
    
    def _compute_action_magnitude_penalty(self, action: Dict[str, float]) -> float:
        """Penalty for large action magnitudes (encourages smooth changes)"""
        bid_adj = abs(action.get("bid_adjustment", 0))
        budget_adj = abs(action.get("budget_adjustment", 0))
        
        total_magnitude = bid_adj + budget_adj
        return -self.config.action_magnitude_penalty * total_magnitude
    
    def _compute_action_frequency_penalty(
        self,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Penalty for frequent actions (encourages stability)"""
        if context is None:
            return 0.0
        
        hours_since_last_action = context.get("hours_since_last_action", 24)
        
        if hours_since_last_action < 4:  # Very frequent
            return -self.config.action_frequency_penalty * 2
        elif hours_since_last_action < 12:  # Frequent
            return -self.config.action_frequency_penalty

        return 0.0

    def _compute_constraint_alignment(
        self,
        metrics_after: Dict[str, float],
        constraints: Dict[str, float],
        state_constraints: Optional[Dict[str, float]],
    ) -> float:
        """
        Bonus for respecting the constraints the policy observed in the state.

        When the state carries normalised constraint features (target_cpa_norm,
        min_roas_norm, daily_budget_limit_norm), we reward the agent for
        producing outcomes that stay within those bounds.

        This teaches the policy to *use* the constraint signals it receives
        in its state vector, closing the loop between constraint-aware state
        representation and constraint-respecting behaviour.
        """
        if state_constraints is None:
            return 0.0

        bonus = 0.0

        # --- CPA alignment ---
        # state stores: target_cpa_norm = log1p(target_cpa) / log1p(1000)
        target_cpa_norm = state_constraints.get("target_cpa_norm", 0.0)
        if target_cpa_norm > 0:
            # Reverse normalisation to get target_cpa
            target_cpa = np.expm1(target_cpa_norm * np.log1p(1000.0))
            actual_cpa = metrics_after.get("cpa", 0.0)
            if actual_cpa > 0 and target_cpa > 0:
                if actual_cpa <= target_cpa:
                    # Under target: small bonus proportional to headroom
                    bonus += 0.05 * (1.0 - actual_cpa / target_cpa)
                else:
                    # Over target: penalty proportional to overshoot
                    bonus -= 0.10 * min((actual_cpa - target_cpa) / target_cpa, 1.0)

        # --- ROAS alignment ---
        # state stores: min_roas_norm = min_roas / 10.0
        min_roas_norm = state_constraints.get("min_roas_norm", 0.0)
        if min_roas_norm > 0:
            min_roas = min_roas_norm * 10.0
            actual_roas = metrics_after.get("roas", 0.0)
            if actual_roas >= min_roas:
                bonus += 0.05 * min((actual_roas - min_roas) / max(min_roas, 0.1), 1.0)
            else:
                bonus -= 0.10 * min((min_roas - actual_roas) / max(min_roas, 0.1), 1.0)

        # --- Budget alignment ---
        # state stores: daily_budget_limit_norm = log1p(budget) / log1p(MAX)
        budget_norm = state_constraints.get("daily_budget_limit_norm", 0.0)
        if budget_norm > 0:
            spend = metrics_after.get("spend", 0.0)
            daily_budget = np.expm1(budget_norm * np.log1p(100_000.0))
            if daily_budget > 0:
                utilisation = spend / daily_budget
                if utilisation <= 1.0:
                    # Within budget — small bonus
                    bonus += 0.02
                else:
                    # Overspend — penalty
                    bonus -= 0.10 * min(utilisation - 1.0, 1.0)

        return bonus

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        # Update running statistics
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_std = np.sqrt(
            (self._reward_std ** 2 * (self._reward_count - 1) + delta * delta2) 
            / self._reward_count
        )
        
        # Normalize
        if self._reward_std > 0:
            return (reward - self._reward_mean) / (self._reward_std + 1e-8)
        return reward
    
    def reset_normalization(self):
        """Reset running statistics"""
        self._reward_mean = 0.0
        self._reward_std = 1.0
        self._reward_count = 0


def compute_simple_reward(
    roas_before: float,
    roas_after: float,
    spend: float
) -> float:
    """
    Simple reward function for basic testing
    
    Returns profit-based reward: revenue - cost
    """
    return (roas_after * spend) - spend


def compute_shaped_reward(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    goal: str = "roas",
    shaping_coefficient: float = 0.1
) -> float:
    """
    Reward shaping to accelerate learning
    
    Adds intermediate rewards for moving toward goal state.
    """
    # Base reward
    if goal == "roas":
        base = metrics_after.get("roas", 0) - metrics_before.get("roas", 0)
    elif goal == "cpa":
        base = metrics_before.get("cpa", 0) - metrics_after.get("cpa", 0)
    elif goal == "conversions":
        base = metrics_after.get("conversions", 0) - metrics_before.get("conversions", 0)
    else:
        base = 0
    
    # Potential-based shaping
    # Φ(s') - γΦ(s) where Φ is potential function
    potential_before = _compute_potential(metrics_before, goal)
    potential_after = _compute_potential(metrics_after, goal)
    shaping = shaping_coefficient * (potential_after - 0.99 * potential_before)
    
    return base + shaping


def _compute_potential(metrics: Dict[str, float], goal: str) -> float:
    """Compute potential function for reward shaping"""
    if goal == "roas":
        return metrics.get("roas", 0) * 10  # Scale
    elif goal == "cpa":
        cpa = metrics.get("cpa", 100)
        return -cpa / 10  # Negative (lower CPA = higher potential)
    elif goal == "conversions":
        return np.log1p(metrics.get("conversions", 0))
    return 0.0
