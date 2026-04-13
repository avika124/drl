"""
M3 — X-Training Data Generation

Transforms per-platform performance history into portfolio-level
(X-State, X-Action, reward, X-NextState) transitions for training
the X-Model (M4).

The core idea:
  - Each *rebalance snapshot* is a timestep in the X-Model MDP.
  - X-State is the portfolio observation at time t.
  - X-Action is the actual allocation that was applied at time t
    (the "behavior policy" — either from heuristic allocator or previous
    X-Model).
  - Reward is the portfolio-level performance improvement observed at t+1.
  - X-NextState is the portfolio observation at time t+1.

Data sources:
  1. ``CrossPlatformOptimizer._allocation_history`` — records of past
     allocation decisions and their portfolio context.
  2. ``PlatformPerformanceTracker._daily_history`` — daily per-platform
     metrics that can be aggregated into portfolio snapshots.

This module does NOT call M1 (P-Training) at runtime, following the
design guidance: "But ideally this should be avoided."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .x_model import (
    XModelState,
    X_PLATFORMS,
    NUM_PLATFORMS,
    build_x_state,
)

logger = logging.getLogger(__name__)


@dataclass
class XTransition:
    """A single transition in the X-Model MDP."""
    state: np.ndarray        # (X_STATE_DIM,)
    action: np.ndarray       # (NUM_PLATFORMS,)  allocation weights
    reward: float
    next_state: np.ndarray   # (X_STATE_DIM,)
    done: bool = False


class XTrainingDataBuilder:
    """
    Builds X-Model training transitions from allocation history.

    Usage::

        builder = XTrainingDataBuilder()

        # Record portfolio snapshots over time
        builder.record_snapshot(portfolio_dict, allocations, total_budget)
        # ... later ...
        builder.record_snapshot(portfolio_dict_t1, allocations_t1, total_budget)

        # Generate transitions
        transitions = builder.build_transitions()
    """

    def __init__(
        self,
        roas_weight: float = 0.50,
        volume_weight: float = 0.25,
        efficiency_weight: float = 0.15,
        stability_weight: float = 0.10,
    ):
        """
        Args:
            roas_weight: Weight for ROAS improvement in the X-reward.
            volume_weight: Weight for total conversion growth.
            efficiency_weight: Weight for spend efficiency improvement.
            stability_weight: Penalty weight for allocation churn.
        """
        self.roas_weight = roas_weight
        self.volume_weight = volume_weight
        self.efficiency_weight = efficiency_weight
        self.stability_weight = stability_weight

        # Ordered list of (portfolio_dict, allocation_weights, total_budget, timestamp)
        self._snapshots: List[Tuple[Dict[str, Any], Dict[str, float], float, str]] = []

    def record_snapshot(
        self,
        portfolio_dict: Dict[str, Any],
        allocation_weights: Dict[str, float],
        total_budget: float,
        timestamp: str = "",
    ) -> None:
        """
        Record a portfolio snapshot at a point in time.

        Args:
            portfolio_dict: Output of PlatformPortfolio.to_dict() or
                equivalent dict with keys ``platforms``, ``portfolio_roas``.
            allocation_weights: Budget shares applied at this timestep,
                keyed by platform name.
            total_budget: Total portfolio budget.
            timestamp: ISO timestamp for ordering.
        """
        self._snapshots.append((portfolio_dict, allocation_weights, total_budget, timestamp))

    def record_from_allocation_history(
        self,
        allocation_history: List[Dict[str, Any]],
        portfolio_snapshots: List[Dict[str, Any]],
    ) -> None:
        """
        Bulk-load from CrossPlatformOptimizer's allocation history and
        matching portfolio snapshots.

        Each entry in *allocation_history* should have:
          - allocations: list of {platform, recommended_share, ...}
          - portfolio_roas, total_budget, timestamp

        Each entry in *portfolio_snapshots* should be a PlatformPortfolio.to_dict().
        """
        for alloc_entry, port_snap in zip(allocation_history, portfolio_snapshots):
            weights = {}
            for a in alloc_entry.get("allocations", []):
                weights[a["platform"]] = a.get("recommended_share", 0.0)

            self.record_snapshot(
                portfolio_dict=port_snap,
                allocation_weights=weights,
                total_budget=alloc_entry.get("total_budget", 0.0),
                timestamp=alloc_entry.get("timestamp", ""),
            )

    def build_transitions(
        self,
        min_snapshots: int = 2,
    ) -> List[XTransition]:
        """
        Build (state, action, reward, next_state) transitions from
        consecutive snapshots.

        Returns:
            List of XTransition objects.
        """
        if len(self._snapshots) < min_snapshots:
            logger.warning(
                f"Only {len(self._snapshots)} snapshots, need >= {min_snapshots}"
            )
            return []

        # Sort by timestamp
        sorted_snaps = sorted(self._snapshots, key=lambda x: x[3])

        transitions: List[XTransition] = []

        for i in range(len(sorted_snaps) - 1):
            port_t, alloc_t, budget_t, ts_t = sorted_snaps[i]
            port_t1, alloc_t1, budget_t1, ts_t1 = sorted_snaps[i + 1]

            # Build X-States
            x_state_t = build_x_state(port_t, budget_t)
            x_state_t1 = build_x_state(port_t1, budget_t1)

            # Build action vector (allocation weights at time t)
            action = np.array(
                [alloc_t.get(p, 0.0) for p in X_PLATFORMS],
                dtype=np.float32,
            )
            # Normalize to valid allocation
            action_sum = action.sum()
            if action_sum > 0:
                action = action / action_sum
            else:
                action = np.ones(NUM_PLATFORMS, dtype=np.float32) / NUM_PLATFORMS

            # Compute reward
            reward = self._compute_reward(port_t, port_t1, alloc_t, alloc_t1)

            transitions.append(XTransition(
                state=x_state_t.to_tensor().numpy(),
                action=action,
                reward=reward,
                next_state=x_state_t1.to_tensor().numpy(),
                done=(i == len(sorted_snaps) - 2),
            ))

        logger.info(f"Built {len(transitions)} X-Model transitions from {len(sorted_snaps)} snapshots")
        return transitions

    def _compute_reward(
        self,
        port_t: Dict[str, Any],
        port_t1: Dict[str, Any],
        alloc_t: Dict[str, float],
        alloc_t1: Dict[str, float],
    ) -> float:
        """
        Compute portfolio-level reward for the X-Model.

        Components:
          1. ROAS improvement (portfolio-level)
          2. Conversion volume growth
          3. Spend efficiency improvement
          4. Allocation stability (penalize excessive churn)
        """
        # 1. ROAS improvement
        roas_t = port_t.get("portfolio_roas", 0.0)
        roas_t1 = port_t1.get("portfolio_roas", 0.0)
        roas_delta = (roas_t1 - roas_t) / max(roas_t, 0.01)
        roas_reward = np.clip(roas_delta, -2.0, 2.0)

        # 2. Conversion volume growth
        platforms_t = port_t.get("platforms", {})
        platforms_t1 = port_t1.get("platforms", {})
        conv_t = sum(platforms_t.get(p, {}).get("total_conversions", 0) for p in X_PLATFORMS)
        conv_t1 = sum(platforms_t1.get(p, {}).get("total_conversions", 0) for p in X_PLATFORMS)
        conv_delta = (conv_t1 - conv_t) / max(conv_t, 1)
        volume_reward = np.clip(conv_delta, -1.0, 1.0)

        # 3. Spend efficiency (revenue per dollar)
        spend_t = sum(platforms_t.get(p, {}).get("total_spend", 0) for p in X_PLATFORMS)
        rev_t = sum(platforms_t.get(p, {}).get("total_revenue", 0) for p in X_PLATFORMS)
        spend_t1 = sum(platforms_t1.get(p, {}).get("total_spend", 0) for p in X_PLATFORMS)
        rev_t1 = sum(platforms_t1.get(p, {}).get("total_revenue", 0) for p in X_PLATFORMS)
        eff_t = rev_t / max(spend_t, 1.0)
        eff_t1 = rev_t1 / max(spend_t1, 1.0)
        eff_delta = (eff_t1 - eff_t) / max(eff_t, 0.01)
        efficiency_reward = np.clip(eff_delta, -1.0, 1.0)

        # 4. Allocation stability (penalize large changes)
        weights_t = np.array([alloc_t.get(p, 0.0) for p in X_PLATFORMS])
        weights_t1 = np.array([alloc_t1.get(p, 0.0) for p in X_PLATFORMS])
        churn = float(np.sum(np.abs(weights_t1 - weights_t)))
        stability_penalty = -churn  # Higher churn → larger penalty

        # Weighted sum
        reward = (
            self.roas_weight * roas_reward
            + self.volume_weight * volume_reward
            + self.efficiency_weight * efficiency_reward
            + self.stability_weight * stability_penalty
        )

        return float(np.clip(reward, -5.0, 5.0))

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()
