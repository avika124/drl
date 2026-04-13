"""
Cross-Platform DRL Engine — Primary X-Model Orchestrator

Wraps CrossPlatformOptimizer and elevates the X-Model (XModelAgent) to
be the **primary** cross-platform allocation strategy, with clean
switching, automatic training data collection, lifecycle management,
and side-by-side benchmarking against the classical heuristic allocator.

Strategy Modes:
    DRL_PRIMARY (default):
        X-Model drives allocation; falls back to heuristic when the
        model is not ready or confidence is too low.

    HEURISTIC_PRIMARY:
        Heuristic BudgetAllocator drives allocation (legacy behavior);
        training data is still collected so the X-Model can learn.

    DUAL_BENCHMARK:
        Both allocators run on every cycle.  DRL result is applied,
        heuristic result is logged.  DualRunResult tracks divergence,
        projected ROAS, and win rate for model validation.

Usage::

    engine = CrossPlatformDRLEngine(
        optimizer=cross_platform_optimizer,
        x_model_agent=x_model_agent,
        strategy_config=CrossPlatformStrategyConfig(
            strategy=AllocationStrategy.DRL_PRIMARY,
        ),
    )
    result = await engine.optimize(org_id, campaigns, budget)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AllocationStrategy, CrossPlatformStrategyConfig
from .cross_platform_optimizer import (
    AllocationRecommendation,
    CrossPlatformOptimizer,
    CrossPlatformResult,
    PlatformPortfolio,
)
from .safe_agent import CampaignContext
from .state_action import CampaignState
from .x_model import XModelAgent, build_x_state
from .x_training import XModelTrainer
from .x_training_data import XTrainingDataBuilder
from .xai_narrator import OptimizationNarrator
from .ab_testing import DRLABTestManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DualRunResult
# ---------------------------------------------------------------------------

@dataclass
class DualRunResult:
    """Side-by-side comparison of DRL and heuristic allocations."""

    timestamp: str = ""
    organization_id: str = ""

    # DRL allocation
    drl_allocations: List[AllocationRecommendation] = field(default_factory=list)
    drl_confidence: float = 0.0
    drl_q_value: float = 0.0
    drl_projected_roas: float = 0.0

    # Heuristic allocation
    heuristic_allocations: List[AllocationRecommendation] = field(default_factory=list)
    heuristic_confidence: float = 0.0
    heuristic_projected_roas: float = 0.0

    # Comparison
    allocation_divergence: float = 0.0   # L1 norm of share differences
    strategy_used: str = ""              # Which was actually applied

    # Optional narrative
    narrative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "organization_id": self.organization_id,
            "drl": {
                "allocations": [a.to_dict() for a in self.drl_allocations],
                "confidence": self.drl_confidence,
                "q_value": self.drl_q_value,
                "projected_roas": self.drl_projected_roas,
            },
            "heuristic": {
                "allocations": [a.to_dict() for a in self.heuristic_allocations],
                "confidence": self.heuristic_confidence,
                "projected_roas": self.heuristic_projected_roas,
            },
            "allocation_divergence": self.allocation_divergence,
            "strategy_used": self.strategy_used,
            "narrative": self.narrative,
        }


# ---------------------------------------------------------------------------
# ModelReadinessChecker
# ---------------------------------------------------------------------------

class ModelReadinessChecker:
    """
    Determines whether the X-Model is ready to serve as the primary
    budget allocator.

    Checks:
      1. Agent exists and has been trained (total_steps >= min_training_steps)
      2. Recent allocation confidence >= min_confidence_for_drl
    """

    def __init__(self, config: CrossPlatformStrategyConfig):
        self.min_steps = config.min_training_steps
        self.min_confidence = config.min_confidence_for_drl

    def is_ready(
        self,
        agent: Optional[XModelAgent],
        recent_confidence: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Returns:
            (is_ready, reason_if_not_ready)
        """
        if agent is None:
            return False, "no X-Model agent configured"

        if agent.total_steps < self.min_steps:
            return (
                False,
                f"X-Model has {agent.total_steps} training steps "
                f"(need >= {self.min_steps})",
            )

        if recent_confidence is not None and recent_confidence < self.min_confidence:
            return (
                False,
                f"recent confidence {recent_confidence:.2f} "
                f"below threshold {self.min_confidence:.2f}",
            )

        return True, "ready"


# ---------------------------------------------------------------------------
# CrossPlatformDRLEngine
# ---------------------------------------------------------------------------

class CrossPlatformDRLEngine:
    """
    Top-level orchestrator that makes the X-Model the primary allocation
    strategy and manages the training lifecycle, benchmarking, and
    automatic training data collection.

    Wraps CrossPlatformOptimizer -- does NOT replace it.  Existing users
    of CrossPlatformOptimizer see no behaviour change.
    """

    def __init__(
        self,
        optimizer: CrossPlatformOptimizer,
        x_model_agent: Optional[XModelAgent] = None,
        strategy_config: Optional[CrossPlatformStrategyConfig] = None,
        narrator: Optional[OptimizationNarrator] = None,
        ab_test_manager: Optional[DRLABTestManager] = None,
    ):
        self.optimizer = optimizer
        self.x_model_agent = x_model_agent
        self.config = strategy_config or CrossPlatformStrategyConfig()
        self.narrator = narrator or OptimizationNarrator()
        self.ab_test_manager = ab_test_manager

        # Training infrastructure (reuses existing M3 / M4 classes)
        self._data_builder = XTrainingDataBuilder()
        self._trainer: Optional[XModelTrainer] = None
        if self.x_model_agent is not None:
            self._trainer = XModelTrainer(self.x_model_agent)

        # Readiness checker
        self._readiness = ModelReadinessChecker(self.config)

        # Benchmark history
        self._benchmark_history: List[DualRunResult] = []

        # Snapshot counter for retrain trigger
        self._snapshots_since_retrain: int = 0

        # Previous allocation weights for narrative context
        self._previous_weights: Optional[Dict[str, float]] = None

        mode = self.config.strategy.value
        ready, reason = self._readiness.is_ready(self.x_model_agent)
        logger.info(
            f"CrossPlatformDRLEngine initialised "
            f"(strategy={mode}, x_model_ready={ready}: {reason})"
        )

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    async def optimize(
        self,
        organization_id: str,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]],
        total_budget: float,
        force_rebalance: bool = False,
    ) -> CrossPlatformResult:
        """
        Run cross-platform optimisation using the configured strategy.

        DRL_PRIMARY:
          Check X-Model readiness → if ready, use X-Model allocation;
          otherwise fall back to heuristic.

        HEURISTIC_PRIMARY:
          Delegate to CrossPlatformOptimizer's heuristic path.
          Still collect training data if enabled.

        DUAL_BENCHMARK:
          Run both allocators, apply DRL result, log comparison.
        """
        strategy = self.config.strategy

        if strategy == AllocationStrategy.HEURISTIC_PRIMARY:
            return await self._heuristic_primary(
                organization_id, campaigns, total_budget, force_rebalance
            )
        elif strategy == AllocationStrategy.DRL_PRIMARY:
            return await self._drl_primary(
                organization_id, campaigns, total_budget, force_rebalance
            )
        elif strategy == AllocationStrategy.DUAL_BENCHMARK:
            return await self._dual_benchmark(
                organization_id, campaigns, total_budget, force_rebalance
            )
        else:
            # Unknown strategy — safe fallback
            return await self._heuristic_primary(
                organization_id, campaigns, total_budget, force_rebalance
            )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    async def _heuristic_primary(
        self,
        organization_id: str,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]],
        total_budget: float,
        force_rebalance: bool,
    ) -> CrossPlatformResult:
        """
        Heuristic-primary strategy (legacy behaviour).

        Temporarily removes the X-Model from the optimizer so that
        Phase 5 takes the BudgetAllocator path, then restores it.
        """
        original_agent = self.optimizer.x_model_agent
        self.optimizer.x_model_agent = None
        try:
            result = await self.optimizer.optimize_portfolio(
                organization_id, campaigns, total_budget, force_rebalance,
            )
        finally:
            self.optimizer.x_model_agent = original_agent

        # Post-optimisation: collect training data
        if self.config.auto_collect_training_data and result.allocations:
            self._collect_training_data(result, total_budget)

        return result

    async def _drl_primary(
        self,
        organization_id: str,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]],
        total_budget: float,
        force_rebalance: bool,
    ) -> CrossPlatformResult:
        """
        DRL-primary strategy.

        Falls back to heuristic if the model is not ready or confidence
        is below threshold.
        """
        is_ready, reason = self._readiness.is_ready(
            self.x_model_agent,
            recent_confidence=self._get_last_confidence(),
        )

        if not is_ready:
            logger.info(
                f"X-Model not ready ({reason}), falling back to heuristic"
            )
            return await self._heuristic_primary(
                organization_id, campaigns, total_budget, force_rebalance,
            )

        # Inject X-Model into optimizer so Phase 5 takes the DRL path
        self.optimizer.x_model_agent = self.x_model_agent
        result = await self.optimizer.optimize_portfolio(
            organization_id, campaigns, total_budget, force_rebalance,
        )

        # Post-optimisation
        if self.config.auto_collect_training_data and result.allocations:
            self._collect_training_data(result, total_budget)
            self._check_retrain_trigger()

        if result.allocations:
            self._generate_and_attach_narrative(result)

        return result

    async def _dual_benchmark(
        self,
        organization_id: str,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]],
        total_budget: float,
        force_rebalance: bool,
    ) -> CrossPlatformResult:
        """
        Dual-benchmark strategy.

        1. Build portfolio snapshot once (shared).
        2. Run heuristic allocation (allocation only, no campaign optimisation).
        3. Run DRL allocation (allocation only).
        4. Build DualRunResult and log.
        5. Apply the DRL result through optimize_portfolio.
        6. Collect training data.
        """
        # Step 1: shared portfolio snapshot
        portfolio, marginal_estimates = self.optimizer.build_portfolio_snapshot(
            organization_id, campaigns, total_budget,
        )

        # Step 2: heuristic allocation (allocation only)
        heuristic_allocs = self.optimizer.heuristic_allocate(
            portfolio, marginal_estimates,
        )

        # Step 3: DRL allocation (if ready)
        is_ready, reason = self._readiness.is_ready(self.x_model_agent)
        drl_allocs: List[AllocationRecommendation] = []

        if is_ready:
            drl_allocs = self.optimizer._x_model_allocate(portfolio, total_budget)

            # Step 4: build comparison
            dual_result = self._build_dual_result(
                organization_id,
                drl_allocs,
                heuristic_allocs,
                portfolio,
                marginal_estimates,
                total_budget,
            )
            self._benchmark_history.append(dual_result)
            self._trim_benchmark_history()

            logger.info(
                f"Dual benchmark: divergence={dual_result.allocation_divergence:.3f}, "
                f"DRL projected ROAS={dual_result.drl_projected_roas:.3f}, "
                f"heuristic projected ROAS={dual_result.heuristic_projected_roas:.3f}"
            )

        # Step 5: run full pipeline with the chosen allocator
        if is_ready:
            self.optimizer.x_model_agent = self.x_model_agent
        else:
            self.optimizer.x_model_agent = None

        result = await self.optimizer.optimize_portfolio(
            organization_id, campaigns, total_budget, force_rebalance,
        )

        # Step 6: collect training data
        if self.config.auto_collect_training_data and result.allocations:
            self._collect_training_data(result, total_budget)
            self._check_retrain_trigger()

        return result

    # ------------------------------------------------------------------
    # Training data collection
    # ------------------------------------------------------------------

    def _collect_training_data(
        self,
        result: CrossPlatformResult,
        total_budget: float,
    ) -> None:
        """
        Record a portfolio snapshot from the optimisation result
        for future X-Model training.

        Prefers the full ``portfolio_snapshot`` stored on the result
        (contains all metrics build_x_state needs).  Falls back to
        a minimal reconstruction from allocations if unavailable.
        """
        if not result.allocations:
            return

        allocation_weights = {
            a.platform: a.recommended_share for a in result.allocations
        }

        # Use full portfolio snapshot when available (includes roas, cpa,
        # ctr, cvr, marginal_roas, trends, segment data, etc.)
        if result.portfolio_snapshot:
            portfolio_dict = result.portfolio_snapshot
        else:
            # Fallback: reconstruct minimal portfolio_dict from allocations
            platforms_dict: Dict[str, Dict[str, Any]] = {}
            for alloc in result.allocations:
                plat_data: Dict[str, Any] = {
                    "current_budget_share": alloc.recommended_share,
                    "total_spend": alloc.recommended_budget,
                }
                plat_results = result.platform_campaign_results.get(
                    alloc.platform, []
                )
                if plat_results:
                    total_spend = sum(
                        r.strategic_action.budget_adjustment
                        * alloc.recommended_budget
                        for r in plat_results
                        if r.strategic_action is not None
                    ) or alloc.recommended_budget
                    plat_data["total_spend"] = total_spend
                platforms_dict[alloc.platform] = plat_data

            portfolio_dict = {
                "portfolio_roas": result.portfolio_roas,
                "platforms": platforms_dict,
            }

        self._data_builder.record_snapshot(
            portfolio_dict=portfolio_dict,
            allocation_weights=allocation_weights,
            total_budget=total_budget,
            timestamp=result.timestamp,
        )

        self._snapshots_since_retrain += 1
        self._previous_weights = allocation_weights

    # ------------------------------------------------------------------
    # Retraining lifecycle
    # ------------------------------------------------------------------

    def _check_retrain_trigger(self) -> None:
        """
        Check if enough new snapshots have accumulated to trigger
        a retraining cycle.  If so, build transitions, train, and
        reset the counter.
        """
        if self._trainer is None or self.x_model_agent is None:
            return

        threshold = self.config.retrain_snapshot_threshold
        if self._snapshots_since_retrain < threshold:
            return

        transitions = self._data_builder.build_transitions(
            min_snapshots=self.config.retrain_min_transitions,
        )

        if len(transitions) < self.config.retrain_min_transitions:
            logger.info(
                f"Only {len(transitions)} transitions, need "
                f"{self.config.retrain_min_transitions} for retrain"
            )
            return

        logger.info(
            f"Auto-retrain triggered: {len(transitions)} transitions "
            f"from {self._snapshots_since_retrain} snapshots"
        )

        self._trainer.load_transitions(transitions)
        history = self._trainer.train(
            num_epochs=self.config.retrain_epochs,
            steps_per_epoch=self.config.retrain_steps_per_epoch,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        self._snapshots_since_retrain = 0
        self._data_builder.clear()

        # Log final training metrics
        critic_loss = history.get("x_critic_loss", [0.0])
        actor_loss = history.get("x_actor_loss", [0.0])
        logger.info(
            f"Auto-retrain complete. "
            f"Final critic_loss={critic_loss[-1]:.4f}, "
            f"actor_loss={actor_loss[-1]:.4f}, "
            f"total_steps={self.x_model_agent.total_steps}"
        )

    def force_retrain(
        self,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Manually trigger a retrain cycle regardless of snapshot count.

        Returns:
            Training history dict, or empty dict if insufficient data.
        """
        if self._trainer is None or self.x_model_agent is None:
            logger.warning("Cannot retrain: no X-Model agent or trainer")
            return {}

        transitions = self._data_builder.build_transitions(min_snapshots=2)
        if len(transitions) < 2:
            logger.warning(
                f"Only {len(transitions)} transitions available for retrain"
            )
            return {}

        logger.info(f"Forced retrain with {len(transitions)} transitions")
        self._trainer.load_transitions(transitions)
        history = self._trainer.train(
            num_epochs=epochs or self.config.retrain_epochs,
            steps_per_epoch=steps_per_epoch or self.config.retrain_steps_per_epoch,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        self._snapshots_since_retrain = 0
        self._data_builder.clear()
        return history

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def _build_dual_result(
        self,
        organization_id: str,
        drl_allocs: List[AllocationRecommendation],
        heuristic_allocs: List[AllocationRecommendation],
        portfolio: PlatformPortfolio,
        marginal_estimates: Dict[str, Tuple[float, float]],
        total_budget: float,
    ) -> DualRunResult:
        """Build a DualRunResult comparing the two allocation strategies."""
        # Compute divergence (L1 norm of share differences)
        drl_map = {a.platform: a.recommended_share for a in drl_allocs}
        heur_map = {a.platform: a.recommended_share for a in heuristic_allocs}
        all_platforms = set(drl_map) | set(heur_map)
        divergence = sum(
            abs(drl_map.get(p, 0.0) - heur_map.get(p, 0.0))
            for p in all_platforms
        )

        # Project ROAS for each
        drl_projected = self.optimizer._project_roas(
            portfolio, drl_allocs, marginal_estimates,
        )
        heur_projected = self.optimizer._project_roas(
            portfolio, heuristic_allocs, marginal_estimates,
        )

        # Confidence
        drl_confidence = (
            float(np.mean([a.confidence for a in drl_allocs]))
            if drl_allocs else 0.0
        )
        heur_confidence = (
            float(np.mean([a.confidence for a in heuristic_allocs]))
            if heuristic_allocs else 0.0
        )

        # Q-value from X-Model
        q_value = 0.0
        if self.x_model_agent is not None:
            x_state = build_x_state(portfolio.to_dict(), total_budget)
            x_action = self.x_model_agent.select_allocation(
                x_state, deterministic=True,
            )
            q_value = x_action.q_value

        # Generate benchmark narrative
        narrative = self._build_benchmark_narrative(
            drl_allocs, heuristic_allocs,
            drl_projected, heur_projected,
            divergence,
        )

        return DualRunResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            organization_id=organization_id,
            drl_allocations=drl_allocs,
            drl_confidence=drl_confidence,
            drl_q_value=q_value,
            drl_projected_roas=drl_projected,
            heuristic_allocations=heuristic_allocs,
            heuristic_confidence=heur_confidence,
            heuristic_projected_roas=heur_projected,
            allocation_divergence=divergence,
            strategy_used="drl",
            narrative=narrative,
        )

    @staticmethod
    def _build_benchmark_narrative(
        drl_allocs: List[AllocationRecommendation],
        heuristic_allocs: List[AllocationRecommendation],
        drl_projected: float,
        heur_projected: float,
        divergence: float,
    ) -> str:
        """Build a short narrative summarising the dual-run comparison."""
        drl_map = {a.platform: a.recommended_share for a in drl_allocs}
        heur_map = {a.platform: a.recommended_share for a in heuristic_allocs}

        lines = [
            f"Dual-run benchmark: DRL projected ROAS {drl_projected:.3f} "
            f"vs heuristic {heur_projected:.3f} "
            f"(divergence {divergence:.3f}).",
        ]

        # Find the biggest allocation difference
        diffs = []
        for plat in set(drl_map) | set(heur_map):
            d = drl_map.get(plat, 0.0) - heur_map.get(plat, 0.0)
            if abs(d) >= 0.03:
                diffs.append((plat, d))

        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        for plat, d in diffs[:3]:
            direction = "more" if d > 0 else "less"
            lines.append(
                f"  {plat}: DRL allocates {abs(d):.0%} {direction} "
                f"than heuristic."
            )

        if drl_projected > heur_projected:
            lines.append("DRL projects higher returns for this cycle.")
        elif drl_projected < heur_projected:
            lines.append(
                "Heuristic projects higher returns — DRL may be "
                "exploring or optimising for longer-term reward."
            )
        else:
            lines.append("Both strategies project equal returns.")

        return "\n".join(lines)

    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        """Return benchmark history as list of dicts."""
        return [r.to_dict() for r in self._benchmark_history]

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics from benchmark history.

        Returns:
            Dict with win rate, avg projected ROAS, avg divergence.
        """
        if not self._benchmark_history:
            return {"num_benchmarks": 0}

        drl_wins = sum(
            1 for r in self._benchmark_history
            if r.drl_projected_roas > r.heuristic_projected_roas
        )

        return {
            "num_benchmarks": len(self._benchmark_history),
            "drl_win_rate": drl_wins / len(self._benchmark_history),
            "avg_drl_projected_roas": float(np.mean([
                r.drl_projected_roas for r in self._benchmark_history
            ])),
            "avg_heuristic_projected_roas": float(np.mean([
                r.heuristic_projected_roas for r in self._benchmark_history
            ])),
            "avg_divergence": float(np.mean([
                r.allocation_divergence for r in self._benchmark_history
            ])),
            "avg_drl_confidence": float(np.mean([
                r.drl_confidence for r in self._benchmark_history
            ])),
            "avg_drl_q_value": float(np.mean([
                r.drl_q_value for r in self._benchmark_history
            ])),
        }

    def _trim_benchmark_history(self) -> None:
        max_size = self.config.benchmark_history_size
        if len(self._benchmark_history) > max_size:
            self._benchmark_history = self._benchmark_history[-max_size:]

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------

    def _generate_and_attach_narrative(
        self, result: CrossPlatformResult,
    ) -> None:
        """
        Generate a portfolio narrative via the xAI narrator and attach
        it to the result.
        """
        allocation_weights = {
            a.platform: a.recommended_share for a in result.allocations
        }

        # Build x_state_dict for the narrator from full portfolio snapshot
        # when available, otherwise fall back to minimal reconstruction
        if result.portfolio_snapshot:
            snap = result.portfolio_snapshot
            platforms_data = snap.get("platforms", {})
            platform_features: Dict[str, Dict[str, float]] = {}
            for plat, pm_data in platforms_data.items():
                platform_features[plat] = {
                    "roas": pm_data.get("roas", 0.0) / 5.0,
                    "marginal_roas": pm_data.get("marginal_roas", 0.0) / 5.0,
                    "roas_trend_7d": pm_data.get("roas_trend_7d", 0.0),
                    "spend_share": pm_data.get("current_budget_share", 0.0),
                }
        else:
            platform_features = {}

        hhi = sum(w ** 2 for w in allocation_weights.values())
        n_active = sum(1 for w in allocation_weights.values() if w >= 0.05)
        x_state_dict: Dict[str, Any] = {
            "portfolio_roas": result.portfolio_roas / 5.0,
            "budget_utilization": (
                result.total_spend / max(result.total_budget, 1.0)
            ),
            "portfolio_hhi": hhi,
            "active_platform_ratio": n_active / 5.0,
            "platform_features": platform_features,
        }

        narrative = self.narrator.generate_portfolio_narrative(
            x_state_dict=x_state_dict,
            allocation_weights=allocation_weights,
            previous_weights=self._previous_weights,
            confidence=result.allocation_confidence,
        )

        # Store the narrative on the result so callers can access it
        result.portfolio_narrative = narrative

        logger.info(
            f"Portfolio narrative: {narrative.allocation_decision}"
        )

    # ------------------------------------------------------------------
    # A/B test integration
    # ------------------------------------------------------------------

    def create_benchmark_experiment(
        self,
        name: Optional[str] = None,
        treatment_ratio: float = 0.5,
        min_duration_days: int = 21,
    ) -> Optional[Any]:
        """
        Create a formal A/B experiment comparing DRL vs heuristic
        using the existing DRLABTestManager.

        Returns:
            ExperimentConfig or None if no A/B test manager.
        """
        if self.ab_test_manager is None:
            logger.warning("No A/B test manager configured")
            return None

        return self.ab_test_manager.create_portfolio_experiment(
            name=name,
            treatment_ratio=treatment_ratio,
            min_duration_days=min_duration_days,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return engine state for debugging."""
        is_ready, reason = self._readiness.is_ready(
            self.x_model_agent,
            recent_confidence=self._get_last_confidence(),
        )

        diag: Dict[str, Any] = {
            "strategy": self.config.strategy.value,
            "x_model_ready": is_ready,
            "x_model_ready_reason": reason,
            "snapshots_since_retrain": self._snapshots_since_retrain,
            "training_data_snapshots": self._data_builder.num_snapshots,
            "benchmark_count": len(self._benchmark_history),
            "benchmark_summary": self.get_benchmark_summary(),
            "optimizer_diagnostics": self.optimizer.get_diagnostics(),
        }

        if self.x_model_agent is not None:
            diag["x_model"] = self.x_model_agent.get_diagnostics()

        return diag

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_last_confidence(self) -> Optional[float]:
        """Get the confidence from the most recent DRL allocation."""
        if self._benchmark_history:
            return self._benchmark_history[-1].drl_confidence
        return None
