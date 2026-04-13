"""
DRL Optimization Engine — singleton that owns the SAC agent lifecycle.

Provides:
- DRLOptimizationEngine: load checkpoint, run inference, record outcomes,
  hot-reload model versions, expose health metrics.

Every inference request is logged with latency and model version.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class _PendingAction:
    """Stored when get_optimization returns; used by record_outcome for real transitions."""
    campaign_id: str
    state_before: Any  # np.ndarray
    continuous_action: Any  # np.ndarray
    discrete_action: Any  # np.ndarray
    metrics_before: Dict[str, float]
    timestamp: datetime

_engine_instance: Optional["DRLOptimizationEngine"] = None


def get_engine() -> "DRLOptimizationEngine":
    """Return the module-level singleton, creating it on first call."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DRLOptimizationEngine()
    return _engine_instance


class DRLOptimizationEngine:
    """Singleton wrapper around the full DRL inference + learning stack.

    Lazy-imports torch / DRL modules so the backend can start without
    them on the import path (e.g. during migration or healthcheck).
    """

    def __init__(
        self,
        state_dim: int = int(os.getenv("DRL_STATE_DIM", "42")),
        model_dir: str = os.getenv("DRL_MODEL_DIR", "models/drl"),
        device: str = os.getenv("DRL_DEVICE", "cpu"),
        min_confidence: float = float(os.getenv("DRL_MIN_CONFIDENCE", "0.7")),
        auto_apply_threshold: float = float(os.getenv("DRL_AUTO_APPLY_THRESHOLD", "0.85")),
        inference_timeout_ms: int = int(os.getenv("DRL_INFERENCE_TIMEOUT_MS", "100")),
    ):
        self.state_dim = state_dim
        self.model_dir = model_dir
        self.device = device
        self.min_confidence = min_confidence
        self.auto_apply_threshold = auto_apply_threshold
        self.inference_timeout_ms = inference_timeout_ms

        self.initialized = False
        self.model_version: Optional[str] = None
        self.checkpoint_date: Optional[str] = None
        self._agent = None
        self._safe_agent = None
        self._hybrid_optimizer = None
        self._narrator = None
        self._reward_computer = None
        self._continuous_engine = None
        self._replay_buffer = None
        self._outcome_tracker_reward = None  # outcome_tracker.RewardComputer for transitions
        self._pending_actions: Dict[str, _PendingAction] = {}
        self._max_pending_actions = 10_000

        self._latencies: deque = deque(maxlen=1000)

        logger.info(
            "DRLOptimizationEngine created (state_dim=%d, model_dir=%s, device=%s)",
            state_dim, model_dir, device,
        )

    # ── Initialization ────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Load checkpoint, build agent stack.  Returns True on success."""
        if self.initialized:
            return True

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._sync_initialize,
            )
        except Exception as exc:
            logger.error("DRL initialization failed: %s", exc, exc_info=True)
            return False

    def _sync_initialize(self) -> bool:
        import torch
        from drl.sac_agent import SACAgent, load_sac_for_inference
        from drl.config import DRLConfig, TrainingConfig, GuardrailConfig
        from drl.safe_agent import SafeDRLAgent
        from drl.hybrid_optimizer import HybridDRLLLMOptimizer
        from drl.xai_narrator import OptimizationNarrator
        from drl.reward_functions import RewardComputer
        from drl.continuous_learning import ContinuousLearningEngine
        from drl.replay_buffer import create_replay_buffer

        ckpt_path = Path(self.model_dir)
        if not (ckpt_path / "agent.pt").exists():
            alt = Path("checkpoints/final_model/agent.pt")
            if alt.exists():
                ckpt_path = alt.parent
            else:
                raise FileNotFoundError(
                    f"No SAC checkpoint at {self.model_dir}/agent.pt "
                    f"or checkpoints/final_model/agent.pt"
                )

        agent, _ = load_sac_for_inference(
            model_dir=str(ckpt_path),
            device=self.device,
            state_dim=self.state_dim,
        )
        self._agent = agent

        drl_cfg = DRLConfig(state_dim=self.state_dim)
        guardrail_cfg = GuardrailConfig()
        self._safe_agent = SafeDRLAgent(
            agent=agent,
            guardrails=guardrail_cfg,
        )
        self._hybrid_optimizer = HybridDRLLLMOptimizer(
            drl_agent=self._safe_agent,
            llm_client=None,
            enable_tactical=True,
        )
        from drl.cross_platform_optimizer import CrossPlatformOptimizer
        self._xp_optimizer = CrossPlatformOptimizer(hybrid_optimizer=self._hybrid_optimizer)
        self._narrator = OptimizationNarrator()
        self._reward_computer = RewardComputer()
        from drl.reward_functions import RewardComputer
        self._outcome_tracker_reward = RewardComputer()
        self._replay_buffer = create_replay_buffer(capacity=100_000, use_prioritized=True)

        learning_mode = os.getenv("CONTINUOUS_LEARNING_MODE", "hybrid")
        threshold = int(os.getenv("CONTINUOUS_LEARNING_THRESHOLD", "1000"))
        self._continuous_engine = ContinuousLearningEngine(
            agent=agent,
            replay_buffer=self._replay_buffer,
            training_config=TrainingConfig(),
            learning_mode=learning_mode,
            update_frequency=threshold,
        )

        ckpt_stat = (ckpt_path / "agent.pt").stat()
        self.checkpoint_date = datetime.fromtimestamp(
            ckpt_stat.st_mtime, tz=timezone.utc,
        ).isoformat()
        self.model_version = ckpt_path.name
        self.initialized = True

        logger.info(
            "DRL engine initialized: version=%s, checkpoint=%s, device=%s",
            self.model_version, self.checkpoint_date, self.device,
        )
        return True

    # ── Inference ─────────────────────────────────────────────────

    async def get_optimization(
        self,
        campaign_state,
        raw_context=None,
        campaign_id: str = "",
    ) -> Dict[str, Any]:
        """Run SAC inference through SafeDRLAgent (guardrails), wrap with narrative.

        Args:
            campaign_state: a ``drl.state_action.CampaignState`` instance
                            (39-dim state vector).
            raw_context: optional CampaignStateSchema or dict with daily_budget,
                         spend, roas, cpa, cpc for building CampaignContext.
            campaign_id: campaign identifier for context.

        Returns:
            dict with keys: action_id, bid_adjustment, budget_adjustment,
            audience_action, creative_action, confidence, reasoning,
            narrative, requires_review, auto_apply, model_version,
            latency_ms.

        Raises:
            RuntimeError if engine is not initialized.
        """
        if not self.initialized:
            ok = await self.initialize()
            if not ok:
                raise RuntimeError("DRL engine not initialized and auto-init failed")

        t0 = time.perf_counter()

        from drl.state_action import (
            ActionSpace, AudienceAction, CreativeAction, DRLDirective,
        )
        from drl.safe_agent import CampaignContext

        # Build CampaignContext for SafeDRLAgent guardrails
        if raw_context is not None:
            schema = raw_context
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump()
            elif not isinstance(schema, dict):
                schema = {}
            current_budget = float(schema.get("daily_budget") or schema.get("total_budget") or 100.0)
            current_bid = float(schema.get("cpc") or 1.0) if schema.get("cpc") else 1.0
            current_roas = float(schema.get("roas") or 2.0)
            current_cpa = float(schema.get("cpa") or 50.0)
            total_spend = float(schema.get("spend") or 0.0)
            metrics_before = {
                "roas": current_roas,
                "cpa": current_cpa,
                "ctr": float(schema.get("ctr") or 0.0),
                "conversions": int(schema.get("conversions") or 0),
                "spend": total_spend,
            }
        else:
            current_budget = 100.0
            current_bid = 1.0
            current_roas = 2.0
            current_cpa = 50.0
            total_spend = 0.0
            metrics_before = {"roas": current_roas, "cpa": current_cpa, "ctr": 0.0, "conversions": 0, "spend": total_spend}

        context = CampaignContext(
            campaign_id=campaign_id or (getattr(campaign_state, "campaign_id", None) or "unknown"),
            current_bid=current_bid,
            current_budget=current_budget,
            last_action_at=None,
            actions_today=0,
            current_roas=current_roas,
            current_cpa=current_cpa,
            target_cpa=current_cpa * 1.2,
            min_roas=2.0,
            total_spend=total_spend,
        )

        campaign_info = raw_context.model_dump() if hasattr(raw_context, "model_dump") else (raw_context or {})
        if not isinstance(campaign_info, dict):
            campaign_info = {}

        # Use HybridDRLLMOptimizer (guardrails + state-aware overrides + directive + narrative)
        opt_result = await self._hybrid_optimizer.optimize(
            state=campaign_state,
            context=context,
            campaign_info=campaign_info,
            generate_tactical=False,
        )
        action = opt_result.action

        requires_review = opt_result.requires_review or (action.confidence < self.min_confidence)
        auto_apply = action.confidence >= self.auto_apply_threshold

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(elapsed_ms)

        action_id = str(uuid4())

        # Store for record_outcome (real transitions)
        import numpy as np
        state_arr = campaign_state.to_tensor().numpy()
        cont_action = np.array([float(action.bid_adjustment), float(action.budget_adjustment)], dtype=np.float32)
        disc_action = np.array([int(action.audience_action), int(action.creative_action)], dtype=np.float32)
        self._pending_actions[action_id] = _PendingAction(
            campaign_id=context.campaign_id,
            state_before=state_arr,
            continuous_action=cont_action,
            discrete_action=disc_action,
            metrics_before=metrics_before,
            timestamp=datetime.now(timezone.utc),
        )
        while len(self._pending_actions) > self._max_pending_actions:
            oldest = min(self._pending_actions.items(), key=lambda x: x[1].timestamp)
            del self._pending_actions[oldest[0]]

        logger.info(
            "DRL inference: action_id=%s confidence=%.3f latency=%.1fms version=%s",
            action_id, opt_result.combined_confidence, elapsed_ms, self.model_version,
        )

        narrative = opt_result.narrative or {}
        narrative_text = narrative.get("full_text", "") if isinstance(narrative, dict) else str(narrative)
        reasoning = narrative.get("reasoning", []) if isinstance(narrative, dict) else []

        return {
            "action_id": action_id,
            "bid_adjustment": round(action.bid_adjustment, 4),
            "budget_adjustment": round(action.budget_adjustment, 4),
            "audience_action": AudienceAction(action.audience_action).name.lower(),
            "creative_action": CreativeAction(action.creative_action).name.lower(),
            "confidence": round(opt_result.combined_confidence, 4),
            "reasoning": reasoning if isinstance(reasoning, list) else [str(reasoning)],
            "narrative": narrative_text,
            "requires_review": requires_review,
            "auto_apply": auto_apply,
            "model_version": self.model_version,
            "latency_ms": round(elapsed_ms, 2),
        }

    # ── Outcome recording ─────────────────────────────────────────

    async def record_outcome(
        self,
        campaign_id: str,
        action_id: str,
        outcome: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute reward from outcome and feed to continuous learning.

        Uses stored (state, action) from get_optimization for real transitions.
        Checks for auto-rollback: ROAS drop >20% or CPA spike >30%.

        Args:
            campaign_id: campaign identifier
            action_id: UUID of the original DRL action (returned by get_optimization)
            outcome: dict with conversions, revenue, spend, roas, ctr

        Returns:
            dict with reward, model_version, retrain_triggered, rollback_required, rollback_action
        """
        if not self.initialized:
            raise RuntimeError("DRL engine not initialized")

        import numpy as np
        from drl.state_action import CampaignState
        from drl.replay_buffer import Transition

        metrics_after = {
            "roas": outcome.get("roas", 0.0),
            "cpa": outcome.get("spend", 0.0) / max(outcome.get("conversions", 1), 1),
            "ctr": outcome.get("ctr", 0.0),
            "conversions": outcome.get("conversions", 0),
            "spend": outcome.get("spend", 0.0),
        }

        # Look up stored action for real transition
        pending = self._pending_actions.pop(action_id, None)
        rollback_required = False
        rollback_action = None

        if pending is not None:
            # Real transition: state, action, reward, next_state
            reward = self._outcome_tracker_reward.compute(
                pending.metrics_before,
                metrics_after,
                context={
                    "bid_adjustment": float(pending.continuous_action[0]),
                    "budget_adjustment": float(pending.continuous_action[1]),
                },
            )
            # Build next_state from outcome (simplified: use metrics to approximate)
            next_state = self._metrics_to_state_vector(metrics_after)
            transition = Transition(
                state=pending.state_before,
                continuous_action=pending.continuous_action,
                discrete_action=pending.discrete_action,
                reward=reward,
                next_state=next_state,
                done=False,
                campaign_id=pending.campaign_id,
                timestamp=pending.timestamp.isoformat(),
            )
            self._continuous_engine.add_transition(transition)

            # Auto-rollback check: ROAS drop >20% or CPA spike >30%
            roas_before = pending.metrics_before.get("roas", 1.0)
            cpa_before = pending.metrics_before.get("cpa", 50.0)
            roas_after = metrics_after.get("roas", 0.0)
            cpa_after = metrics_after.get("cpa", 0.0)
            if roas_before > 0 and roas_after < roas_before * 0.80:
                rollback_required = True
                rollback_action = {
                    "bid_adjustment": -float(pending.continuous_action[0]),
                    "budget_adjustment": -float(pending.continuous_action[1]),
                    "reason": f"ROAS dropped {100*(1 - roas_after/roas_before):.0f}%",
                }
            elif cpa_before > 0 and cpa_after > cpa_before * 1.30:
                rollback_required = True
                rollback_action = {
                    "bid_adjustment": -float(pending.continuous_action[0]),
                    "budget_adjustment": -float(pending.continuous_action[1]),
                    "reason": f"CPA spiked {100*(cpa_after/cpa_before - 1):.0f}%",
                }
        else:
            # Fallback: no stored action (e.g. old client or expired), use reward-only
            reward = self._outcome_tracker_reward.compute({}, metrics_after, {})
            dummy = CampaignState()
            dummy_state = dummy.to_tensor().numpy()
            transition = Transition(
                state=dummy_state,
                continuous_action=np.array([0.0, 0.0]),
                discrete_action=np.array([0, 0]),
                reward=reward,
                next_state=dummy_state,
                done=False,
            )
            self._continuous_engine.add_transition(transition)

        retrain_triggered = False
        threshold = int(os.getenv("CONTINUOUS_LEARNING_THRESHOLD", "1000"))
        if len(self._replay_buffer) >= threshold and len(self._replay_buffer) % threshold == 0:
            retrain_triggered = True
            logger.info("Continuous learning threshold reached — triggering batch update")
            self._continuous_engine.run_batch_update(n_steps=10)

        logger.info(
            "Outcome recorded: campaign=%s action=%s reward=%.4f buffer_size=%d rollback=%s",
            campaign_id, action_id, reward, len(self._replay_buffer), rollback_required,
        )

        result = {
            "reward": round(reward, 4),
            "model_version": self.model_version,
            "retrain_triggered": retrain_triggered,
            "buffer_size": len(self._replay_buffer),
        }
        if rollback_required and rollback_action:
            result["rollback_required"] = True
            result["rollback_action"] = rollback_action
        return result

    def _metrics_to_state_vector(self, metrics: Dict[str, Any]) -> "np.ndarray":
        """Build a 39-dim state vector from outcome metrics (simplified approximation)."""
        from drl.state_action import CampaignState
        roas = min(metrics.get("roas", 2.0) / 10.0, 1.0)
        cpa = 1.0 / (1.0 + metrics.get("cpa", 50.0) / 100.0)
        ctr = min(metrics.get("ctr", 0.02), 1.0)
        s = CampaignState(roas=roas, cpa=cpa, ctr=ctr)
        return s.to_tensor().numpy()

    # ── Model management ──────────────────────────────────────────

    def get_model_version(self) -> Dict[str, Any]:
        """Return current model metadata."""
        p95 = 0.0
        if self._latencies:
            sorted_l = sorted(self._latencies)
            idx = int(len(sorted_l) * 0.95)
            p95 = sorted_l[min(idx, len(sorted_l) - 1)]

        return {
            "model_version": self.model_version,
            "checkpoint_date": self.checkpoint_date,
            "state_dim": self.state_dim,
            "initialized": self.initialized,
            "device": self.device,
            "inference_count": len(self._latencies),
            "inference_latency_p95_ms": round(p95, 2),
        }

    async def reload_checkpoint(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Hot-swap to a different checkpoint without process restart.

        Args:
            version: subdirectory name under model_dir. If None, reloads
                     the current checkpoint (useful after in-place retrain).
        """
        old_version = self.model_version
        if version:
            self.model_dir = str(Path(self.model_dir).parent / version)

        self.initialized = False
        ok = await self.initialize()

        if not ok:
            raise RuntimeError(f"Failed to reload checkpoint: {self.model_dir}")

        logger.info("Checkpoint reloaded: %s → %s", old_version, self.model_version)
        return {
            "reloaded": True,
            "old_version": old_version,
            "new_version": self.model_version,
        }

    def __repr__(self) -> str:
        return (
            f"DRLOptimizationEngine(initialized={self.initialized}, "
            f"version={self.model_version}, device={self.device})"
        )
