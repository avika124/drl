"""
Continuous Learning Engine

Implements online learning pipeline for DRL agent:
- Real-time outcome tracking
- Online experience collection
- Incremental model updates
- Performance monitoring and drift detection
- Automatic model retraining triggers
"""

import asyncio
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
import logging
import json
from pathlib import Path
from enum import Enum

from .config import DRLConfig, TrainingConfig, RewardConfig
from .sac_agent import SACAgent
from .safe_agent import SafeDRLAgent, CampaignContext
from .replay_buffer import PrioritizedReplayBuffer, Transition
from .reward_functions import RewardComputer, MultiObjectiveReward
from .state_action import CampaignState, ActionSpace
from .forecast_feedback import ForecastFeedbackLoop

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning mode for continuous learning"""
    ONLINE = "online"           # Update on every transition
    BATCH = "batch"             # Batch updates at intervals
    TRIGGERED = "triggered"     # Update when performance drops
    HYBRID = "hybrid"           # Combination of modes


@dataclass
class OutcomeRecord:
    """Record of action outcome for learning"""
    campaign_id: str
    action_timestamp: str
    
    # State at action time
    state_before: np.ndarray
    action_continuous: np.ndarray
    action_discrete: np.ndarray
    
    # Observed outcome
    state_after: Optional[np.ndarray] = None
    reward: Optional[float] = None
    reward_breakdown: Optional[Dict[str, float]] = None
    
    # Timing
    action_applied_at: Optional[datetime] = None
    outcome_observed_at: Optional[datetime] = None
    observation_delay_hours: float = 0.0
    
    # Status
    is_complete: bool = False
    is_terminal: bool = False
    
    def to_transition(self) -> Optional[Transition]:
        """Convert to Transition if complete"""
        if not self.is_complete or self.state_after is None or self.reward is None:
            return None
        
        return Transition(
            state=self.state_before,
            continuous_action=self.action_continuous,
            discrete_action=self.action_discrete,
            reward=self.reward,
            next_state=self.state_after,
            done=self.is_terminal,
            campaign_id=self.campaign_id,
            timestamp=self.action_timestamp,
        )


class OutcomeTracker:
    """
    Tracks action outcomes for delayed reward learning
    
    Advertising has delayed feedback:
    - Impressions: Immediate
    - Clicks: Minutes to hours
    - Conversions: Hours to days
    - Revenue: Days to weeks
    
    This tracker manages the asynchronous outcome collection.
    """
    
    def __init__(
        self,
        observation_window_hours: float = 24.0,
        reward_computer: Optional[RewardComputer] = None,
        max_pending: int = 10000
    ):
        """
        Args:
            observation_window_hours: Hours to wait for outcome
            reward_computer: Reward function computer
            max_pending: Maximum pending observations
        """
        self.observation_window = timedelta(hours=observation_window_hours)
        self.reward_computer = reward_computer or RewardComputer()
        self.max_pending = max_pending
        
        # Pending outcomes by campaign_id
        self.pending: Dict[str, OutcomeRecord] = {}
        
        # Completed transitions ready for learning
        self.completed_queue: deque[Transition] = deque(maxlen=10000)
        
        # Statistics
        self.stats = {
            "actions_tracked": 0,
            "outcomes_received": 0,
            "transitions_completed": 0,
            "transitions_expired": 0,
        }
    
    def register_action(
        self,
        campaign_id: str,
        state: CampaignState,
        action: ActionSpace,
        context: CampaignContext
    ) -> str:
        """
        Register an action for outcome tracking
        
        Args:
            campaign_id: Campaign identifier
            state: State at action time
            action: Action taken
            context: Campaign context
            
        Returns:
            Tracking ID
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        tracking_id = f"{campaign_id}_{timestamp}"
        
        record = OutcomeRecord(
            campaign_id=campaign_id,
            action_timestamp=timestamp,
            state_before=state.to_tensor().numpy(),
            action_continuous=np.array([action.bid_adjustment, action.budget_adjustment]),
            action_discrete=np.array([action.audience_action, action.creative_action]),
            action_applied_at=datetime.now(timezone.utc),
        )
        
        self.pending[tracking_id] = record
        self.stats["actions_tracked"] += 1
        
        # Cleanup old pending if at capacity
        if len(self.pending) > self.max_pending:
            self._cleanup_expired()
        
        return tracking_id
    
    def record_outcome(
        self,
        tracking_id: str,
        current_state: CampaignState,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        goal: str,
        constraints: Dict[str, float],
        is_terminal: bool = False
    ) -> Optional[Transition]:
        """
        Record outcome for a tracked action
        
        Args:
            tracking_id: Tracking ID from register_action
            current_state: Current campaign state
            metrics_before: Metrics at action time
            metrics_after: Current metrics
            goal: Optimization goal
            constraints: Campaign constraints
            is_terminal: Whether campaign ended
            
        Returns:
            Completed Transition if ready
        """
        if tracking_id not in self.pending:
            logger.warning(f"Unknown tracking ID: {tracking_id}")
            return None
        
        record = self.pending[tracking_id]
        
        # Compute reward
        from .config import OptimizationGoal
        reward_result = self.reward_computer.compute(
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            action={
                "bid_adjustment": record.action_continuous[0],
                "budget_adjustment": record.action_continuous[1],
            },
            goal=OptimizationGoal(goal),
            constraints=constraints,
            context={
                "hours_since_last_action": record.observation_delay_hours,
            }
        )
        
        # Update record
        record.state_after = current_state.to_tensor().numpy()
        record.reward = reward_result.total
        record.reward_breakdown = reward_result.to_dict()
        record.outcome_observed_at = datetime.now(timezone.utc)
        record.observation_delay_hours = (
            record.outcome_observed_at - record.action_applied_at
        ).total_seconds() / 3600
        record.is_terminal = is_terminal
        record.is_complete = True
        
        self.stats["outcomes_received"] += 1
        
        # Convert to transition
        transition = record.to_transition()
        if transition:
            self.completed_queue.append(transition)
            self.stats["transitions_completed"] += 1
            
            # Remove from pending
            del self.pending[tracking_id]
        
        return transition
    
    def get_completed_transitions(
        self,
        max_count: Optional[int] = None
    ) -> List[Transition]:
        """
        Get completed transitions for learning
        
        Args:
            max_count: Maximum number to return
            
        Returns:
            List of completed Transitions
        """
        if max_count is None:
            transitions = list(self.completed_queue)
            self.completed_queue.clear()
        else:
            transitions = []
            for _ in range(min(max_count, len(self.completed_queue))):
                transitions.append(self.completed_queue.popleft())
        
        return transitions
    
    def _cleanup_expired(self):
        """Remove expired pending records"""
        now = datetime.now(timezone.utc)
        expired_ids = []
        
        for tracking_id, record in self.pending.items():
            if record.action_applied_at:
                age = now - record.action_applied_at
                if age > self.observation_window * 2:  # 2x observation window
                    expired_ids.append(tracking_id)
        
        for tracking_id in expired_ids:
            del self.pending[tracking_id]
            self.stats["transitions_expired"] += 1
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired tracking records")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        return {
            **self.stats,
            "pending_count": len(self.pending),
            "completed_queue_size": len(self.completed_queue),
        }


class PerformanceMonitor:
    """
    Monitors model performance for drift detection and retraining triggers
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.15,
        min_samples: int = 100
    ):
        """
        Args:
            window_size: Size of rolling window for metrics
            drift_threshold: Performance drop threshold for alert
            min_samples: Minimum samples before drift detection
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        
        # Rolling metrics
        self.rewards = deque(maxlen=window_size)
        self.q_values = deque(maxlen=window_size)
        self.td_errors = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        
        # Baseline metrics (from initial training)
        self.baseline_reward: Optional[float] = None
        self.baseline_q_value: Optional[float] = None
        
        # Drift detection
        self.drift_detected = False
        self.drift_history: List[Dict[str, Any]] = []
    
    def record_transition(
        self,
        reward: float,
        q_value: float,
        td_error: float,
        confidence: float
    ):
        """Record metrics from a transition"""
        self.rewards.append(reward)
        self.q_values.append(q_value)
        self.td_errors.append(td_error)
        self.confidences.append(confidence)
    
    def set_baseline(self, reward: float, q_value: float):
        """Set baseline metrics from training"""
        self.baseline_reward = reward
        self.baseline_q_value = q_value
        logger.info(f"Baseline set: reward={reward:.4f}, q_value={q_value:.4f}")
    
    def check_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for performance drift
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if len(self.rewards) < self.min_samples:
            return False, {"reason": "insufficient_samples"}
        
        if self.baseline_reward is None:
            return False, {"reason": "no_baseline"}
        
        current_reward = np.mean(list(self.rewards)[-self.min_samples:])
        reward_change = (current_reward - self.baseline_reward) / abs(self.baseline_reward + 1e-8)
        
        current_td = np.mean(list(self.td_errors)[-self.min_samples:])
        
        drift_info = {
            "current_reward": current_reward,
            "baseline_reward": self.baseline_reward,
            "reward_change": reward_change,
            "current_td_error": current_td,
            "sample_count": len(self.rewards),
        }
        
        # Check for significant performance drop
        if reward_change < -self.drift_threshold:
            self.drift_detected = True
            drift_info["drift_type"] = "performance_degradation"
            self.drift_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **drift_info
            })
            logger.warning(f"Performance drift detected: {reward_change:.2%}")
            return True, drift_info
        
        # Check for high TD error (value function inaccuracy)
        if current_td > 1.0:  # High TD error threshold
            drift_info["drift_type"] = "value_estimation_error"
            self.drift_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **drift_info
            })
            logger.warning(f"High TD error detected: {current_td:.4f}")
            return True, drift_info
        
        self.drift_detected = False
        return False, drift_info
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.rewards:
            return {"status": "no_data"}
        
        return {
            "reward_mean": np.mean(self.rewards),
            "reward_std": np.std(self.rewards),
            "q_value_mean": np.mean(self.q_values),
            "td_error_mean": np.mean(self.td_errors),
            "confidence_mean": np.mean(self.confidences),
            "sample_count": len(self.rewards),
            "drift_detected": self.drift_detected,
            "drift_count": len(self.drift_history),
        }


class ContinuousLearningEngine:
    """
    Main continuous learning engine coordinating online updates
    """
    
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: PrioritizedReplayBuffer,
        training_config: TrainingConfig,
        learning_mode: LearningMode = LearningMode.HYBRID,
        update_frequency: int = 100,
        batch_interval_minutes: int = 60,
        forecast_feedback: Optional[ForecastFeedbackLoop] = None,
    ):
        """
        Args:
            agent: SAC agent to update
            replay_buffer: Prioritized replay buffer
            training_config: Training configuration
            learning_mode: Learning mode
            update_frequency: Steps between online updates
            batch_interval_minutes: Minutes between batch updates
            forecast_feedback: Optional forecast feedback loop for forecaster refit
        """
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.training_config = training_config
        self.learning_mode = learning_mode
        self.update_frequency = update_frequency
        self.batch_interval = timedelta(minutes=batch_interval_minutes)

        # Components
        self.outcome_tracker = OutcomeTracker()
        self.performance_monitor = PerformanceMonitor()
        self.forecast_feedback = forecast_feedback
        
        # State
        self.steps_since_update = 0
        self.last_batch_update = datetime.now(timezone.utc)
        self.is_running = False
        
        # Statistics
        self.learning_stats = {
            "online_updates": 0,
            "batch_updates": 0,
            "triggered_updates": 0,
            "transitions_learned": 0,
        }
        
        logger.info(f"ContinuousLearningEngine initialized with mode: {learning_mode.value}")
    
    async def start(self):
        """Start the continuous learning loop"""
        self.is_running = True
        logger.info("Continuous learning engine started")
        
        while self.is_running:
            try:
                await self._learning_step()
                await asyncio.sleep(1)  # 1 second between checks
            except Exception as e:
                logger.error(f"Learning step error: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    def stop(self):
        """Stop the continuous learning loop"""
        self.is_running = False
        logger.info("Continuous learning engine stopped")
    
    async def _learning_step(self):
        """Execute one learning step based on mode"""
        # Collect completed transitions
        transitions = self.outcome_tracker.get_completed_transitions(max_count=100)
        
        # Add to replay buffer
        for transition in transitions:
            self.replay_buffer.push(transition)
            self.learning_stats["transitions_learned"] += 1
        
        self.steps_since_update += len(transitions)
        
        # Determine if update needed
        should_update = False
        update_type = None
        
        if self.learning_mode == LearningMode.ONLINE:
            should_update = self.steps_since_update >= self.update_frequency
            update_type = "online"
            
        elif self.learning_mode == LearningMode.BATCH:
            if datetime.now(timezone.utc) - self.last_batch_update > self.batch_interval:
                should_update = True
                update_type = "batch"
                
        elif self.learning_mode == LearningMode.TRIGGERED:
            drift_detected, _ = self.performance_monitor.check_drift()
            if drift_detected:
                should_update = True
                update_type = "triggered"
                
        elif self.learning_mode == LearningMode.HYBRID:
            # Online updates at frequency
            if self.steps_since_update >= self.update_frequency:
                should_update = True
                update_type = "online"
            # Batch updates at interval
            elif datetime.now(timezone.utc) - self.last_batch_update > self.batch_interval:
                should_update = True
                update_type = "batch"
            # Triggered updates on drift
            else:
                drift_detected, _ = self.performance_monitor.check_drift()
                if drift_detected:
                    should_update = True
                    update_type = "triggered"
        
        # Execute update
        if should_update and len(self.replay_buffer) >= self.training_config.min_buffer_size:
            await self._execute_update(update_type)
    
    async def _execute_update(self, update_type: str):
        """Execute model update"""
        logger.debug(f"Executing {update_type} update")
        
        # Number of gradient steps based on update type
        if update_type == "online":
            num_steps = 1
            self.learning_stats["online_updates"] += 1
        elif update_type == "batch":
            num_steps = 10
            self.learning_stats["batch_updates"] += 1
            self.last_batch_update = datetime.now(timezone.utc)
        elif update_type == "triggered":
            num_steps = 50  # More steps for recovery
            self.learning_stats["triggered_updates"] += 1
        else:
            num_steps = 1
        
        # Perform gradient steps
        for _ in range(num_steps):
            metrics = self.agent.update(self.replay_buffer)
            
            if metrics:
                # Record for monitoring
                self.performance_monitor.record_transition(
                    reward=metrics.get("q_value", 0),
                    q_value=metrics.get("q_value", 0),
                    td_error=metrics.get("critic_loss", 0),
                    confidence=1.0 - metrics.get("entropy", 0),
                )
        
        self.steps_since_update = 0
        logger.debug(f"Completed {num_steps} gradient steps")
    
    def register_action(
        self,
        campaign_id: str,
        state: CampaignState,
        action: ActionSpace,
        context: CampaignContext
    ) -> str:
        """Register action for outcome tracking"""
        return self.outcome_tracker.register_action(
            campaign_id=campaign_id,
            state=state,
            action=action,
            context=context
        )
    
    def record_outcome(
        self,
        tracking_id: str,
        current_state: CampaignState,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        goal: str,
        constraints: Dict[str, float],
        is_terminal: bool = False
    ):
        """Record outcome for learning and forecast feedback."""
        self.outcome_tracker.record_outcome(
            tracking_id=tracking_id,
            current_state=current_state,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            goal=goal,
            constraints=constraints,
            is_terminal=is_terminal
        )

        # Forward actuals to forecast feedback loop (if wired)
        if self.forecast_feedback is not None:
            self.forecast_feedback.record_actual(tracking_id, metrics_after)
            self.forecast_feedback.maybe_refit()
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get engine diagnostics"""
        diag = {
            "learning_mode": self.learning_mode.value,
            "is_running": self.is_running,
            "buffer_size": len(self.replay_buffer),
            "steps_since_update": self.steps_since_update,
            "learning_stats": self.learning_stats,
            "outcome_tracker_stats": self.outcome_tracker.get_stats(),
            "performance_summary": self.performance_monitor.get_summary(),
            "agent_diagnostics": self.agent.get_diagnostics(),
        }
        if self.forecast_feedback is not None:
            diag["forecast_feedback"] = self.forecast_feedback.get_diagnostics()
        return diag
    
    def save_checkpoint(self, path: str):
        """Save learning state checkpoint"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save agent
        self.agent.save(str(path / "agent"))
        
        # Save stats
        stats = {
            "learning_stats": self.learning_stats,
            "outcome_tracker_stats": self.outcome_tracker.get_stats(),
            "performance_summary": self.performance_monitor.get_summary(),
            "last_batch_update": self.last_batch_update.isoformat(),
        }
        
        with open(path / "learning_state.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load learning state checkpoint"""
        path = Path(path)
        
        # Load agent
        self.agent.load(str(path / "agent"))
        
        # Load stats
        state_path = path / "learning_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                stats = json.load(f)
            self.learning_stats = stats.get("learning_stats", self.learning_stats)
            self.last_batch_update = datetime.fromisoformat(
                stats.get("last_batch_update", datetime.now(timezone.utc).isoformat())
            )
        
        logger.info(f"Loaded checkpoint from {path}")


class ModelVersionManager:
    """
    Manages model versions for safe deployment and rollback
    """
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions: List[Dict[str, Any]] = []
        self._load_version_history()
    
    def save_version(
        self,
        agent: SACAgent,
        metrics: Dict[str, float],
        description: str = ""
    ) -> str:
        """
        Save new model version
        
        Returns:
            Version ID
        """
        version_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_path = self.model_dir / version_id
        
        agent.save(str(version_path))
        
        version_info = {
            "version_id": version_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "description": description,
            "path": str(version_path),
        }
        
        self.versions.append(version_info)
        self._save_version_history()
        
        logger.info(f"Saved model version: {version_id}")
        return version_id
    
    def load_version(self, agent: SACAgent, version_id: str) -> bool:
        """Load specific model version"""
        for version in self.versions:
            if version["version_id"] == version_id:
                agent.load(version["path"])
                logger.info(f"Loaded model version: {version_id}")
                return True
        
        logger.error(f"Version not found: {version_id}")
        return False
    
    def get_best_version(self, metric: str = "reward_mean") -> Optional[str]:
        """Get version with best performance"""
        if not self.versions:
            return None
        
        best = max(
            self.versions,
            key=lambda v: v.get("metrics", {}).get(metric, float("-inf"))
        )
        return best["version_id"]
    
    def _load_version_history(self):
        """Load version history from disk"""
        history_path = self.model_dir / "version_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.versions = json.load(f)
    
    def _save_version_history(self):
        """Save version history to disk"""
        history_path = self.model_dir / "version_history.json"
        with open(history_path, "w") as f:
            json.dump(self.versions, f, indent=2)
