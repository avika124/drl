"""
M4 — X-Model Training Pipeline

Trains the XModelAgent on portfolio-level transitions produced by M3
(x_training_data.py).

Pipeline:
  1. Load or receive X-transitions (from XTrainingDataBuilder)
  2. Populate a replay buffer
  3. Run SAC updates on batches
  4. Save checkpoints periodically

Also supports a behavioral-cloning warm-start (like the P-Model pipeline)
to initialize the X-Model actor from historical allocation behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .x_model import XModelAgent, X_STATE_DIM, X_ACTION_DIM, NUM_PLATFORMS
from .x_training_data import XTransition

logger = logging.getLogger(__name__)


class XModelReplayBuffer:
    """Simple replay buffer for X-Model transitions."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._next_states: List[np.ndarray] = []
        self._dones: List[bool] = []
        self._pos = 0

    def add(self, t: XTransition) -> None:
        if len(self._states) < self.capacity:
            self._states.append(t.state)
            self._actions.append(t.action)
            self._rewards.append(t.reward)
            self._next_states.append(t.next_state)
            self._dones.append(t.done)
        else:
            idx = self._pos % self.capacity
            self._states[idx] = t.state
            self._actions[idx] = t.action
            self._rewards[idx] = t.reward
            self._next_states[idx] = t.next_state
            self._dones[idx] = t.done
        self._pos += 1

    def add_batch(self, transitions: List[XTransition]) -> None:
        for t in transitions:
            self.add(t)

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        n = min(len(self._states), batch_size)
        indices = np.random.choice(len(self._states), size=n, replace=False)

        return {
            "states": torch.tensor(
                np.stack([self._states[i] for i in indices]),
                dtype=torch.float32, device=device,
            ),
            "actions": torch.tensor(
                np.stack([self._actions[i] for i in indices]),
                dtype=torch.float32, device=device,
            ),
            "rewards": torch.tensor(
                np.array([self._rewards[i] for i in indices]),
                dtype=torch.float32, device=device,
            ),
            "next_states": torch.tensor(
                np.stack([self._next_states[i] for i in indices]),
                dtype=torch.float32, device=device,
            ),
            "dones": torch.tensor(
                np.array([float(self._dones[i]) for i in indices]),
                dtype=torch.float32, device=device,
            ),
        }

    def __len__(self) -> int:
        return min(len(self._states), self.capacity)


class XModelTrainer:
    """
    M4 training pipeline for the X-Model.

    Usage::

        trainer = XModelTrainer(agent)
        trainer.load_transitions(transitions)  # from M3
        history = trainer.train(num_epochs=20, steps_per_epoch=100)
        trainer.save_checkpoint("models/x_model")
    """

    def __init__(
        self,
        agent: XModelAgent,
        buffer_capacity: int = 100_000,
        min_buffer_size: int = 50,
        batch_size: int = 64,
    ):
        self.agent = agent
        self.buffer = XModelReplayBuffer(capacity=buffer_capacity)
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size

    def load_transitions(self, transitions: List[XTransition]) -> None:
        """Load transitions from M3 into the replay buffer."""
        self.buffer.add_batch(transitions)
        logger.info(f"Loaded {len(transitions)} X-transitions; buffer size={len(self.buffer)}")

    def behavior_cloning_pretrain(
        self,
        transitions: List[XTransition],
        epochs: int = 5,
        batch_size: int = 64,
    ) -> None:
        """
        Supervised pre-training of the X-Model actor to match historical
        allocation behavior.  Warm-starts the policy before SAC training.
        """
        if not transitions:
            return

        states = torch.tensor(
            np.stack([t.state for t in transitions]),
            dtype=torch.float32,
        )
        actions = torch.tensor(
            np.stack([t.action for t in transitions]),
            dtype=torch.float32,
        )

        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, dataset_size, batch_size):
                batch_idx = indices[start: start + batch_size]
                batch_states = states[batch_idx].to(self.agent.device)
                batch_actions = actions[batch_idx].to(self.agent.device)

                # Forward pass: get predicted allocation weights
                pred_weights, _, _ = self.agent.actor.sample(batch_states, deterministic=True)

                loss = F.mse_loss(pred_weights, batch_actions)

                self.agent.actor_optimizer.zero_grad()
                loss.backward()
                self.agent.actor_optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"X-Model BC pretrain epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    def train(
        self,
        num_epochs: int = 20,
        steps_per_epoch: int = 100,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Run SAC training on the X-Model replay buffer.

        Returns:
            Dict mapping metric names to lists of per-epoch values.
        """
        if len(self.buffer) < self.min_buffer_size:
            logger.warning(
                f"Buffer size {len(self.buffer)} < min {self.min_buffer_size}; "
                f"skipping training"
            )
            return {}

        history: Dict[str, List[float]] = {
            "x_critic_loss": [],
            "x_actor_loss": [],
            "x_entropy": [],
            "x_q_value": [],
        }

        for epoch in range(num_epochs):
            epoch_metrics: Dict[str, List[float]] = {k: [] for k in history}

            for _step in range(steps_per_epoch):
                batch = self.buffer.sample(self.batch_size, self.agent.device)
                metrics = self.agent.update(
                    states=batch["states"],
                    actions=batch["actions"],
                    rewards=batch["rewards"],
                    next_states=batch["next_states"],
                    dones=batch["dones"],
                )
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

            # Average epoch metrics
            for k in history:
                vals = epoch_metrics.get(k, [])
                avg = float(np.mean(vals)) if vals else 0.0
                history[k].append(avg)

            logger.info(
                f"X-Model epoch {epoch+1}/{num_epochs}: "
                f"critic_loss={history['x_critic_loss'][-1]:.4f}, "
                f"actor_loss={history['x_actor_loss'][-1]:.4f}, "
                f"q_value={history['x_q_value'][-1]:.4f}"
            )

            # Checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_dir)

        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir)

        return history

    def save_checkpoint(self, path: str) -> None:
        self.agent.save(path)

    def load_checkpoint(self, path: str) -> None:
        self.agent.load(path)
