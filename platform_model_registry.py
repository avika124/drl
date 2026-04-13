"""
Per-Platform P-Model Registry (M1/M2 infrastructure)

Manages separate SAC agents per advertising platform, enabling each platform
to learn its own policy (P-Model) while sharing the same network architecture.

Key capabilities:
  - Load/save per-platform SAC checkpoints
  - Route inference requests to the correct platform agent
  - Fall back to a global agent when no platform-specific model exists
  - Track per-platform training metadata

Architecture reference (P & X Model HL Design):
  M1 — P-Training:  One DRL model trained per platform on that platform's
        historical (state, action, reward, next_state) transitions.
  M2 — P-Execution:  At inference time, route the CampaignState to the
        platform-specific agent and return (ActionSpace, forecast).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import DRLConfig, TrainingConfig
from .sac_agent import SACAgent
from .state_action import CampaignState, ActionSpace

logger = logging.getLogger(__name__)


# Platforms we maintain separate models for.  Additional platforms fall
# back to the global model.
SUPPORTED_PLATFORMS: List[str] = ["meta", "google", "tiktok", "amazon", "walmart"]


@dataclass
class PlatformModelMeta:
    """Metadata stored alongside each per-platform checkpoint."""
    platform: str
    state_dim: int = 42
    total_training_steps: int = 0
    last_trained_at: str = ""
    training_transitions: int = 0
    checkpoint_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "state_dim": self.state_dim,
            "total_training_steps": self.total_training_steps,
            "last_trained_at": self.last_trained_at,
            "training_transitions": self.training_transitions,
            "checkpoint_path": self.checkpoint_path,
        }


class PlatformModelRegistry:
    """
    Registry managing one SAC agent per advertising platform.

    Usage::

        registry = PlatformModelRegistry(model_root="models/platforms")
        registry.load_all()

        # M2 — Inference
        action = registry.select_action("google", campaign_state)

        # M1 — Training (per-platform)
        agent = registry.get_or_create("meta")
        # … train agent …
        registry.save("meta")
    """

    def __init__(
        self,
        model_root: str = "models/platforms",
        state_dim: int = 42,
        device: str = "cpu",
        global_fallback_dir: Optional[str] = None,
    ):
        """
        Args:
            model_root: Root directory containing per-platform sub-directories.
            state_dim: State dimension all agents share.
            device: Torch device string.
            global_fallback_dir: Optional path to a pre-trained global model
                used when no platform-specific checkpoint exists.
        """
        self.model_root = Path(model_root)
        self.state_dim = state_dim
        self.device = device
        self.global_fallback_dir = global_fallback_dir

        # platform -> SACAgent
        self._agents: Dict[str, SACAgent] = {}
        # platform -> metadata
        self._meta: Dict[str, PlatformModelMeta] = {}
        # Optional global fallback agent
        self._global_agent: Optional[SACAgent] = None

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def _make_config(self) -> Tuple[DRLConfig, TrainingConfig]:
        """Create standard configs for a per-platform agent."""
        config = DRLConfig(state_dim=self.state_dim)
        training_config = TrainingConfig()
        return config, training_config

    def get_or_create(self, platform: str) -> SACAgent:
        """
        Return the agent for *platform*, creating a fresh one if needed.
        """
        platform = platform.lower()
        if platform in self._agents:
            return self._agents[platform]

        config, train_cfg = self._make_config()
        agent = SACAgent(config, train_cfg, device=self.device)

        self._agents[platform] = agent
        self._meta[platform] = PlatformModelMeta(
            platform=platform,
            state_dim=self.state_dim,
        )
        logger.info(f"Created new P-Model agent for platform={platform}")
        return agent

    def get(self, platform: str) -> Optional[SACAgent]:
        """Return the agent for *platform* or ``None``."""
        return self._agents.get(platform.lower())

    def has_platform(self, platform: str) -> bool:
        return platform.lower() in self._agents

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _platform_dir(self, platform: str) -> Path:
        return self.model_root / platform.lower()

    def save(self, platform: str) -> None:
        """Save a single platform agent + metadata."""
        platform = platform.lower()
        agent = self._agents.get(platform)
        if agent is None:
            raise ValueError(f"No agent loaded for platform={platform}")

        out_dir = self._platform_dir(platform)
        agent.save(str(out_dir))

        meta = self._meta.get(platform, PlatformModelMeta(platform=platform))
        meta.total_training_steps = agent.total_steps
        meta.checkpoint_path = str(out_dir)
        meta_path = out_dir / "platform_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        logger.info(f"Saved P-Model for platform={platform} to {out_dir}")

    def save_all(self) -> None:
        """Save all loaded platform agents."""
        for platform in list(self._agents.keys()):
            self.save(platform)

    def load(self, platform: str) -> bool:
        """
        Load a single platform agent from disk.

        Returns True if a checkpoint was found and loaded, False otherwise.
        """
        platform = platform.lower()
        ckpt_dir = self._platform_dir(platform)
        agent_pt = ckpt_dir / "agent.pt"

        if not agent_pt.exists():
            return False

        agent = self.get_or_create(platform)
        agent.load(str(ckpt_dir))
        agent.actor.eval()
        agent.critic.eval()

        # Load metadata if present
        meta_path = ckpt_dir / "platform_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                data = json.load(f)
            self._meta[platform] = PlatformModelMeta(**data)

        logger.info(f"Loaded P-Model for platform={platform} from {ckpt_dir}")
        return True

    def load_all(self) -> Dict[str, bool]:
        """
        Attempt to load all supported platform agents.

        Returns dict mapping platform -> whether load succeeded.
        """
        results: Dict[str, bool] = {}
        for platform in SUPPORTED_PLATFORMS:
            results[platform] = self.load(platform)
        return results

    def _ensure_global_fallback(self) -> Optional[SACAgent]:
        """Lazily load the global fallback agent."""
        if self._global_agent is not None:
            return self._global_agent

        if self.global_fallback_dir is None:
            return None

        fallback_pt = Path(self.global_fallback_dir) / "agent.pt"
        if not fallback_pt.exists():
            return None

        config, train_cfg = self._make_config()
        agent = SACAgent(config, train_cfg, device=self.device)
        agent.load(self.global_fallback_dir)
        agent.actor.eval()
        agent.critic.eval()
        self._global_agent = agent
        logger.info(f"Loaded global fallback agent from {self.global_fallback_dir}")
        return agent

    # ------------------------------------------------------------------
    # M2 — P-Execution (per-platform inference)
    # ------------------------------------------------------------------

    def select_action(
        self,
        platform: str,
        state: CampaignState,
        deterministic: bool = True,
    ) -> ActionSpace:
        """
        M2 entry-point: route *state* to the platform-specific P-Model
        and return an ActionSpace.

        Falls back to the global model when no platform checkpoint exists.
        """
        platform = platform.lower()
        agent = self._agents.get(platform)

        if agent is None:
            agent = self._ensure_global_fallback()

        if agent is None:
            logger.warning(
                f"No P-Model for {platform} and no global fallback; "
                f"returning default action"
            )
            return ActionSpace()

        return agent.select_action(state, deterministic=deterministic)

    def evaluate_q(
        self,
        platform: str,
        state: CampaignState,
    ) -> float:
        """
        Evaluate Q-value for a state using the platform's critic.

        Used by M5 (X-Execution) to get per-platform value estimates.
        """
        platform = platform.lower()
        agent = self._agents.get(platform) or self._ensure_global_fallback()
        if agent is None:
            return 0.0

        with torch.no_grad():
            state_tensor = state.to_tensor(agent.device).unsqueeze(0)
            continuous, discrete_soft, _, _, _ = agent.actor.sample(
                state_tensor, deterministic=True
            )
            q1, q2 = agent.critic(state_tensor, continuous, discrete_soft)
            return torch.min(q1, q2).item()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Summary of all loaded P-Models."""
        diag: Dict[str, Any] = {
            "loaded_platforms": list(self._agents.keys()),
            "has_global_fallback": self._global_agent is not None,
            "model_root": str(self.model_root),
        }
        for platform, meta in self._meta.items():
            diag[f"platform_{platform}"] = meta.to_dict()
        return diag

    @property
    def platforms(self) -> List[str]:
        """Return list of platforms with loaded agents."""
        return list(self._agents.keys())
