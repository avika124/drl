"""
Soft Actor-Critic (SAC) Agent

===== STEP 1 (Training) & STEP 2 (Execution) =====
Training: Used by train.py, OfflineTrainer; consumes transitions from replay buffer.
Execution: Used by SafeDRLAgent, load_sac_for_inference(); consumes CampaignState.

Implements the SAC algorithm with:
- Automatic entropy tuning
- Twin Q-networks for stable learning
- Support for both continuous and discrete actions (Gumbel-Softmax compatible)
- Conservative Q-Learning (CQL) regularization for offline training
"""
# QA/Testing: Set True to enable input/output logging for traceability
_QA_IO_LOGGING = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging
import json
from datetime import datetime

from .config import DRLConfig, TrainingConfig
from .networks import ActorNetwork, CriticNetwork, create_networks
from .replay_buffer import PrioritizedReplayBuffer, Transition
from .state_action import CampaignState, ActionSpace

logger = logging.getLogger(__name__)


class SACAgent:
    """
    [SACAgent]
    Description: Core RL agent - selects actions from state (inference) or updates from replay buffer (training).
    Input (inference): CampaignState from campaign metrics. Input (training): batch from replay buffer.
    Output: ActionSpace (inference) or metrics dict (training); checkpoint files (save).
    """
    
    """
    Soft Actor-Critic Agent for Campaign Optimization
    
    SAC is an off-policy actor-critic algorithm that:
    - Maximizes expected reward + entropy (for exploration)
    - Uses twin Q-networks to reduce overestimation
    - Automatically tunes the entropy coefficient
    
    This implementation supports hybrid action spaces (continuous + discrete)
    using Gumbel-Softmax relaxation for differentiable discrete actions.
    """
    
    def __init__(
        self,
        config: DRLConfig,
        training_config: TrainingConfig,
        device: str = "auto"
    ):
        """
        Args:
            config: DRL model configuration
            training_config: Training hyperparameters
            device: Device to use ("cpu", "cuda", or "auto")
        """
        self.config = config
        self.training_config = training_config
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing SAC agent on device: {self.device}")
        
        # Create networks
        self.actor, self.critic, self.critic_target = create_networks(config)
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)
        
        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=training_config.weight_decay
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=training_config.weight_decay
        )
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = config.auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha],
                lr=config.alpha_lr
            )
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha
        
        # Training state
        self.total_steps = 0
        self.training_info = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "q_values": [],
            "entropy": [],
        }
    
    def select_action(
        self,
        state: CampaignState,
        deterministic: bool = False
    ) -> ActionSpace:
        """
        Select action for given state
        
        Args:
            state: Current campaign state
            deterministic: If True, return mean action
            
        Returns:
            ActionSpace with selected actions and confidence
        """
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT select_action: state.ctr={state.ctr:.4f}, roas={state.roas:.4f}, deterministic={deterministic}")
        with torch.no_grad():
            state_tensor = state.to_tensor(self.device).unsqueeze(0)
            
            # Unpack 5 values from Gumbel-Softmax Actor
            # continuous: (B, cont_dim)
            # discrete_soft: (B, total_disc_dim) - Differentiable one-hots
            # discrete_indices: (B, num_heads) - Hard indices
            # log_prob: (B,)
            # entropy: (B,)
            continuous, discrete_soft, discrete_indices, log_prob, entropy = self.actor.sample(
                state_tensor, deterministic=deterministic
            )
            
            # Get Q-value estimate using Soft actions (Critic handles both soft/hard)
            q1, q2 = self.critic(state_tensor, continuous, discrete_soft)
            q_value = torch.min(q1, q2).item()
            
            # Compute confidence from entropy (lower entropy = higher confidence)
            max_entropy = np.log(4) * len(self.config.discrete_action_dims) + self.config.continuous_action_dim
            confidence = 1.0 - (entropy.item() / max_entropy)
            confidence = np.clip(confidence, 0.0, 1.0)
        
        action = ActionSpace(
            bid_adjustment=continuous[0, 0].item(),
            budget_adjustment=continuous[0, 1].item(),
            audience_action=discrete_indices[0, 0].item(),
            creative_action=discrete_indices[0, 1].item(),
            confidence=confidence,
            entropy=entropy.item(),
            q_value=q_value,
        )
        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] OUTPUT select_action: bid_adj={action.bid_adjustment:.4f}, budget_adj={action.budget_adjustment:.4f}, confidence={action.confidence:.4f} | Next: SafeDRLAgent, env.step()")
        return action
    
    def update(
        self,
        replay_buffer: PrioritizedReplayBuffer,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform one update step
        
        Args:
            replay_buffer: Experience replay buffer
            batch_size: Batch size (uses config default if None)
            
        Returns:
            Dictionary of training metrics
        """
        if batch_size is None:
            batch_size = self.training_config.batch_size
        
        if len(replay_buffer) < self.training_config.min_buffer_size:
            return {}
        
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT update: buffer_size={len(replay_buffer)}, batch_size={batch_size or self.training_config.batch_size} | From: replay_buffer (train.py, OfflineTrainer)")
        # Sample batch
        transitions, indices, weights = replay_buffer.sample(batch_size)
        batch = replay_buffer.to_tensors(transitions, weights, self.device)
        
        # Update critic
        critic_loss, td_errors = self._update_critic(batch)
        
        # Update priorities in replay buffer
        replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
        
        # Update actor
        actor_loss, entropy = self._update_actor(batch)
        
        # Update entropy coefficient
        alpha_loss = 0.0
        if self.auto_entropy_tuning:
            alpha_loss = self._update_alpha(entropy)
        
        # Soft update target network
        self._soft_update_target()
        
        self.total_steps += 1
        
        # Log metrics (use q_values key to match training_info init)
        with torch.no_grad():
            q1, q2 = self.critic(batch["states"], batch["continuous_actions"], batch["discrete_actions"])
            q_val = torch.min(q1, q2).mean().item()
        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": entropy.mean().item(),
            "q_values": q_val,
        }
        
        for key, value in metrics.items():
            self.training_info[key].append(value)
        
        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING and self.total_steps % 100 == 0:
            logger.info(f"[IO] OUTPUT update: step={self.total_steps}, actor_loss={metrics.get('actor_loss', 0):.4f}, critic_loss={metrics.get('critic_loss', 0):.4f} | Internal: training_info")
        return metrics
    
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, torch.Tensor]:
        """Update critic networks"""
        states = batch["states"]
        continuous_actions = batch["continuous_actions"]
        discrete_actions = batch["discrete_actions"] # Indices from buffer
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        weights = batch["weights"]
        
        with torch.no_grad():
            # Sample next actions from current policy
            # We use soft discrete actions for the target computation
            next_continuous, next_discrete_soft, _, next_log_prob, _ = self.actor.sample(next_states)
            
            # Target Q-values
            next_q1, next_q2 = self.critic_target(next_states, next_continuous, next_discrete_soft)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        # Current Q-values
        # Critic automatically detects discrete_actions are indices (integers) and encodes them
        current_q1, current_q2 = self.critic(states, continuous_actions, discrete_actions)
        
        # TD errors for prioritized replay
        td_errors = torch.abs(current_q1 - target_q).detach()
        
        # Critic loss (weighted MSE for PER)
        critic_loss = (
            weights * F.mse_loss(current_q1, target_q, reduction='none') +
            weights * F.mse_loss(current_q2, target_q, reduction='none')
        ).mean()
        
        # CQL regularization for offline training
        if self.training_config.use_cql:
            # Pass indices (discrete_actions) for the "data" part of CQL
            cql_loss = self._compute_cql_loss(states, continuous_actions, discrete_actions)
            critic_loss = critic_loss + self.training_config.cql_alpha * cql_loss
        
        # Update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.training_config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.training_config.gradient_clip
            )
        self.critic_optimizer.step()
        
        return critic_loss.item(), td_errors
    
    def _compute_cql_loss(
        self,
        states: torch.Tensor,
        continuous_actions: torch.Tensor,
        discrete_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Conservative Q-Learning loss for offline training
        
        Penalizes Q-values for out-of-distribution actions
        """
        batch_size = states.shape[0]
        
        # Current Q-values for dataset actions
        q1_data, q2_data = self.critic(states, continuous_actions, discrete_actions)
        
        # Q-values for random actions
        random_continuous = torch.rand(
            batch_size, self.config.continuous_action_dim, device=self.device
        ) * 2 - 1  # Uniform [-1, 1]
        
        # Random discrete indices
        random_discrete = torch.randint(
            0, 4, (batch_size, len(self.config.discrete_action_dims)),
            device=self.device
        )
        q1_random, q2_random = self.critic(states, random_continuous, random_discrete)
        
        # Q-values for policy actions (Use Soft Discrete for differentiation)
        policy_continuous, policy_discrete_soft, _, _, _ = self.actor.sample(states)
        q1_policy, q2_policy = self.critic(states, policy_continuous, policy_discrete_soft)
        
        # CQL loss: minimize Q for OOD actions, maximize for dataset actions
        cql_q1_loss = (
            torch.logsumexp(torch.stack([q1_random, q1_policy], dim=0), dim=0).mean()
            - q1_data.mean()
        )
        cql_q2_loss = (
            torch.logsumexp(torch.stack([q2_random, q2_policy], dim=0), dim=0).mean()
            - q2_data.mean()
        )
        
        return (cql_q1_loss + cql_q2_loss) / 2
    
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, torch.Tensor]:
        """Update actor network"""
        states = batch["states"]
        
        # Sample actions from current policy
        # Use SOFT discrete actions to allow gradients to flow from Critic to Actor
        continuous, discrete_soft, _, log_prob, entropy = self.actor.sample(states)
        
        # Q-value for sampled actions
        q1, q2 = self.critic(states, continuous, discrete_soft)
        min_q = torch.min(q1, q2)
        
        # Actor loss: maximize Q - alpha * log_prob
        actor_loss = (self.alpha * log_prob - min_q).mean()
        
        # Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.training_config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.training_config.gradient_clip
            )
        self.actor_optimizer.step()
        
        return actor_loss.item(), entropy
    
    def _update_alpha(self, entropy: torch.Tensor) -> float:
        """Update entropy coefficient"""
        alpha_loss = -(
            self.log_alpha.exp() * (entropy + self.target_entropy).detach()
        ).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        return alpha_loss.item()
    
    def _soft_update_target(self):
        """Soft update target network parameters"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1 - self.config.tau) * target_param.data
            )
    
    def save(self, path: str):
        """
        [save]
        Description: Persists actor, critic, optimizers, and training_info to disk.
        Input: path (e.g. checkpoints/final_model) - from train(), OfflineTrainer.
        Output: path/agent.pt, path/training_info.json | Next: load(), load_sac_for_inference().
        """
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT save: path={path}, total_steps={self.total_steps}")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy_tuning else None,
            "alpha_optimizer_state_dict": (
                self.alpha_optimizer.state_dict() 
                if self.auto_entropy_tuning else None
            ),
            "total_steps": self.total_steps,
            "config": self.config.__dict__,
            "training_config": self.training_config.__dict__,
        }, path / "agent.pt")
        
        # Save training info
        with open(path / "training_info.json", "w") as f:
            json.dump(self.training_info, f)
        
        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] OUTPUT save: {path}/agent.pt, {path}/training_info.json | Next: load_sac_for_inference(), SafeDRLAgent")
        logger.info(f"Saved agent to {path}")
    
    def load(self, path: str):
        """
        [load]
        Description: Loads checkpoint from disk into this agent instance.
        Input: path (e.g. checkpoints/final_model) - from load_sac_for_inference(), train_bigquery_offline.
        Output: In-memory agent state | Next: select_action(), SafeDRLAgent.get_action().
        """
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT load: path={path}")
        path = Path(path)
        agent_pt = path / "agent.pt"

        # Robust loading for different PyTorch versions
        try:
            checkpoint = torch.load(agent_pt, map_location=self.device, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only
            checkpoint = torch.load(agent_pt, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        # Safe alpha restoration
        log_alpha = checkpoint.get("log_alpha")
        if log_alpha is not None and self.auto_entropy_tuning:
            self.log_alpha = log_alpha
            alpha_opt_state = checkpoint.get("alpha_optimizer_state_dict")
            if alpha_opt_state is not None and hasattr(self, "alpha_optimizer"):
                self.alpha_optimizer.load_state_dict(alpha_opt_state)
            self.alpha = self.log_alpha.exp().item()

        self.total_steps = checkpoint.get("total_steps", 0)

        # Load training info
        training_info_path = path / "training_info.json"
        if training_info_path.exists():
            with open(training_info_path, "r") as f:
                self.training_info = json.load(f)

        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] OUTPUT load: agent restored from {path} | Next: select_action(), SafeDRLAgent")
        logger.info(f"Loaded agent from {path}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information"""
        if not self.training_info["actor_loss"]:
            return {}
        
        n = min(100, len(self.training_info["actor_loss"]))
        
        return {
            "total_steps": self.total_steps,
            "actor_loss_mean": np.mean(self.training_info["actor_loss"][-n:]),
            "critic_loss_mean": np.mean(self.training_info["critic_loss"][-n:]),
            "alpha": self.alpha,
            "entropy_mean": np.mean(self.training_info["entropy"][-n:]),
            "q_value_mean": np.mean(self.training_info["q_values"][-n:]) if self.training_info["q_values"] else 0,
        }


class CQLLoss:
    """
    Conservative Q-Learning loss module
    
    Standalone implementation for flexibility in training pipelines.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        num_action_samples: int = 10,
        importance_sample: bool = True
    ):
        self.alpha = alpha
        self.num_action_samples = num_action_samples
        self.importance_sample = importance_sample
    
    def __call__(
        self,
        critic: CriticNetwork,
        actor: ActorNetwork,
        states: torch.Tensor,
        actions_continuous: torch.Tensor,
        actions_discrete: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CQL loss
        
        Args:
            critic: Critic network
            actor: Actor network
            states: Batch of states
            actions_continuous: Dataset continuous actions
            actions_discrete: Dataset discrete actions (indices)
            
        Returns:
            CQL loss tensor
        """
        batch_size = states.shape[0]
        device = states.device
        
        # Q-values for dataset actions (using indices)
        q1_data, q2_data = critic(states, actions_continuous, actions_discrete)
        
        # Repeat states for multiple action samples
        states_repeated = states.unsqueeze(1).repeat(
            1, self.num_action_samples, 1
        ).view(-1, states.shape[-1])
        
        # Sample random actions
        random_cont = torch.rand(
            batch_size * self.num_action_samples,
            actions_continuous.shape[-1],
            device=device
        ) * 2 - 1
        random_disc = torch.randint(
            0, 4,
            (batch_size * self.num_action_samples, actions_discrete.shape[-1]),
            device=device
        )
        
        # Sample policy actions
        # Unpack 5-tuple: we need policy_cont and policy_disc_soft
        policy_cont, policy_disc_soft, _, log_probs, _ = actor.sample(states_repeated)
        
        # Compute Q-values
        q1_random, q2_random = critic(states_repeated, random_cont, random_disc)
        q1_policy, q2_policy = critic(states_repeated, policy_cont, policy_disc_soft)
        
        # Reshape for logsumexp
        q1_random = q1_random.view(batch_size, self.num_action_samples)
        q2_random = q2_random.view(batch_size, self.num_action_samples)
        q1_policy = q1_policy.view(batch_size, self.num_action_samples)
        q2_policy = q2_policy.view(batch_size, self.num_action_samples)
        
        if self.importance_sample:
            log_probs = log_probs.view(batch_size, self.num_action_samples)
            q1_policy = q1_policy - log_probs
            q2_policy = q2_policy - log_probs
        
        # CQL loss
        q1_logsumexp = torch.logsumexp(
            torch.cat([q1_random, q1_policy], dim=1), dim=1
        )
        q2_logsumexp = torch.logsumexp(
            torch.cat([q2_random, q2_policy], dim=1), dim=1
        )
        
        cql_loss = (
            (q1_logsumexp - q1_data).mean() +
            (q2_logsumexp - q2_data).mean()
        ) / 2
        
        return self.alpha * cql_loss


def load_sac_for_inference(
    model_dir: str,
    device: str = "cpu",
    state_dim: int = 42,
) -> Tuple[SACAgent, List[str]]:
    """
    [load_sac_for_inference]
    Description: Loads trained SAC checkpoint for inference (no training).
    Input: model_dir (e.g. checkpoints/final_model), device, state_dim - from backend, run scripts.
    Output: (SACAgent, feature_list) | Next: SafeDRLAgent(agent=...), HybridDRLLLMOptimizer.

    Args:
        model_dir: Path to saved checkpoint directory.
        device: Torch device string.
        state_dim: State dimension the checkpoint was trained with.

    Returns:
        (agent, feature_list) tuple.  feature_list is the ordered feature
        names stored at training time (if available), otherwise empty.
    """
    config = DRLConfig(state_dim=state_dim)
    training_config = TrainingConfig()
    agent = SACAgent(config, training_config, device=device)
    # ----- INPUT LOGGING -----
    if _QA_IO_LOGGING:
        logger.info(f"[IO] INPUT load_sac_for_inference: model_dir={model_dir}, device={device}, state_dim={state_dim}")
    agent.load(model_dir)
    agent.actor.eval()
    agent.critic.eval()

    features: List[str] = []
    features_path = Path(model_dir) / "features.json"
    if features_path.exists():
        with open(features_path, "r") as f:
            features = json.load(f)

    # ----- OUTPUT LOGGING -----
    if _QA_IO_LOGGING:
        logger.info(f"[IO] OUTPUT load_sac_for_inference: agent loaded, features={len(features)} | Next: SafeDRLAgent, HybridDRLLLMOptimizer")
    return agent, features