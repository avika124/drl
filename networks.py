"""
Neural Network Architectures for Soft Actor-Critic (SAC)

Implements:
- ActorNetwork: Policy network with Gaussian output for continuous actions
- CriticNetwork: Twin Q-networks for stable value estimation
- ValueNetwork: State value network (optional, for SAC v1)
- Discrete action heads for audience/creative decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple, List, Optional
import numpy as np

from .config import DRLConfig


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    use_layer_norm: bool = True,
    dropout_rate: float = 0.0
) -> nn.Sequential:
    """
    Create a multi-layer perceptron
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        output_activation: Optional activation for output layer
        use_layer_norm: Whether to use layer normalization
        dropout_rate: Dropout probability
    """
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    
    act_fn = activations.get(activation, nn.ReLU)
    layers = []
    
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    if output_activation:
        out_act = activations.get(output_activation, nn.Identity)
        layers.append(out_act())
    
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """
    Patched Actor Network with Gumbel-Softmax for Differentiable Discrete Actions
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(
        self,
        state_dim: int,
        continuous_action_dim: int,
        discrete_action_dims: List[int],
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        activation: str = "relu",
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.state_dim = state_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = discrete_action_dims
        
        # Shared feature extractor
        hidden_dims = [hidden_dim] * num_hidden_layers
        self.shared = create_mlp(
            input_dim=state_dim,
            output_dim=hidden_dim,
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate
        )
        
        # Continuous action heads
        self.mean_head = nn.Linear(hidden_dim, continuous_action_dim)
        self.log_std_head = nn.Linear(hidden_dim, continuous_action_dim)
        
        # Discrete action heads (logits)
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in discrete_action_dims
        ])
        
        self._init_weights()

    def _init_weights(self):
        # [Keep original initialization logic]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        for head in self.discrete_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)

    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        discrete_logits = [head(features) for head in self.discrete_heads]
        return mean, log_std, discrete_logits
    
    def sample(
        self, 
        state: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0  # Temperature for Gumbel-Softmax
    ):
        """
        Samples actions using Reparameterization (Continuous) and Gumbel-Softmax (Discrete)
        """
        mean, log_std, discrete_logits = self.forward(state)
        
        # --- Continuous Actions (Reparameterization Trick) ---
        if deterministic:
            continuous_action = torch.tanh(mean)
            log_prob_continuous = torch.zeros(state.shape[0], device=state.device)
            cont_entropy = torch.zeros(state.shape[0], device=state.device)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            continuous_action = torch.tanh(x_t)
            
            # Log prob with tanh correction
            log_prob_continuous = normal.log_prob(x_t).sum(dim=-1)
            log_prob_continuous -= torch.log(1 - continuous_action.pow(2) + 1e-6).sum(dim=-1)
            cont_entropy = normal.entropy().sum(dim=-1)

        # --- Discrete Actions (Gumbel-Softmax Trick) ---
        discrete_actions_soft = []  # For Critic update (differentiable)
        discrete_actions_hard = []  # For Environment (indices)
        log_prob_discrete = 0
        disc_entropy = 0
        
        for logits in discrete_logits:
            if deterministic:
                # Greedy selection
                action_idx = logits.argmax(dim=-1)
                # Create hard one-hot for consistency if needed, but we usually return indices for deterministic
                one_hot = F.one_hot(action_idx, num_classes=logits.shape[-1]).float()
                discrete_actions_soft.append(one_hot)
                discrete_actions_hard.append(action_idx)
            else:
                # Gumbel-Softmax: Returns differentiable soft one-hot vector
                # hard=False ensures gradients flow through the soft probabilities
                soft_one_hot = F.gumbel_softmax(logits, tau=temperature, hard=False)
                discrete_actions_soft.append(soft_one_hot)
                
                # For environment execution, we still need the hard index
                discrete_actions_hard.append(logits.argmax(dim=-1))
                
                # Calculate log_prob of the sample
                dist = Categorical(logits=logits)
                # Note: We estimate log_prob using the hard action for entropy calculation
                # This is standard hybrid SAC practice
                action_idx = logits.argmax(dim=-1) 
                log_prob_discrete = log_prob_discrete + dist.log_prob(action_idx)
                disc_entropy = disc_entropy + dist.entropy()

        # Concatenate soft vectors for the Critic (Batch, Sum of Discrete Dims)
        discrete_action_tensor = torch.cat(discrete_actions_soft, dim=-1)
        
        # Stack indices for the Environment/Buffer (Batch, Num Heads)
        discrete_indices = torch.stack(discrete_actions_hard, dim=-1)
        
        total_log_prob = log_prob_continuous + log_prob_discrete
        total_entropy = cont_entropy + disc_entropy
        
        return continuous_action, discrete_action_tensor, discrete_indices, total_log_prob, total_entropy


class CriticNetwork(nn.Module):
    """
    Patched Critic Network that accepts both Indices (from Buffer) and Soft Vectors (from Actor)
    """
    def __init__(
        self,
        state_dim: int,
        continuous_action_dim: int,
        discrete_action_dims: List[int],
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        activation: str = "relu",
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.discrete_action_dims = discrete_action_dims
        
        # Total action dimension: Continuous + Sum of all Discrete one-hot dims
        discrete_total_dim = sum(discrete_action_dims)
        input_dim = state_dim + continuous_action_dim + discrete_total_dim
        
        hidden_dims = [hidden_dim] * num_hidden_layers
        self.q1 = create_mlp(input_dim, 1, hidden_dims, activation, use_layer_norm, dropout_rate)
        self.q2 = create_mlp(input_dim, 1, hidden_dims, activation, use_layer_norm, dropout_rate)
        
    def forward(
        self,
        state: torch.Tensor,
        continuous_action: torch.Tensor,
        discrete_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            discrete_action: 
                - If shape is (Batch, Num_Heads): treated as INDICES (from Replay Buffer).
                - If shape is (Batch, Total_Discrete_Dim): treated as ONE-HOT/SOFT (from Actor).
        """
        # Auto-detect input type
        if discrete_action.shape[-1] == len(self.discrete_action_dims) and discrete_action.dtype in [torch.long, torch.int, torch.int64]:
             # It's indices from the Replay Buffer -> One-hot encode it
            discrete_encoded = self._encode_indices(discrete_action)
        else:
             # It's already a soft/hard one-hot vector from the Actor -> Use as is
            discrete_encoded = discrete_action
            
        sa = torch.cat([state, continuous_action, discrete_encoded], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def _encode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Helper to encode indices into concatenated one-hot vectors"""
        encoded = []
        for i, dim in enumerate(self.discrete_action_dims):
            one_hot = F.one_hot(indices[:, i].long(), num_classes=dim)
            encoded.append(one_hot.float())
        return torch.cat(encoded, dim=-1)

class ValueNetwork(nn.Module):
    """
    State Value Network (optional, for SAC v1)
    
    Estimates V(s) = E[Q(s,a) - alpha * log(pi(a|s))]
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        activation: str = "relu",
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        hidden_dims = [hidden_dim] * num_hidden_layers
        
        self.network = create_mlp(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: Batch of states
            
        Returns:
            value: State values
        """
        return self.network(state).squeeze(-1)


def create_networks(config: DRLConfig) -> Tuple[ActorNetwork, CriticNetwork, CriticNetwork]:
    """
    Factory function to create actor and critic networks from config
    
    Args:
        config: DRL configuration
        
    Returns:
        actor: Actor network
        critic: Critic network
        critic_target: Target critic network (copy of critic)
    """
    actor = ActorNetwork(
        state_dim=config.state_dim,
        continuous_action_dim=config.continuous_action_dim,
        discrete_action_dims=config.discrete_action_dims,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        dropout_rate=config.dropout_rate
    )
    
    critic = CriticNetwork(
        state_dim=config.state_dim,
        continuous_action_dim=config.continuous_action_dim,
        discrete_action_dims=config.discrete_action_dims,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        dropout_rate=config.dropout_rate
    )
    
    # Create target network as a copy
    critic_target = CriticNetwork(
        state_dim=config.state_dim,
        continuous_action_dim=config.continuous_action_dim,
        discrete_action_dims=config.discrete_action_dims,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        dropout_rate=config.dropout_rate
    )
    critic_target.load_state_dict(critic.state_dict())
    
    return actor, critic, critic_target
