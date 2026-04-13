"""
Prioritized Experience Replay Buffer

Implements:
- Standard replay buffer for uniform sampling
- Prioritized replay buffer with TD-error based priorities
- Segment tree for efficient O(log n) priority updates
- Importance sampling weights for unbiased learning
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import random
from collections import deque


@dataclass
class Transition:
    """
    Single transition tuple for experience replay
    
    Stores (s, a_continuous, a_discrete, r, s', done) along with metadata
    """
    state: np.ndarray
    continuous_action: np.ndarray
    discrete_action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    
    # Metadata
    campaign_id: str = ""
    timestamp: str = ""
    td_error: float = 0.0
    priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.tolist(),
            "continuous_action": self.continuous_action.tolist(),
            "discrete_action": self.discrete_action.tolist(),
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "campaign_id": self.campaign_id,
            "timestamp": self.timestamp,
            "td_error": self.td_error,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transition":
        return cls(
            state=np.array(data["state"]),
            continuous_action=np.array(data["continuous_action"]),
            discrete_action=np.array(data["discrete_action"]),
            reward=data["reward"],
            next_state=np.array(data["next_state"]),
            done=data["done"],
            campaign_id=data.get("campaign_id", ""),
            timestamp=data.get("timestamp", ""),
            td_error=data.get("td_error", 0.0),
            priority=data.get("priority", 1.0),
        )


class SumTree:
    """
    Sum Tree data structure for efficient priority-based sampling
    
    Allows O(log n) insertion and sampling operations.
    Each leaf node stores a priority, internal nodes store sum of children.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf node for given cumulative sum"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    @property
    def total(self) -> float:
        """Total priority sum"""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """Add new transition with given priority"""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority at given tree index"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get transition for given cumulative sum
        
        Returns:
            Tuple of (tree_index, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    
    @property
    def min(self) -> float:
        """Minimum priority (for importance sampling)"""
        return np.min(self.tree[-self.capacity:self.capacity - 1 + self.n_entries])
    
    @property
    def max(self) -> float:
        """Maximum priority"""
        return np.max(self.tree[-self.capacity:self.capacity - 1 + self.n_entries])


class MinTree:
    """
    Min Tree for efficient minimum priority lookup
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.full(2 * capacity - 1, float('inf'), dtype=np.float64)
    
    def _propagate(self, idx: int):
        parent = (idx - 1) // 2
        left = 2 * parent + 1
        right = left + 1
        
        if right < len(self.tree):
            self.tree[parent] = min(self.tree[left], self.tree[right])
        else:
            self.tree[parent] = self.tree[left]
        
        if parent != 0:
            self._propagate(parent)
    
    def update(self, idx: int, priority: float):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)
    
    @property
    def min(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Samples transitions with probability proportional to their TD-error.
    Uses importance sampling weights to correct for the sampling bias.
    
    Reference: Schaul et al., "Prioritized Experience Replay" (2015)
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100_000,
        epsilon: float = 1e-6
    ):
        """
        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            beta_anneal_steps: Steps to anneal beta from start to end
            epsilon: Small constant added to priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        
        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)
        
        self._max_priority = 1.0
        self._step = 0
    
    @property
    def beta(self) -> float:
        """Current importance sampling exponent"""
        fraction = min(self._step / self.beta_anneal_steps, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def __len__(self) -> int:
        return self.sum_tree.n_entries
    
    def push(self, transition: Transition, td_error: Optional[float] = None):
        """
        Add transition to buffer
        
        Args:
            transition: Transition to add
            td_error: Optional TD-error for priority (uses max if not provided)
        """
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = self._max_priority
        
        transition.priority = priority
        self.sum_tree.add(priority, transition)
        self.min_tree.update(self.sum_tree.write_idx, priority)
        
        self._max_priority = max(self._max_priority, priority)
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch of transitions with prioritized sampling
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            transitions: List of sampled transitions
            indices: Tree indices for priority updates
            weights: Importance sampling weights
        """
        transitions = []
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Divide total priority into segments for stratified sampling
        segment = self.sum_tree.total / batch_size
        
        # Min priority for weight normalization
        min_prob = self.min_tree.min / self.sum_tree.total
        max_weight = (min_prob * len(self)) ** (-self.beta)
        
        for i in range(batch_size):
            # Sample uniformly from segment
            low = segment * i
            high = segment * (i + 1)
            s = random.uniform(low, high)
            
            idx, priority, transition = self.sum_tree.get(s)
            
            # Calculate importance sampling weight
            prob = priority / self.sum_tree.total
            weight = (prob * len(self)) ** (-self.beta)
            weights[i] = weight / max_weight  # Normalize
            
            indices[i] = idx
            transitions.append(transition)
        
        self._step += 1
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled transitions
        
        Args:
            indices: Tree indices from sampling
            td_errors: New TD-errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.sum_tree.update(idx, priority)
            
            data_idx = idx - self.capacity + 1
            self.min_tree.update(data_idx, priority)
            
            self._max_priority = max(self._max_priority, priority)
    
    def to_tensors(
        self,
        transitions: List[Transition],
        weights: np.ndarray,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Convert transitions to PyTorch tensors
        
        Returns:
            Dictionary with state, action, reward, next_state, done, weights tensors
        """
        states = np.stack([t.state for t in transitions])
        continuous_actions = np.stack([t.continuous_action for t in transitions])
        discrete_actions = np.stack([t.discrete_action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        
        return {
            "states": torch.tensor(states, dtype=torch.float32, device=device),
            "continuous_actions": torch.tensor(continuous_actions, dtype=torch.float32, device=device),
            "discrete_actions": torch.tensor(discrete_actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_states": torch.tensor(next_states, dtype=torch.float32, device=device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=device),
            "weights": torch.tensor(weights, dtype=torch.float32, device=device),
        }


class UniformReplayBuffer:
    """
    Standard replay buffer with uniform sampling
    
    Simpler alternative to prioritized replay for comparison/debugging.
    """
    
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def push(self, transition: Transition, td_error: Optional[float] = None):
        """Add transition to buffer"""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch uniformly
        
        Returns same interface as PrioritizedReplayBuffer for compatibility
        """
        transitions = random.sample(list(self.buffer), batch_size)
        indices = np.zeros(batch_size, dtype=np.int64)  # Not used
        weights = np.ones(batch_size, dtype=np.float32)  # Uniform weights
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """No-op for uniform buffer"""
        pass
    
    def to_tensors(
        self,
        transitions: List[Transition],
        weights: np.ndarray,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """Convert to tensors (same as prioritized)"""
        states = np.stack([t.state for t in transitions])
        continuous_actions = np.stack([t.continuous_action for t in transitions])
        discrete_actions = np.stack([t.discrete_action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        
        return {
            "states": torch.tensor(states, dtype=torch.float32, device=device),
            "continuous_actions": torch.tensor(continuous_actions, dtype=torch.float32, device=device),
            "discrete_actions": torch.tensor(discrete_actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_states": torch.tensor(next_states, dtype=torch.float32, device=device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=device),
            "weights": torch.tensor(weights, dtype=torch.float32, device=device),
        }


def create_replay_buffer(
    capacity: int = 1_000_000,
    use_prioritized: bool = True,
    **kwargs
) -> PrioritizedReplayBuffer | UniformReplayBuffer:
    """
    Factory function to create replay buffer
    
    Args:
        capacity: Buffer capacity
        use_prioritized: Whether to use prioritized replay
        **kwargs: Additional arguments for PrioritizedReplayBuffer
        
    Returns:
        Replay buffer instance
    """
    if use_prioritized:
        return PrioritizedReplayBuffer(capacity=capacity, **kwargs)
    else:
        return UniformReplayBuffer(capacity=capacity)
