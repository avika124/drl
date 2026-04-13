"""
===== STEP 1: P-MODEL TRAINING =====
[train.py]
Description: On-policy SAC training using MockCampaignEnv (synthetic data).
Data Flow: MockCampaignEnv -> SACAgent -> checkpoints/final_model
"""
# QA/Testing: Set True to enable input/output logging for traceability
_QA_IO_LOGGING = True

import os
import numpy as np
import torch
import logging
from datetime import datetime
from collections import deque

# Import your modules
from .config import DRLConfig, TrainingConfig, GuardrailConfig
from .sac_agent import SACAgent
from .replay_buffer import PrioritizedReplayBuffer, Transition, create_replay_buffer
from .state_action import CampaignState, ActionSpace
from .networks import ActorNetwork, CriticNetwork

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingLoop")

class MockCampaignEnv:
    """
    [MockCampaignEnv]
    Description: Simulates ad platform dynamics for synthetic training (no external data).
    Input: Internal random state; no files/tables.
    Output: CampaignState observations -> consumed by SACAgent.select_action(), env.step().
    """
    def __init__(self):
        self.state_dim = 42  # Matches CampaignState.state_dim()
        self.current_step = 0
        self.max_steps = 100

        # Internal hidden state
        self.true_cvr = 0.05
        self.true_ctr = 0.02
        self.market_price = 1.0

        # Spend simulation
        self.daily_spend = 0.0
        self.daily_budget = 0.0
        self.total_spend = 0.0

    def reset(self) -> CampaignState:
        self.current_step = 0
        self.true_cvr = np.random.uniform(0.03, 0.07)
        self.daily_spend = np.random.uniform(100, 2000)
        self.daily_budget = self.daily_spend * np.random.uniform(1.0, 3.0)
        self.total_spend = 0.0
        return self._get_observation()

    def _get_observation(self) -> CampaignState:
        # Create a dummy state with some noise
        return CampaignState(
            ctr=self.true_ctr + np.random.normal(0, 0.001),
            cvr=self.true_cvr + np.random.normal(0, 0.002),
            roas=2.5 + np.random.normal(0, 0.5),
            cpa=15.0 + np.random.normal(0, 2.0),
            hour_of_day=0.5,
            day_of_week=0.1,
            budget_utilization=self.current_step / self.max_steps,
            log_daily_spend=np.log1p(self.daily_spend) / np.log1p(100_000),
            log_total_campaign_spend=np.log1p(self.total_spend) / np.log1p(10_000_000),
            log_daily_budget=np.log1p(self.daily_budget) / np.log1p(100_000),
            # Audience segmentation
            segment_count=np.random.uniform(0.1, 0.5),
            top_segment_roas=np.random.uniform(0.3, 0.8),
            avg_frequency=np.random.uniform(0.1, 0.4),
            # Constraints (simulated targets)
            target_cpa_norm=np.log1p(25.0) / np.log1p(1000.0),
            min_roas_norm=0.2,
            daily_budget_limit_norm=np.log1p(self.daily_budget) / np.log1p(100_000),
        )

    def step(self, action: ActionSpace):
        self.current_step += 1

        # Update spend tracking based on budget adjustment
        self.daily_spend *= (1 + action.budget_adjustment * 0.1)
        self.daily_spend = max(1.0, self.daily_spend)
        self.total_spend += self.daily_spend

        # --- Simulate Dynamics ---
        # 1. Bid Adjustment Impact
        # Higher bid -> More volume, Higher CPA, slightly better CTR (better placement)
        bid_impact = action.bid_adjustment
        self.market_price *= (1 + bid_impact * 0.1)
        
        # 2. Audience Action Impact
        # Refine (2) -> Higher CVR, Lower Volume
        # Expand (1) -> Lower CVR, Higher Volume
        if action.audience_action == 2: # Refine
            self.true_cvr *= 1.05
        elif action.audience_action == 1: # Expand
            self.true_cvr *= 0.95
            
        # 3. Creative Action Impact
        # Rotate (1) -> Temporary CTR boost
        if action.creative_action == 1:
            self.true_ctr *= 1.02

        # --- Calculate Reward ---
        # Simple proxy reward: ROAS + Volume Bonus - CPA Penalty
        next_state = self._get_observation()
        
        # Reward function: Revenue - Cost
        # Simulated Revenue = Volume * CVR * Value
        # Simulated Cost = Volume * Market Price
        revenue = 100 * self.true_cvr * 50 # Assume $50 AOV
        cost = 100 * self.market_price
        
        profit = revenue - cost
        reward = profit / 100.0 # Normalize roughly to [-1, 1] range
        
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done, {}

def train():
    """
    [train]
    Description: Main training loop - runs SAC on MockCampaignEnv, saves checkpoint.
    Input: DRLConfig, TrainingConfig (internal defaults), MockCampaignEnv (synthetic).
    Output: checkpoints/final_model/agent.pt -> used by load_sac_for_inference(), SafeDRLAgent.
    """
    # ----- INPUT LOGGING -----
    if _QA_IO_LOGGING:
        logger.info("[IO] INPUT: No external files. Params: state_dim=42, batch_size=64, min_buffer=100, use_per=True")
    
    # 1. Configuration
    drl_config = DRLConfig(
        state_dim=42,
        continuous_action_dim=2, # Bid, Budget
        discrete_action_dims=[4, 4], # Audience(4), Creative(4)
        hidden_dim=256,
        auto_entropy_tuning=True
    )
    train_config = TrainingConfig(
        batch_size=64,
        min_buffer_size=100, # Small for testing
        use_per=True
    )
    
    # 2. Initialization
    env = MockCampaignEnv()
    agent = SACAgent(drl_config, train_config, device="cpu") # Force CPU for demo
    buffer = create_replay_buffer(
        capacity=10000, 
        use_prioritized=True
    )
    
    logger.info("Initialization Complete. Starting Training Loop...")
    
    # 3. Training Loop
    num_episodes = 50
    global_step = 0
    recent_rewards = deque(maxlen=10)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # A. Select Action
            # During training, we use stochastic policy (exploration)
            # SafeDRLAgent wrapper would usually handle this in prod
            action_obj = agent.select_action(state, deterministic=False)
            
            # B. Execute in Environment
            # Note: action_obj contains indices for discrete actions
            next_state, reward, done, _ = env.step(action_obj)
            
            # C. Store Transition
            # We must store the raw indices for discrete actions to save space
            # The patched Critic will handle encoding them later
            transition = Transition(
                state=state.to_tensor().numpy(), # Convert to numpy for buffer
                continuous_action=np.array([action_obj.bid_adjustment, action_obj.budget_adjustment]),
                discrete_action=np.array([action_obj.audience_action, action_obj.creative_action]),
                reward=reward,
                next_state=next_state.to_tensor().numpy(),
                done=done
            )
            buffer.push(transition)
            
            # D. Update Agent
            if len(buffer) > train_config.min_buffer_size:
                metrics = agent.update(buffer, batch_size=train_config.batch_size)
                
                if global_step % 100 == 0:
                    logger.info(f"Step {global_step} | Actor Loss: {metrics.get('actor_loss', 0):.4f} | Critic Loss: {metrics.get('critic_loss', 0):.4f}")

            state = next_state
            episode_reward += reward
            global_step += 1
            
        recent_rewards.append(episode_reward)
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        if episode % 5 == 0:
            logger.info(f"Episode {episode} | Reward: {episode_reward:.2f} | Avg Reward (10): {avg_reward:.2f}")

    # 4. Save Model (support M1_OUTPUT_DIR for pipeline runs)
    out_dir = os.environ.get("M1_OUTPUT_DIR", "checkpoints/final_model")
    agent.save(out_dir)
    # ----- OUTPUT LOGGING -----
    if _QA_IO_LOGGING:
        logger.info(f"[IO] OUTPUT: {out_dir}/agent.pt, training_info.json | Next: load_sac_for_inference(), SafeDRLAgent")
    logger.info("Training Complete. Model Saved.")

if __name__ == "__main__":
    train()