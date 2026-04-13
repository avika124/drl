# DRL Advertising Optimization System -- Architecture Reference

**Module**: `backend/services/ai_optimization/drl/`
**Version**: 2.1.0
**Files**: 25 Python modules

---

## Table of Contents

1. [System Overview and M1-M5 Architecture](#1-system-overview-and-m1-m5-architecture)
2. [File Index and Dependencies](#2-file-index-and-dependencies)
3. [Configuration Layer](#3-configuration-layer)
4. [State, Action, and Data Structures](#4-state-action-and-data-structures)
5. [Network Architectures](#5-network-architectures)
6. [Core Agent: SAC Implementation](#6-core-agent-sac-implementation)
7. [Replay Buffers](#7-replay-buffers)
8. [Reward Functions](#8-reward-functions)
9. [Training Pipelines](#9-training-pipelines)
10. [Safety and Production Layer](#10-safety-and-production-layer)
11. [Hybrid DRL+LLM Architecture](#11-hybrid-drlllm-architecture)
12. [Cross-Platform Optimization](#12-cross-platform-optimization)
13. [X-Model (Cross-Platform DRL Agent)](#13-x-model-cross-platform-drl-agent)
14. [Platform Model Registry (P-Models)](#14-platform-model-registry-p-models)
15. [Cross-Platform DRL Engine (Orchestrator)](#15-cross-platform-drl-engine-orchestrator)
16. [Continuous Learning](#16-continuous-learning)
17. [A/B Testing Framework](#17-ab-testing-framework)
18. [Explainable AI (xAI) Narrator](#18-explainable-ai-xai-narrator)
19. [Audience Constraints](#19-audience-constraints)
20. [Campaign Forecasting](#20-campaign-forecasting)
21. [Forecast Feedback Loop](#21-forecast-feedback-loop)
22. [BigQuery Data Loader](#22-bigquery-data-loader)
23. [Cross-File Data Flow](#23-cross-file-data-flow)

---

## 1. System Overview and M1-M5 Architecture

### Hierarchical Design

The system uses a two-tier DRL architecture -- per-platform P-Models for campaign-level optimization and a cross-platform X-Model for portfolio-level budget allocation -- integrated with an LLM layer for tactical execution.

```
M5 -- X-Execution: XModelAgent selects cross-platform allocation
    |
M2 -- P-Execution: Per-platform P-Models (via PlatformModelRegistry)
    |
DRL Macro Layer: Campaign-level strategic decisions
    |
LLM Micro Layer: Tactical execution (creative, messaging, offers)
```

### Training Pipeline

```
M1 -- P-Training: Per-platform SAC training (one model per platform)
M3 -- X-Training Data: Portfolio-level transitions from platform outcomes
M4 -- X-Training: XModelAgent training on M3 data
```

### Module Flow

```
M1: train.py / train_bigquery_offline.py --> sac_agent.py --> platform_model_registry.py
M2: platform_model_registry.select_action() --> hybrid_optimizer.py --> safe_agent.py
M3: x_training_data.py (XTrainingDataBuilder) collects portfolio snapshots
M4: x_training.py (XModelTrainer) trains XModelAgent from M3 transitions
M5: cross_platform_drl_engine.py uses x_model.XModelAgent for allocation
```

---

## 2. File Index and Dependencies

| File | Lines | Purpose | Key Imports From |
|------|-------|---------|------------------|
| `__init__.py` | 150 | Public API, exports 60+ symbols | All other modules |
| `config.py` | 362 | Central configuration (6 dataclasses, 2 enums) | numpy |
| `state_action.py` | 570 | State/action definitions (42-dim state, hybrid action) | config |
| `networks.py` | 362 | Neural network architectures (Actor, Critic, Value) | torch |
| `sac_agent.py` | 573 | SAC agent with CQL support | networks, config, replay_buffer |
| `replay_buffer.py` | 423 | PER with SumTree/MinTree | state_action |
| `reward_functions.py` | 593 | Multi-objective reward (11 components) | config |
| `offline_trainer.py` | 654 | Offline CQL training from historical data | sac_agent, replay_buffer, reward_functions |
| `safe_agent.py` | 610 | Production safety wrapper + rollback | sac_agent, config |
| `hybrid_optimizer.py` | 684 | DRL+LLM hybrid pipeline | safe_agent, state_action, xai_narrator, benchmark_model |
| `continuous_learning.py` | 740 | Online/batch learning engine | sac_agent, replay_buffer, forecast_feedback |
| `ab_testing.py` | 1114 | Experiment lifecycle + statistical analysis | config, xai_narrator |
| `cross_platform_optimizer.py` | 1619 | Heuristic budget allocation + portfolio orchestration | hybrid_optimizer, safe_agent, audience_constraints, platform_model_registry, x_model |
| `cross_platform_drl_engine.py` | 832 | Top-level DRL orchestrator (3 strategy modes) | cross_platform_optimizer, x_model, x_training, x_training_data, xai_narrator, ab_testing |
| `x_model.py` | 559 | Cross-platform DRL agent (70-dim state, 5-dim action) | torch |
| `x_training.py` | 248 | M4 trainer (behavior cloning + SAC) | x_model |
| `x_training_data.py` | 260 | M3 transition builder from portfolio snapshots | x_model |
| `xai_narrator.py` | 668 | Plain-English narrative generation | (standalone) |
| `audience_constraints.py` | 268 | Audience segment budget allocation + frequency capping | state_action |
| `benchmark_model.py` | 215 | Ridge regression campaign forecaster | numpy |
| `forecast_feedback.py` | 255 | Forecast-vs-actual accuracy tracking + auto-refit | benchmark_model |
| `bigquery_loader.py` | 208 | BigQuery data extraction + normalization | google.cloud.bigquery |
| `platform_model_registry.py` | 308 | One SACAgent per platform (M1/M2 infra) | sac_agent, config |
| `train.py` | 192 | Mock environment + training loop | sac_agent, replay_buffer |
| `train_bigquery_offline.py` | 394 | End-to-end BigQuery offline training pipeline | bigquery_loader, sac_agent, reward_functions |

---

## 3. Configuration Layer

**File**: `config.py` (362 lines)

### Enums

**`OptimizationGoal(Enum)`**
- `ROAS`, `CPA`, `CONVERSIONS`, `CTR`, `REVENUE`, `PROFIT`

**`ActionType(Enum)`**
- `BID_ADJUSTMENT`, `BUDGET_ADJUSTMENT`, `AUDIENCE_EXPANSION`, `AUDIENCE_REFINEMENT`, `CREATIVE_ROTATION`, `PLATFORM_SHIFT`

**`AllocationStrategy(Enum)`**
- `DRL_PRIMARY` -- X-Model primary, heuristic fallback
- `HEURISTIC_PRIMARY` -- Heuristic primary (legacy default)
- `DUAL_BENCHMARK` -- Run both, compare, apply DRL

### DRLConfig

Core model hyperparameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `state_dim` | int | 42 | 33 original + 3 spend + 3 audience + 3 constraint |
| `continuous_action_dim` | int | 2 | bid_adjustment, budget_adjustment |
| `discrete_action_dims` | List[int] | [4, 4] | audience (4 options), creative (4 options) |
| `hidden_dim` | int | 256 | Hidden layer size |
| `num_hidden_layers` | int | 3 | Number of hidden layers |
| `activation` | str | "relu" | Activation function |
| `use_layer_norm` | bool | True | Layer normalization |
| `dropout_rate` | float | 0.1 | Dropout |
| `gamma` | float | 0.99 | Discount factor |
| `tau` | float | 0.005 | Soft target update coefficient |
| `alpha` | float | 0.2 | Entropy coefficient |
| `auto_entropy_tuning` | bool | True | Automatic entropy tuning |
| `target_entropy` | Optional[float] | None | Auto-computed if None |
| `actor_lr` | float | 3e-4 | Actor learning rate |
| `critic_lr` | float | 3e-4 | Critic learning rate |
| `alpha_lr` | float | 3e-4 | Alpha learning rate |
| `model_dir` | str | "models/drl" | Checkpoint directory |
| `checkpoint_frequency` | int | 1000 | Steps between checkpoints |

`__post_init__`: Computes hybrid target entropy as `-dim(A_cont) + 0.98 * sum(ln(d_i))` where d_i are discrete action dimensions.

### TrainingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | 256 | Training batch size |
| `replay_buffer_size` | int | 1,000,000 | Maximum buffer capacity |
| `min_buffer_size` | int | 10,000 | Minimum samples before training |
| `num_offline_epochs` | int | 100 | Offline training epochs |
| `steps_per_epoch` | int | 1,000 | Gradient steps per epoch |
| `gradient_steps_per_update` | int | 1 | Gradient steps per env step |
| `online_update_frequency` | int | 4 | Update every N env steps |
| `target_update_frequency` | int | 1 | Target net update frequency |
| `use_cql` | bool | True | Conservative Q-Learning |
| `cql_alpha` | float | 1.0 | CQL regularization weight |
| `cql_num_samples` | int | 10 | CQL action samples |
| `cql_importance_sample` | bool | True | CQL importance sampling |
| `use_per` | bool | True | Prioritized Experience Replay |
| `per_alpha` | float | 0.6 | PER prioritization exponent |
| `per_beta_start` | float | 0.4 | IS weight start |
| `per_beta_end` | float | 1.0 | IS weight end |
| `per_beta_anneal_steps` | int | 100,000 | Beta annealing steps |
| `weight_decay` | float | 1e-4 | L2 regularization |
| `gradient_clip` | float | 1.0 | Gradient clipping norm |
| `initial_exploration_steps` | int | 10,000 | Pure exploration steps |
| `exploration_noise_std` | float | 0.1 | Exploration noise |
| `validation_frequency` | int | 5,000 | Steps between validation |
| `validation_episodes` | int | 100 | Validation episodes |
| `early_stopping_patience` | int | 10 | Epochs without improvement |
| `early_stopping_min_delta` | float | 0.01 | Minimum improvement |

### GuardrailConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_bid_increase_pct` | float | 0.50 | Max 50% bid increase |
| `max_bid_decrease_pct` | float | 0.30 | Max 30% bid decrease |
| `min_bid` | float | 0.01 | Minimum bid ($) |
| `max_bid` | float | 100.0 | Maximum bid ($) |
| `max_budget_increase_pct` | float | 0.30 | Max 30% budget increase |
| `max_budget_decrease_pct` | float | 0.30 | Max 30% budget decrease |
| `min_daily_budget` | float | 10.0 | Minimum daily budget ($) |
| `max_daily_budget` | float | 100,000 | Maximum daily budget ($) |
| `min_hours_between_actions` | float | 4.0 | Cooldown period |
| `max_actions_per_day` | int | 6 | Rate limit |
| `min_confidence_for_action` | float | 0.7 | Minimum confidence to act |
| `min_confidence_for_auto_apply` | float | 0.85 | Auto-apply threshold |
| `max_spend_increase_per_action` | float | 1,000 | Max spend increase ($) |
| `emergency_stop_roas_threshold` | float | 0.5 | ROAS emergency stop |
| `emergency_stop_cpa_multiplier` | float | 3.0 | CPA emergency stop |
| `max_exploration_rate` | float | 0.2 | Maximum exploration |
| `min_exploration_rate` | float | 0.01 | Minimum exploration |
| `exploration_decay_rate` | float | 0.995 | Decay per step |
| `enable_auto_rollback` | bool | True | Auto-rollback enabled |
| `rollback_observation_hours` | int | 24 | Observation window |
| `rollback_performance_threshold` | float | -0.15 | Rollback if -15% |

Methods:
- `validate_bid_adjustment(current_bid, adjustment) -> (constrained_adjustment, was_constrained)` -- Enforces percentage bounds and absolute limits
- `validate_budget_adjustment(current_budget, adjustment) -> (constrained_adjustment, was_constrained)` -- Same for budget

### RewardConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `roas_weight` | float | 0.4 | Primary ROAS weight |
| `cpa_weight` | float | 0.3 | Primary CPA weight |
| `conversion_weight` | float | 0.2 | Primary conversion weight |
| `ctr_weight` | float | 0.1 | Primary CTR weight |
| `efficiency_bonus_weight` | float | 0.1 | ROAS efficiency bonus |
| `volume_bonus_weight` | float | 0.05 | Conversion volume bonus |
| `ltv_bonus_weight` | float | 0.15 | LTV bonus |
| `budget_violation_penalty` | float | -1.0 | Budget constraint violation |
| `cpa_violation_penalty` | float | -0.5 | CPA constraint violation |
| `roas_violation_penalty` | float | -0.5 | ROAS constraint violation |
| `action_magnitude_penalty` | float | 0.01 | Large action penalty |
| `action_frequency_penalty` | float | 0.05 | Frequent action penalty |
| `roas_target` | float | 2.0 | Target ROAS for bonus |
| `roas_excellent` | float | 4.0 | Excellent ROAS threshold |
| `cpa_target_multiplier` | float | 1.5 | CPA penalty multiplier |
| `spend_efficiency_weight` | float | 0.1 | Spend efficiency component |
| `diminishing_return_threshold` | float | 0.7 | Diminishing returns start |
| `spend_efficiency_alpha` | float | 0.5 | Concavity parameter |
| `gamma` | float | 0.99 | Temporal discount |
| `normalize_rewards` | bool | True | Running normalization |
| `reward_scale` | float | 1.0 | Reward scaling |
| `reward_clip` | Optional[float] | 10.0 | Reward clipping bound |

### FeatureConfig

| Field | Type | Default |
|-------|------|---------|
| `include_hour_of_day` | bool | True |
| `include_day_of_week` | bool | True |
| `include_day_of_month` | bool | True |
| `include_is_weekend` | bool | True |
| `include_is_holiday` | bool | True |
| `rolling_windows` | List[int] | [1, 7, 14, 30] |
| `metric_features` | List[str] | [ctr, cvr, cpc, cpm, roas, cpa, impressions, clicks, conversions, spend, revenue] |
| `include_trend_features` | bool | True |
| `include_volatility_features` | bool | True |
| `include_competitive_features` | bool | True |
| `include_audience_features` | bool | True |
| `include_creative_features` | bool | True |
| `include_predicted_ltv` | bool | True |
| `include_predicted_cvr` | bool | True |
| `include_fatigue_score` | bool | True |
| `include_audience_quality` | bool | True |
| `normalize_features` | bool | True |
| `normalization_method` | str | "standard" |
| `clip_outliers` | bool | True |
| `outlier_std` | float | 3.0 |

### CrossPlatformStrategyConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | AllocationStrategy | DRL_PRIMARY | Strategy selection |
| `min_confidence_for_drl` | float | 0.40 | Fallback threshold |
| `min_training_steps` | int | 500 | Minimum X-Model steps |
| `auto_collect_training_data` | bool | True | Auto training data |
| `retrain_snapshot_threshold` | int | 50 | Retrain after N snapshots |
| `retrain_min_transitions` | int | 30 | Minimum transitions |
| `retrain_epochs` | int | 10 | Retrain epochs |
| `retrain_steps_per_epoch` | int | 100 | Steps per epoch |
| `benchmark_history_size` | int | 500 | Dual-run history |
| `checkpoint_dir` | str | "models/x_model" | Model persistence |

---

## 4. State, Action, and Data Structures

**File**: `state_action.py` (570 lines)

### Constants

```python
MAX_DAILY_SPEND = 100_000
MAX_TOTAL_SPEND = 10_000_000
MAX_DAILY_BUDGET = 100_000
```

### Discrete Action Enums

**`AudienceAction(IntEnum)`**: HOLD=0, EXPAND=1, REFINE=2, EXCLUDE=3

**`CreativeAction(IntEnum)`**: HOLD=0, ROTATE=1, PAUSE_UNDERPERFORMING=2, TEST_NEW=3

**`MessagingTone(IntEnum)`**: CONSISTENT=0, AGGRESSIVE_GROWTH=1, EFFICIENCY_FOCUSED=2, URGENCY=3, FRESH_ANGLE=4

### CampaignState (42-dimensional)

The core state representation. All values are normalized to approximately [0, 1].

| Index | Field | Category |
|-------|-------|----------|
| 0 | `ctr` | Core metrics |
| 1 | `cvr` | Core metrics |
| 2 | `roas` | Core metrics |
| 3 | `cpa` | Core metrics |
| 4 | `cpc` | Core metrics |
| 5 | `cpm` | Core metrics |
| 6 | `spend_velocity` | Volume |
| 7 | `impression_volume` | Volume |
| 8 | `click_volume` | Volume |
| 9 | `conversion_volume` | Volume |
| 10 | `hour_of_day` | Temporal |
| 11 | `day_of_week` | Temporal |
| 12 | `day_of_month` | Temporal |
| 13 | `is_weekend` | Temporal |
| 14 | `is_holiday` | Temporal |
| 15 | `days_remaining` | Temporal |
| 16 | `ctr_trend_7d` | Trends |
| 17 | `cvr_trend_7d` | Trends |
| 18 | `roas_trend_7d` | Trends |
| 19 | `cpa_trend_7d` | Trends |
| 20 | `spend_trend_7d` | Trends |
| 21 | `impression_share` | Competitive |
| 22 | `auction_pressure` | Competitive |
| 23 | `competitive_position` | Competitive |
| 24 | `audience_quality_score` | ML features |
| 25 | `creative_fatigue_score` | ML features |
| 26 | `predicted_cvr` | ML features |
| 27 | `predicted_ltv` | ML features |
| 28 | `propensity_score` | ML features |
| 29 | `optimization_goal_encoding` | Context |
| 30 | `platform_encoding` | Context |
| 31 | `campaign_maturity` | Context |
| 32 | `budget_utilization` | Context |
| 33 | `log_daily_spend` | Absolute spend (log1p normalized) |
| 34 | `log_total_campaign_spend` | Absolute spend (log1p normalized) |
| 35 | `log_daily_budget` | Absolute spend (log1p normalized) |
| 36 | `segment_count` | Audience segmentation (count/10) |
| 37 | `top_segment_roas` | Audience segmentation |
| 38 | `avg_frequency` | Audience segmentation (freq/10) |
| 39 | `target_cpa_norm` | Constraint features |
| 40 | `min_roas_norm` | Constraint features |
| 41 | `daily_budget_limit_norm` | Constraint features |

Methods:
- `to_tensor(device) -> Tensor` -- Converts all fields to tensor in index order
- `from_tensor(tensor) -> CampaignState` -- Reconstructs from tensor
- `state_dim() -> int` -- Returns 42
- `from_campaign_metrics(metrics, ...) -> CampaignState` -- Factory method from raw campaign data

### ActionSpace (Hybrid: Continuous + Discrete)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `bid_adjustment` | float | [-1, 1] | Continuous bid change |
| `budget_adjustment` | float | [-1, 1] | Continuous budget change |
| `audience_action` | int | [0, 3] | Discrete audience action |
| `creative_action` | int | [0, 3] | Discrete creative action |
| `confidence` | float | [0, 1] | Agent confidence |
| `entropy` | float | >= 0 | Action entropy |
| `q_value` | float | any | Critic Q-value |
| `action_id` | str | UUID | Unique identifier |
| `timestamp` | str | ISO | Creation time |

Methods:
- `to_tensor() -> (continuous_tensor, discrete_tensor)` -- Separate continuous [2] and discrete [2] tensors
- `from_tensors(continuous, discrete, confidence, entropy, q_value) -> ActionSpace` -- Reconstruct
- `scale_to_bounds(guardrails) -> ActionSpace` -- Asymmetric scaling: positive bid scaled by `max_bid_increase_pct`, negative by `max_bid_decrease_pct`; same for budget

### DRLDirective (DRL-to-LLM Bridge)

Strategic output that constrains LLM tactical execution. The DRL decides WHAT to do; the LLM decides HOW to communicate it.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `budget_allocation` | float | [-1, 1] | Budget change percentage (from action) |
| `bid_strategy` | float | [-1, 1] | Bid change percentage (from action) |
| `audience_priority` | str | AudienceAction names | "hold"/"expand"/"refine"/"exclude" |
| `creative_direction` | str | CreativeAction names | "hold"/"rotate"/"pause_underperforming"/"test_new" |
| `messaging_tone` | str | MessagingTone names | Derived from action+state: "consistent"/"aggressive_growth"/"efficiency_focused"/"urgency"/"fresh_angle" |
| `urgency_level` | float | [0, 1] | Derived: 0.8 if growth, 0.3 if efficiency, 0.5 if fatigued |
| `value_emphasis` | float | [0, 1] | 0=growth-focused, 1=efficiency-focused |
| `max_offer_discount` | float | [0, 0.25] | Max discount LLM can offer (based on ROAS: 0.25 if >0.7, 0.15 if >0.5, 0.05 otherwise) |
| `product_focus` | str | - | Product category to emphasize |
| `audience_segment` | str | - | Target segment for personalization |
| `strategic_confidence` | float | [0, 1] | DRL confidence (from action) |
| `recommended_test` | bool | - | True if confidence < 0.7, suggests A/B test |

Methods:
- `from_action(action, state, campaign_context) -> DRLDirective` -- Derives messaging tone from bid/ROAS/fatigue signals, max discount from ROAS headroom
- `to_llm_prompt_context() -> str` -- Formats as structured text for LLM prompt injection (budget direction, bid direction, audience/creative strategy, tone, urgency, constraints, confidence)

### Transition (Replay Buffer Record)

| Field | Type | Description |
|-------|------|-------------|
| `state` | ndarray | State vector (42-dim) |
| `continuous_action` | ndarray | Continuous actions [2] |
| `discrete_action` | ndarray | Discrete action indices [2] |
| `reward` | float | Scalar reward |
| `next_state` | ndarray | Next state vector (42-dim) |
| `done` | bool | Episode termination flag |
| `campaign_id` | Optional[str] | Campaign identifier |
| `timestamp` | Optional[str] | Action timestamp |
| `td_error` | float | TD error for PER |
| `priority` | float | Sampling priority |

---

## 5. Network Architectures

**File**: `networks.py` (362 lines)

### Utility: create_mlp()

```
create_mlp(input_dim, output_dim, hidden_dims, activation, use_layer_norm, dropout_rate) -> nn.Sequential
```

Supported activations: relu, tanh, leaky_relu, elu, gelu. Applies LayerNorm before activation, Dropout after.

### ActorNetwork

The actor outputs both continuous and discrete actions through a shared feature extractor with separate heads.

```
Architecture:
    Input: state [batch, state_dim=42]
        |
    Shared Backbone: create_mlp(42 -> 256 -> 256, no output layer)
        |
    +--> mean_head: Linear(256 -> 2)        [continuous mean]
    +--> log_std_head: Linear(256 -> 2)      [continuous log-std, clamped to [-20, 2]]
    +--> discrete_heads: [Linear(256 -> 4), Linear(256 -> 4)]  [discrete logits]
```

Weight initialization: `orthogonal_(gain=sqrt(2))` for hidden layers, `orthogonal_(gain=0.01)` for output heads.

**`sample(state, temperature=1.0, deterministic=False)`** returns:
1. `continuous_action` -- Gaussian reparameterization + tanh squashing
2. `discrete_soft` -- Gumbel-Softmax (tau=temperature, hard=False)
3. `discrete_indices` -- argmax of soft outputs
4. `total_log_prob` -- Sum of continuous log-prob (with tanh correction: `-log(1 - tanh(x)^2)`) and discrete log-prob
5. `total_entropy` -- Sum of continuous entropy (from Gaussian) and discrete entropy (from categorical)

### CriticNetwork (Twin Q)

Two independent Q-networks for the double-Q trick.

```
Architecture (each Q-network):
    Input: concat(state, continuous_actions, one_hot(discrete_actions))
           [batch, 42 + 2 + 4 + 4 = 52]
        |
    create_mlp(52 -> 256 -> 256 -> 256 -> 1)

Returns: (q1, q2)  [batch, 1] each
```

Auto-detects discrete input format: if indices (from buffer), applies `_encode_indices()` for one-hot encoding; if already soft vectors (from actor), passes through directly.

### ValueNetwork

Optional SAC v1 state value network.

```
Architecture:
    Input: state [batch, 42]
        |
    create_mlp(42 -> 256 -> 256 -> 256 -> 1)

Returns: v [batch, 1]
```

### Factory: create_networks()

```python
create_networks(config: DRLConfig) -> (ActorNetwork, CriticNetwork, CriticNetwork)
```
Returns (actor, critic, critic_target) where critic_target is initialized as a copy of critic.

---

## 6. Core Agent: SAC Implementation

**File**: `sac_agent.py` (573 lines)

### SACAgent

The core Soft Actor-Critic agent supporting hybrid action spaces and Conservative Q-Learning.

**Constructor**: `__init__(config, training_config, guardrail_config, device)`
- Creates actor, critic, critic_target via `create_networks()`
- Creates Adam optimizers with `weight_decay` for actor, critic
- If `auto_entropy_tuning=True`: creates `log_alpha` parameter (init=0) and alpha optimizer

**`select_action(state: CampaignState, deterministic=False) -> ActionSpace`**
- Converts state to tensor, runs `actor.sample()`
- Computes confidence: `1 - entropy / max_entropy` where `max_entropy = -target_entropy`
- Returns ActionSpace with bid/budget adjustments, discrete actions, confidence, entropy, Q-value

**`update(replay_buffer, batch_size) -> Dict[str, float]`**
1. Samples from PER buffer: `(transitions, indices, weights)`
2. `_update_critic()`: Computes TD target `r + gamma * (1-d) * (min(Q1', Q2') - alpha * log_prob')`, MSE loss weighted by IS weights, optional CQL loss
3. `_update_actor()`: Maximize `min(Q1, Q2) - alpha * log_prob`
4. `_update_alpha()`: Minimize `alpha * (log_prob + target_entropy).detach()`
5. `_soft_update_target()`: Polyak averaging `theta_target = tau * theta + (1-tau) * theta_target`
6. Updates PER priorities with new TD errors

**CQL Loss computation** (`_compute_cql_loss()`):
- Samples `cql_num_samples` random and policy actions
- Computes `logsumexp(Q(s, a_random), Q(s, a_policy)) - Q(s, a_data)`
- Optional importance sampling correction

### CQLLoss (Standalone)

Separate class for CQL loss with importance sampling, using repeated state expansion for multiple action evaluations.

### load_sac_for_inference()

```python
load_sac_for_inference(checkpoint_path, device="cpu") -> (SACAgent, List[str])
```
Loads checkpoint, sets eval mode. Returns agent and feature list.

---

## 7. Replay Buffers

**File**: `replay_buffer.py` (423 lines)

### SumTree

Binary tree for O(log n) priority-based sampling. Methods:
- `update(index, priority)` -- Update leaf and propagate
- `sample(value) -> index` -- Find leaf for cumulative priority value
- Properties: `total`, `min`, `max`

### MinTree

Binary tree for O(log n) minimum priority lookup.

### PrioritizedReplayBuffer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity` | 1,000,000 | Maximum transitions |
| `alpha` | 0.6 | Prioritization exponent (0=uniform, 1=full priority) |
| `beta_start` | 0.4 | Initial IS weight correction |
| `beta_end` | 1.0 | Final IS weight correction |
| `beta_anneal_steps` | 100,000 | Steps to anneal beta |

Key methods:
- `push(transition)` -- Add with max existing priority (or 1.0 initial)
- `sample(batch_size) -> (List[Transition], List[int], Tensor)` -- Stratified sampling: divides total priority into `batch_size` segments, samples one from each. Returns transitions, buffer indices, IS weights (normalized by min priority)
- `update_priorities(indices, td_errors)` -- Updates priorities as `|td_error| + epsilon` raised to alpha
- `to_tensors(transitions) -> Dict[str, Tensor]` -- Batch conversion

### UniformReplayBuffer

Simple deque-based buffer with identical interface, uniform random sampling.

### create_replay_buffer()

Factory function: returns PrioritizedReplayBuffer if `use_per=True`, else UniformReplayBuffer.

---

## 8. Reward Functions

**File**: `reward_functions.py` (593 lines)

### RewardComponent (Enum)

`PRIMARY`, `EFFICIENCY_BONUS`, `VOLUME_BONUS`, `LTV_BONUS`, `BUDGET_VIOLATION`, `CPA_VIOLATION`, `ROAS_VIOLATION`, `ACTION_MAGNITUDE`, `ACTION_FREQUENCY`, `SPEND_EFFICIENCY`, `CONSTRAINT_ALIGNMENT`

### MultiObjectiveReward

| Field | Type | Description |
|-------|------|-------------|
| `total` | float | Final scalar reward |
| `primary` | float | Goal-specific improvement |
| `efficiency_bonus` | float | ROAS exceeding targets |
| `volume_bonus` | float | Conversion volume |
| `ltv_bonus` | float | Predicted LTV delta |
| `budget_violation` | float | Budget constraint penalty |
| `cpa_violation` | float | CPA constraint penalty |
| `roas_violation` | float | ROAS constraint penalty |
| `action_magnitude` | float | Large action penalty |
| `action_frequency` | float | Frequent action penalty |
| `spend_efficiency` | float | Diminishing returns signal |
| `constraint_alignment` | float | Constraint respect bonus |
| `metadata` | Dict | Additional info |

### RewardComputer

**`compute(prev_state, action, curr_state, context) -> MultiObjectiveReward`**

11-component reward computation:

1. **Primary** (`_compute_primary_reward`): Goal-specific improvement
   - ROAS: `(curr.roas - prev.roas) / max(prev.roas, 0.01)`
   - CPA: `(prev.cpa - curr.cpa) / max(prev.cpa, 0.01)` (inverted: lower is better)
   - Conversions: `(curr.conversion_volume - prev.conversion_volume) / max(prev, 0.01)`
   - CTR: `(curr.ctr - prev.ctr) / max(prev.ctr, 0.01)`

2. **Efficiency bonus**: `0.1 * (roas - 2.0)` if ROAS > target, extra bonus at 4.0+

3. **Volume bonus**: `0.05 * log1p(conversion_increase)` for positive conversion changes

4. **LTV bonus**: `0.15 * (curr.predicted_ltv - prev.predicted_ltv)`

5. **Spend efficiency** (`_compute_spend_efficiency`):
   - Diminishing returns penalty: `-weight * (utilization - threshold)^alpha` when utilization > threshold
   - Efficiency-at-scale bonus: `+weight * roas * log1p(spend)` when ROAS > 1.5

6. **Budget/CPA/ROAS violation penalties**: -1.0 / -0.5 / -0.5 for constraint breaches

7. **Action magnitude penalty**: `-0.01 * (|bid_adj| + |budget_adj|)` discourages large changes

8. **Action frequency penalty**: `-0.05` if actions within cooldown period

9. **Constraint alignment** (`_compute_constraint_alignment`):
   - CPA within +/-5% of target: +0.1 bonus; 10%+ over: -0.1 penalty
   - ROAS within +/-5% of minimum: +0.1 bonus; 10%+ under: -0.1 penalty
   - Budget within +/-2% of limit: +0.05 bonus; 10%+ over: -0.15 penalty

Running normalization via Welch's online algorithm (mean/variance tracking), reward clipping to +/-10.

### Helper Functions

- `compute_simple_reward(prev, curr, goal) -> float` -- Lightweight single-metric reward
- `compute_shaped_reward(prev, curr, goal, action) -> float` -- Potential-based reward shaping using `gamma * potential(curr) - potential(prev)`

---

## 9. Training Pipelines

### Offline Training (CQL)

**File**: `offline_trainer.py` (654 lines)

**`HistoricalCampaignData`** dataclass:
- `campaign_id`, `organization_id`, `platform`, `optimization_goal`, `daily_metrics` (list of dicts), `actions_taken` (list of dicts), `constraints` (dict)

**`OfflineDataExtractor`**:
- `extract_from_csv(path) -> List[HistoricalCampaignData]`
- `build_transitions(campaigns) -> List[Transition]` -- Iterates campaigns, builds state-action-reward tuples from consecutive daily metrics
- `_extract_state(metrics, campaign) -> CampaignState` -- Computes trends via linear regression on rolling windows, temporal features from dates
- `_infer_action(prev_metrics, curr_metrics) -> ActionSpace` -- Derives actions from CPC and budget changes
- Goal encoding: roas=0.0, cpa=0.25, conversions=0.5, ctr=0.75, revenue=1.0
- Platform encoding: meta=0.0, google=0.2, tiktok=0.4, amazon=0.6, walmart=0.8

**`OfflineTrainer`**:
- `train(transitions, config) -> Dict[str, Any]` -- Epoch loop with CQL loss, PER buffer, validation, early stopping (patience=10), checkpointing every 10 epochs

### BigQuery Offline Training

**File**: `train_bigquery_offline.py` (394 lines)

End-to-end pipeline: BigQuery -> transitions -> behavioral cloning -> CQL offline training.

Key functions:
- `_infer_actions(prev, curr)` -- bid from CPC delta, budget from 0.6*volume_change + 0.4*spend_velocity_change, audience from quality/impression_share, creative from fatigue/CTR
- `_compute_reward(prev, curr, config)` -- Uses RewardComputer with optional CPA penalty, spend-scaled clipping +/-5
- `build_transitions(loader, config)` -- Fetches from BigQuery, orders by campaign_id+created_at, outlier filtering, optional action log CSV lookup
- `behavior_cloning_pretrain(agent, transitions, epochs=5)` -- MSE on `tanh(actor.mean)` for continuous actions, cross-entropy for discrete heads
- `train_from_bigquery(project_id, table, ...)` -- Full pipeline with config: state_dim=42, auto_entropy_tuning=False, alpha=0.2, cql_alpha=0.5, batch_size=256, min_buffer_size=500

### Mock Training

**File**: `train.py` (192 lines)

- `MockCampaignEnv`: Simulated ad platform with state_dim=42. Dynamics: bid increases volume but raises CPA/CTR, audience refine +5% CVR, expand -5% CVR, creative rotate +2% CTR. Reward: `(revenue - cost) / 100`
- `train()`: 50 episodes x 100 steps, batch_size=64, PER buffer size=10K, forces CPU

### X-Model Training (M4)

**File**: `x_training.py` (248 lines)

**`XModelReplayBuffer`**: Simple list-based replay buffer (capacity=100K) with random sampling.

**`XModelTrainer`**:
- `behavior_cloning_pretrain(agent, buffer, epochs=5, batch_size=64)` -- MSE loss on predicted vs actual allocation weights (softmax output vs recorded allocations)
- `train(agent, buffer, epochs, steps_per_epoch, batch_size=64)` -- SAC updates on X-Model buffer with periodic logging. Default: min_buffer_size=50

---

## 10. Safety and Production Layer

**File**: `safe_agent.py` (610 lines)

### ActionStatus (Enum)

`APPROVED`, `MODIFIED`, `BLOCKED`, `REQUIRES_REVIEW`

### ActionValidationResult

| Field | Type | Description |
|-------|------|-------------|
| `original_action` | ActionSpace | Pre-validation action |
| `validated_action` | ActionSpace | Post-validation action |
| `status` | ActionStatus | Validation result |
| `modifications` | List[str] | Applied modifications |
| `blocking_reason` | Optional[str] | Reason if blocked |
| `requires_human_review` | bool | Needs human approval |

### CampaignContext

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `campaign_id` | str | - | Campaign identifier |
| `current_bid` | float | - | Current bid price |
| `current_budget` | float | - | Current daily budget |
| `last_action_at` | Optional[datetime] | None | Last action timestamp |
| `actions_today` | int | 0 | Actions taken today |
| `current_roas` | float | 0 | Current ROAS |
| `current_cpa` | float | 0 | Current CPA |
| `target_cpa` | Optional[float] | None | Target CPA constraint |
| `min_roas` | Optional[float] | None | Minimum ROAS constraint |
| `is_new_campaign` | bool | False | New campaign flag |
| `total_spend` | float | 0 | Total campaign spend |

### ActionValidator

Validation chain:
1. **Cooldown check**: Block if `< min_hours_between_actions` since last action
2. **Rate limit check**: Block if `>= max_actions_per_day`
3. **Confidence threshold**: Block if `< min_confidence_for_action` (0.7)
4. **Bid constraint**: Clamp via `guardrails.validate_bid_adjustment()`
5. **Budget constraint**: Clamp via `guardrails.validate_budget_adjustment()`
6. **Emergency conditions**: Block + flag for review if ROAS < 0.5 or CPA > 3x target
7. **Spend increase cap**: Limit single-action spend increase to $1,000

### SafeDRLAgent

Production wrapper combining:
- **SACAgent** -- Core policy
- **ActionValidator** -- Safety guardrails
- **Bounded exploration**: Gaussian noise clamped to +/-0.2 (bid), +/-0.15 (budget), 10% discrete exploration
- **Audit trail**: Circular buffer of last 10,000 actions with campaign_id, timestamp, status, modifications
- **Rollback capability**: Via `RollbackManager`

### RollbackManager

Monitors pending observations for performance degradation.
- Observation window: 24 hours
- Performance threshold: -15%
- `add_observation(campaign_id, action, pre_metrics)`
- `check_and_rollback(campaign_id, post_metrics) -> Optional[rollback_action]`

---

## 11. Hybrid DRL+LLM Architecture

**File**: `hybrid_optimizer.py` (684 lines)

### OptimizationType (Enum)

`BUDGET_ALLOCATION`, `BID_STRATEGY`, `AUDIENCE_TARGETING`, `CREATIVE_OPTIMIZATION`, `MESSAGING`, `OFFER_GENERATION`

### TacticalExecution

LLM output structure:

| Field | Type |
|-------|------|
| `headline` | str |
| `body_copy` | str |
| `call_to_action` | str |
| `offer_text` | Optional[str] |
| `product_highlights` | List[str] |
| `urgency_elements` | List[str] |
| `personalization_tokens` | List[str] |
| `metadata` | Dict |

### OptimizationResult

Complete optimization output:

| Field | Type | Description |
|-------|------|-------------|
| `directive` | DRLDirective | Strategic DRL output |
| `action` | ActionSpace | Raw DRL action |
| `validation` | ActionValidationResult | Safety validation result |
| `tactical` | TacticalExecution | LLM tactical output |
| `strategic_confidence` | float | DRL confidence |
| `tactical_confidence` | float | LLM confidence |
| `combined_confidence` | float | 0.7 * strategic + 0.3 * tactical |
| `recommended_changes` | List[Dict] | Actionable recommendations |
| `narrative` | Optional[RunNarrative] | xAI explanation |
| `forecast` | Optional[CampaignForecast] | Performance forecast |

### HybridDRLLLMOptimizer

4-phase pipeline:

1. **DRL Strategic Decision**: `SafeDRLAgent.select_action()` -> `ActionSpace` -> `DRLDirective.from_action()`
2. **LLM Tactical Execution**: Builds constrained prompt from `directive.to_llm_prompt_context()`, sends to LLM (GPT-4/Claude placeholder), parses response into `TacticalExecution`
3. **xAI Narrative**: `OptimizationNarrator.narrate_campaign_run()` generates plain-English explanation
4. **Campaign Forecast**: `CampaignForecaster.predict()` generates performance projections

Combined confidence: `0.7 * strategic + 0.3 * tactical`

Recommendations built for: bid adjustment, budget adjustment, audience action, creative action.

### BatchOptimizer

Async parallel campaign optimization:
- `optimize_batch(campaigns, max_concurrent=10) -> List[OptimizationResult]`
- Semaphore-based concurrency control

---

## 12. Cross-Platform Optimization

**File**: `cross_platform_optimizer.py` (1619 lines)

### Platform (Enum)

`META`, `GOOGLE`, `TIKTOK`, `AMAZON`, `WALMART`

### PLATFORM_ENCODING

```python
{"meta": 0.0, "google": 0.2, "tiktok": 0.4, "amazon": 0.6, "walmart": 0.8}
```

### CrossPlatformConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_single_shift_pct` | float | 0.20 | Max per-platform shift per cycle |
| `min_platform_budget_pct` | float | 0.05 | Floor: every platform >= 5% |
| `max_platform_budget_pct` | float | 0.80 | Ceiling: no platform > 80% |
| `rebalance_cooldown_hours` | float | 24.0 | Min hours between rebalances |
| `min_campaigns_for_signal` | int | 1 | Min campaigns for signal |
| `lookback_days` | int | 14 | Response curve data window |
| `smoothing_alpha` | float | 0.3 | EMA smoothing for marginal ROAS |
| `roas_weight` | float | 0.50 | Optimization weight |
| `volume_weight` | float | 0.25 | Conversion volume weight |
| `diversification_weight` | float | 0.10 | Concentration penalty |
| `momentum_weight` | float | 0.15 | Trend following weight |
| `min_confidence_for_shift` | float | 0.60 | Min confidence for rebalance |
| `emergency_roas_floor` | float | 0.5 | Pull budget if ROAS below |

### BudgetRecommendationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `candidate_budgets` | List[float] | [50, 100, ..., 50000] | Budget candidates for Q-sweep |
| `min_q_value_threshold` | float | -10.0 | Reject budgets below this Q |
| `use_portfolio_context` | bool | True | Portfolio constraints |
| `campaign_duration_days` | int | 30 | Default duration |

### PlatformMetrics

| Field | Type | Description |
|-------|------|-------------|
| `platform` | str | Platform name |
| `num_campaigns` | int | Campaign count |
| `total_spend` | float | Total spend |
| `total_revenue` | float | Total revenue |
| `total_conversions` | int | Total conversions |
| `total_clicks` | int | Total clicks |
| `total_impressions` | int | Total impressions |
| `roas` | float | Derived: revenue/spend |
| `cpa` | float | Derived: spend/conversions |
| `ctr` | float | Derived: clicks/impressions |
| `cvr` | float | Derived: conversions/clicks |
| `marginal_roas` | float | Estimated marginal return |
| `marginal_roas_confidence` | float | Estimate confidence |
| `roas_trend_7d` | float | 7-day ROAS trend |
| `spend_trend_7d` | float | 7-day spend trend |
| `conversion_trend_7d` | float | 7-day conversion trend |
| `segment_count` | int | Audience segments |
| `top_segment_roas` | float | Best segment ROAS |
| `avg_frequency` | float | Average impression frequency |
| `max_frequency` | float | Max impression frequency |
| `current_budget_share` | float | Current portfolio share |

### PlatformPortfolio

| Field | Type |
|-------|------|
| `organization_id` | str |
| `total_budget` | float |
| `platforms` | Dict[str, PlatformMetrics] |
| `timestamp` | str |
| `campaign_states` | Dict[str, List[Tuple[CampaignState, CampaignContext, Dict]]] |

Methods: `active_platforms()`, `total_spend()`, `total_revenue()`, `portfolio_roas()`, `to_dict()`

### AllocationRecommendation

| Field | Type | Description |
|-------|------|-------------|
| `platform` | str | Target platform |
| `current_share` | float | Current portfolio fraction |
| `recommended_share` | float | Target fraction |
| `shift_pct` | float | recommended - current |
| `current_budget` | float | Current $ allocation |
| `recommended_budget` | float | Target $ allocation |
| `rationale` | str | Human-readable explanation |
| `confidence` | float | Recommendation confidence |

### CrossPlatformResult

| Field | Type |
|-------|------|
| `organization_id` | str |
| `timestamp` | str |
| `portfolio_roas` | float |
| `total_budget` | float |
| `total_spend` | float |
| `allocations` | List[AllocationRecommendation] |
| `allocation_confidence` | float |
| `platform_campaign_results` | Dict[str, List[OptimizationResult]] |
| `projected_portfolio_roas` | float |
| `projected_incremental_revenue` | float |
| `rebalance_triggered` | bool |
| `blocked_reason` | Optional[str] |
| `audience_constraints` | Optional[Dict[str, Any]] |
| `portfolio_narrative` | Optional[Any] |
| `portfolio_snapshot` | Optional[Dict] |

### MarginalReturnEstimator

Log-linear response model: `Revenue(spend) = a * ln(1 + spend/b)`

Marginal ROAS = `a / (b + spend)` -- captures diminishing returns.

Methods:
- `record_observation(platform, daily_spend, daily_revenue)` -- Appends to history (90-day cap)
- `estimate(platform, current_daily_spend, lookback_days=14) -> (marginal_roas, confidence)` -- Splits data at median spend, compares high-spend vs low-spend ROAS, applies EMA smoothing (alpha=0.3). Confidence = `min(1, n/14) * max(0.3, 1 - CV)`
- `get_all_estimates(portfolio, lookback_days) -> Dict[str, (float, float)]`

### BudgetAllocator

Constrained simplex optimization via projected gradient ascent.

**Objective**:
```
max  sum_p [w_roas * marginal_roas_p * budget_p
          + w_vol  * log(1 + conv_p) * budget_p / spend_p
          + w_div  * entropy(allocation)
          + w_mom  * trend_p * budget_p]
```

**Constraints**:
- `sum(budget_p) = total_budget`
- `min_pct * total <= budget_p <= max_pct * total` for all p
- `|budget_p - current_p| <= max_shift * total` for all p

Implementation:
1. Compute composite score per platform (weighted sum of normalized marginal ROAS, volume, trend, scaled by confidence)
2. Convert scores to allocations via softmax (temperature=0.5)
3. Apply constraints iteratively: clamp shifts -> clamp floor/ceiling -> project to simplex -> final normalization
4. `_project_simplex(v)`: Standard simplex projection algorithm

### PlatformPerformanceTracker

- `record_daily_metrics(org_id, platform, metrics)` -- Stores daily data (90-day rolling window)
- `build_portfolio(org_id, campaigns, total_budget) -> PlatformPortfolio` -- Aggregates campaigns by platform, computes derived metrics, calculates trends from 7d/14d windows, collects audience segment features
- `can_rebalance(org_id, cooldown_hours) -> (bool, reason)` -- Cooldown check
- `mark_rebalance(org_id)` -- Records rebalance timestamp

### CrossPlatformOptimizer

Top-level orchestrator combining allocation + per-campaign optimization.

**Constructor**:
```python
__init__(hybrid_optimizer, config, max_concurrent_campaigns=10,
         audience_manager=None, platform_registry=None, x_model_agent=None)
```

**`optimize_portfolio(org_id, campaigns, total_budget, force_rebalance=False) -> CrossPlatformResult`**

7-phase pipeline:
1. Build portfolio snapshot via `PlatformPerformanceTracker`
2. Check cooldown (24h default)
3. Estimate marginal returns per platform
4. Emergency override: force low marginal for platforms with ROAS < 0.5
5. Solve allocation (X-Model if available, else heuristic BudgetAllocator)
6. Compute projected impact (ROAS projection from marginal rates)
7. Adjust campaign budgets proportionally, apply audience constraints, run per-campaign DRL

**M2 execution** (`_run_campaign_optimization`): When `PlatformModelRegistry` is configured, swaps in per-platform P-Model agent before running each platform's campaigns, then restores the original agent.

**M5 execution** (`_x_model_allocate`): Builds XModelState from portfolio, runs `XModelAgent.select_allocation()`, converts to AllocationRecommendation list with shift constraints.

**Budget recommendation** (`recommend_budget`): Q-function sweep over candidate budgets for new campaigns. Builds synthetic CampaignState, evaluates Q(s,a) via critic, scores with `0.5*Q_norm + 0.3*marginal_roas + 0.2*confidence`, applies portfolio constraints.

---

## 13. X-Model (Cross-Platform DRL Agent)

**File**: `x_model.py` (559 lines)

### Constants

```python
X_PLATFORMS = ["meta", "google", "tiktok", "amazon", "walmart"]
NUM_PLATFORMS = 5
FEATURES_PER_PLATFORM = 13
GLOBAL_FEATURES = 5
X_STATE_DIM = 70  # 5 * 13 + 5
X_ACTION_DIM = 5  # allocation weight per platform
```

### XModelState (70-dimensional)

Per-platform features (13 per platform, 5 platforms = 65):

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | roas | /5.0, clipped [0,1] |
| 1 | cpa_inv | 1/(1+cpa), [0,1] |
| 2 | ctr | raw, [0,1] |
| 3 | cvr | raw, [0,1] |
| 4 | marginal_roas | /5.0, clipped [0,1] |
| 5 | spend_share | [0,1] |
| 6 | spend_trend_7d | clipped [-1,1] |
| 7 | roas_trend_7d | clipped [-1,1] |
| 8 | conversion_trend_7d | clipped [-1,1] |
| 9 | num_campaigns_norm | /20, clipped [0,1] |
| 10 | segment_count_norm | /10, clipped [0,1] |
| 11 | top_segment_roas | /5, clipped [0,1] |
| 12 | avg_frequency_ratio | avg_freq/max_freq, [0,1] |

Global features (5):

| Index | Feature | Normalization |
|-------|---------|---------------|
| 65 | portfolio_roas | /5.0, clipped [0,1] |
| 66 | total_budget_log | log1p(budget)/log1p(1e6) |
| 67 | budget_utilization | spend/budget, [0,1] |
| 68 | portfolio_hhi | Herfindahl-Hirschman Index of spend shares |
| 69 | active_platform_ratio | active_count / 5 |

Methods: `to_tensor(device) -> Tensor`, `from_tensor(tensor) -> XModelState`

### XModelAction

| Field | Type | Description |
|-------|------|-------------|
| `allocation_weights` | Dict[str, float] | Per-platform weights (sum to 1.0) |
| `confidence` | float | Agent confidence |
| `entropy` | float | Action entropy |
| `q_value` | float | Critic value |

### XModelActor

```
Architecture:
    Input: state [batch, 70]
        |
    Layer 1: Linear(70, 256) -> LayerNorm -> ReLU
    Layer 2: Linear(256, 256) -> LayerNorm -> ReLU
    Layer 3: Linear(256, 128) -> LayerNorm -> ReLU
        |
    mean_head: Linear(128, 5)
    log_std_head: Linear(128, 5)  [clamped to [-5, 2]]
```

`sample()`: Gaussian reparameterization -> softmax normalization to get allocation weights summing to 1.0.

### XModelCritic (Twin Q)

```
Architecture (each Q-network):
    Input: concat(state, action) [batch, 75]
        |
    Layer 1: Linear(75, 256) -> LayerNorm -> ReLU
    Layer 2: Linear(256, 256) -> LayerNorm -> ReLU
    Output: Linear(256, 1)
```

### XModelAgent

Dedicated SAC agent for cross-platform allocation.

**`select_allocation(state, deterministic=False, min_share=0.05, max_share=0.80) -> XModelAction`**
- Runs actor, applies min/max share constraints, re-normalizes to sum=1.0
- Confidence = `1 - entropy / max_entropy`

**`update(batch) -> Dict[str, float]`**
- Standard SAC update: critic loss (TD + min-Q), actor loss (max Q - alpha * log_prob), alpha tuning, soft target update

### build_x_state()

```python
build_x_state(portfolio_dict: Dict, total_budget: float) -> XModelState
```
Converts portfolio dictionary (from `PlatformPortfolio.to_dict()`) into normalized XModelState. Handles missing platforms by zero-filling. Computes HHI from spend shares.

---

## 14. Platform Model Registry (P-Models)

**File**: `platform_model_registry.py` (308 lines)

### Constants

```python
SUPPORTED_PLATFORMS = ["meta", "google", "tiktok", "amazon", "walmart"]
```

### PlatformModelMeta

| Field | Type | Default |
|-------|------|---------|
| `platform` | str | - |
| `state_dim` | int | 42 |
| `total_training_steps` | int | 0 |
| `last_trained_at` | Optional[str] | None |
| `training_transitions` | int | 0 |
| `checkpoint_path` | Optional[str] | None |

### PlatformModelRegistry

One SACAgent per platform, with global fallback.

Methods:
- `get_or_create(platform) -> SACAgent` -- Returns existing or creates new agent for platform
- `get(platform) -> SACAgent` -- Gets agent (raises if not exists)
- `has_platform(platform) -> bool` -- Check existence
- `save(platform, checkpoint_dir)` -- Save per-platform checkpoint
- `load(platform, checkpoint_dir)` -- Load per-platform checkpoint
- `load_all(checkpoint_dir)` -- Load all saved platforms
- `_ensure_global_fallback()` -- Lazy loads a global fallback agent
- `select_action(platform, state) -> ActionSpace` -- M2 execution: routes to platform agent or global fallback
- `evaluate_q(platform, state, action) -> float` -- Q-value for X-Model value estimation
- `get_diagnostics() -> Dict` -- Per-platform step counts, training timestamps

---

## 15. Cross-Platform DRL Engine (Orchestrator)

**File**: `cross_platform_drl_engine.py` (832 lines)

### DualRunResult

Side-by-side comparison of DRL vs heuristic allocation.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | str | When comparison was run |
| `drl_allocations` | Dict[str, float] | X-Model allocation weights |
| `heuristic_allocations` | Dict[str, float] | BudgetAllocator weights |
| `drl_confidence` | float | X-Model confidence |
| `drl_q_value` | float | X-Model Q-value |
| `heuristic_projected_roas` | float | Heuristic projected ROAS |
| `drl_projected_roas` | float | DRL projected ROAS |
| `divergence` | float | L1 norm of allocation difference |

### ModelReadinessChecker

Checks if X-Model is ready for production use:
- `agent.total_steps >= min_training_steps` (default: 500)
- `confidence >= min_confidence_for_drl` (default: 0.40)

### CrossPlatformDRLEngine

Top-level orchestrator wrapping `CrossPlatformOptimizer`, `XModelAgent`, `XModelTrainer`, and `XTrainingDataBuilder`.

**Three strategy modes**:

1. **DRL_PRIMARY**: X-Model first; if ModelReadinessChecker passes, use X-Model allocation; otherwise fall back to heuristic BudgetAllocator
2. **HEURISTIC_PRIMARY**: Always use heuristic allocation, but collect training data for X-Model in background
3. **DUAL_BENCHMARK**: Run both X-Model and heuristic, apply DRL allocation, log DualRunResult for comparison

**Automatic training lifecycle**:
- Snapshot counting: After each `optimize_portfolio()` call, records portfolio snapshot via `XTrainingDataBuilder`
- Retrain trigger: When `snapshot_count >= retrain_snapshot_threshold` (50) and `transitions >= retrain_min_transitions` (30)
- Retraining: Calls `XModelTrainer.train()` with configured epochs/steps
- Reset: Clears snapshot counter after retraining

**Key methods**:
- `optimize_portfolio(org_id, campaigns, total_budget, force_rebalance=False) -> CrossPlatformResult` -- Delegates to appropriate strategy, collects training data, checks retrain triggers
- `_try_auto_retrain()` -- Automatic X-Model retraining
- `get_diagnostics()` -- Comprehensive system diagnostics
- `get_benchmark_history() -> List[DualRunResult]`

Integration points:
- Narrative generation via `OptimizationNarrator.narrate_portfolio_run()`
- A/B test integration via `DRLABTestManager.create_portfolio_experiment()`

---

## 16. Continuous Learning

**File**: `continuous_learning.py` (740 lines)

### LearningMode (Enum)

`ONLINE`, `BATCH`, `TRIGGERED`, `HYBRID`

### OutcomeRecord

| Field | Type | Description |
|-------|------|-------------|
| `campaign_id` | str | Campaign identifier |
| `action` | ActionSpace | Action taken |
| `pre_state` | CampaignState | State before action |
| `post_state` | Optional[CampaignState] | State after action |
| `reward` | Optional[float] | Computed reward |
| `outcome_delay` | str | Expected delay type |

### OutcomeTracker

Manages asynchronous outcome collection with different delay profiles:
- Impressions: minutes
- Clicks: hours
- Conversions: days

Features:
- Pending observations with 24h window
- Completed queue (10,000 max)
- `record_outcome(campaign_id, post_state) -> Optional[Transition]` -- Matches pending to completed, computes reward

### PerformanceMonitor

Rolling window tracking (size=1000) for:
- `rewards`: Recent reward values
- `q_values`: Critic estimates
- `td_errors`: Temporal difference errors
- `confidences`: Agent confidence scores

Drift detection:
- Performance degradation: -15% threshold
- High TD error: > 1.0 mean

### ContinuousLearningEngine

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | HYBRID | Learning mode |
| `online_update_frequency` | 4 | Update every N outcomes |
| `batch_update_interval` | 3600 | Batch update interval (seconds) |
| `min_batch_size` | 64 | Minimum batch for update |

Hybrid mode behavior:
- **Online**: Single gradient step per `online_update_frequency` outcomes
- **Batch**: Multi-step update every `batch_update_interval` (10 gradient steps)
- **Triggered**: On drift detection (50 gradient steps)
- **Hybrid**: All three modes active simultaneously

Integrates `ForecastFeedbackLoop` for periodic forecaster refitting.

### ModelVersionManager

- `save_version(agent, metrics) -> version_id`
- `load_version(version_id) -> agent`
- `get_best_version(metric_name) -> version_id`

---

## 17. A/B Testing Framework

**File**: `ab_testing.py` (1114 lines)

### ExperimentStatus (Enum)

`DRAFT`, `RUNNING`, `PAUSED`, `COMPLETED`, `STOPPED`

### AssignmentMethod (Enum)

`DETERMINISTIC` (SHA256 hash), `RANDOM`, `STRATIFIED`

### ExperimentConfig

| Field | Type | Default |
|-------|------|---------|
| `experiment_id` | str | - |
| `name` | str | - |
| `variants` | List[Dict] | [{name: "baseline", weight: 0.80}, {name: "drl", weight: 0.20}] |
| `assignment_method` | AssignmentMethod | DETERMINISTIC |
| `target_sample_size` | int | 1,000 |
| `min_detectable_effect` | float | 0.05 |
| `significance_level` | float | 0.05 |
| `power` | float | 0.8 |
| `primary_metric` | str | "roas" |
| `secondary_metrics` | List[str] | ["cpa", "conversions", "ctr"] |

### VariantAssigner

Hash-based deterministic assignment using SHA256 for stability across sessions.

### StatisticalAnalyzer

- Welch's t-test (unequal variance)
- Confidence intervals
- Welch-Satterthwaite degrees of freedom
- Sample size computation for desired power

### DRLABTestManager

Full experiment lifecycle:
- `create_experiment(name, config) -> ExperimentConfig`
- `create_platform_experiment(platform) -> ExperimentConfig` -- P-Model vs Global agent
- `create_portfolio_experiment(org_id) -> ExperimentConfig` -- X-Model vs Heuristic at account level
- `assign_variant(experiment_id, entity_id) -> str`
- `record_observation(experiment_id, variant, metrics)`
- `analyze(experiment_id) -> ExperimentResult`
- Generates human-readable narratives using `ParameterGlossary`

### ExperimentResult

Contains per-variant summary statistics, t-test results (t_stat, p_value, significant), confidence intervals, lift estimate, and narrative summary.

---

## 18. Explainable AI (xAI) Narrator

**File**: `xai_narrator.py` (668 lines)

### RunNarrative (Campaign-level)

| Field | Type |
|-------|------|
| `situation_summary` | str |
| `decision_summary` | str |
| `reasoning` | List[str] |
| `confidence_explanation` | str |
| `reasonability_check` | str |
| `full_narrative` | str |

### PortfolioNarrative (Portfolio-level)

| Field | Type |
|-------|------|
| `portfolio_summary` | str |
| `allocation_decision` | str |
| `platform_reasoning` | Dict[str, str] |
| `confidence_explanation` | str |
| `risk_assessment` | str |
| `full_narrative` | str |

### OptimizationNarrator

**`narrate_campaign_run(state, action, validation, context) -> RunNarrative`**

Generates:
1. **Situation summary**: Describes state health (trends up/down/stable, fatigue low/moderate/high, utilization, impression share)
2. **Decision summary**: Maps action to human language (bid increase/decrease, budget increase/decrease, audience action, creative action)
3. **Reasoning bullets**: "trigger -> action -> because rationale" format
4. **Confidence explanation**: Maps confidence level to language (very high > 0.85, high > 0.7, moderate > 0.5, low)
5. **Reasonability check**: Validates action against guardrails, flags concerns

**`narrate_portfolio_run(portfolio, allocations, x_action) -> PortfolioNarrative`**

Generates:
1. **Portfolio summary**: Describes ROAS, HHI concentration, utilization, platform trends
2. **Allocation decision**: Lists platform shifts with amounts
3. **Per-platform reasoning**: Explains each shift direction
4. **Confidence**: Maps Q-value and confidence to language
5. **Risk assessment**: Flags concentration risk (HHI > 0.3), large shifts (> 10%), low confidence

### ParameterGlossary

Dictionary of 9 key parameters:

| Parameter | Full Name | Normal Range |
|-----------|-----------|--------------|
| `roas` | Return on Ad Spend | 1.5x - 5.0x |
| `cpa` | Cost per Acquisition | $5 - $100 |
| `ctr` | Click-Through Rate | 0.5% - 5% |
| `budget_utilization` | Budget Utilization | 70% - 95% |
| `creative_fatigue_score` | Creative Fatigue Score | 0 - 0.3 |
| `strategic_confidence` | Strategic Confidence | 0.7 - 1.0 |
| `allocation_weight` | Platform Allocation Weight | 5% - 50% |
| `portfolio_hhi` | Portfolio Herfindahl-Hirschman Index | 0.2 - 0.5 |
| `marginal_roas` | Marginal Return on Ad Spend | 0.5x - 3.0x |

Each entry includes: `full_name`, `definition`, `formula`, `normal_range`, `impact`.

---

## 19. Audience Constraints

**File**: `audience_constraints.py` (268 lines)

### AudienceSegment

| Field | Type | Default |
|-------|------|---------|
| `segment_id` | str | - |
| `name` | str | - |
| `platform` | str | - |
| `min_budget_pct` | float | 0.0 |
| `max_budget_pct` | float | 1.0 |
| `max_exposures_per_user` | int | 10 |
| `max_daily_frequency` | int | 3 |

### SegmentAllocation

| Field | Type |
|-------|------|
| `segment_id` | str |
| `recommended_budget_pct` | float |
| `recommended_daily_frequency` | int |
| `rationale` | str |

### AudienceConstraintResult

| Field | Type |
|-------|------|
| `allocations` | List[SegmentAllocation] |
| `total_budget_check_passed` | bool |
| `violations` | List[str] |
| `narrative` | str |

### AudienceConstraintManager

**`allocate_budget(platform_budget, action, performance_signals) -> AudienceConstraintResult`**

1. Validates constraints (min sums <= 1, min <= max)
2. Scores segments: `0.5 * roas + 0.3 * cvr + 0.2 * ctr`
3. Applies DRL audience action modifiers:
   - EXPAND: +10% to best-scoring segment
   - REFINE: +/-5% based on CVR ranking
   - EXCLUDE: Remove worst segment, redistribute
   - HOLD: No changes
4. Applies floor/ceiling from segment constraints
5. Frequency capping based on segment configuration
6. Rolling performance history updates

---

## 20. Campaign Forecasting

**File**: `benchmark_model.py` (215 lines)

### MetricForecast

| Field | Type |
|-------|------|
| `metric` | str |
| `mean` | float |
| `p10` | float |
| `p90` | float |

### CampaignForecast

| Field | Type |
|-------|------|
| `metric_forecasts` | List[MetricForecast] |
| `estimated_conversions` | float |
| `thresholds` | Dict[str, float] |

### LinearModel

| Field | Type |
|-------|------|
| `coef` | ndarray |
| `intercept` | float |
| `resid_std` | float |

Method: `predict(X) -> ndarray`

### Key Functions

- `fit_ridge(X, y, l2=1e-2) -> LinearModel` -- Closed-form ridge regression: `(X'X + l2*I)^(-1) X'y`
- `bootstrap_interval(X, y, n_boot=200) -> (mean, p10, p90)` -- Bootstrap resampling for prediction intervals
- `compute_thresholds(history)` -- Percentile-based decision thresholds: cpc_stop@p90, ctr_stop@p10, roas_increase_budget@p90, cpa_decrease_bid@p90

### CampaignForecaster

- `fit(feature_matrix, outcome_matrix)` -- Fits one LinearModel per outcome column
- `predict(feature_vector) -> CampaignForecast` -- Returns per-metric forecasts with bootstrap intervals

**OUTCOME_COLUMNS**: `[impressions, clicks, cpc, ctr, cvr, roas, cpa]`

---

## 21. Forecast Feedback Loop

**File**: `forecast_feedback.py` (255 lines)

### ForecastRecord

| Field | Type |
|-------|------|
| `campaign_id` | str |
| `tracking_id` | str (UUID) |
| `forecast` | CampaignForecast |
| `feature_vector` | ndarray |
| `actuals` | Optional[Dict[str, float]] |
| `outcome_timestamp` | Optional[str] |

### AccuracyMetrics

| Field | Type |
|-------|------|
| `metric` | str |
| `mae` | float |
| `mape` | float |
| `coverage_p10_p90` | float |
| `n_samples` | int |

### ForecastFeedbackLoop

- `register_forecast(campaign_id, forecast, feature_vector) -> tracking_id`
- `record_actual(tracking_id, actuals)` -- Matches forecast to actual outcome
- `compute_accuracy() -> List[AccuracyMetrics]` -- Per-metric MAE, MAPE, p10-p90 coverage
- Automatic refit every 200 new actuals (minimum 50 completed records)

---

## 22. BigQuery Data Loader

**File**: `bigquery_loader.py` (208 lines)

### DEFAULT_FEATURES

42 features matching CampaignState dimensions, with column aliases:
```
ctr, cvr, roas, cpa, cpc, cpm,
spend_velocity, impression_volume, click_volume, conversion_volume,
hour_of_day, day_of_week, day_of_month, is_weekend, is_holiday, days_remaining,
ctr_trend_7d, cvr_trend_7d, roas_trend_7d, cpa_trend_7d, spend_trend_7d,
impression_share, auction_pressure, competitive_position,
audience_quality_score OR audience_quality,
creative_fatigue_score OR creative_fatigue,
predicted_cvr, predicted_ltv, propensity_score,
optimization_goal_encoding OR goal_encoding,
platform_encoding, campaign_maturity, budget_utilization,
log_daily_spend, log_total_campaign_spend, log_daily_budget,
segment_count, top_segment_roas, avg_frequency,
target_cpa_norm, min_roas_norm, daily_budget_limit_norm
```

### BigQueryDataLoader

| Field | Type |
|-------|------|
| `project_id` | str |
| `table` | str |
| `credentials_path` | Optional[str] |
| `features` | List[str/Tuple[str,str]] |

Methods:
- `_client()` -- Creates BigQuery client with GCP credentials
- `_resolve_features()` -- Handles (column, alias) tuples
- `fetch_dataframe(where=None, limit=None) -> DataFrame` -- SQL query with optional filter
- `fit_normalizer(df) -> (mean_dict, std_dict)` -- Per-column mean/std
- `transform(df, mean_dict, std_dict) -> DataFrame` -- Standardization with NaN fill (0.0)
- `load_states(where, limit) -> (ndarray, feature_list)` -- End-to-end: fetch + normalize

---

## 23. Cross-File Data Flow

### Full Optimization Pipeline (Single Request)

```
1. API Request (org_id, campaigns, total_budget)
       |
2. CrossPlatformDRLEngine.optimize_portfolio()
       |
3. CrossPlatformOptimizer.optimize_portfolio()
       |
   3a. PlatformPerformanceTracker.build_portfolio() -> PlatformPortfolio
   3b. MarginalReturnEstimator.get_all_estimates() -> Dict[platform, (marginal, conf)]
   3c. Emergency override check (ROAS < 0.5)
       |
   3d. Allocation decision:
       IF X-Model available:
           build_x_state(portfolio) -> XModelState [70-dim]
           XModelAgent.select_allocation() -> XModelAction
           Convert to AllocationRecommendation list
       ELSE:
           BudgetAllocator.allocate() -> AllocationRecommendation list
       |
   3e. _adjust_campaign_budgets() -- Distribute platform budgets to campaigns
       |
   3f. AudienceConstraintManager.allocate_budget() (if configured)
       |
   3g. _run_campaign_optimization() -- Per-platform:
       |
       FOR each platform:
           PlatformModelRegistry.get(platform) -> SACAgent (M2)
           |
           FOR each campaign:
               HybridDRLLLMOptimizer.optimize()
               |
               Phase 1: SafeDRLAgent.select_action(CampaignState)
                   -> SACAgent.select_action()
                       -> ActorNetwork.sample() -> (continuous, discrete, log_prob, entropy)
                       -> CriticNetwork(state, action) -> Q-value
                   -> ActionValidator.validate() -> ActionValidationResult
                   -> DRLDirective.from_action()
               |
               Phase 2: LLMClient.generate(directive.to_llm_prompt_context())
                   -> TacticalExecution
               |
               Phase 3: OptimizationNarrator.narrate_campaign_run()
                   -> RunNarrative
               |
               Phase 4: CampaignForecaster.predict()
                   -> CampaignForecast
               |
               -> OptimizationResult
       |
4. OptimizationNarrator.narrate_portfolio_run() -> PortfolioNarrative
       |
5. XTrainingDataBuilder.record_snapshot() -- M3 data collection
6. Auto-retrain check -> XModelTrainer.train() if triggered
       |
7. Return CrossPlatformResult
```

### Training Data Flow

```
Historical Data (BigQuery/CSV)
       |
BigQueryDataLoader.load_states() -> normalized feature matrix
       |
train_bigquery_offline.build_transitions() -> List[Transition]
       |
   1. behavior_cloning_pretrain() -- Warm-start SAC
   2. OfflineTrainer.train() -- CQL offline training
       |
SACAgent (per-platform) -> PlatformModelRegistry.save()

Portfolio Snapshots (runtime)
       |
XTrainingDataBuilder.record_snapshot() -> builds XTransition
       |
XModelTrainer.train() -> XModelAgent update
```

### Continuous Learning Flow

```
Production Action
       |
OutcomeTracker.record_action(campaign_id, action, pre_state)
       |  [waits for delayed outcome]
       |
OutcomeTracker.record_outcome(campaign_id, post_state) -> Transition
       |
ReplayBuffer.push(transition)
       |
ContinuousLearningEngine:
   - Online: SACAgent.update() every 4 outcomes (1 gradient step)
   - Batch: SACAgent.update() every 60min (10 gradient steps)
   - Triggered: SACAgent.update() on drift (50 gradient steps)
       |
PerformanceMonitor: drift detection
   - Performance degradation: -15% threshold -> triggered update
   - High TD error: > 1.0 -> triggered update
       |
ModelVersionManager: checkpoint and rollback
ForecastFeedbackLoop: periodic forecaster refit
```

### Key Inter-Module Dependencies

| Module | Depends On | Provides To |
|--------|-----------|-------------|
| `config.py` | (standalone) | All modules |
| `state_action.py` | config | All modules |
| `networks.py` | (torch only) | sac_agent |
| `sac_agent.py` | networks, config, replay_buffer | safe_agent, platform_model_registry |
| `replay_buffer.py` | state_action | sac_agent, offline_trainer, continuous_learning |
| `reward_functions.py` | config | offline_trainer, continuous_learning, train_bigquery_offline |
| `safe_agent.py` | sac_agent, config | hybrid_optimizer |
| `hybrid_optimizer.py` | safe_agent, xai_narrator, benchmark_model | cross_platform_optimizer |
| `cross_platform_optimizer.py` | hybrid_optimizer, audience_constraints, platform_model_registry, x_model | cross_platform_drl_engine |
| `x_model.py` | (torch only) | x_training, x_training_data, cross_platform_optimizer, cross_platform_drl_engine |
| `x_training_data.py` | x_model | cross_platform_drl_engine |
| `x_training.py` | x_model | cross_platform_drl_engine |
| `cross_platform_drl_engine.py` | cross_platform_optimizer, x_model, x_training, x_training_data, xai_narrator, ab_testing | (top-level entry point) |
| `platform_model_registry.py` | sac_agent, config | cross_platform_optimizer |
| `xai_narrator.py` | (standalone) | hybrid_optimizer, cross_platform_drl_engine, ab_testing |
| `audience_constraints.py` | state_action | cross_platform_optimizer |
| `benchmark_model.py` | (numpy only) | hybrid_optimizer, forecast_feedback |
| `forecast_feedback.py` | benchmark_model | continuous_learning |
| `bigquery_loader.py` | google.cloud.bigquery | train_bigquery_offline |

---

## Appendix: Public API (__all__)

```python
# Config
DRLConfig, TrainingConfig, GuardrailConfig, AllocationStrategy, CrossPlatformStrategyConfig

# State/Action
CampaignState, ActionSpace, DRLDirective

# Networks
ActorNetwork, CriticNetwork, ValueNetwork

# Agent
SACAgent, SafeDRLAgent, ActionValidator

# Training
PrioritizedReplayBuffer, Transition, OfflineTrainer, CQLLoss

# Reward
RewardComputer, MultiObjectiveReward

# Hybrid
HybridDRLLLMOptimizer, OptimizationResult

# Cross-Platform
CrossPlatformOptimizer, CrossPlatformConfig, CrossPlatformResult,
PlatformPortfolio, PlatformMetrics, PlatformPerformanceTracker,
MarginalReturnEstimator, BudgetAllocator, AllocationRecommendation,
Platform, BudgetRecommendation, BudgetRecommendationConfig

# Learning
ContinuousLearningEngine, OutcomeTracker

# Testing
DRLABTestManager, ExperimentResult

# xAI Narrative
OptimizationNarrator, RunNarrative, PortfolioNarrative, ParameterGlossary

# Audience Constraints
AudienceConstraintManager, AudienceConstraintResult

# Forecasting
CampaignForecaster, CampaignForecast

# Forecast Feedback
ForecastFeedbackLoop, AccuracyMetrics

# Inference Helper
load_sac_for_inference

# P & X Model Architecture
PlatformModelRegistry, PlatformModelMeta,
XModelAgent, XModelState, XModelAction, build_x_state,
XTrainingDataBuilder, XTransition, XModelTrainer

# DRL Engine
CrossPlatformDRLEngine, DualRunResult, ModelReadinessChecker
```
