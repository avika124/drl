# Deep Reinforcement Learning Module for Campaign Optimization

A production-ready DRL system for advertising campaign optimization using Soft Actor-Critic (SAC) with offline pre-training, online learning, and DRL+LLM hybrid architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Campaign Optimization Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐│
│  │   Campaign  │───▶│  DRL Macro   │───▶│  Strategic Directive││
│  │    State    │    │    Layer     │    │  (Budget, Bid, etc.)││
│  └─────────────┘    └──────────────┘    └──────────┬──────────┘│
│                                                     │           │
│                                                     ▼           │
│                                          ┌─────────────────────┐│
│                                          │    LLM Micro Layer  ││
│                                          │ (Creative, Messaging││
│                                          └──────────┬──────────┘│
│                                                     │           │
│                                                     ▼           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐│
│  │   Action    │◀───│    Safety    │◀───│  Tactical Execution ││
│  │  Execution  │    │  Guardrails  │    │                     ││
│  └─────────────┘    └──────────────┘    └─────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
drl/
├── __init__.py              # Package exports
├── config.py                # Configuration classes
├── state_action.py          # MDP definitions (State, Action, Directive)
├── networks.py              # Neural network architectures
├── replay_buffer.py         # Prioritized experience replay
├── reward_functions.py      # Multi-objective reward computation
├── sac_agent.py             # Soft Actor-Critic implementation
├── offline_trainer.py       # Historical data training pipeline
├── safe_agent.py            # Production agent with guardrails
├── hybrid_optimizer.py      # DRL + LLM integration
├── continuous_learning.py   # Online learning engine
├── ab_testing.py            # A/B testing framework
├── xai_narrator.py          # Explainable AI narrative layer
├── audience_constraints.py  # Audience segmentation constraints
├── drl_integration.py       # Async integration layer
└── outcome_tracker.py       # Outcome tracking & reward computation
```

## Key Components

### 1. State Space (39 dimensions)

| Category | Features |
|----------|----------|
| Core Metrics | CTR, CVR, ROAS, CPA, CPC, CPM |
| Volume | Spend velocity, impressions, clicks, conversions |
| Temporal | Hour, day of week, day of month, weekend, holiday, days remaining |
| Trends | 7-day trends for CTR, CVR, ROAS, CPA, spend |
| Competitive | Impression share, auction pressure, competitive position |
| ML-Derived | Audience quality, creative fatigue, predicted CVR/LTV, propensity |
| Context | Goal encoding, platform encoding, campaign maturity, budget utilization |
| Absolute Spend | Log-normalized daily spend, total campaign spend, daily budget |
| Audience Segments | Segment count, top segment ROAS, average frequency |

### 2. Action Space (Hybrid)

**Continuous Actions:**
- `bid_adjustment`: [-50%, +50%]
- `budget_adjustment`: [-30%, +30%]

**Discrete Actions:**
- `audience_action`: HOLD, EXPAND, REFINE, EXCLUDE
- `creative_action`: HOLD, ROTATE, PAUSE_UNDERPERFORMING, TEST_NEW

### 3. Reward Function (Multi-Objective)

```python
reward = primary_objective_reward     # Based on campaign goal (ROAS/CPA/conversions)
       + efficiency_bonus             # Bonus for ROAS > target
       + volume_bonus                 # Bonus for conversion growth
       + ltv_bonus                    # Bonus for high-LTV customers
       - budget_violation_penalty     # Penalty for exceeding budget
       - cpa_violation_penalty        # Penalty for exceeding CPA target
       - action_magnitude_penalty     # Smoothness penalty
```

## Usage

### Basic Usage

```python
from drl import (
    DRLConfig, TrainingConfig, GuardrailConfig,
    SACAgent, SafeDRLAgent, CampaignState, CampaignContext
)

# Initialize configuration
config = DRLConfig(
    state_dim=39,
    continuous_action_dim=2,
    discrete_action_dims=[4, 4],
    hidden_dim=256,
)

training_config = TrainingConfig(
    batch_size=256,
    use_cql=True,
    use_per=True,
)

# Create agent
agent = SACAgent(config, training_config)

# Wrap with safety guardrails for production
safe_agent = SafeDRLAgent(
    agent=agent,
    guardrails=GuardrailConfig(),
    exploration_rate=0.1
)

# Get action for a campaign
state = CampaignState.from_campaign_metrics(...)
context = CampaignContext(
    campaign_id="camp_123",
    current_bid=1.50,
    current_budget=1000.0,
    ...
)

action, validation = await safe_agent.get_action(state, context)
```

### Offline Training

```python
from drl import OfflineTrainer, OfflineDataExtractor, RewardComputer

# Extract historical data
extractor = OfflineDataExtractor()
campaigns = extractor.extract_from_csv("campaign_metrics.csv")

# Build transitions
reward_computer = RewardComputer()
transitions = extractor.build_transitions(campaigns, reward_computer)

# Train offline
trainer = OfflineTrainer(agent, training_config)
trainer.load_transitions(transitions)
history = trainer.train(
    num_epochs=100,
    steps_per_epoch=1000,
    checkpoint_dir="models/drl"
)
```

### Hybrid DRL + LLM Optimization

```python
from drl import HybridDRLLLMOptimizer, LLMClient

# Initialize hybrid optimizer
optimizer = HybridDRLLLMOptimizer(
    drl_agent=safe_agent,
    llm_client=LLMClient(model="gpt-4"),
    enable_tactical=True
)

# Run optimization
result = await optimizer.optimize(
    state=campaign_state,
    context=campaign_context,
    campaign_info={
        "product_name": "Premium Widget",
        "brand_name": "WidgetCo",
        "target_audience": "Tech enthusiasts 25-45",
    }
)

# Access results
print(f"Strategic: {result.directive.to_dict()}")
print(f"Tactical: {result.tactical.to_dict()}")
print(f"Recommendations: {result.recommended_changes}")
```

### A/B Testing

```python
from drl import DRLABTestManager

# Create test manager
ab_manager = DRLABTestManager(storage_dir="experiments")

# Create experiment
experiment = ab_manager.create_experiment(
    name="DRL vs Baseline Q4",
    treatment_ratio=0.2,  # 20% DRL, 80% baseline
    primary_metric="roas",
    min_duration_days=14
)

# Start experiment
ab_manager.start_experiment(experiment.experiment_id)

# Get assignment for campaign
variant, config = ab_manager.get_assignment(
    experiment.experiment_id,
    campaign_id="camp_123"
)

# Record metrics
ab_manager.record_metrics(
    experiment.experiment_id,
    variant_name=variant,
    metrics={"roas": 2.5, "cpa": 45.0, "conversions": 100}
)

# Analyze results
result = ab_manager.analyze_experiment(experiment.experiment_id)
print(f"Recommendation: {result.recommendation}")
print(f"Confidence: {result.confidence_in_recommendation:.0%}")
```

### Continuous Learning

```python
from drl import ContinuousLearningEngine, LearningMode

# Initialize engine
learning_engine = ContinuousLearningEngine(
    agent=agent,
    replay_buffer=replay_buffer,
    training_config=training_config,
    learning_mode=LearningMode.HYBRID,
    update_frequency=100,
    batch_interval_minutes=60
)

# Register action for outcome tracking
tracking_id = learning_engine.register_action(
    campaign_id="camp_123",
    state=state,
    action=action,
    context=context
)

# Later: Record outcome
learning_engine.record_outcome(
    tracking_id=tracking_id,
    current_state=new_state,
    metrics_before={"roas": 2.0, "cpa": 50.0},
    metrics_after={"roas": 2.3, "cpa": 45.0},
    goal="roas",
    constraints={"target_cpa": 50.0}
)

# Start continuous learning loop
await learning_engine.start()
```

## Safety Guardrails

The `SafeDRLAgent` enforces production safety:

| Guardrail | Default Value |
|-----------|---------------|
| Max bid increase | +50% |
| Max bid decrease | -30% |
| Max budget increase | +30% |
| Max budget decrease | -30% |
| Min hours between actions | 4 hours |
| Max actions per day | 6 |
| Min confidence for action | 0.7 |
| Min confidence for auto-apply | 0.85 |
| Emergency ROAS threshold | 0.5 |
| Emergency CPA multiplier | 3x |

## Performance Targets

| Metric | Target |
|--------|--------|
| Inference latency | < 50ms |
| ROAS lift vs baseline | > 15% |
| Guardrail violations | < 1% |
| A/B test duration | ≥ 14 days |
| Minimum confidence | 95% |

## Algorithm Details

### Soft Actor-Critic (SAC)

- **Entropy regularization**: Prevents premature convergence
- **Twin Q-networks**: Reduces overestimation bias
- **Automatic entropy tuning**: Adapts exploration-exploitation tradeoff
- **Reparameterization trick**: Enables gradient flow through sampling

### Conservative Q-Learning (CQL)

For offline training, CQL regularization prevents overestimation of out-of-distribution actions:

```
L_CQL = α * (logsumexp(Q(s, a_random), Q(s, a_policy)) - Q(s, a_data))
```

### Prioritized Experience Replay

- Priority based on TD-error: `p_i = |δ_i| + ε`
- Importance sampling correction: `w_i = (N * P(i))^(-β)`
- Beta annealing from 0.4 to 1.0

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
```

## License

Proprietary - AI Advertising Platform
