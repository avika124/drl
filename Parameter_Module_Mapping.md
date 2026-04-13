# DRL Advertising Optimization System — Parameter Module Mapping

## Overview

Parameters are sourced from: `config.py`, environment variables, and CLI arguments.

---

## M1: P-Training Parameters

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| **DRLConfig** | config.py | | |
| `state_dim` | DRLConfig | 42 | State vector dimension |
| `continuous_action_dim` | DRLConfig | 2 | Bid, budget |
| `discrete_action_dims` | DRLConfig | [4, 4] | Audience, creative action counts |
| `hidden_dim` | DRLConfig | 256 | Network hidden size |
| `gamma` | DRLConfig | 0.99 | Discount factor |
| `tau` | DRLConfig | 0.005 | Target network soft update |
| `alpha` | DRLConfig | 0.2 | Entropy coefficient |
| `actor_lr` | DRLConfig | 3e-4 | Actor learning rate |
| `critic_lr` | DRLConfig | 3e-4 | Critic learning rate |
| **TrainingConfig** | config.py | | |
| `batch_size` | TrainingConfig | 256 | Training batch size |
| `replay_buffer_size` | TrainingConfig | 1_000_000 | Buffer capacity |
| `min_buffer_size` | TrainingConfig | 10_000 | Min samples before training |
| `num_offline_epochs` | TrainingConfig | 100 | Offline epochs |
| `steps_per_epoch` | TrainingConfig | 1000 | Gradient steps per epoch |
| `use_cql` | TrainingConfig | True | CQL regularization |
| `cql_alpha` | TrainingConfig | 1.0 | CQL weight |
| `use_per` | TrainingConfig | True | Prioritized replay |
| `per_alpha` | TrainingConfig | 0.6 | PER exponent |
| `gradient_clip` | TrainingConfig | 1.0 | Gradient clipping |
| **Env vars** | | | |
| `DRL_BQ_PROJECT_ID` | train_bigquery_offline | "" | BigQuery project |
| `DRL_BQ_TABLE` | train_bigquery_offline | ad_metrics.campaign_states | BigQuery table |
| `GOOGLE_APPLICATION_CREDENTIALS` | train_bigquery_offline | "" | GCP credentials path |
| `DRL_OUTPUT_DIR` | train_bigquery_offline | models/bq_run | Checkpoint output dir |
| `DRL_ACTION_LOGS_PATH` | train_bigquery_offline | "" | Synthetic action logs CSV |

---

## M2: P-Execution Parameters

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| **GuardrailConfig** | config.py | | |
| `max_bid_increase_pct` | GuardrailConfig | 0.50 | Max 50% bid increase |
| `max_bid_decrease_pct` | GuardrailConfig | 0.30 | Max 30% bid decrease |
| `max_budget_increase_pct` | GuardrailConfig | 0.30 | Max 30% budget increase |
| `max_budget_decrease_pct` | GuardrailConfig | 0.30 | Max 30% budget decrease |
| `min_bid` | GuardrailConfig | 0.01 | Minimum bid ($) |
| `max_bid` | GuardrailConfig | 100.0 | Maximum bid ($) |
| `min_daily_budget` | GuardrailConfig | 10.0 | Min daily budget |
| `max_daily_budget` | GuardrailConfig | 100_000.0 | Max daily budget |
| `min_hours_between_actions` | GuardrailConfig | 4.0 | Cooldown (hours) |
| `max_actions_per_day` | GuardrailConfig | 6 | Max actions per campaign/day |
| `min_confidence_for_action` | GuardrailConfig | 0.7 | Min confidence to act |
| `min_confidence_for_auto_apply` | GuardrailConfig | 0.85 | Auto-apply threshold |
| `max_spend_increase_per_action` | GuardrailConfig | 1000.0 | Max spend increase |
| `emergency_stop_roas_threshold` | GuardrailConfig | 0.5 | ROAS floor for emergency stop |
| `emergency_stop_cpa_multiplier` | GuardrailConfig | 3.0 | CPA ceiling multiplier |
| `max_exploration_rate` | GuardrailConfig | 0.2 | Max exploration |
| `min_exploration_rate` | GuardrailConfig | 0.01 | Min exploration |
| `exploration_decay_rate` | GuardrailConfig | 0.995 | Exploration decay |
| **CLI / Env** | | | |
| `DRL_STATE_DIM` | run_drl_on_csv, run_cross_platform | 42 | State dimension for checkpoint |
| `--model-dir` | run_drl_on_csv | checkpoints/final_model | SAC checkpoint path |
| `--sac-model-dir` | run_cross_platform_optimizer | models/bq_run/final | SAC checkpoint path |

---

## M3: X-Training Data Parameters

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| **XTrainingDataBuilder** | x_training_data.py | | |
| `roas_weight` | XTrainingDataBuilder | 0.50 | Portfolio ROAS weight |
| `volume_weight` | XTrainingDataBuilder | 0.25 | Conversion volume weight |
| `efficiency_weight` | XTrainingDataBuilder | 0.15 | Efficiency weight |
| `stability_weight` | XTrainingDataBuilder | 0.10 | Allocation stability weight |
| **Transformation** | | | |
| Aggregation | XTrainingDataBuilder | Per-platform metrics | Portfolio snapshot |
| Transition logic | XTrainingDataBuilder | Consecutive snapshots | (s,a,r,s') tuples |

---

## M4: X-Training Parameters

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| **X-Model** | x_model.py | | |
| `X_STATE_DIM` | x_model.py | 70 | Portfolio state dimension |
| `X_ACTION_DIM` | x_model.py | 5 | Allocation weights (5 platforms) |
| `FEATURES_PER_PLATFORM` | x_model.py | 13 | Per-platform feature count |
| **CrossPlatformStrategyConfig** | config.py | | |
| `retrain_snapshot_threshold` | CrossPlatformStrategyConfig | 50 | Retrain after N snapshots |
| `retrain_min_transitions` | CrossPlatformStrategyConfig | 30 | Min transitions for retrain |
| `retrain_epochs` | CrossPlatformStrategyConfig | 10 | Epochs per retrain |
| `retrain_steps_per_epoch` | CrossPlatformStrategyConfig | 100 | Steps per epoch |
| `checkpoint_dir` | CrossPlatformStrategyConfig | models/x_model | X-Model checkpoint dir |
| `min_confidence_for_drl` | CrossPlatformStrategyConfig | 0.40 | Fallback threshold |
| `min_training_steps` | CrossPlatformStrategyConfig | 500 | Min steps for DRL use |

---

## M5: X-Execution Parameters

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| **CrossPlatformConfig** | cross_platform_optimizer.py | | |
| `max_single_shift_pct` | CrossPlatformConfig | 0.20 | Max budget shift per platform |
| `min_platform_budget_pct` | CrossPlatformConfig | 0.05 | Floor: 5% per platform |
| `max_platform_budget_pct` | CrossPlatformConfig | 0.80 | Ceiling: 80% per platform |
| `rebalance_cooldown_hours` | CrossPlatformConfig | 24.0 | Min hours between rebalances |
| `lookback_days` | CrossPlatformConfig | 14 | Days for marginal ROAS |
| `smoothing_alpha` | CrossPlatformConfig | 0.3 | EMA smoothing |
| `roas_weight` | CrossPlatformConfig | 0.50 | Allocation objective |
| `volume_weight` | CrossPlatformConfig | 0.25 | Conversion volume |
| `min_confidence_for_shift` | CrossPlatformConfig | 0.60 | Min confidence for shift |
| `emergency_roas_floor` | CrossPlatformConfig | 0.5 | Pull budget if ROAS < this |
| **CLI** | run_cross_platform_optimizer.py | | |
| `--project-id` | CLI | ad-metrics-pipeline | BigQuery project |
| `--table` | CLI | ad_metrics.cross_platform_states | BigQuery table |
| `--sample-input-csv` | CLI | Training_Data/input_template.csv | CSV input (--sample) |
| `--per-platform-limit` | CLI | 120 | Max campaigns per platform |
| `--total-budget` | CLI | 0 (inferred) | Portfolio budget |
| `--output-json` | CLI | Training_Data/outputs/cross_platform/cross_platform_result.json | Output path |
