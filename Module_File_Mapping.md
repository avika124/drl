# DRL Advertising Optimization System — Module File Mapping

## Overview

This document maps Python files to the 5-module architecture (M1–M5).

| Module | Description | Primary Role |
|--------|-------------|--------------|
| **M1** | P-Training | Single-platform SAC model training |
| **M2** | P-Execution | Single-platform campaign optimization (inference) |
| **M3** | X-Training Data | Portfolio-level transition generation |
| **M4** | X-Training | Cross-platform X-Model training |
| **M5** | X-Execution | Cross-platform orchestration & allocation |

---

## M1: P-Training (Single-Platform Training)

| File | Role | Description |
|------|------|-------------|
| `train.py` | Entry point | On-policy SAC training with MockCampaignEnv (synthetic) |
| `train_bigquery_offline.py` | Entry point | Offline SAC training from BigQuery campaign states |
| `offline_trainer.py` | Core | Offline training pipeline (CQL, PER, validation) |
| `bigquery_loader.py` | Data | Fetches & normalizes campaign states from BigQuery |
| `generate_synthetic_action_logs.py` | Data | Creates synthetic (campaign_id, date, actions) CSV |
| `create_cross_platform_states.py` | Data | Builds 42-dim state vectors from raw metrics |
| `sac_agent.py` | Core | SAC agent (Actor, Critic, CQL, PER) |
| `networks.py` | Core | ActorNetwork, CriticNetwork |
| `replay_buffer.py` | Core | PrioritizedReplayBuffer, Transition |
| `reward_functions.py` | Core | RewardComputer, multi-objective reward |
| `state_action.py` | Core | CampaignState, ActionSpace definitions |
| `config.py` | Config | DRLConfig, TrainingConfig |

---

## M2: P-Execution (Single-Platform Execution)

| File | Role | Description |
|------|------|-------------|
| `hybrid_optimizer.py` | Entry point | DRL + LLM hybrid (strategic + tactical) |
| `sac_agent.py` | Core | select_action(), load_sac_for_inference() |
| `safe_agent.py` | Core | SafeDRLAgent, ActionValidator, guardrails |
| `run_drl_on_csv.py` | Entry point | Run SAC on CSV input → output actions CSV |
| `state_action.py` | Core | CampaignState, ActionSpace, DRLDirective |
| `xai_narrator.py` | Output | OptimizationNarrative, human-readable explanations |
| `benchmark_model.py` | Output | CampaignForecaster, outcome predictions |
| `config.py` | Config | GuardrailConfig |

---

## M3: X-Training Data (Portfolio Data Preparation)

| File | Role | Description |
|------|------|-------------|
| `x_training_data.py` | Core | XTrainingDataBuilder, XTransition |
| `create_cross_platform_states.py` | Data | State vector construction (shared with M1) |
| `cross_platform_optimizer.py` | Source | Allocation history, PlatformPerformanceTracker |
| `x_model.py` | Core | build_x_state(), XModelState |

---

## M4: X-Training (Cross-Platform Model Training)

| File | Role | Description |
|------|------|-------------|
| `x_training.py` | Core | XModelTrainer, XModelReplayBuffer |
| `x_model.py` | Core | XModelAgent, X_STATE_DIM, X_ACTION_DIM |
| `cross_platform_drl_engine.py` | Orchestrator | Retrain triggers, training lifecycle |
| `config.py` | Config | CrossPlatformStrategyConfig |

---

## M5: X-Execution (Cross-Platform Orchestration)

| File | Role | Description |
|------|------|-------------|
| `cross_platform_optimizer.py` | Core | CrossPlatformOptimizer, BudgetAllocator, MarginalReturnEstimator |
| `cross_platform_drl_engine.py` | Orchestrator | XModelAgent primary allocation, dual benchmark |
| `run_cross_platform_optimizer.py` | Entry point | CLI: BigQuery or CSV → cross-platform result JSON |
| `platform_model_registry.py` | Registry | Per-platform P-Model loading |
| `hybrid_optimizer.py` | Delegation | Per-campaign P-Execution via HybridDRLLLMOptimizer |
| `audience_constraints.py` | Constraints | Segment allocation, AudienceConstraintManager |
| `x_model.py` | Core | XModelAgent.select_allocation() |

---

## Shared / Supporting Files

| File | Used By | Description |
|------|---------|-------------|
| `config.py` | M1, M2, M4, M5 | DRLConfig, TrainingConfig, GuardrailConfig, etc. |
| `state_action.py` | M1, M2, M5 | CampaignState, ActionSpace |
| `drl_integration.py` | Backend | Integration with FastAPI backend |
| `continuous_learning.py` | M1, M2 | OutcomeTracker, retrain triggers |
| `ab_testing.py` | M4, M5 | DRLABTestManager, model validation |
| `forecast_feedback.py` | M2 | ForecastFeedbackLoop |
| `quantile_budget_model.py` | M5 | Quantile regression forecasting |

---

## Entry Points Summary

| Module | CLI / Entry | Command Example |
|--------|-------------|-----------------|
| M1 | `train.py` | `python -m drl.train` |
| M1 | `train_bigquery_offline.py` | `python -m drl.train_bigquery_offline` |
| M2 | `run_drl_on_csv.py` | `python -m drl.run_drl_on_csv --input X --output Y` |
| M2 | `hybrid_optimizer.py` | Via API / CrossPlatformOptimizer |
| M5 | `run_cross_platform_optimizer.py` | `python -m drl.run_cross_platform_optimizer --sample --sample-input-csv X` |
