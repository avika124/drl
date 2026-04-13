# Data Flow: M1 (P-Training) → M2 (P-Execution)

This document traces each data element from source to destination across the M1 → M2 pipeline.

---

## Pipeline Overview

```
M1: P-Training                    M2: P-Execution
┌─────────────────────┐          ┌─────────────────────────────────────┐
│ MockCampaignEnv     │          │ load_sac_for_inference(m1_outputs)  │
│   or                │          │         ↓                           │
│ BigQuery states     │          │ SafeDRLAgent + HybridDRLLLMOptimizer │
│         ↓           │          │         ↓                           │
│ Transitions         │   →      │ Per-campaign OptimizationResult     │
│         ↓           │          │         ↓                           │
│ SACAgent.train()    │          │ CrossPlatformOptimizer.optimize_     │
│         ↓           │          │   portfolio()                      │
│ agent.pt checkpoint │ ────────→│         ↓                           │
└─────────────────────┘          │ cross_platform_result.json         │
                                  └─────────────────────────────────────┘
```

---

## Step-by-Step Data Flow

### STEP 1: M1 — MockCampaignEnv generates states

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| State | Internal (MockCampaignEnv) | CampaignState with ctr, cvr, roas, cpa, etc. (42-dim) | SACAgent.select_action() |
| Action | SACAgent.select_action() | ActionSpace (bid_adj, budget_adj, audience, creative) | env.step() |
| Reward | env.step() | profit/100 normalized | Transition.reward |
| Next state | env.step() | CampaignState | Transition.next_state |

**Parameters:** config.py DRLConfig, TrainingConfig; train.py overrides (batch_size=64, min_buffer=100)

---

### STEP 2: M1 — Build transitions and train SAC

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| Transitions | Replay buffer (from Step 1) | (state, action, reward, next_state, done) | agent.update() |
| Batch | replay_buffer.sample(64) | Tensor batch | _update_critic(), _update_actor() |
| Actor/Critic grads | SAC loss | Backprop | actor.pt, critic state_dict |

**Parameters:** batch_size=64, gamma=0.99, tau=0.005, use_per=True

---

### STEP 3: M1 — Save checkpoint

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| agent.pt | SACAgent (actor, critic, optimizers) | torch.save() | drl/m1_outputs/agent.pt |
| training_info.json | agent.training_info | json.dump() | drl/m1_outputs/training_info.json |

**Output:** M1 checkpoint → consumed by M2 load_sac_for_inference()

---

### STEP 4: M2 — Load SAC for inference

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| agent.pt | drl/m1_outputs/ (from M1) | torch.load(), agent.load() | SACAgent (eval mode) |
| state_dim | DRL_STATE_DIM env or default 42 | Config | DRLConfig |

**Output:** SACAgent instance → SafeDRLAgent(agent=...)

---

### STEP 5: M2 — Load campaign state vectors

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| CSV | --sample-input-csv (e.g. _demo_cross_platform_input_overrides.csv) | pd.read_csv(), _load_campaigns_from_sample_csv() | DataFrame |
| Campaign tuples | DataFrame | _build_campaign_tuples() | List[(CampaignState, CampaignContext, campaign_info)] |

**Columns used:** ctr, cvr, roas, cpa, cpc, cpm, platform_encoding, campaign_id, impressions, clicks, etc.

---

### STEP 6: M2 — Per-campaign optimization (HybridDRLLLMOptimizer)

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| CampaignState | _build_campaign_tuples | — | SafeDRLAgent.get_action() |
| CampaignContext | _build_campaign_tuples | current_bid, current_budget, etc. | ActionValidator.validate() |
| Raw action | SACAgent.select_action(state) | ActionSpace | SafeDRLAgent |
| Validated action | ActionValidator.validate() | Clipped to GuardrailConfig bounds | OptimizationResult.action |
| Tactical (LLM) | LLMClient.generate() | TacticalExecution (headline, body, CTA) | OptimizationResult.tactical |
| Narrative | OptimizationNarrator | RunNarrative | OptimizationResult.narrative |
| Forecast | CampaignForecaster | CampaignForecast | OptimizationResult.forecast |

**Parameters:** GuardrailConfig (max_bid_increase_pct=0.5, min_confidence_for_action=0.7)

---

### STEP 7: M2 — Cross-platform portfolio allocation

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| Platform metrics | Per-campaign results | Aggregate by platform | MarginalReturnEstimator |
| Total budget | --total-budget or inferred spend | — | BudgetAllocator |
| Allocation | BudgetAllocator.allocate() | Recommended share per platform | allocations[] |
| Portfolio ROAS | Weighted avg of platform ROAS | — | portfolio_roas |

**Parameters:** CrossPlatformConfig (max_single_shift_pct=0.2, emergency_roas_floor=0.5)

---

### STEP 8: M2 — Final output

| Element | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|---------|--------------|----------------|--------------------|
| Result payload | CrossPlatformResult, allocations, platform_metrics | result.to_dict() | cross_platform_result.json |
| JSON file | --output-json (e.g. drl/m2_outputs/cross_platform_result.json) | json.dump() | File system |

**Output:** cross_platform_result.json → API, reporting, downstream systems

---

## Alternative M1 Data Sources (BigQuery Path)

| Step | INPUT SOURCE | TRANSFORMATION | OUTPUT DESTINATION |
|------|--------------|----------------|--------------------|
| States | BigQuery (DRL_BQ_TABLE) | BigQueryDataLoader.fetch_dataframe(), transform() | Normalized state vectors |
| Action logs | DRL_ACTION_LOGS_PATH (optional) | generate_synthetic_action_logs.py or real logs | build_transitions() action lookup |
| Transitions | build_transitions() | Consecutive (s,a,r,s') per campaign | OfflineTrainer.load_transitions() |
| Checkpoint | agent.save() | models/bq_run/final/ | M2 --sac-model-dir |

---

## File Artifacts Summary

| Artifact | Produced By | Consumed By |
|----------|-------------|-------------|
| agent.pt | M1 train.py / train_bigquery_offline | M2 load_sac_for_inference() |
| training_info.json | M1 | Diagnostics, monitoring |
| synthetic_action_logs.csv | generate_synthetic_action_logs.py | train_bigquery_offline (DRL_ACTION_LOGS_PATH) |
| cross_platform_states.csv | create_cross_platform_states.py | run_cross_platform_optimizer (alternative input) |
| cross_platform_result.json | M2 run_cross_platform_optimizer | API, dashboards |
