# Testing Strategy for DRL Ad Optimization System

## Test Categories & Coverage

### 1. **Unit Tests** (Individual Component Validation)

#### State & Action Space (state_action.py)
```python
test_campaign_state_42dim_construction()
  - Verify 42-dim tensor creation (was 39)
  - Check constraint features: target_cpa_norm, min_roas_norm, daily_budget_limit_norm
  - Ensure normalization ranges [0, 1]

test_state_backward_compatibility()
  - Load legacy 36-dim tensors
  - Verify from_tensor() fills missing dims with defaults
  - Round-trip: create → to_tensor() → from_tensor() → identical

test_constraint_feature_encoding()
  - target_cpa_norm: log1p(target_cpa) / log1p(1000) ∈ [0, 1]
  - min_roas_norm: min_roas / 10.0 ∈ [0, 1]
  - daily_budget_limit_norm: log1p(limit) / log1p(MAX) ∈ [0, 1]

test_action_space_bounds()
  - Bid adjustment: [-30%, +50%]
  - Budget adjustment: [±30%]
  - Audience action: 0-3 (HOLD, EXPAND, REFINE, EXCLUDE)
  - Creative action: 0-3 (HOLD, ROTATE, PAUSE, TEST_NEW)
```

#### Safe DRL Agent (safe_agent.py)
```python
test_bid_clipping()
  - Input: +67% → Output: +50% (clipped)
  - Input: -45% → Output: -30% (clipped)
  - Input: +25% → Output: +25% (passthrough)

test_budget_clipping()
  - Input: ±45% → Output: ±30%
  - Input: ±15% → Output: ±15%

test_confidence_threshold()
  - confidence=0.92 → approved (≥0.7)
  - confidence=0.65 → requires_review=True (<0.7)
  - confidence=0.5 → requires_review=True

test_cooldown_enforcement()
  - Action applied at 10:00 AM
  - Next action blocked at 1:59 PM (cooldown=4h)
  - Action allowed at 2:00 PM
```

#### SAC Agent (sac_agent.py)
```python
test_sac_forward_pass()
  - Input: 42-dim state
  - Output: action (bid_adj, budget_adj, audience, creative)
  - Verify output bounds (tanh squashing)
  - Verify deterministic=True produces same action twice

test_sac_model_loading()
  - Load pre-trained .pt checkpoint
  - Verify weights loaded correctly
  - Inference <100ms on CPU

test_sac_inference_consistency()
  - Same state → same action (deterministic mode)
  - Different seeds → different actions (stochastic mode)
```

---

### 2. **Integration Tests** (Component Interaction)

#### Hybrid Optimizer (hybrid_optimizer.py - 5-Phase Pipeline)
```python
test_phase1_drl_decision()
  # DRL action generation
  - Input: CampaignState, CampaignContext
  - Verify action contains all 4 components
  - Verify validation.status is APPROVED or REQUIRES_REVIEW

test_phase2_tactical_execution()
  # LLM creative generation
  - Input: DRLDirective from Phase 1
  - Verify tactical output contains: headline, body_copy, cta, offer
  - Verify generation_time_ms < 500ms

test_phase3_xai_narrative()
  # Explainability
  - Verify narrative_dict has 5 keys: situation, decision, reasoning, confidence, reasonability
  - Verify narrative is human-readable (>20 words, <500 words)

test_phase4_forecasting()
  # NEW: Outcome predictions
  - Verify forecast is not None (if forecaster fitted)
  - Check forecast structure: expected_roas, expected_cpa, confidence_interval
  - Verify confidence_interval[0] < confidence_interval[1]
  - Verify convergence_days > 0

test_phase5_optimization_result()
  # Packaging
  - Verify OptimizationResult.to_dict() JSON serializable
  - Verify all phases included: directive, tactical, narrative, forecast
  - Verify total_latency_ms < 1000ms

test_full_hybrid_pipeline()
  # End-to-end
  - Run full 5-phase pipeline
  - Verify outputs at each phase
  - Measure total latency (target: <1s)
```

#### Cross-Platform Optimizer (cross_platform_optimizer.py)
```python
test_marginal_return_estimation()
  # Phase 3
  - Feed platform historical data (spend, revenue)
  - Verify estimates ∈ [0.5, 10.0] ROAS range
  - Confidence scores ∈ [0.0, 1.0]

test_budget_allocation()
  # Phase 5
  - Input: $100k total budget, 5 platforms with different ROAS
  - Verify allocations sum to total_budget (±$1 tolerance)
  - Verify high-ROAS platforms get more budget

test_state_re_derivation()
  # Phase 7 - NEW
  - Original budget: $1000
  - New budget: $1500
  - Verify log_daily_budget updated
  - Verify daily_budget_limit_norm updated
  - Verify values reflect new_budget, not original

test_parallel_campaign_optimization()
  # BatchOptimizer
  - Input: 100 campaigns
  - Run concurrently (not sequential)
  - Verify total time ≈ single_campaign_time (not 100x)
  - Verify all 100 results returned

test_cross_platform_result_structure()
  # Output validation
  - portfolio_snapshot: dict with all platform metrics
  - allocations: List[AllocationRecommendation]
  - platform_campaign_results: {platform: [OptimizationResult]}
  - audience_constraints: {platform: AudienceConstraintResult} (if manager enabled)
```

---

### 3. **Constraint Compliance Tests** (NEW - Updated Features)

#### Constraint Feature Encoding
```python
test_constraint_features_in_state()
  # Verify constraints are encoded in state (indices 39-41)
  state = CampaignState(
    target_cpa_norm=0.8,      # index 39
    min_roas_norm=0.5,        # index 40
    daily_budget_limit_norm=0.7  # index 41
  )
  tensor = state.to_tensor()
  assert tensor[39] == 0.8
  assert tensor[40] == 0.5
  assert tensor[41] == 0.7

test_constraint_aware_decision()
  # SAC should learn to respect constraints
  - state.target_cpa_norm = 1.0 (at limit)
  - state.current_cpa = target_cpa (tight)
  - Expect SAC to decrease spend (conservative action)

  - state.min_roas_norm = 0.1 (below threshold)
  - Expect SAC to decrease spend or improve efficiency

test_post_hoc_guardrails()
  # SafeDRLAgent still catches edge cases
  - SAC suggests: bid +60% (violates +50% max)
  - SafeDRLAgent clips to +50%
  - Guardrails log warning

test_constraint_feature_normalization()
  # All constraints should normalize to [0, 1]
  for target_cpa in [10, 50, 100, 500, 1000, 5000]:
    norm = log1p(target_cpa) / log1p(1000)
    assert 0 <= norm <= 1.5  # allow slight overflow
```

---

### 4. **Forecasting Integration Tests** (NEW)

```python
test_forecast_generation()
  # Phase 4 of hybrid optimizer
  - forecaster.predict(state) returns CampaignForecast
  - Forecast has: expected_roas, expected_cpa, confidence_interval, days_to_convergence

test_forecast_in_result()
  # OptimizationResult.forecast populated
  result = await hybrid.optimize(state, context, info)
  assert result.forecast is not None
  assert "expected_roas" in result.forecast
  assert "expected_cpa" in result.forecast
  assert "confidence_interval" in result.forecast

test_forecast_accuracy()
  # Backtesting (if historical outcomes available)
  - Run forecast on historical state
  - Compare predicted_roas vs actual_roas
  - Calculate MAPE (mean absolute percentage error)
  - Target: MAPE < 15%

test_forecast_confidence_bounds()
  # Verify confidence intervals are realistic
  - lower_bound < expected < upper_bound
  - width = upper_bound - lower_bound ∈ [0.1, 0.5]  (not too tight, not too wide)

test_forecast_convergence_time()
  # Verify days_to_convergence is reasonable
  - Most campaigns: 5-14 days
  - Outliers acceptable: 1-30 days
```

---

### 5. **End-to-End Pipeline Tests**

#### Full Portfolio Optimization Flow
```python
test_full_portfolio_optimization()
  # Simulate entire workflow
  organization_id = "test_org"
  campaigns = [
    (state_google_1, ctx_1, info_1),
    (state_google_2, ctx_2, info_2),
    (state_meta_1, ctx_3, info_3),
    ...  # 10 campaigns across 5 platforms
  ]
  total_budget = $100_000

  result = await cross_platform_optimizer.optimize_portfolio(
    organization_id=organization_id,
    campaigns=campaigns,
    total_budget=total_budget,
    force_rebalance=True
  )

  # Verify
  assert result.allocations is not None
  assert sum(a.recommended_budget for a in result.allocations) ≈ total_budget
  assert len(result.platform_campaign_results["google"]) == 2
  assert len(result.platform_campaign_results["meta"]) == 1
  assert result.projected_portfolio_roas > result.portfolio_roas  # improvement

test_portfolio_with_audience_constraints()
  # If AudienceConstraintManager enabled
  result = await optimizer.optimize_portfolio(...)
  assert result.audience_constraints is not None
  for platform, aud_result in result.audience_constraints.items():
    assert "segment_allocations" in aud_result
    assert sum(seg_budget for _, seg_budget in aud_result["segment_allocations"]) ≈ platform_budget

test_portfolio_result_json_serialization()
  # Verify API response is JSON-serializable
  result_dict = result.to_dict()
  import json
  json_str = json.dumps(result_dict)
  assert len(json_str) > 0
```

---

### 6. **Performance & Scale Tests**

```python
test_inference_latency_single_campaign()
  # SAC forward pass
  - Input: 42-dim state
  - Measure time
  - Target: <100ms CPU

test_llm_generation_latency()
  # Mock LLM
  - Target: <10ms (mock)

  # Real LLM (future)
  - Target: <500ms (Claude/GPT-4)

test_end_to_end_latency()
  # Full 5-phase pipeline
  - Target: <1000ms (with mock LLM)

test_batch_throughput()
  # BatchOptimizer parallel execution
  - 100 campaigns in parallel
  - Measure total time
  - Verify ≈ single_campaign_time (parallelization working)

  - 1000 campaigns
  - Target: 1000 campaigns / 100ms ≈ 100 campaigns/sec

test_memory_usage()
  # Monitor during optimization
  - State tensors
  - Model weights
  - Result accumulation
  - Target: <500MB for 1000 campaigns
```

---

### 7. **Continuous Learning Tests** (outcome_tracker.py, continuous_learning.py)

```python
test_action_recording()
  # OutcomeTracker.register_action()
  - Action applied to campaign
  - record_action() stores: action_id, campaign_id, state_before, timestamp

test_outcome_collection()
  # POST /record-outcome
  - New metrics received
  - state_after computed
  - outcome stored

test_reward_computation()
  # RewardComputer.compute()
  - Input: state_before, state_after, action
  - Output: reward (float)
  - Verify formula: ROAS_delta(1.0) + CPA_delta(1.0) + Conv_delta(0.8) + CTR_delta(0.5)

test_per_buffer_update()
  # Prioritized Experience Replay
  - Transition added to buffer
  - Priority computed (TD-error)
  - High-priority transitions sampled more often

test_retrain_trigger()
  # ContinuousLearningEngine
  - Mode: BATCH → retrain after 1000 outcomes
  - Mode: ONLINE → retrain after each action
  - Mode: TRIGGERED → retrain if reward_drop > 15%
  - Mode: HYBRID → adaptive

test_ab_testing_validation()
  # ab_testing.py
  - Train new model
  - A/B test: new_model vs current_model
  - Welch t-test on outcome metrics
  - Verify statistical significance (p < 0.05)
```

---

### 8. **Regression Tests** (Ensure Updates Don't Break Anything)

```python
test_old_models_still_load()
  # Legacy checkpoint (39-dim)
  - Load old .pt file
  - Verify loads without error
  - Verify inference works

test_state_dimension_migration()
  # Old state (39-dim) → New state (42-dim)
  - Load 39-dim tensor
  - Call from_tensor() → should fill dims 39-41 with defaults
  - to_tensor() should produce 42-dim

test_forecast_graceful_fallback()
  # If forecaster not fitted
  - forecast_dict = None (silently)
  - Pipeline continues without error
  - Result JSON still valid (forecast=null)

test_no_regression_in_latency()
  # With new constraints & forecasting
  - Measure latency vs baseline
  - Verify <10% regression
  - Target: still <1s end-to-end
```

---

### 9. **Safety & Guardrail Tests**

```python
test_bid_never_exceeds_max()
  for _ in range(1000):
    action = sac.sample_action(random_state)
    clipped = safe_agent.validate(action)
    assert -0.30 <= clipped.bid_change <= 0.50

test_budget_never_exceeds_max()
  for _ in range(1000):
    clipped = safe_agent.validate(action)
    assert -0.30 <= clipped.budget_change <= 0.30

test_cooldown_prevents_action_spam()
  # Action 1: allowed at 10:00
  # Action 2: rejected at 10:30 (cooldown=4h)
  # Action 3: allowed at 14:00

test_confidence_threshold_enforcement()
  # Low confidence → requires_review
  # High confidence → auto-apply
```

---

## Running Tests

### Framework
```bash
# pytest + pytest-asyncio for async tests
pip install pytest pytest-asyncio pytest-cov pytest-benchmark

# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v

# With coverage report
pytest --cov=drl tests/
```

### Test Structure
```
tests/
├── unit/
│   ├── test_state_action.py
│   ├── test_sac_agent.py
│   ├── test_safe_agent.py
│   └── test_reward_functions.py
├── integration/
│   ├── test_hybrid_optimizer.py
│   ├── test_cross_platform_optimizer.py
│   ├── test_constraint_compliance.py
│   └── test_forecasting.py
├── e2e/
│   ├── test_full_portfolio_optimization.py
│   ├── test_continuous_learning.py
│   └── test_ab_testing.py
├── performance/
│   ├── test_latency.py
│   ├── test_throughput.py
│   └── test_memory.py
└── conftest.py  # pytest fixtures
```

---

## Success Criteria

| Test Category | Pass Criteria |
|---|---|
| **Unit Tests** | 100% pass rate |
| **Integration Tests** | 100% pass rate |
| **Constraint Compliance** | All constraints respected post-clipping |
| **Forecasting** | MAPE < 15% on historical data |
| **Latency (SAC)** | <100ms per inference |
| **Latency (E2E)** | <1000ms with mock LLM |
| **Throughput** | 100 campaigns/sec with parallelization |
| **Safety** | No guardrail violations in 1000 samples |
| **Backward Compatibility** | Legacy 39-dim models load & infer correctly |
| **Memory** | <500MB for 1000 campaigns |

---

## Recommended Test Execution Order

1. **Unit Tests** (fast, foundational)
2. **Constraint Compliance Tests** (verify new features)
3. **Forecasting Integration Tests** (verify new integration)
4. **Integration Tests** (component interaction)
5. **E2E Tests** (full pipeline)
6. **Regression Tests** (ensure no breakage)
7. **Performance Tests** (latency, throughput)
8. **Safety Tests** (guardrails)
9. **Continuous Learning Tests** (feedback loop)

---

## CI/CD Integration

```yaml
# GitHub Actions example
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/ --cov=drl --cov-report=xml
      - uses: codecov/codecov-action@v2
```

---

## Quick Start: Run Essential Tests

```bash
# Fast validation (5 min)
pytest tests/unit/ tests/integration/test_constraint_compliance.py tests/integration/test_forecasting.py -v

# Full validation (15 min)
pytest tests/ -v --cov=drl

# Performance baseline
pytest tests/performance/ -v --benchmark-only
```
