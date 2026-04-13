# DRL Training Data Sources Guide

A comprehensive guide to acquiring training data for the 33-dimensional campaign optimization DRL system.

## Reality Check: No Single Source Has Everything

**The hard truth**: No public dataset contains all 39 features you need. Your state space combines:
- Platform metrics (CTR, CVR, ROAS, CPA)
- Competitive intelligence (impression share, auction pressure)
- ML-derived predictions (audience quality, creative fatigue, propensity)
- Temporal/contextual features

This means you'll need a **multi-source strategy**:

---

## Option 1: Platform APIs (Best for Real Data)

### Google Ads API
**Coverage**: ~60% of features

| Feature Category | Available | API Resource |
|-----------------|-----------|--------------|
| Core Metrics (CTR, CVR, CPA, CPC, CPM) | ✅ | `campaign`, `metrics` |
| Volume (impressions, clicks, conversions, spend) | ✅ | `metrics` |
| Temporal (date, segments) | ✅ | `segments.date` |
| Budget/Bid | ✅ | `campaign.target_cpa`, `campaign_budget` |
| Impression Share | ✅ | `metrics.search_impression_share` |
| Auction Insights | ✅ | `auction_insights` |
| Historical Trends | ⚠️ Compute yourself | Daily metrics over time |
| ML Predictions | ❌ | Not available |
| Creative Fatigue | ❌ | Not available |

```python
# Example GAQL Query for Campaign-Level Data
query = """
SELECT
  campaign.id,
  campaign.name,
  campaign.status,
  campaign.target_cpa.target_cpa_micros,
  campaign_budget.amount_micros,
  segments.date,
  metrics.impressions,
  metrics.clicks,
  metrics.conversions,
  metrics.cost_micros,
  metrics.conversions_value,
  metrics.average_cpc,
  metrics.ctr,
  metrics.search_impression_share,
  metrics.search_rank_lost_impression_share
FROM campaign
WHERE segments.date DURING LAST_90_DAYS
"""
```

### Meta Marketing API
**Coverage**: ~55% of features

| Feature Category | Available | Endpoint |
|-----------------|-----------|----------|
| Core Metrics | ✅ | `/insights` |
| Volume | ✅ | `/insights` |
| Frequency/Reach | ✅ | `frequency`, `reach` |
| Budget/Bid | ✅ | Campaign/AdSet objects |
| Audience Insights | ⚠️ Limited | `/delivery_insights` |
| Creative Performance | ✅ | Ad-level insights |

### Amazon Ads API
**Coverage**: ~50% of features
- Sponsored Products/Brands/Display metrics
- ACOS, ROAS, conversion data
- Share of voice (for search terms)

### The Trade Desk API
**Coverage**: ~45% of features
- Programmatic campaign metrics
- Bid landscape data
- Audience segment performance

---

## Option 2: Public Datasets (For Pre-training)

### Criteo Datasets (Best Available)

**1. Criteo 1TB Click Logs**
- **Size**: 4 billion events, 1TB
- **Source**: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
- **Features**: 13 numerical + 26 categorical (anonymized)
- **Use Case**: CTR prediction pre-training

**2. CriteoPrivateAd Dataset (NEW - Feb 2025)**
- **Size**: 100M displays, 30 days
- **Source**: https://arxiv.org/html/2502.12103v1
- **Features**: 100+ features including:
  - Click/conversion labels
  - User/publisher/campaign IDs (hashed)
  - Time deltas
- **Best for**: Bidding model training

**3. Criteo Sponsored Search Conversion Log**
- **Size**: 90 days of traffic
- **Source**: https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/
- **Features**: Product characteristics, user behavior, conversion timing
- **Best for**: Conversion prediction, LTV modeling

### Kaggle Datasets

| Dataset | Features | Records | Link |
|---------|----------|---------|------|
| Marketing Campaign Performance | CTR, CVR, spend, revenue | 10K+ | kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset |
| Google Analytics Sample | Traffic, transactions, sessions | 1M+ | BigQuery public dataset |
| Advertising Dataset | TV/Radio/Newspaper spend vs sales | 200 | kaggle.com/datasets/ashydv/advertising-dataset |

### Feature Coverage Gap

| Your Feature | Public Dataset Coverage |
|--------------|------------------------|
| CTR, CVR | ✅ Criteo, Kaggle |
| ROAS, CPA | ⚠️ Must compute from spend/revenue |
| Impression Share | ❌ Not in public data |
| Auction Pressure | ❌ Not in public data |
| Audience Quality Score | ❌ Must train separately |
| Creative Fatigue | ❌ Must compute |
| Predicted CVR/LTV | ❌ Must train separately |
| Propensity Score | ❌ Must train separately |

---

## Option 3: Synthetic Data Generation (Recommended for Full Coverage)

Given the feature gaps, **synthetic data generation is your best path to a complete 33-feature dataset**.

### Approach 1: Hybrid Real + Synthetic

```python
# 1. Use Criteo data for base patterns
# 2. Generate missing features synthetically

import numpy as np
from scipy import stats

def generate_synthetic_campaign_state(real_metrics):
    """
    Extend real campaign metrics with synthetic features
    """
    state = {}
    
    # === Real features from API/dataset ===
    state['ctr'] = real_metrics['clicks'] / max(real_metrics['impressions'], 1)
    state['cvr'] = real_metrics['conversions'] / max(real_metrics['clicks'], 1)
    state['cpa'] = real_metrics['spend'] / max(real_metrics['conversions'], 1)
    state['roas'] = real_metrics['revenue'] / max(real_metrics['spend'], 1)
    
    # === Synthetic competitive features ===
    # Model impression share as function of bid competitiveness
    bid_competitiveness = real_metrics.get('cpc', 1.0) / 2.0  # Normalize
    state['impression_share'] = np.clip(
        0.3 + 0.5 * bid_competitiveness + np.random.normal(0, 0.1),
        0.05, 0.95
    )
    
    # Auction pressure correlates with impression share
    state['auction_pressure'] = np.clip(
        1 - state['impression_share'] + np.random.normal(0, 0.1),
        0.1, 0.9
    )
    
    # === Synthetic ML-derived features ===
    # Audience quality correlates with CVR
    state['audience_quality_score'] = np.clip(
        state['cvr'] * 10 + np.random.normal(0, 0.1),
        0, 1
    )
    
    # Creative fatigue increases with campaign age
    campaign_age = real_metrics.get('campaign_age_days', 30)
    state['creative_fatigue_score'] = np.clip(
        0.1 + (campaign_age / 90) * 0.6 + np.random.normal(0, 0.1),
        0, 1
    )
    
    # Predicted CVR with noise around actual
    state['predicted_cvr'] = np.clip(
        state['cvr'] + np.random.normal(0, state['cvr'] * 0.2),
        0, 1
    )
    
    # LTV correlates with ROAS and audience quality
    state['predicted_ltv'] = np.clip(
        state['roas'] * 0.3 + state['audience_quality_score'] * 0.5 + 
        np.random.normal(0, 0.1),
        0, 1
    )
    
    # Propensity correlates with CVR and audience quality
    state['propensity_score'] = np.clip(
        state['cvr'] * 5 + state['audience_quality_score'] * 0.3 + 
        np.random.normal(0, 0.1),
        0, 1
    )
    
    return state
```

### Approach 2: Full Simulation Environment

Build a campaign simulation that generates realistic trajectories:

```python
class CampaignSimulator:
    """
    Simulates advertising campaign dynamics for DRL training
    """
    
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Initialize new campaign"""
        # Sample campaign characteristics
        self.platform = np.random.choice(['google', 'meta', 'amazon', 'tiktok'])
        self.goal = np.random.choice(['roas', 'cpa', 'conversions'])
        self.daily_budget = np.random.uniform(100, 10000)
        self.target_cpa = np.random.uniform(10, 100)
        
        # Initialize state
        self.day = 0
        self.total_days = np.random.randint(30, 120)
        self.base_ctr = np.random.uniform(0.005, 0.05)
        self.base_cvr = np.random.uniform(0.01, 0.10)
        
        return self._get_state()
    
    def step(self, action):
        """
        Take action, advance simulation, return (next_state, reward, done)
        
        Actions:
        - bid_adjustment: [-0.5, 0.5]
        - budget_adjustment: [-0.3, 0.3]
        - audience_action: [0, 1, 2, 3]
        - creative_action: [0, 1, 2, 3]
        """
        bid_adj = action['bid_adjustment']
        budget_adj = action['budget_adjustment']
        
        # Model effect of bid change on impressions/CTR
        impression_multiplier = 1 + bid_adj * 0.8
        cpc_multiplier = 1 + bid_adj * 0.6
        
        # Model effect of budget change
        spend_multiplier = 1 + budget_adj
        
        # Model audience action effects
        if action['audience_action'] == 1:  # EXPAND
            impressions_boost = 1.2
            ctr_penalty = 0.9  # Broader = lower CTR
        elif action['audience_action'] == 2:  # REFINE
            impressions_boost = 0.8
            ctr_penalty = 1.1  # Narrower = higher CTR
        else:
            impressions_boost = 1.0
            ctr_penalty = 1.0
        
        # Model creative action effects
        if action['creative_action'] == 1:  # ROTATE
            self.creative_fatigue *= 0.7  # Reduce fatigue
        elif action['creative_action'] == 3:  # TEST_NEW
            self.creative_fatigue *= 0.5
        
        # Compute new metrics with noise
        self.impressions = int(
            self.base_impressions * impression_multiplier * 
            impressions_boost * (1 + np.random.normal(0, 0.1))
        )
        self.ctr = self.base_ctr * ctr_penalty * (1 - self.creative_fatigue * 0.3)
        self.clicks = int(self.impressions * self.ctr)
        self.cvr = self.base_cvr * (1 + np.random.normal(0, 0.1))
        self.conversions = int(self.clicks * self.cvr)
        
        # Update fatigue
        self.creative_fatigue = min(1.0, self.creative_fatigue + 0.02)
        
        self.day += 1
        done = self.day >= self.total_days
        
        # Compute reward
        reward = self._compute_reward()
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        """Return full 33-dimensional state"""
        # ... construct state vector
        pass
    
    def _compute_reward(self):
        """Multi-objective reward"""
        # ... reward computation
        pass
```

### Approach 3: GAN-Based Generation

Use Generative Adversarial Networks to create realistic campaign data:

```python
# Libraries for synthetic data generation
# pip install sdv ctgan nbsynthetic

from sdv.tabular import CTGAN
import pandas as pd

# Train on real campaign data (even partial)
real_data = pd.read_csv('campaign_metrics.csv')

# Initialize and train CTGAN
model = CTGAN()
model.fit(real_data)

# Generate synthetic campaigns
synthetic_data = model.sample(num_rows=100000)

# Validate statistical similarity
from sdv.evaluation import evaluate
score = evaluate(synthetic_data, real_data)
print(f"Similarity score: {score}")
```

---

## Option 4: Data Partnerships & Purchase

### Commercial Data Providers

| Provider | Data Type | Cost | Coverage |
|----------|-----------|------|----------|
| **LiveRamp** | Identity, attribution | $$$$ | Audience quality, cross-device |
| **Nielsen** | Competitive intel | $$$$ | Market share, benchmarks |
| **Similarweb** | Traffic, engagement | $$$ | Competitive metrics |
| **SpyFu / SEMrush** | PPC competitive | $$ | Bid estimates, impression share |
| **Pathmatics** | Creative intelligence | $$$ | Ad spend, creative performance |

### Agency Partnerships

Approach digital marketing agencies who manage multiple clients:
- WPP, Omnicom, Publicis, Dentsu
- Offer anonymized/aggregated data in exchange for optimization tools
- Most agencies have 3+ years of campaign data across platforms

### Retail Media Networks

Given your platform focus, partner with:
- **Walmart Connect** - First-party retail data
- **Instacart Ads** - CPG campaign performance
- **Kroger Precision Marketing** - Grocery retail media
- **Amazon DSP** - E-commerce attribution

---

## Recommended Data Strategy

### Phase 1: Bootstrap (Weeks 1-4)

1. **Generate synthetic dataset** (100K campaigns)
   - Use simulator with realistic dynamics
   - Ensure proper correlations between features
   - Validate against published benchmarks

2. **Augment with Criteo data** for CTR/CVR patterns
   - Extract feature distributions
   - Use as prior for synthetic generation

3. **Pre-train DRL** on synthetic data
   - Focus on learning general optimization patterns
   - Don't expect production-quality policies

### Phase 2: Real Data Integration (Weeks 5-12)

1. **Connect platform APIs**
   - Google Ads → BigQuery pipeline
   - Meta Marketing API integration
   - Build daily ETL for metrics

2. **Fill gaps with ML models**
   - Train audience quality model (LTV prediction)
   - Train creative fatigue model (CTR decay over time)
   - Train propensity model (conversion likelihood)

3. **Fine-tune DRL** on real campaign data
   - Start with conservative exploration
   - Use CQL for offline fine-tuning

### Phase 3: Production Learning (Ongoing)

1. **Continuous data collection** from live campaigns
2. **Online learning** with safety guardrails
3. **A/B testing** to validate improvements

---

## Feature Engineering Recommendations

### Compute Missing Features

```python
def compute_derived_features(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing features from raw metrics
    """
    df = daily_metrics.copy()
    
    # === Trends (7-day rolling) ===
    for metric in ['ctr', 'cvr', 'roas', 'cpa', 'spend']:
        df[f'{metric}_trend_7d'] = df.groupby('campaign_id')[metric].transform(
            lambda x: x.pct_change(periods=7)
        )
    
    # === Creative Fatigue Proxy ===
    # CTR decay relative to first 7 days
    first_week_ctr = df.groupby('campaign_id')['ctr'].transform(
        lambda x: x.head(7).mean()
    )
    df['creative_fatigue_score'] = 1 - (df['ctr'] / first_week_ctr)
    df['creative_fatigue_score'] = df['creative_fatigue_score'].clip(0, 1)
    
    # === Budget Utilization ===
    df['budget_utilization'] = df['spend'] / df['daily_budget']
    
    # === Spend Velocity ===
    df['spend_velocity'] = df.groupby('campaign_id')['spend'].transform(
        lambda x: x / x.rolling(7).mean()
    )
    
    # === Competitive Position (if you have benchmarks) ===
    industry_avg_ctr = 0.02  # Example
    industry_avg_cvr = 0.03
    df['competitive_position'] = (
        (df['ctr'] / industry_avg_ctr) * 0.5 +
        (df['cvr'] / industry_avg_cvr) * 0.5
    ).clip(0, 1)
    
    return df
```

---

## Summary: Data Source Matrix

| Feature Group | Best Source | Alternative | Synthetic Fallback |
|--------------|-------------|-------------|-------------------|
| Core Metrics | Platform APIs | Criteo | ✅ |
| Volume | Platform APIs | Criteo | ✅ |
| Temporal | Platform APIs | Any | ✅ |
| Trends | Compute from daily | - | ✅ |
| Competitive | Google Auction Insights | SpyFu/SEMrush | ✅ |
| ML-Derived | Train your own models | - | ✅ |
| Campaign Context | Platform APIs | - | ✅ |

**Bottom Line**: Start with synthetic data to validate your architecture, then progressively incorporate real data as you build platform integrations. The DRL algorithm will transfer learn from synthetic to real with proper fine-tuning.
