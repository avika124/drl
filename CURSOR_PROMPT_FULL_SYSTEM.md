# DRL Campaign Optimization Engine — Complete System Build

## **PROJECT OVERVIEW**

Build a **complete, production-ready campaign optimization system** that:
1. **Pulls real campaign data** from multiple platforms (Google Ads, Facebook, TikTok)
2. **Transforms data into 42-dimensional state vectors** using DRL encoding
3. **Runs SAC agent inference** to generate optimization recommendations
4. **Applies safety guardrails** to ensure constraints are met
5. **Generates actionable moves** with reasoning and confidence
6. **Tracks performance** of applied recommendations
7. **Shows ROI impact** of optimizations

User sees: **"Here's your campaign data → Here's what DRL recommends → Here's why → Here's the predicted impact → Here's the actual results"**

---

## **SYSTEM ARCHITECTURE (END-TO-END)**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        USER OPENS DASHBOARD                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA INGESTION (Pull from ad platforms)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Google Ads API     Facebook Ads API     TikTok Ads API     LinkedIn API    │
│       ↓                    ↓                    ↓                ↓          │
│  [Get campaigns]   [Get campaigns]    [Get campaigns]   [Get campaigns]    │
│  [Get daily stats] [Get daily stats] [Get daily stats] [Get daily stats]   │
│  [Get conversions] [Get conversions][Get conversions] [Get conversions]    │
│                                                                              │
│  Raw Data Returned:                                                        │
│  - spend, impressions, clicks, conversions                                 │
│  - CTR, CPA, ROAS, CPM, CPC                                               │
│  - audience segments, creative IDs, placements                             │
│  - bid strategies, budget allocation                                       │
│  - historical performance (last 7/14/30 days)                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: STATE ENCODING (Convert to 42-dimensional DRL state)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  For each campaign:                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ INPUT: Google Ads campaign "Summer Sale - Search"                 │    │
│  │ - Spend: $5,432.50                                                │    │
│  │ - Impressions: 142,340                                            │    │
│  │ - Clicks: 4,210                                                   │    │
│  │ - Conversions: 287                                                │    │
│  │ - Conversion Value: $43,050                                       │    │
│  │                                                                    │    │
│  │ ENCODED STATE VECTOR (42 dimensions):                             │    │
│  │ [0.0280, 0.0320, 2.30, 24.50, ...]                              │    │
│  │  ↓       ↓       ↓     ↓                                           │    │
│  │  CTR    CVR    ROAS   CPA   ...                                   │    │
│  │                                                                    │    │
│  │ INDEX 0-5 (Core Metrics):                                         │    │
│  │   [CTR, CVR, ROAS, CPA, CPC, CPM]                               │    │
│  │                                                                    │    │
│  │ INDEX 6-9 (Volume):                                              │    │
│  │   [spend_velocity, impression_vol, click_vol, conv_vol]          │    │
│  │                                                                    │    │
│  │ INDEX 10-15 (Temporal):                                          │    │
│  │   [hour_of_day, day_of_week, day_of_month, is_weekend,          │    │
│  │    is_holiday, days_remaining]                                   │    │
│  │                                                                    │    │
│  │ INDEX 16-20 (Trends - 7 day):                                    │    │
│  │   [ctr_trend, cvr_trend, roas_trend, cpa_trend, spend_trend]    │    │
│  │                                                                    │    │
│  │ INDEX 21-23 (Competitive):                                       │    │
│  │   [impression_share, auction_pressure, competitive_position]    │    │
│  │                                                                    │    │
│  │ INDEX 24-28 (ML Scores):                                         │    │
│  │   [audience_quality, creative_fatigue, predicted_cvr,            │    │
│  │    predicted_ltv, propensity_score]                              │    │
│  │                                                                    │    │
│  │ INDEX 29-32 (Context):                                           │    │
│  │   [optimization_goal, platform, campaign_maturity, budget_util]  │    │
│  │                                                                    │    │
│  │ INDEX 33-35 (Spend):                                             │    │
│  │   [log_daily_spend, log_total_spend, log_daily_budget]           │    │
│  │                                                                    │    │
│  │ INDEX 36-38 (Audience):                                          │    │
│  │   [segment_count, top_segment_roas, avg_frequency]               │    │
│  │                                                                    │    │
│  │ INDEX 39-41 (Constraints):                                       │    │
│  │   [target_cpa_norm, min_roas_norm, daily_budget_limit_norm]     │    │
│  │                                                                    │    │
│  │ READY FOR DRL: state_vector (42,) ✅                             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: DRL INFERENCE (Run trained SAC agent)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: state_vector (42,)                                                │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │ LOAD SAC AGENT FROM CHECKPOINT                              │          │
│  │ - agent.pt (trained weights)                                │          │
│  │ - Actor network: 42-d input → continuous action space      │          │
│  │ - Critic network: 42-d input + action → Q-value            │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                            ↓                                               │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │ FORWARD PASS                                                │          │
│  │ action_tensor = actor(state_vector)                         │          │
│  │ → actor outputs continuous values for each campaign action  │          │
│  │ → maps to: [bid_multiplier, budget_delta, creative_id]     │          │
│  │                                                              │          │
│  │ Example output:                                             │          │
│  │ action = [1.15, 500.0, 3]                                  │          │
│  │          ↓     ↓      ↓                                      │          │
│  │        bid   budget creative                                │          │
│  │        +15%  +$500   swap to ID 3                          │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                            ↓                                               │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │ EXTRACT Q-VALUE (Confidence Score)                          │          │
│  │ q_value = critic(state_vector, action_tensor)              │          │
│  │ → Higher Q-value = higher confidence in recommendation     │          │
│  │ confidence_score = sigmoid(q_value) → [0, 1]              │          │
│  │                                                              │          │
│  │ In this case: q_value = 2.34 → confidence = 0.91 (91%)    │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  OUTPUT: action_vector (continuous) + confidence_score (0-1)               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: SAFETY GUARDRAILS (SafeDRLAgent validation)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: action_vector [1.15, 500.0, 3], constraints {target_cpa: $25}     │
│                                                                              │
│  CHECK 1: Bid Multiplier Bounds                                           │
│  ├─ Action says: increase bid by 15% (1.15x)                             │
│  ├─ Guardrail limit: max ±50% (0.5 to 1.5)                              │
│  ├─ Status: ✅ PASS (1.15 is within bounds)                             │
│  └─ Clipped value: 1.15 (no change)                                      │
│                                                                              │
│  CHECK 2: Budget Increase Bounds                                          │
│  ├─ Action says: increase budget by $500                                 │
│  ├─ Current daily budget: $2,000                                         │
│  ├─ Guardrail limit: ±30% of current budget = ±$600                    │
│  ├─ Status: ✅ PASS ($500 < $600)                                       │
│  └─ Clipped value: $500 (no change)                                      │
│                                                                              │
│  CHECK 3: CPA Constraint                                                  │
│  ├─ Current CPA: $24.50                                                  │
│  ├─ Target CPA: $25.00                                                   │
│  ├─ Predicted CPA after bid increase: $23.80                            │
│  ├─ Status: ✅ PASS (stays below $25 target)                            │
│  └─ Safe to apply                                                         │
│                                                                              │
│  CHECK 4: Cooldown Period                                                 │
│  ├─ Last optimization: 6 hours ago                                       │
│  ├─ Cooldown limit: 4 hours minimum between changes                     │
│  ├─ Status: ✅ PASS (6 > 4)                                             │
│  └─ Can optimize now                                                      │
│                                                                              │
│  OUTPUT: validated_action [1.15, 500.0, 3] ✅ All checks passed           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: HYBRID OPTIMIZER (Generate human-readable recommendations)          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: Translate action to business moves                              │
│  ┌────────────────────────────────────────────────────────────┐           │
│  │ action [1.15, 500.0, 3] →                                 │           │
│  │                                                             │           │
│  │ MOVE 1: BID ADJUSTMENT                                    │           │
│  │ ├─ Platform: Google Ads                                   │           │
│  │ ├─ Campaign: "Summer Sale - Search"                       │           │
│  │ ├─ Action: Increase bid by 15%                           │           │
│  │ ├─ Current avg CPC bid: $3.20                            │           │
│  │ ├─ New bid: $3.68 (3.20 × 1.15)                          │           │
│  │ └─ Target segments: Desktop, Male 25-34, High-intent     │           │
│  │                                                             │           │
│  │ MOVE 2: BUDGET INCREASE                                   │           │
│  │ ├─ Current daily budget: $2,000                          │           │
│  │ ├─ New daily budget: $2,500 (+$500)                      │           │
│  │ ├─ Impact: Can serve ~250 more clicks/day               │           │
│  │ └─ Monthly cost increase: ~$15,000                        │           │
│  │                                                             │           │
│  │ MOVE 3: CREATIVE SWAP                                     │           │
│  │ ├─ Current creative: ID 1 (Creative fatigue score: 6.2)  │           │
│  │ ├─ Swap to: ID 3 (Fresh creative, CTR +8%)              │           │
│  │ ├─ Reason: Creative fatigue detected, new one has better │           │
│  │ │          historical CTR on similar audience              │           │
│  │ └─ Rollout: 20% traffic test for 1 day first             │           │
│  └────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  PHASE 2: Generate narrative explanation (xAI)                            │
│  ┌────────────────────────────────────────────────────────────┐           │
│  │ SITUATION:                                                 │           │
│  │ Campaign has strong click-through (CTR 2.8%) but moderate │           │
│  │ conversion rate (CVR 3.2%). Budget is 70% utilized.       │           │
│  │ Competitor impression share is 35%. ROAS is 2.3x.        │           │
│  │                                                             │           │
│  │ DECISION:                                                  │           │
│  │ Increase spend on top-performing segments while improving │           │
│  │ creative engagement.                                       │           │
│  │                                                             │           │
│  │ REASONING:                                                 │           │
│  │ The SAC agent was trained on 50,000+ transitions where:   │           │
│  │ - When CTR_trend_7d > 2.1% AND                           │           │
│  │   top_segment_roas > 3.0,                                 │           │
│  │ - Bidding up returns 1.3x reward (higher conversions)    │           │
│  │                                                             │           │
│  │ Since your top segment has ROAS 3.1x, bidding 15% more   │           │
│  │ should yield ~+8% conversion increase while staying       │           │
│  │ within CPA targets ($23.80 predicted vs $25 limit).      │           │
│  │                                                             │           │
│  │ Creative swap (ID 1 → ID 3) based on:                    │           │
│  │ - ID 1 has been live 45 days (fatigue detected)         │           │
│  │ - ID 3 has similar audience targeting but +8% historical │           │
│  │   CTR on "Male 25-34" segment                            │           │
│  │                                                             │           │
│  │ CONFIDENCE: 89%                                            │           │
│  │ (Based on SAC actor entropy + Q-value consensus)          │           │
│  │                                                             │           │
│  │ REASONABILITY: ✅ PASS                                     │           │
│  │ ✓ Aligns with business goal (ROAS optimization, 40%)     │           │
│  │ ✓ Respects CPA constraint ($25 max)                      │           │
│  │ ✓ Respects budget constraint ($2,500 daily max)          │           │
│  │ ✓ Respects bid constraint (±15% < ±50% limit)           │           │
│  │ ✓ Respects cooldown period (6 hours since last change)   │           │
│  └────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  PHASE 3: Generate forecast (predict outcomes)                            │
│  ┌────────────────────────────────────────────────────────────┐           │
│  │ PREDICTED IMPACT (7-day forecast):                         │           │
│  │                                                             │           │
│  │ Metric          Current      Predicted    Change           │           │
│  │ ─────────────────────────────────────────────────────────  │           │
│  │ ROAS            2.30x        2.58x        +12.2% ✅        │           │
│  │ CPA             $24.50       $22.10       -9.8% ✅         │           │
│  │ CTR             2.80%        2.95%        +5.4% ✅         │           │
│  │ CVR             3.20%        3.45%        +7.8% ✅         │           │
│  │ Volume          1,240 conv   1,337 conv   +7.8% ✅        │           │
│  │ Daily Spend     $2,000       $2,500       +$500            │           │
│  │ Daily Revenue   $4,600       $5,287       +$687 ✅        │           │
│  │ Daily Profit    $2,600       $2,787       +$187 ✅        │           │
│  │                                                             │           │
│  │ RISK ASSESSMENT: LOW ✅                                    │           │
│  │ - CPA stays well below target                             │           │
│  │ - ROAS improvement has 89% confidence                     │           │
│  │ - Budget increase is gradual and reversible               │           │
│  │ - Creative swap is A/B tested (20% traffic first)        │           │
│  │                                                             │           │
│  │ RECOMMENDATION: ✅ APPLY IMMEDIATELY                       │           │
│  └────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: APPLY RECOMMENDATIONS (Optional: Auto-apply or Manual Review)       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User sees dashboard with recommendations and clicks "APPLY"               │
│                                                                              │
│  For each move:                                                            │
│  ┌────────────────────────────────────────────────────────────┐            │
│  │ MOVE 1: Bid increase (Google Ads)                          │            │
│  │ ├─ Call Google Ads API                                    │            │
│  │ ├─ Update campaign keyword bids                           │            │
│  │ ├─ Status: ✅ APPLIED (2 mins)                            │            │
│  │ └─ Timestamp: 2024-11-08 14:23:15 UTC                     │            │
│  │                                                             │            │
│  │ MOVE 2: Budget increase (Google Ads)                       │            │
│  │ ├─ Call Google Ads API                                    │            │
│  │ ├─ Update daily budget cap                                │            │
│  │ ├─ Status: ✅ APPLIED (1 min)                             │            │
│  │ └─ Timestamp: 2024-11-08 14:24:30 UTC                     │            │
│  │                                                             │            │
│  │ MOVE 3: Creative swap (Google Ads)                         │            │
│  │ ├─ Call Google Ads API                                    │            │
│  │ ├─ Update ad group to use new creative (20% rollout)      │            │
│  │ ├─ Status: ✅ APPLIED (1 min)                             │            │
│  │ └─ Timestamp: 2024-11-08 14:25:45 UTC                     │            │
│  │                                                             │            │
│  │ ALL MOVES APPLIED ✅ (4 mins total)                        │            │
│  └────────────────────────────────────────────────────────────┘            │
│                                                                              │
│  Updates stored in database:                                               │
│  - optimization_id: opt_1730966995                                         │
│  - campaign_id: cmp_7f3a                                                   │
│  - actions_applied: [bid_increase, budget_increase, creative_swap]        │
│  - timestamp_applied: 2024-11-08 14:25:45                                  │
│  - confidence_score: 0.89                                                  │
│  - predicted_metrics: {roas: 2.58, cpa: 22.10, ...}                       │
│  - status: "applied"                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: MONITOR & TRACK RESULTS (Next 7 days)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dashboard shows real-time performance vs predicted:                       │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────┐            │
│  │ OPTIMIZATION TRACKING (opt_1730966995)                     │            │
│  │ Applied: 2024-11-08 14:25:45                              │            │
│  │ Elapsed: 6 hours, 23 minutes                              │            │
│  │                                                             │            │
│  │                 PREDICTED  ACTUAL    VARIANCE             │            │
│  │ ROAS            2.58x      2.51x     -2.7% (OK)            │            │
│  │ CPA             $22.10     $23.45    +6.1% (OK)            │            │
│  │ CTR             2.95%      2.88%     -2.4% (OK)            │            │
│  │ CVR             3.45%      3.38%     -2.0% (OK)            │            │
│  │ Conversions     28/day     27/day    -3.6% (trending OK)   │            │
│  │                                                             │            │
│  │ STATUS: ON TRACK ✅                                        │            │
│  │ → Predictions are within 5% variance                      │            │
│  │ → CPA still well below $25 target                         │            │
│  │ → ROAS improvement starting to show                       │            │
│  │                                                             │            │
│  │ ESTIMATED FINAL (7-day):                                  │            │
│  │ → Daily revenue +$187 (annualized: +$68k)                │            │
│  │ → Daily profit +$143 (annualized: +$52k)                 │            │
│  │                                                             │            │
│  │ IMPACT SCORE: 8.7/10 ✅                                   │            │
│  └────────────────────────────────────────────────────────────┘            │
│                                                                              │
│  After 7 days, final comparison:                                           │
│  ┌────────────────────────────────────────────────────────────┐            │
│  │ FINAL RESULTS vs PREDICTED                                 │            │
│  │                                                             │            │
│  │ Metric          Predicted  Actual    Accuracy             │            │
│  │ ─────────────────────────────────────────────────────────  │            │
│  │ ROAS            2.58x      2.62x     +1.5% ✅             │            │
│  │ CPA             $22.10     $21.80    +1.4% ✅             │            │
│  │ CTR             2.95%      2.98%     +1.0% ✅             │            │
│  │ CVR             3.45%      3.52%     +2.0% ✅             │            │
│  │ Conversions     196 conv   202 conv  +3.1% ✅             │            │
│  │ 7-day spend     $17,500    $17,502   +0.01% ✅            │            │
│  │ 7-day revenue   $37,100    $38,240   +3.1% ✅             │            │
│  │ 7-day profit    $19,600    $20,738   +5.8% ✅             │            │
│  │                                                             │            │
│  │ ✅ PREDICTION ACCURACY: 94.3%                             │            │
│  │ ✅ OPTIMIZATION SUCCESSFUL                                │            │
│  │ ✅ ROI IMPROVEMENT: +$1,138 over 7 days                   │            │
│  │ ✅ CONFIDENCE MAINTAINED: 89% (unchanged)                 │            │
│  │                                                             │            │
│  │ RECOMMENDATION FOR NEXT OPTIMIZATION:                      │            │
│  │ → Monitor another 3 days before next change               │            │
│  │ → Based on new data, next rec: expand to audiences with   │            │
│  │   similar top-segment profile                             │            │
│  │ → Estimated additional gain: +$4.2k/month                │            │
│  └────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: NEXT OPTIMIZATION CYCLE                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dashboard shows:                                                          │
│  - Summary of last optimization impact                                     │
│  - New recommended moves (based on updated campaign data)                  │
│  - Confidence score + predicted impact                                     │
│  - Option to apply or schedule for later                                   │
│                                                                              │
│  User can now:                                                             │
│  ✅ Apply new recommendations immediately                                 │
│  ✅ Schedule for specific time (e.g., 2 PM tomorrow)                      │
│  ✅ Request alternative recommendations (conservative/aggressive)         │
│  ✅ Export full optimization report (PDF)                                 │
│  ✅ Share with team (Slack notification with summary)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## **DASHBOARD LAYOUT (What User Sees)**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DRL CAMPAIGN OPTIMIZER                                  [Settings] [Logout]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────┐ ┌──────────────────────────────────────┐   │
│  │ QUICK STATS                │ │ ACTIVE CAMPAIGNS                   │   │
│  ├────────────────────────────┤ ├──────────────────────────────────────┤   │
│  │ Total ROAS: 2.45x          │ │ Google Ads                         │   │
│  │ Total CPA: $23.20          │ │ ├─ Summer Sale - Search      ⭐⭐⭐ │   │
│  │ Daily Spend: $14,230       │ │ ├─ Summer Sale - Display     ⭐⭐   │   │
│  │ Daily Revenue: $32,450     │ │ ├─ Brand - Search            ⭐⭐⭐ │   │
│  │ Daily Profit: $18,220      │ │ │                                  │   │
│  │                            │ │ Facebook Ads                       │   │
│  │ Optimizations Applied: 47  │ │ ├─ Summer Sale Carousel      ⭐⭐   │   │
│  │ Avg Improvement: +8.3%     │ │ ├─ Summer Sale Collection   ⭐⭐⭐ │   │
│  └────────────────────────────┘ │ │                                  │   │
│                                  │ TikTok Ads                        │   │
│  ┌────────────────────────────┐ │ ├─ Summer Sale - For You     ⭐⭐⭐ │   │
│  │ RECENT OPTIMIZATIONS       │ │ │                                  │   │
│  ├────────────────────────────┤ │ Manually Refresh:                 │   │
│  │ 1 hour ago ✅              │ │ [🔄 Update Now]                   │   │
│  │   Campaign: Summer Sale    │ │                                    │   │
│  │   Action: Bid +12%, Budget │ │ Next recommended optimization:    │   │
│  │           +$500            │ │ [🤖 Get Recommendations]  (3 mins)│   │
│  │   Impact: ROAS +2.1%, CPA  │ │                                    │   │
│  │           -$0.70           │ │                                    │   │
│  │                            │ │                                    │   │
│  │ 6 hours ago ✅             │ │ [View All Campaigns] [View History]│   │
│  │   Campaign: Brand - Search │ │                                    │   │
│  │   Action: Creative swap    │ └──────────────────────────────────────┘   │
│  │   Impact: CTR +3.2%        │                                             │
│  │                            │ ┌──────────────────────────────────────┐   │
│  │ 1 day ago ✅               │ │ CURRENT OPTIMIZATION RUNNING...      │   │
│  │   Campaign: Carousel       │ ├──────────────────────────────────────┤   │
│  │   Action: Audience expand  │ │ Campaign: Summer Sale - Search       │   │
│  │   Impact: Volume +5.8%     │ │ Status: Monitoring (Day 3 of 7)      │   │
│  │                            │ │                                      │   │
│  └────────────────────────────┘ │ Predicted Impact:                   │   │
│                                  │ • ROAS: 2.30x → 2.58x (+12.2%)     │   │
│                                  │ • CPA: $24.50 → $22.10 (-9.8%)     │   │
│                                  │ • Daily profit: +$187 ✅            │   │
│                                  │                                      │   │
│                                  │ Actual Performance (so far):        │   │
│                                  │ • ROAS: 2.51x (97% of target) ✅   │   │
│                                  │ • CPA: $23.45 (95% on track) ✅    │   │
│                                  │ • Volume: 27/day (tracking OK) ✅   │   │
│                                  │                                      │   │
│                                  │ [View Full Details]                 │   │
│                                  └──────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## **TECH IMPLEMENTATION**

### **Frontend (Next.js + React)**

**Pages:**
- `/dashboard` — Main overview
- `/campaign/[id]` — Campaign details & optimization history
- `/recommendation/[id]` — Detailed recommendation view
- `/history` — All past optimizations with results
- `/settings` — API integrations (Google Ads, Facebook, TikTok)

**Components:**
- `CampaignCard` — Quick campaign status
- `RecommendationPanel` — Shows moves + reasoning
- `PerformanceChart` — Predicted vs actual charts
- `OptimizationTimeline` — History of optimizations
- `NarrativeExplainer` — Human-readable explanation

### **Backend API Routes**

```
POST /api/campaigns/sync
  → Pull fresh data from ad platforms
  → Store in database
  → Return: campaign_id, metrics, state_vector

POST /api/optimize
  → Input: campaign_id (or state_vector)
  → Run SAC inference
  → Run HybridOptimizer
  → Return: recommendations with confidence & forecast

POST /api/recommendations/apply
  → Input: recommendation_id
  → Push changes to ad platform APIs
  → Log applied action
  → Return: status, timestamp

GET /api/recommendations/[id]/results
  → Track performance of applied recommendation
  → Compare predicted vs actual
  → Return: metrics, variance, impact

GET /api/optimization-history
  → Return: all past optimizations with results
  → Filter by date, campaign, impact

POST /api/forecast/recompute
  → Given new campaign data, update forecasts
  → Show updated predicted impact
```

### **Database Schema**

```
Table: campaigns
├─ id (PK)
├─ platform (google_ads | facebook | tiktok)
├─ name
├─ status (active | paused)
├─ daily_budget
├─ target_metrics {cpa_target, min_roas, ...}
└─ timestamps

Table: campaign_daily_metrics
├─ id (PK)
├─ campaign_id (FK)
├─ date
├─ spend, impressions, clicks, conversions
├─ ctr, cpa, roas, cvr, cpm
├─ state_vector (42,) JSON
└─ timestamp

Table: recommendations
├─ id (PK)
├─ campaign_id (FK)
├─ state_vector_used (42,) JSON
├─ sac_action JSON
├─ confidence_score (0-1)
├─ moves [{type, params, reasoning}] JSON
├─ narrative {situation, decision, reasoning, ...} JSON
├─ predicted_metrics {roas, cpa, ...}
├─ status (pending | applied | archived)
├─ created_at
└─ applied_at

Table: optimizations_applied
├─ id (PK)
├─ recommendation_id (FK)
├─ moves_applied JSON
├─ timestamp_applied
├─ status (success | failed)
└─ api_response JSON

Table: optimization_results
├─ id (PK)
├─ optimization_id (FK)
├─ day (1-7)
├─ predicted_metrics JSON
├─ actual_metrics JSON
├─ variance_pct
├─ timestamp_measured
└─ final_impact_score
```

---

## **COMPLETE USER JOURNEY**

### **Morning: Check Dashboard**
1. User opens dashboard.vercel.app
2. Sees summary: "Your 47 optimizations have generated +$18k profit over 30 days"
3. Sees 5 active campaigns with status badges
4. Sees 3 past optimizations tracking their results
5. Sees 1 new recommendation ready to review

### **Review Recommendation**
1. Clicks "Summer Sale - Search" campaign
2. Sees recommendation:
   - 🎯 **The Moves:** Bid +15%, Budget +$500, Creative swap
   - 💭 **Why:** "CTR trending up, top segment ROI is 3.1x, competitors weak"
   - 📊 **Predicted Impact:** ROAS +12%, CPA -10%, Daily profit +$187
   - ✅ **Confidence:** 89%, All constraints respected
   - 📈 **Track:** See last optimization results (ROAS +2.1%, verified!)

### **Apply & Monitor**
1. Clicks "Apply Recommendation"
2. System pushes changes to Google Ads API
3. Changes confirmed in 2-4 minutes
4. Dashboard shows: "✅ Applied 6 minutes ago, monitoring..."
5. Real-time metrics update every hour
6. After 3 days, user sees: "On track for $187 daily profit increase"
7. After 7 days, final results: "ROAS +2.6% (predicted 2.58x), CPA -$2.70"

### **Next Week**
1. Dashboard shows 6 optimizations applied, avg impact +8.7%
2. System suggests next recommendation: "Expand to similar audiences"
3. Monthly report: "+$68k profit vs last month"
4. User receives Slack notification: "Daily profit milestone: $20k/day"

---

## **KEY FEATURES TO BUILD**

✅ **Real Data Integration**
- Google Ads API connector
- Facebook Ads API connector
- TikTok Ads API connector
- Automatic daily sync of metrics

✅ **State Encoding**
- Convert 20+ ad platform metrics → 42-d DRL state
- Handle different platforms (normalization)
- Cache states for quick access

✅ **DRL Inference**
- Load trained SAC checkpoint
- Run forward pass (42-d input → action)
- Extract confidence (Q-value)
- Fast inference (<500ms)

✅ **Safety & Validation**
- Check all constraints
- Clip actions to guardrails
- Verify cooldown periods
- Prevent dangerous combinations

✅ **Hybrid Optimization**
- Convert raw actions to business moves
- Generate narratives (xAI)
- Produce forecasts (predict impact)
- Score reasonability

✅ **Apply & Track**
- Push recommendations to platform APIs
- Log applied optimizations
- Track predicted vs actual metrics
- Display final impact & accuracy

✅ **Dashboard UI**
- Campaign overview cards
- Real-time recommendation panel
- Performance comparison charts
- Optimization history
- Detailed explanation narratives

✅ **Deployment**
- Next.js on Vercel
- Environment variables for API keys
- Database (Postgres or MongoDB)
- Scheduled syncs (daily metrics pull)

---

## **START BUILDING**

1. **Setup (30 mins)**
   - Create Next.js project
   - Setup Vercel deployment
   - Create database schema

2. **API Integration (2 hours)**
   - Build Google Ads connector
   - Build Facebook Ads connector
   - Implement metric sync logic

3. **State Encoding (1 hour)**
   - Convert raw metrics → 42-d state
   - Normalization & feature engineering
   - Caching strategy

4. **DRL Integration (1 hour)**
   - Load SAC checkpoint
   - Implement inference endpoint
   - Add confidence scoring

5. **Safety & Hybrid (1 hour)**
   - Implement guardrails
   - Generate narratives
   - Produce forecasts

6. **Frontend (3 hours)**
   - Build dashboard components
   - Real-time metric updates
   - Recommendation display
   - Charts & visualizations

7. **Apply & Monitor (1 hour)**
   - API calls to platforms
   - Track applied optimizations
   - Compare predicted vs actual

8. **Test & Deploy (1 hour)**
   - End-to-end testing
   - Deploy to Vercel
   - Monitor performance

**Total: ~10 hours to MVP**

---

**This is a COMPLETE, WORKING system. Everything connects. User pulls campaign data → DRL analyzes → System recommends moves → User applies → System tracks results. Go build! 🚀**
