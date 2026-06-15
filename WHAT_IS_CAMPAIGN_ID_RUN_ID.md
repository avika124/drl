# What is campaign_id and run_id? - Simple Explanation

## **The 2 Most Important IDs in the System**

### **1. campaign_id = Which Campaign Are We Optimizing?**

**What is it?**
- A **unique code** that identifies ONE advertising campaign
- Like a **license plate** for a campaign
- Each campaign has its own ID

**Real Examples:**
```
cmp_7f3a        → Campaign A (Summer Sale - Google Search)
cmp_abc123      → Campaign B (Back to School - Facebook)
cmp_xyz789      → Campaign C (Black Friday - TikTok)
```

**What does it mean?**
- `cmp_7f3a` = "This is campaign #7f3a"
- `cmp_` = prefix meaning "campaign"
- `7f3a` = unique identifier (like a serial number)

**Why do we need it?**
```
Imagine you have 10 campaigns running:
- Google Ads campaign for Summer Sale
- Facebook campaign for Summer Sale
- TikTok campaign for Summer Sale
- Google Ads campaign for Back to School
- ... and 6 more

The campaign_id tells the system:
"Hey, we're working on campaign #7f3a (Summer Sale - Google)"
```

---

### **2. run_id = Which Training/Test Run Is This?**

**What is it?**
- A **unique code** for each time you **train the AI** or **run the system**
- Like a **timestamp + counter** for different experiments
- Each run has its own ID so you can track history

**Real Examples:**
```
run_2026_04_03       → Run from April 3, 2026
run_2026_04_03_r2    → Second run on April 3, 2026
run_2026_04_03_r3    → Third run on April 3, 2026
run_2026_04_10       → Different day (April 10, 2026)
```

**What does it mean?**
- `run_2026_04_03` = "This is the run from 2026-04-03" (April 3, 2026)
- `run_2026_04_03_r2` = "Second run on that day"
- `run_2026_04_03_r3` = "Third run on that day"
- `_r2`, `_r3` = "run 2", "run 3" (repeated experiments)

**Why do we need it?**
```
Imagine you're testing the AI multiple times:

Day 1 (April 3):
- Morning: Train AI with batch_size=64 → run_2026_04_03
- Afternoon: Train AI with batch_size=128 → run_2026_04_03_r2
- Evening: Train AI with batch_size=256 → run_2026_04_03_r3

Day 2 (April 10):
- Train AI again → run_2026_04_10

The run_id tells you:
"When was this run? What number attempt was it?"
```

---

## **How They Work Together**

### **Real-World Scenario:**

You have:
- **3 campaigns** running
- **Multiple training runs** for each campaign

```
Campaign: cmp_7f3a (Summer Sale - Google)
├─ Training Run 1: run_2026_04_03
│  └─ Trained AI model, found it's okay
├─ Training Run 2: run_2026_04_03_r2
│  └─ Tried different settings, much better!
└─ Training Run 3: run_2026_04_10
   └─ Retrained after 1 week, improved further

Campaign: cmp_abc123 (Back to School - Facebook)
├─ Training Run 1: run_2026_04_03
│  └─ Trained AI model
├─ Training Run 2: run_2026_04_03_r2
│  └─ Tried again
└─ Training Run 3: run_2026_04_10
   └─ Updated model
```

**What the system does:**
```
Input: campaign_id = "cmp_7f3a", run_id = "run_2026_04_03"

System thinks:
"Okay, I'm working on:
 - Campaign: Summer Sale (Google Ads)
 - Run: First training attempt on April 3
 
Get the data for THIS campaign's THIS run."

Output: 
- Campaign data (spend, clicks, conversions)
- 42-dimensional state vector
- Reward score
```

---

## **Step-by-Step: What Actually Happens**

### **Step 1: You Start the System**
```
"I want to optimize campaign cmp_7f3a using run_2026_04_03"
```

### **Step 2: Node 1 (MockCampaignEnv) Receives Input**
```
Input:
  campaign_id = "cmp_7f3a"
  run_id = "run_2026_04_03"
```

### **Step 3: Node 1 Does Lookup**
```
System looks up in database:
"Give me all data for campaign cmp_7f3a from run_2026_04_03"

Database returns:
  Spend: $5,432.50
  Impressions: 142,340
  Clicks: 4,210
  Conversions: 287
  Conversion Value: $43,050
  ... and more metrics
```

### **Step 4: Node 1 Converts to 42-D State**
```
Takes raw metrics and converts to 42 numbers:
[0.028, 0.032, 2.30, 24.50, 0.08, ..., 0.95, 0.87, 0.42]
  ↓      ↓      ↓     ↓     ↓         ↓     ↓     ↓
  CTR   CVR   ROAS   CPA   CPM   ... [other metrics]
```

### **Step 5: Node 2 (Replay Buffer) Receives This**
```
Gets state vector + reward:
state = [0.028, 0.032, 2.30, 24.50, ...]
reward = 0.034
```

### **Step 6: Node 3 (AI Brain) Learns**
```
AI learns from this example:
"When state looks like THIS, 
 and I choose THIS action,
 I get THIS reward (0.034)"
```

### **Step 7: System Makes Recommendation**
```
After analyzing campaign cmp_7f3a from run 2026_04_03:

"I recommend:
 - Increase bid by 15%
 - Increase budget by $500
 - Swap creative to ID 3
 
Expected result: ROAS 2.3x → 2.58x"
```

---

## **Why Both IDs Are Important**

### **Without campaign_id:**
```
❌ System wouldn't know which campaign to optimize
❌ Would mix data from 10 different campaigns
❌ Recommendations would be wrong
```

### **Without run_id:**
```
❌ System wouldn't know WHEN the data is from
❌ Would mix old data (April 3) with new data (April 10)
❌ Would train on outdated campaign metrics
```

### **With BOTH:**
```
✅ System knows EXACTLY which campaign at WHICH time
✅ Can track improvements over multiple runs
✅ Can compare: "April 3 run was better than April 10 run"
✅ Can A/B test: run_2026_04_03 vs run_2026_04_03_r2
```

---

## **Real Example: What You're Seeing**

In the screenshot, the system is processing:

```
campaign_id: cmp_7f3a
run_id: run_2026_04_03

Node 1 pulls this data:
- Campaign: cmp_7f3a (Summer Sale - Google Search)
- Date: April 3, 2026
- Metrics: spend, clicks, conversions, CTR, CVR, ROAS, CPA, etc.

Sample Data Shown:
cmp_7f3a        | run_2026_04_03     ← First record
cmp_7f3a - r2   | run_2026_04_03_r2  ← Second experiment same day
cmp_7f3a - r3   | run_2026_04_03_r3  ← Third experiment same day
cmp_7f3a - r10  | run_2026_04_03_r10 ← Tenth experiment same day
```

---

## **The Complete Picture**

```
START
  ↓
[Input] campaign_id + run_id
  ↓
[Node 1] Fetch campaign data & convert to 42-d state
  ↓
[Node 2] Store in replay buffer
  ↓
[Node 3] AI learns: "If state looks like this, bid higher"
  ↓
[Node 4] Save trained model
  ↓
[Node 5] Load model for predictions
  ↓
[Node 6] Safety check: "Is this safe?"
  ↓
[Node 7] Make final recommendation
  ↓
[Output] "Increase bid 15%, budget +$500, use creative 3"
  ↓
END → Apply to campaign cmp_7f3a
```

---

## **Summary**

| What | Meaning | Example |
|------|---------|---------|
| **campaign_id** | Which campaign to optimize | cmp_7f3a (Summer Sale) |
| **run_id** | When/which training run | run_2026_04_03 (April 3) |
| **Together** | Work on THIS campaign FROM THIS run | Optimize cmp_7f3a from run April 3 |

---

## **In One Sentence**

> "campaign_id says WHICH campaign you're optimizing, and run_id says WHEN/HOW MANY TIMES you've trained the AI on that campaign."

---

## **What The System Does**

1. Takes campaign_id (which campaign?)
2. Takes run_id (which training run?)
3. Looks up the data
4. Converts to 42-d state vector
5. Trains AI to make better decisions
6. Makes recommendations
7. Applies to the real campaign

**Result:** Better ROAS, lower CPA, more profit! 🎯
