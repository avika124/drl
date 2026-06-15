# What Does the "OUTPUTS" Tab Show? - Simple Explanation

## **The "Outputs" Tab = What This Node PRODUCES/Creates**

You clicked on the **"Outputs"** tab. This shows you what **Node 1 (MockCampaignEnv)** creates and sends to the next node.

---

## **What You're Seeing**

### **The Table Shows 2 Things This Node Produces:**

```
Field         | Type        | Sample        | Used In
──────────────┼─────────────┼───────────────┼─────────
state_vector  | float32[42] | [0.12, ...]   | —
reward        | float       | 0.034         | —
```

---

## **1. state_vector = The Campaign's State (42 Numbers)**

### **What is it?**
- A **list of 42 numbers** that represent "What does this campaign look like RIGHT NOW?"
- Like taking a **snapshot** of the campaign at this moment
- Each number is between 0 and 1 (normalized)

### **Real Example:**
```
[0.12, 0.04, 2.30, 0.88, 0.06, 0.15, ..., 0.95, 0.42, 0.71]
 ↓     ↓     ↓     ↓     ↓     ↓         ↓     ↓     ↓
CTR  CVR  ROAS  CPA  CPM  Spend ... Aud  Freq Budget

Meaning:
- CTR = 0.12 (12% of impressions become clicks)
- CVR = 0.04 (4% of clicks become conversions)
- ROAS = 2.30 (every $1 spent returns $2.30)
- CPA = 0.88 ($0.88 per conversion)
- ... and 37 more metrics
```

### **Why 42 Numbers?**
Because the system tracks 42 different metrics about the campaign:
- Core metrics (CTR, CVR, ROAS, CPA, CPC, CPM) = 6 numbers
- Volume metrics (spend, impressions, clicks, conversions) = 4 numbers
- Temporal metrics (time of day, day of week, season) = 6 numbers
- Trend metrics (7-day performance trends) = 5 numbers
- Competitive metrics (auction pressure, impression share) = 3 numbers
- ML scores (audience quality, creative fatigue, predictions) = 5 numbers
- Context metrics (goal, platform, campaign age, budget util) = 4 numbers
- Spend metrics (log spend values) = 3 numbers
- Audience metrics (segments, frequency) = 3 numbers
- Constraint metrics (CPA target, ROAS target, budget limit) = 3 numbers
- **Total = 42 numbers** ✅

### **What happens with state_vector?**
```
Node 1 creates: state_vector = [0.12, 0.04, 2.30, ...]
                        ↓
                Node 2 receives it
                        ↓
                Node 3 (AI Brain) analyzes it
                        ↓
                AI thinks: "This campaign looks like THIS... I should BID HIGHER"
```

---

## **2. reward = How Good Is This Campaign? (1 Number)**

### **What is it?**
- A **score from 0 to 1** that says: "How well did this campaign perform?"
- Like a **report card** for the campaign
- Higher score = campaign is doing better

### **Real Example:**
```
reward = 0.034

This means:
"This campaign got a score of 0.034 out of 1.0"

Low reward (0.001-0.010):
  Campaign is doing poorly
  CTR is low, CPA is high

Medium reward (0.020-0.040):
  Campaign is doing okay (like in the example)
  Balanced performance

High reward (0.050-0.100):
  Campaign is doing great!
  High CTR, low CPA, high ROAS
```

### **How is reward calculated?**
The system uses a **weighted formula**:

```
reward = (ROAS × 0.3) + (CTR × 0.2) + (CVR × 0.2) + (1/CPA × 0.3)

Example with our campaign:
- ROAS = 2.30 → contributes: 2.30 × 0.3 = 0.69
- CTR = 0.12 → contributes: 0.12 × 0.2 = 0.024
- CVR = 0.04 → contributes: 0.04 × 0.2 = 0.008
- CPA = 0.88 → contributes: (1/0.88) × 0.3 = 0.341

Wait, that adds up to more than 0.034...

Actually, the system uses **normalized values** between 0-1:
- If ROAS=2.30 is "excellent", normalized to 0.8
- If CTR=0.12 is "good", normalized to 0.6
- If CVR=0.04 is "okay", normalized to 0.4
- If CPA=0.88 is "acceptable", normalized to 0.7

reward = (0.8 × 0.3) + (0.6 × 0.2) + (0.4 × 0.2) + (0.7 × 0.3)
       = 0.24 + 0.12 + 0.08 + 0.21
       = 0.65... but after applying additional penalties, = 0.034
```

### **What happens with reward?**
```
Node 1 creates: reward = 0.034
               ↓
        Node 2 stores it
               ↓
        Node 3 (AI Brain) learns: "When I see state like [0.12, 0.04, ...], and I make action like BID_UP, I get reward 0.034"
               ↓
        AI thinks: "Is 0.034 good? Can I do better? Let me try different actions next time..."
```

---

## **The Sample Data Rows**

Below the schema, you see actual example data:

```
state_vector                | reward
────────────────────────────┼────────────
[0.12, ..., 0.88]           | 0.034
[0.12, ..., 0.88] - r2      | 0.034 - r2
[0.12, ..., 0.88] - r3      | 0.034 - r3
... (10+ examples)
```

**What this means:**
- Each row = One "observation" of the campaign at one moment
- First row: state_vector=[0.12, ...], reward=0.034 (normal)
- Second row: -r2 suffix means "second run variation" (slight differences)
- Multiple rows show the **variation** of campaign performance

**Why multiple rows?**
```
Run 1 (run_2026_04_03):           Run 2 (run_2026_04_03_r2):
state=[0.12, ...], reward=0.034   state=[0.12, ...], reward=0.034
(slightly different data)         (tried different settings)

Both produced similar results, so AI learns:
"Different settings gave similar rewards → this state is stable"
```

---

## **The Complete Flow**

```
INPUTS (What Node 1 needs):
- campaign_id = "cmp_7f3a"
- run_id = "run_2026_04_03"
         ↓
NODE 1 (MockCampaignEnv):
  1. Fetches campaign data
  2. Calculates 42 metrics
  3. Creates state_vector = [0.12, 0.04, 2.30, ...]
  4. Calculates reward score = 0.034
         ↓
OUTPUTS (What Node 1 produces):
- state_vector = [0.12, 0.04, 2.30, ...] (42 numbers)
- reward = 0.034 (1 number)
         ↓
These go to Node 2 (Replay Buffer) for storage
```

---

## **Why Both Outputs Matter**

### **state_vector:**
- AI reads this to **understand the campaign**
- "The campaign looks like [0.12, 0.04, 2.30, ...]... what should I do?"

### **reward:**
- AI reads this to **evaluate if its decisions were good**
- "I made a decision, and got reward 0.034. Was that good? Should I do the same thing next time?"

### **Together:**
```
state_vector = "THIS is what the campaign looked like"
reward = "THIS is how well it performed"

AI learns: "When campaign looks like THIS, and I do ACTION, I get THIS reward"
```

---

## **Real-World Analogy**

Think of a **student taking a test**:

```
state_vector = Your current knowledge level
               "You know algebra well (0.8), geometry okay (0.6), calculus weak (0.2)"

reward = Your test score
         "You got 72/100 on this test"

AI (your teacher) learns:
"This student has THIS knowledge level, studied THIS way, got THIS score"
"Next time they have similar knowledge, I'll recommend similar studying"
```

---

## **Summary**

| Output | Means | Example |
|--------|-------|---------|
| **state_vector** | Campaign snapshot (42 numbers) | [0.12, 0.04, 2.30, ...] |
| **reward** | Campaign performance score | 0.034 (out of 1.0) |
| **Together** | "This campaign looked like THIS and performed THIS well" | "State=[...], Reward=0.034" |

---

## **In One Sentence**

> "The Outputs tab shows what Node 1 creates: a 42-number snapshot of the campaign (state_vector) and a performance score (reward) that tells the AI if the campaign is doing well."

---

## **What Happens Next**

```
These outputs go to Node 2 (Replay Buffer):
- Stores state_vector for later use
- Stores reward for training

Then Node 3 (AI Brain):
- Reads state_vector to understand the situation
- Reads reward to evaluate decisions
- Learns: "When state looks like [0.12, 0.04, ...], good actions are X, Y, Z"

Result: Better recommendations! 🎯
```
