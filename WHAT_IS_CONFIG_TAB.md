# What is the "CONFIG" Tab? - Simple Explanation

## **The "Config" Tab = Settings/Tuning Parameters for This Node**

You clicked on the **"Config"** tab. This shows you all the **settings that control how this node works**.

---

## **What You're Seeing**

### **The Configuration Table**

```
Parameter  | Value  | Type  | Source
───────────┼────────┼───────┼──────────────────
state_dim  | 42     | int   | config.yaml
max_steps  | 100    | int   | train.py
```

This means:
- Node 1 has **2 settings** you can adjust
- Each setting has a **current value**
- Each setting has a **type** (what kind of value)
- Each setting comes from a **file/source**

---

## **Breaking Down Each Setting**

### **1. state_dim = 42**

**What is it?**
- `state_dim` = "state dimension"
- The **number of metrics** in the 42-dimensional state vector
- Tells the system: "Each campaign snapshot has 42 numbers"

**What it means:**
```
state_dim = 42 means:
"When I describe a campaign, I'll use 42 numbers"

If it was state_dim = 36, would mean:
"Only use 36 numbers" (fewer metrics)

If it was state_dim = 50, would mean:
"Use 50 numbers" (more detailed)
```

**Type: int (integer)**
- Must be a whole number (42, not 42.5)

**Source: config.yaml**
- This setting comes from a **configuration file** called `config.yaml`
- Not hardcoded in the Python code
- Can be changed easily by editing that file

**Why is this important?**
```
If you change state_dim from 42 to 36:
- Node 1 creates only 36 numbers instead of 42
- Node 3 (AI Brain) expects 36, not 42
- Everything must match!

So state_dim is LOCKED at 42 for the whole system.
```

---

### **2. max_steps = 100**

**What is it?**
- `max_steps` = "maximum number of steps"
- The **maximum number of times** the node runs before stopping
- Tells the system: "Run this node for up to 100 iterations"

**What it means:**
```
max_steps = 100 means:
"Run up to 100 times" (fetch 100 data points)

If it was max_steps = 50:
"Only run 50 times" (less data, faster)

If it was max_steps = 1000:
"Run 1000 times" (more data, slower)
```

**Type: int (integer)**
- Must be a whole number (100, not 100.5)

**Source: train.py (MockCampaignEnv)**
- This setting comes from the **Python code file** `train.py`
- In the `MockCampaignEnv` class
- Can be changed by editing that Python file

**Why is this important?**
```
If you change max_steps from 100 to 50:
- Node 1 only fetches 50 campaign observations
- Node 2 (Replay Buffer) stores fewer examples
- Node 3 (AI Brain) learns from 50 instead of 100
- Model quality might be lower (less training data)

If you change max_steps from 100 to 1000:
- Node 1 fetches 1000 observations
- Takes longer to run (more data to process)
- But AI learns better (more examples)
```

---

## **Parameter vs Output: What's the Difference?**

### **PARAMETERS (Config Tab):**
```
Things YOU CAN CHANGE to control how the node works:
- state_dim = 42
- max_steps = 100

"Use these settings when running this node"
```

### **OUTPUTS (Outputs Tab):**
```
Things the NODE PRODUCES based on those parameters:
- state_vector = [0.12, 0.04, 2.30, ...] (42 numbers)
- reward = 0.034

"Here's what the node created using those settings"
```

### **Real Analogy:**

Think of a **recipe**:

```
PARAMETERS (Config) = Ingredients & Instructions
- Use 2 cups of flour
- Use 3 eggs
- Bake at 350°F for 30 minutes

OUTPUTS = The Result
- A delicious cake
- Golden brown color
- Ready to eat

Change the parameters → different output!
Change flour to 4 cups → cake is thicker
Change temperature to 400°F → cake burns faster
```

---

## **What Each Column Means**

| Column | Means | Example |
|--------|-------|---------|
| **Parameter** | Name of the setting | `state_dim`, `max_steps` |
| **Value** | Current setting | `42`, `100` |
| **Type** | What kind of value | `int` (number), `float` (decimal), `bool` (true/false), `string` (text) |
| **Source** | Where it comes from | `config.yaml` (file), `train.py` (code) |

---

## **Why Show Parameters in the UI?**

### **For Understanding:**
- "What settings control this node?"
- "Can I change the settings?"
- "Where do these settings come from?"

### **For Debugging:**
- "Is max_steps set correctly?"
- "Are parameters from the right file?"
- "What happens if I change state_dim?"

### **For Optimization:**
```
Current setting: max_steps = 100
Thinking: "The AI model quality is low. Maybe I need more training data?"
Solution: "Change max_steps to 500 to get more observations"
Result: Better AI model!
```

---

## **The Complete Picture: 3 Tabs Explained**

### **1. INPUTS Tab**
```
What does Node 1 NEED to run?
- campaign_id (which campaign?)
- run_id (which training run?)
```

### **2. OUTPUTS Tab**
```
What does Node 1 CREATE after running?
- state_vector (42 numbers representing campaign)
- reward (performance score)
```

### **3. CONFIG Tab**
```
What SETTINGS control Node 1?
- state_dim = 42 (how many numbers?)
- max_steps = 100 (how many times to run?)
```

---

## **Flow Diagram**

```
CONFIG (Settings)
    ↓
state_dim = 42  ┐
max_steps = 100 ├─→ [Node 1 Runs]
    ↑           ├─→ INPUTS
INPUTS          │
campaign_id     │
run_id          ↓
            OUTPUTS
            state_vector [42 numbers]
            reward 0.034
                ↓
            [Goes to Node 2]
```

---

## **Real Example: Changing a Parameter**

### **Current Setting:**
```
max_steps = 100
```

### **What Happens:**
```
Node 1 fetches 100 data points from campaign cmp_7f3a
Creates 100 state vectors
Creates 100 rewards
Stores in Node 2
```

### **If You Change to max_steps = 500:**
```
Node 1 fetches 500 data points from campaign cmp_7f3a
Creates 500 state vectors
Creates 500 rewards
Stores in Node 2

Result: AI has more data to learn from → better model!
```

### **If You Change to max_steps = 10:**
```
Node 1 fetches only 10 data points
Creates 10 state vectors
Creates 10 rewards

Result: AI has very little data → poor model quality!
```

---

## **Summary**

| What | Purpose | Example |
|------|---------|---------|
| **CONFIG Parameters** | Control how node works | state_dim=42, max_steps=100 |
| **Type** | What kind of value | int, float, bool, string |
| **Source** | Where setting comes from | config.yaml, train.py |
| **INPUTS** | What node needs | campaign_id, run_id |
| **OUTPUTS** | What node creates | state_vector, reward |

---

## **In One Sentence**

> "The Config tab shows the settings that control how this node works—like knobs you can turn to change its behavior."

---

## **Quick Reference**

```
⚙️ CONFIG TAB = Settings/Tuning Parameters
   └─ state_dim = 42 (number of metrics)
   └─ max_steps = 100 (how many times to run)

📥 INPUTS TAB = What node needs
   └─ campaign_id (which campaign?)
   └─ run_id (which run?)

📤 OUTPUTS TAB = What node creates
   └─ state_vector (42 numbers)
   └─ reward (performance score)
```

**All 3 tabs together show: What you give it → How it works → What it produces!**
