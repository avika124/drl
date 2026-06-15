# DRL Workflow Studio — Data Explorer Explained (Simple English)

## **What You're Looking At**

You can see the **DRL Workflow Studio** with a **Data Explorer modal** (popup window) open. This is a tool that lets you understand exactly what data flows through your campaign optimization system.

---

## **The Big Picture**

### **What is the DRL Workflow Studio?**

Think of it like a **visual map of a factory**:
- Each **box (node)** is a machine/worker that does a job
- **Arrows** show data flowing from one machine to the next
- You can **click on any machine** to see what it takes as input and what it produces as output

In our case, we have 7 machines (n1 through n7):
- **n1 (MockCampaignEnv)**: Gets campaign data from Google Ads / Facebook / TikTok
- **n2 (Replay Buffer)**: Stores training examples
- **n3 (SACAgent)**: The AI brain that learns
- **n4 (Checkpoint)**: Saves the trained brain
- **n5 (Load SAC)**: Wakes up the saved brain
- **n6 (SafeDRLAgent)**: Safety guard that prevents bad decisions
- **n7 (HybridOptimizer)**: Makes final recommendations

---

## **What You See in the Screenshot**

### **1. Left Side: Workflow Canvas**
```
This shows the flow of data:

MockCampaignEnv / BigQuery (n1)
           ↓
       [other nodes below]
           ↓
load_sac_for_inference (n5)
           ↓
       [more nodes]
```

**Status Indicator:**
- **"SUCCESS"** = Node ran successfully
- **"Last run: 37 ms"** = The node took 37 milliseconds to execute

### **2. Top Right: Action Buttons**

Three buttons appear when you select a node:

**🎯 "View Data" (Blue)**
- Click to open the **Data Explorer modal**
- Shows what input the node needs
- Shows what output the node produces
- Shows node configuration/settings

**⚙️ "Edit Params" (Orange)**
- Click to change settings for this node
- Example: Change batch_size from 64 to 128
- Example: Change learning_rate from 0.0003 to 0.0001

**▶ "Run Node" (Green)**
- Click to execute just this one node
- Updates the outputs without running the whole pipeline
- Shows how long it took to run

### **3. Center: Data Explorer Modal (The Popup)**

This is the **information window** that shows you exactly what this node does.

**Header:**
```
📊 Data Explorer: MockCampaignEnv / BigQuery (n1)
```
This tells you which node's data you're looking at.

**Three Tabs:**
```
[📥 Inputs] [📤 Outputs] [⚙️ Config]
```
- **Inputs Tab** (currently selected) = What data comes INTO this node
- **Outputs Tab** = What data comes OUT of this node
- **Config Tab** = What settings/parameters this node uses

---

## **Breaking Down the "Inputs" Tab**

### **What is "Inputs schema"?**

"Schema" = **the blueprint of what data looks like**

Think of it like a **recipe card** that says:
- What ingredients you need
- What type of ingredient (string, number, etc.)
- An example of what it looks like
- Where it comes from

### **The Table: Field | Type | Sample | Source**

```
Field      | Type   | Sample          | Source
───────────┼────────┼─────────────────┼──────────────────
campaign_id| string | cmp_7f3a        | external: seed
run_id     | string | run_2026_04_03  | MockCampaignEnv
```

**Let me explain each column:**

#### **Column 1: Field**
- The **name** of the data
- `campaign_id` = the campaign's unique ID
- `run_id` = which training run this is for

#### **Column 2: Type**
- What **kind of data** it is
- `string` = text (like a campaign name)
- `float` = decimal number (like 2.3)
- `int` = whole number (like 64)
- `Tensor` = array of numbers (like [0.1, 0.2, 0.3, ...])

#### **Column 3: Sample**
- **Example data** showing what it looks like
- `cmp_7f3a` = example campaign ID (real data)
- `run_2026_04_03` = example run ID (means a run from 2026-04-03)

#### **Column 4: Source**
- **Where this data comes from**
- `external: seed` = provided from outside the system
- `MockCampaignEnv` = created by the MockCampaignEnv node
- `config.yaml` = comes from a configuration file

---

## **The "Sample Data Rows" Section**

Below the schema table, you see actual real example data:

```
campaign_id          | run_id
─────────────────────┼──────────────────────
cmp_7f3a             | run_2026_04_03
cmp_7f3a - r2        | run_2026_04_03 - r2
cmp_7f3a - r3        | run_2026_04_03 - r3
cmp_7f3a - r4        | run_2026_04_03 - r4
... (more rows)
```

**What this means:**
- These are **actual data examples** from when the node ran
- Shows you 10+ rows so you can see what real data looks like
- Each row is one "record" of data

**The "View Full Table" Button:**
- Click this to see **ALL the data** (if there are 1000s of rows)
- Opens a bigger window showing everything

---

## **What The Other Tabs Show**

### **"Outputs" Tab**

This shows what this node **produces/creates**:

```
Field       | Type      | Sample     | Used In
────────────┼───────────┼────────────┼──────────────
state_vector| float[42] | [0.028...] | n2 replay buffer
reward      | float     | 0.034      | n2 transitions
```

**Translation:**
- This node creates 2 outputs:
  1. **state_vector** = a list of 42 numbers describing the campaign
  2. **reward** = a score (0.034) showing if the campaign did well

### **"Config" Tab**

This shows what **settings/parameters** this node uses:

```
Parameter   | Value  | Type   | Source
────────────┼────────┼────────┼──────────────
state_dim   | 42     | int    | config.yaml
max_steps   | 100    | int    | train.py
```

**Translation:**
- This node has 2 settings:
  1. **state_dim = 42** = state vectors have 42 numbers
  2. **max_steps = 100** = run for maximum 100 steps

---

## **Real-World Analogy**

Think of each node like a **restaurant kitchen station**:

### **Node 1: Data Input Station**
```
INPUT:  "I need a campaign ID and run ID"
OUTPUT: "Here's the campaign data (42-d state vector)"
```

### **Node 2: Storage Station (Replay Buffer)**
```
INPUT:  "I get campaign states and rewards"
OUTPUT: "Here's a batch of 64 examples for training"
```

### **Node 3: AI Training Station**
```
INPUT:  "Here's 64 training examples"
OUTPUT: "I learned something! Here's the loss score"
```

**Each station:**
- Has specific **inputs** it needs
- Does some **work** 
- Produces specific **outputs**
- Uses specific **settings/parameters** to do its job

---

## **How to Use the Data Explorer**

### **Step 1: Select a Node**
```
Click on any box in the workflow (e.g., "MockCampaignEnv")
```

### **Step 2: Click "View Data"**
```
The blue button in the inspector panel (right side)
```

### **Step 3: Explore the Data**
```
Read the Inputs tab  → see what data comes in
Read the Outputs tab → see what data goes out
Read the Config tab  → see what settings are used
```

### **Step 4: Look at Sample Data**
```
Scroll down to see actual example data
Click "View Full Table" to see all the data
```

### **Step 5: Close the Modal**
```
Click "Close" button or click outside the modal
```

---

## **Why This Matters**

### **For Understanding the System:**
- You can see **what each node expects** as input
- You can see **what each node produces** as output
- You can verify **data is flowing correctly** through the system

### **For Debugging:**
- If something breaks, you can check:
  - "Is node n1 producing the right 42-d state vector?"
  - "Is node n2 getting the right input from n1?"
  - "Are all parameters set correctly for node n3?"

### **For Learning:**
- You understand the **entire data pipeline**
- You see **real example data** flowing through
- You know **what each parameter does**

---

## **Quick Reference: What Each Column Means**

| Column | Means | Example |
|--------|-------|---------|
| **Field** | Name of the data | `campaign_id`, `reward` |
| **Type** | Kind of data | `string`, `float`, `Tensor` |
| **Sample** | Example value | `cmp_7f3a`, `0.034` |
| **Source** | Where it comes from | `external: seed`, `n1`, `config.yaml` |

---

## **Summary in 1 Sentence**

**The Data Explorer lets you click on any part of the workflow and see exactly what data flows in, what data flows out, and what settings it uses.**

---

## **Now You Can Explain It!**

When someone asks "What is this modal showing?"

You can say:
> "This is the **Data Explorer**. It shows you what data this node (machine) takes as input, what output it produces, and what settings it uses. You can see the data structure (schema) and real example data. It helps you understand how data flows through the entire optimization system."

**Done! ✅**
