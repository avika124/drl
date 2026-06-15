# DRL Campaign Optimization Engine — Cursor Implementation Guide

## **PROJECT BRIEF**
Build a **real-time campaign optimization dashboard** that:
- 📊 **Pulls LIVE campaign data** from Google Ads, Facebook, TikTok, etc.
- 🤖 **Runs DRL inference** on real campaign states
- 💡 **Suggests optimizations** (bid changes, budget reallocations, creative swaps, audience targeting)
- ✅ **Shows the IMPACT** of previous recommendations
- 📈 **Displays ROI improvements** with confidence scores
- 🎯 **Recommends specific MOVES** (e.g., "Increase bid by 15% on Desktop Male 25-34")
- 📱 **Deployable on Vercel** (Next.js + real backend integration)

---

## **TECH STACK**
- **Framework:** Next.js 14+ (App Router)
- **Styling:** Tailwind CSS + shadcn/ui components
- **Visualization:** Recharts (graphs), D3 (flow diagrams)
- **State:** React Context + Zustand
- **Deployment:** Vercel
- **Real-time:** WebSocket simulation (for demo)

---

## **FOLDER STRUCTURE**
```
drl-dashboard/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   ├── api/
│   │   ├── optimize/route.ts         # Simulated DRL optimization endpoint
│   │   ├── train/route.ts            # M1 training simulation
│   │   └── metrics/route.ts          # Performance metrics
│   └── components/
│       ├── Header.tsx
│       ├── Sidebar.tsx
│       ├── WorkflowVisualizer.tsx    # M1 → M2 flow diagram
│       ├── ParameterPanel.tsx        # Config + presets
│       ├── ExecutionMonitor.tsx      # Live execution tracking
│       ├── ResultsDisplay.tsx        # Outputs + explanations
│       └── Dashboard.tsx             # Main layout
├── lib/
│   ├── types.ts
│   ├── simulations.ts               # DRL simulation logic
│   └── utils.ts
├── hooks/
│   └── useWorkflow.ts              # Workflow state management
├── public/
│   └── icons/
├── tailwind.config.ts
├── next.config.js
└── vercel.json
```

---

## **CORE PAGES & COMPONENTS**

### **1. HOMEPAGE / DASHBOARD** (`app/page.tsx`)
**Layout:** 3-column responsive design
```
┌─────────────────────────────────────────────────┐
│ HEADER: "DRL Cross-Platform Optimizer"          │
├──────────┬─────────────────────┬────────────────┤
│          │                     │                │
│ SIDEBAR  │   WORKFLOW VIZ      │  RESULTS &     │
│          │   (M1 → M2 flow)    │  EXPLANATIONS  │
│          │                     │                │
│ - Presets│   [Live Execution   │ [Output Tables]│
│ - Params │    Progress Bars]   │ [Graphs]       │
│ - Status │                     │ [Narratives]   │
│          │                     │                │
└──────────┴─────────────────────┴────────────────┘
```

---

### **2. SIDEBAR COMPONENT** — Parameter Configuration
**Features:**
- **Preset buttons:** "Balanced", "Aggressive", "Conservative" (one click load)
- **Collapsible parameter groups:**
  - **Training:** batch_size, gamma, tau, max_steps
  - **Safety:** max_bid_pct, max_budget_pct, cooldown
  - **Rewards:** ROAS weight, CPA weight, conversion weight, CTR weight
  - **Data:** state_dim (42), device (CPU/GPU), model_dir

**UI Elements:**
```
┌──────────────────────┐
│ ⚙️ PARAMETERS       │
├──────────────────────┤
│ [Balanced]           │ <- Preset buttons
│ [Aggressive]         │
│ [Conservative]       │
├──────────────────────┤
│ 📚 Training       ▼  │ <- Collapsible
│  batch_size: [64 ] ◎ │
│  gamma: [0.99   ] ◎ │
│  tau: [0.005   ] ◎ │
│                      │
│ 🛡️ Safety        ▼  │
│  max_bid%: [0.5  ] ◎ │
│  max_budget%: [0.3] │
│                      │
│ 💰 Rewards       ▼  │
│  ROAS: [0.4  ] ◎    │
│  CPA: [0.3   ] ◎    │
│                      │
│ 💾 Data          ▼  │
│  State dim: 42 (fix) │
│  Device: [CPU ▼]    │
│                      │
│ [📥 Import] [📤 Exp]│ <- Buttons
└──────────────────────┘
```

**Interactions:**
- Sliders update parameter values in real-time
- Changed parameters highlighted in yellow
- Live count of modified parameters
- Import/export JSON config modal

---

### **3. WORKFLOW VISUALIZER** — Visual Data Flow
**Shows the complete M1 → M2 pipeline with animated data flow**

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ M1 — TRAINING PIPELINE                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│  │ Campaign │ --> │  Replay  │ --> │   SAC    │             │
│  │  Data    │     │  Buffer  │     │ Training │             │
│  │ (42-dim) │     │  (64)    │     │  (loss)  │             │
│  └──────────┘     └──────────┘     └──────────┘             │
│       ↓                 ↓                ↓                    │
│   Input: campaign    Input: transitions Input: batches       │
│   id, 42 features   Reward signal      Policy update         │
│                                                              │
│                                                              │
│                   ┌──────────────────┐                       │
│                   │   CHECKPOINT     │                       │
│                   │  agent.pt saved  │                       │
│                   │ training_info.json                       │
│                   └──────────────────┘                       │
│                          ↓                                   │
├─────────────────────────────────────────────────────────────┤
│ M2 — INFERENCE PIPELINE                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│  │  Load    │ --> │  Safety  │ --> │  Hybrid  │             │
│  │   SAC    │     │ Guardrails│     │Optimizer │             │
│  │ (eval)   │     │ (clip)   │     │ (LLM)    │             │
│  └──────────┘     └──────────┘     └──────────┘             │
│       ↓                 ↓                ↓                    │
│   Input: checkpoint Input: raw action  Input: 42-dim state   │
│   Output: policy    Output: validated  Output: directive     │
│                     Output: confidence Output: forecast      │
│                                                              │
│  [Execution Progress: ████████░░ 80%]  [Time: 245ms]       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Interactive Elements:**
- Click any node → show node details (inputs, outputs, config)
- Animated arrows showing data flow during execution
- Progress bars for each phase (M1 training, M2 inference)
- Execution time displayed
- Live logs/status messages

**Data Labels on Connections:**
```
Campaign Data (42-d vector)
  ├─ Core metrics: CTR, CVR, ROAS, CPA, CPC, CPM
  ├─ Volume: spend velocity, impressions, clicks, conversions
  ├─ Temporal: hour, day, weekend, holiday, days_remaining
  ├─ Trends: 7-day trends
  ├─ Competitive: impression share, auction pressure
  ├─ ML scores: audience quality, creative fatigue, predicted CVR/LTV
  ├─ Context: goal encoding, platform, campaign age, budget util
  ├─ Spend: log spend values
  ├─ Audience: segment count, top segment ROAS, frequency
  └─ Constraints: target_cpa_norm, min_roas_norm, daily_budget_norm
```

---

### **4. EXECUTION MONITOR** — Live Status & Logs
**Real-time execution tracker**

```
┌──────────────────────────────┐
│ 🚀 EXECUTION MONITOR         │
├──────────────────────────────┤
│ Status: Running              │ <- Status indicator
│ Progress: ████████░░ 80%     │ <- Overall progress
│ Time: 245 ms                 │ <- Elapsed time
│                              │
│ Node Execution Status:       │
│ ✅ n1 (campaign env)      5ms│ <- Completed
│ ✅ n2 (replay buffer)    12ms│
│ ✅ n3 (SAC training)     80ms│
│ ✅ n4 (checkpoint)        8ms│
│ ⏳ n5 (load SAC)         18ms│ <- In progress
│ ⬜ n6 (safety guardrails)  —  │ <- Pending
│ ⬜ n7 (hybrid optimizer)   —  │
│                              │
│ Live Log Output:             │
│ [12:34:01] Batch created    │
│ [12:34:02] Policy loss: 0.04│
│ [12:34:03] Guardrail check  │
│ [12:34:04] Constraints OK   │
│                              │
│ [🔄 Refresh] [📋 Download] │
└──────────────────────────────┘
```

---

### **5. RESULTS & EXPLANATIONS** — Output Visualization
**Show what the optimization produced and why**

```
┌─────────────────────────────────────────────┐
│ 📊 OPTIMIZATION RESULTS                     │
├─────────────────────────────────────────────┤
│                                             │
│ RECOMMENDED ACTIONS (from HybridOptimizer): │
│                                             │
│ Platform: Google Ads                        │
│ ┌─────────────────────────────────────────┐│
│ │ ✅ Increase bid: +12.3%                 ││
│ │    Confidence: 87%                      ││
│ │    Reason: Low auction pressure + strong││
│ │    CTR trend → higher probability of win││
│ │                                         ││
│ │ ✅ Increase daily budget: +$500         ││
│ │    Confidence: 72%                      ││
│ │    Reason: ROAS target achievable at    ││
│ │    higher spend; still meets CPA limits ││
│ │                                         ││
│ │ ❌ DO NOT adjust creative (fatigue low) ││
│ │    Creative fatigue score: 2.1/10       ││
│ │    Creative has 45% impressions left    ││
│ └─────────────────────────────────────────┘│
│                                             │
│ PREDICTED OUTCOMES (from forecaster):       │
│ ┌─────────────────────────────────────────┐│
│ │ Expected ROAS (7-day): 2.8 (vs 2.3 now)││
│ │ Expected CPA (7-day): $22.50 ✅ < $25  ││
│ │ Expected Conv Rate: +8.2%               ││
│ │ Budget utilization: 96% (safe)          ││
│ │ Risk assessment: LOW                    ││
│ └─────────────────────────────────────────┘│
│                                             │
│ NARRATIVE EXPLANATION (xAI):               │
│ ┌─────────────────────────────────────────┐│
│ │ SITUATION:                              ││
│ │ Campaign has strong click-through but   ││
│ │ moderate conversion. Budget is 70%      ││
│ │ utilized. Competitors are active.       ││
│ │                                         ││
│ │ DECISION:                               ││
│ │ Increase spend on high-performing       ││
│ │ audience segments (top_segment_roas=3.2)││
│ │                                         ││
│ │ REASONING:                              ││
│ │ SAC learned from 50k transitions that   ││
│ │ when CTR_trend_7d > 2.1% AND            ││
│ │ top_segment_roas > 3.0, bidding up      ││
│ │ returns 1.3x reward. Safety check:      ││
│ │ CPA constraint is 40 bps away from limit││
│ │                                         ││
│ │ CONFIDENCE: 87%                         ││
│ │ (SAC actor entropy + ensemble agreement)││
│ │                                         ││
│ │ REASONABILITY: ✅ Recommendation aligns ││
│ │ with business goals (ROAS 40% weight)   ││
│ │ and constraints (CPA target = $25)      ││
│ └─────────────────────────────────────────┘│
│                                             │
│ PERFORMANCE METRICS:                        │
│ ┌──────────┬──────────┬──────────┐        │
│ │ Metric   │ Current  │ Predicted││
│ ├──────────┼──────────┼──────────┤        │
│ │ ROAS     │ 2.3x     │ 2.8x     │ ↑      │
│ │ CPA      │ $24.50   │ $22.50   │ ↓      │
│ │ CTR      │ 2.8%     │ 3.1%     │ ↑      │
│ │ CVR      │ 3.2%     │ 3.8%     │ ↑      │
│ │ Volume   │ 1,240    │ 1,540    │ ↑      │
│ └──────────┴──────────┴──────────┘        │
│                                             │
│ [💾 Save] [📧 Share] [🔄 Re-run]         │
└─────────────────────────────────────────────┘
```

---

### **6. PARAMETER IMPACT GRAPHS**
**Show how parameters affect outcomes**

```
Three side-by-side Recharts graphs:

[Graph 1: Batch Size → Buffer Size]
    Size (MB)
    |     ╱────
    |    ╱
    |   ╱
    |  ╱────────────
    | ╱
    |╱_____________→ Batch Size (16-512)

[Graph 2: Gamma → Policy Loss]
    Loss
    |  ╲
    |   ╲────────
    |    ╲
    |     ╲      ╲
    |      ╲      ╲___
    |_______╲__________→ Gamma (0.9-1.0)

[Graph 3: ROAS Weight → Target]
    Target ROAS
    | ___
    |╱
    |/────────────
    |            ╲
    |             ╲___
    |________________→ ROAS Weight (0-1)
```

---

## **API ROUTES** (Simulated Backend)

### **1. POST `/api/optimize`**
**Input:**
```json
{
  "campaign_id": "cmp_7f3a",
  "state": [0.028, 0.032, 2.3, 24.50, ...],  // 42-d vector
  "constraints": {
    "target_cpa": 25,
    "min_roas": 2.0,
    "daily_budget": 2000
  },
  "parameters": {
    "batch_size": 64,
    "gamma": 0.99,
    "max_bid_pct": 0.5,
    "reward_roas": 0.4
  }
}
```

**Output:**
```json
{
  "directive": {
    "bid_change": "+12.3%",
    "budget_change": "$500",
    "creative_action": "hold"
  },
  "tactical": {
    "headlines": ["Fast Delivery Guaranteed", "Risk-Free 30 Day Trial"],
    "descriptions": ["Order today, ship tomorrow", "We stand behind our products"]
  },
  "narrative": {
    "situation": "Strong CTR, moderate CVR, budget 70% utilized",
    "decision": "Increase spend on top segments",
    "reasoning": "SAC learned positive correlation...",
    "confidence": 0.87,
    "reasonability": "Aligns with ROAS target (40% weight)"
  },
  "forecast": {
    "expected_roas_7d": 2.8,
    "expected_cpa_7d": 22.50,
    "expected_cvr": 0.038,
    "confidence_interval": [0.85, 0.92]
  },
  "execution_time_ms": 245,
  "schema_version": "1.0"
}
```

### **2. POST `/api/train`**
**Simulates M1 training**

Input: training config
Output: Progress events + final agent.pt metadata

### **3. GET `/api/metrics`**
**Returns:**
- Node execution times
- Parameter impact curves
- Historical optimization results

---

## **STATE MANAGEMENT** (Zustand Hook)

```typescript
// lib/store.ts
interface WorkflowStore {
  // Parameters
  parameters: Record<string, string>;
  setParameter: (key: string, value: string) => void;
  loadPreset: (preset: 'balanced' | 'aggressive' | 'conservative') => void;
  
  // Execution
  isRunning: boolean;
  nodeStatuses: Record<string, 'pending' | 'running' | 'success' | 'error'>;
  executionLogs: LogEntry[];
  setNodeStatus: (nodeId: string, status: string) => void;
  addLog: (entry: LogEntry) => void;
  
  // Results
  lastResult: OptimizationResult | null;
  setResult: (result: OptimizationResult) => void;
  
  // UI
  selectedNode: string | null;
  showDetailPanel: boolean;
}
```

---

## **VISUAL DESIGN SYSTEM**

### **Color Palette:**
```
Primary Blue:     #2563eb (flow, active)
Success Green:    #10b981 (completed, safe)
Warning Orange:   #f59e0b (in progress, attention)
Error Red:        #ef4444 (failed, violations)
Neutral Gray:     #6b7280 (background, text)
Light BG:         #f9fafb (panels)
Dark BG:          #111827 (header)
```

### **Component Library:**
- Use **shadcn/ui** buttons, cards, badges, progress bars, tooltips
- Use **Recharts** for all graphs and charts
- Use **Lucide icons** for status indicators
- Use **Tailwind** for responsive grid (sm/md/lg/xl breakpoints)

### **Responsive Breakpoints:**
- **Mobile** (sm): Sidebar becomes drawer, single column layout
- **Tablet** (md): Two-column layout (sidebar + main)
- **Desktop** (lg+): Three-column layout (sidebar + workflow + results)

---

## **KEY INTERACTIONS & FLOWS**

### **Flow 1: Load Preset & Run**
1. User clicks "Aggressive" preset
2. All parameters update with yellow highlight
3. Modified count increases
4. User clicks "Run All"
5. Workflow starts executing
6. Nodes change color: blue (running) → green (success)
7. Real-time logs appear
8. Results panel populates with outputs
9. Graphs update to show parameter impact
10. Narrative explains what happened and why

### **Flow 2: Manual Parameter Tuning**
1. User adjusts sliders (batch_size, gamma, etc.)
2. Node parameter cards update in real-time
3. Impact graphs recalculate
4. User clicks "Run All"
5. Execution uses new parameters
6. Results show how changes affected outcomes

### **Flow 3: Save & Share Configuration**
1. User modifies parameters
2. Clicks "Export Config"
3. JSON downloaded
4. User shares JSON with team
5. Colleague clicks "Import Config"
6. Pastes JSON
7. All parameters load
8. Can now run with shared configuration

---

## **DEPLOYMENT ON VERCEL**

### **vercel.json:**
```json
{
  "buildCommand": "next build",
  "outputDirectory": ".next",
  "installCommand": "npm install",
  "env": {
    "NEXT_PUBLIC_API_URL": "@drl_api_url"
  }
}
```

### **Environment Variables (.env.local):**
```
NEXT_PUBLIC_API_URL=http://localhost:3000
# In production, set to your API endpoint
```

### **Deployment Steps:**
```bash
npm install
npm run build
npm run start
# Or just: vercel deploy
```

### **GitHub Integration:**
- Push to GitHub repo
- Connect to Vercel
- Auto-deploy on every push to main
- Preview deployments for PRs

---

## **FILE STRUCTURE - DETAILED**

### **app/components/WorkflowVisualizer.tsx**
- Renders M1 → M2 pipeline diagram
- Shows animated data flow during execution
- Display node statuses (pending/running/success/error)
- Click node → show details panel
- Real-time parameter updates from sidebar

### **app/components/ParameterPanel.tsx**
- Collapsible parameter groups
- Preset buttons with dropdown
- Sliders for numeric values, selects for categorical
- Import/export modal
- Change tracking (yellow highlight)
- Statistics (modified count)

### **app/components/ResultsDisplay.tsx**
- Tabs for different result views (Directive, Narrative, Forecast, Metrics)
- Card-based layout for each recommendation
- Confidence indicators (progress bars)
- Tables for metrics comparison
- Narrative text with formatting (bold, colors)

### **app/components/ExecutionMonitor.tsx**
- Node execution timeline with durations
- Overall progress bar
- Live log stream (scrollable)
- Status badges (✅, ⏳, ⬜)
- Real-time clock

### **lib/simulations.ts**
- Simulated M1 training (generates random but realistic loss curves)
- Simulated M2 inference (produces OptimizationResult based on state + parameters)
- Forecast engine (predict outcomes based on parameter changes)
- Parameter impact calculations

### **hooks/useWorkflow.ts**
- Manages entire workflow state
- Handles API calls to /api/optimize, /api/train
- Manages node statuses, logs, results
- Parameter loading/saving

---

## **EXAMPLE: WHAT RUNS WHEN USER CLICKS "RUN ALL"**

```
1. Frontend: setState(isRunning = true)
2. Frontend: Clear previous logs & results
3. Frontend: POST /api/optimize with current parameters & campaign state
4. Backend (simulated): 
   - Load SAC agent from checkpoint
   - Run inference on 42-d state
   - Apply SafeDRLAgent guardrails
   - Generate recommendations via HybridOptimizer
   - Run forecaster for predictions
   - Return OptimizationResult
5. Frontend: Receive result
6. Frontend: Parse and display:
   - Directive (bid + budget actions)
   - Tactical (creatives if enabled)
   - Narrative (5-part explanation)
   - Forecast (expected metrics)
   - Execution time
7. Frontend: Update node statuses to "success"
8. Frontend: Update results panel with outputs
9. Frontend: Display confidence indicators
10. Frontend: Animate graphs showing parameter impact
11. Frontend: setState(isRunning = false)
12. User can now modify parameters and re-run OR export results
```

---

## **SUCCESS CRITERIA**

✅ **Functionality:**
- All parameters editable with live updates
- Presets load instantly
- Workflow execution simulates M1 → M2 pipeline
- Results show inputs, processing, outputs, and explanations
- Import/export JSON configurations work
- Parameter impact graphs update in real-time

✅ **Visual Design:**
- Clean, professional UI with consistent colors
- Responsive layout (mobile, tablet, desktop)
- Clear data flow diagrams (no confusion about what connects to what)
- Status indicators (colors, icons, progress bars)
- Readable explanatory text (narratives, tooltips)

✅ **Deployment:**
- Builds successfully on Vercel
- No console errors
- All API routes functional
- Environment variables configured
- Fast load time (< 2s)

✅ **Interactivity:**
- Clicking nodes shows detailed info
- Dragging parameter sliders updates graphs
- Preset buttons work smoothly
- Real-time log streaming during execution
- Modal dialogs for import/export

---

## **BONUS FEATURES (OPTIONAL)**

- [ ] Dark mode toggle
- [ ] Export results as PDF report
- [ ] Save optimization history (last 10 runs)
- [ ] A/B test comparison (run two presets side-by-side)
- [ ] 3D visualization of 42-d state space
- [ ] Real integration with actual DRL backend (FastAPI)
- [ ] User authentication & multi-tenant support
- [ ] Real database for storing campaign optimization history

---

## **START HERE**

1. **Create Next.js project:**
   ```bash
   npx create-next-app@latest drl-dashboard --typescript --tailwind
   cd drl-dashboard
   npm install recharts zustand lucide-react
   npm install -D @types/node
   ```

2. **Copy this project structure** and component templates

3. **Build components in this order:**
   - Header & Layout
   - Sidebar (ParameterPanel)
   - WorkflowVisualizer (basic flow)
   - ExecutionMonitor
   - ResultsDisplay
   - Connect with useWorkflow hook
   - Add API routes (/api/optimize, etc.)
   - Deploy to Vercel

4. **Test flows:**
   - Load preset → Run
   - Modify parameter → See impact graph update
   - Export/Import config
   - Check responsive design on mobile

---

**This prompt gives you everything needed to build a professional, visual DRL optimization dashboard that explains WHAT, HOW, and WHY at every step. Go build! 🚀**
