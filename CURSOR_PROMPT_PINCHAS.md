# DRL Workflow QA/Process UI — Pinchas Requirements

## **PROJECT BRIEF**

Build a **QA/Process UI** for the DRL pipeline that allows:
1. **View & Explore** all data flowing through each node (inputs, outputs, configurations)
2. **Edit & Configure** node parameters directly from the UI
3. **Run Individual Nodes** (not just "Run All") with one click
4. **Track Execution** with real-time status updates
5. **Separate visualization** of all key processing steps (P-Training, P-Execution, etc.)

---

## **PINCHAS'S CORE REQUIREMENTS**

From his email guidance on "Process & QA UI Notes":

### **Requirement 1: Support Running Processes from UI**
```
The UI should support running the processes:
- Each process (P-training, P-execution, etc.) can be run, activated from the UI 
  as long as inputs are ready and it is fully configured
- When a node is executed from the UI its inputs are unchanged 
  but the outputs may change
```

**Implementation:**
- Add "▶ Run Node" button for each workflow node
- Button should be enabled when node inputs are configured
- On click, execute the node and update its outputs
- Show execution time and status

---

### **Requirement 2: Include All Key Processing Steps as Separate Nodes**
```
Include in the process flow all key processing steps as separate nodes:
- For example: the P-training step should be shown separately from 
  the P-execution step
- Separate nodes for data creation vs. modeling (dev / execution) nodes
```

**Implementation:**
- Show these nodes as separate boxes:
  - **n1:** MockCampaignEnv / Data Source
  - **n2:** Replay Buffer / Data Collection
  - **n3:** SACAgent.train / Model Training
  - **n4:** Checkpoint / Artifacts (saved model)
  - **n5:** load_sac_for_inference / Model Loading
  - **n6:** SafeDRLAgent / Safety Layer
  - **n7:** HybridOptimizer / Decision Engine
- Each node has ports (input/output connectors)
- Edges show data flow between nodes

---

### **Requirement 3: Each Node Show Clear Input(s) & Output(s) That Can Be Viewed/Explored**
```
Each node should have clear input(s) & output(s), that can be viewed / explored

Example: Most nodes have at least one dataset or table as output, 
which the UI should enable exploring easily - all columns (NOT an aggregate 
single column that users have to select), 10+ rows of data shown, 
and the option to view the full table.
```

**Implementation:**
- Click node → Inspector panel shows node details
- Add "📊 View Data" button
- Click opens modal showing:
  - **INPUTS Tab:**
    - Table with columns: Field | Type | Sample | Source
    - All input fields listed with examples
    - Shows where data comes from
  - **OUTPUTS Tab:**
    - Table with columns: Field | Type | Sample | Used In
    - All output fields listed with examples
    - Shows where output data flows next
  - Show at least 3 sample rows of data
  - Add "View Full Table" option for large datasets

---

### **Requirement 4: Each Node Should Have Clear Set of Parameters That Can Be Viewed/Explored & Configured – Modified from the UI**
```
Each node should have clear set of Parameters that can be viewed / explored 
& configured – Modified from the UI

Example: Numerical parameters – such as threshold, algorithm parameters 
(numerical or labels for method option names...)
```

**Implementation:**
- Click node → Inspector panel shows node details
- Add "⚙️ Edit Params" button
- Click opens modal showing:
  - All configurable parameters in a grid/list
  - Each parameter shows:
    - Parameter name
    - Current value (editable)
    - Type (number, text, boolean, select)
    - Source (config file, env var, etc.)
  - Type-aware inputs:
    - Numbers: `<input type="number">`
    - Booleans: `<input type="checkbox">`
    - Selects: `<select>`
    - Text: `<input type="text">`
  - "Apply & Update Node" button to save changes
  - Changes immediately reflected in node

---

## **UI ARCHITECTURE**

```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER: DRL Workflow Studio                                     │
│ [Run all] [Step] [Reset] [Add node] [Export] [Import] [Fit]   │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────┬────────────────────────┬───────────────┐
│                        │                        │               │
│  WORKFLOW CANVAS       │   WORKFLOW CANVAS      │   INSPECTOR   │
│  (Left + Center)       │   CONTINUED            │   (Right)     │
│                        │                        │               │
│ ┌─────┐   ┌─────┐    │ ┌─────┐   ┌─────┐     │ ┌───────────┐ │
│ │ n1  │──→│ n2  │    │ │ n3  │──→│ n4  │     │ │ Node Info │ │
│ └─────┘   └─────┘    │ └─────┘   └─────┘     │ │           │ │
│   ↓        ↓         │                        │ │ Status    │ │
│ n1 title  n2 title   │  M1: Training          │ │ Inputs    │ │
│ [status]  [status]   │                        │ │ Outputs   │ │
│                        │                        │ │           │ │
│                        │ ┌─────┐   ┌─────┐    │ │ Buttons:  │ │
│ M1: Training           │ │ n5  │──→│ n6  │    │ │ [Data]    │ │
│                        │ └─────┘   └─────┘    │ │ [Params]  │ │
│                        │   ↓        ↓         │ │ [Run]     │ │
│                        │  n5 title n6 title   │ │           │ │
│                        │  [status] [status]   │ │ Data/      │ │
│                        │                      │ │ Config/    │ │
│                        │ ┌─────────────────┐  │ │ Code/      │ │
│                        │ │ n7: Optimizer   │  │ │ Logs/      │ │
│                        │ └─────────────────┘  │ │ Meta       │ │
│                        │                      │ │           │ │
│                        │ M2: Inference       │ │ └───────────┘ │
│                        │                      │               │
└────────────────────────┴────────────────────────┴───────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STATUS BAR: "Ready" | Last action | Execution time              │
└─────────────────────────────────────────────────────────────────┘
```

---

## **KEY UI COMPONENTS**

### **1. Workflow Canvas**
- Display all nodes in two sections: M1 (training) and M2 (inference)
- Nodes positioned to show data flow top → bottom
- Edges (arrows) show data flowing from output of one node to input of next
- Swimlanes with labels: "M1 — Training" and "M2 — Inference"
- Draggable nodes (optional) to rearrange
- Click node → select it (highlight) and show inspector on right

### **2. Inspector Panel (Right Side)**
**When node selected:**
```
┌─────────────────────────┐
│ Node Title              │ ← h2 element
│ [idle] Last: — ms       │ ← Status badge
│ id: n3                  │ ← Node ID
│                         │
│ [📊 View Data]          │ ← NEW: Click to see inputs/outputs
│ [⚙️ Edit Params]        │ ← NEW: Click to edit configuration
│ [▶ Run Node]            │ ← NEW: Click to execute node
│                         │
│ [Data] [Config] [Code]  │ ← Existing tabs
│ [Lineage] [Logs] [Meta] │
│                         │
│ ... tab content ...     │
└─────────────────────────┘
```

### **3. Data Explorer Modal**
```
┌──────────────────────────────────────┐
│ 📊 Data Explorer: SACAgent.train (n3)│
├──────────────────────────────────────┤
│ [Inputs] [Outputs] [Config]          │
│                                      │
│ INPUTS:                              │
│ ┌────────────────────────────────┐  │
│ │ Field  │ Type  │ Sample │ From │  │
│ ├────────┼───────┼────────┼──────┤  │
│ │ batch  │ Tensor│ (64,42)│ n2   │  │
│ │ target │ float │ 0.034  │ n2   │  │
│ └────────────────────────────────┘  │
│                                      │
│ OUTPUTS:                             │
│ ┌────────────────────────────────┐  │
│ │ Field │ Type  │ Sample │ Used  │  │
│ ├───────┼───────┼────────┼───────┤  │
│ │ loss  │ float │ 0.045  │ logs  │  │
│ │ grad  │ Tensor│ (...)  │ update│  │
│ └────────────────────────────────┘  │
│                                      │
│ [Close]                              │
└──────────────────────────────────────┘
```

### **4. Parameter Editor Modal**
```
┌──────────────────────────────────────┐
│ ⚙️ Edit Parameters: SACAgent (n3)    │
├──────────────────────────────────────┤
│                                      │
│ batch_size    [64        ]           │
│ gamma         [0.99      ]           │
│ tau           [0.005     ]           │
│ learning_rate [0.0003    ]           │
│                                      │
│ use_entropy   [✓]                    │
│                                      │
│ device        [cpu ▼]                │
│                                      │
│ [Apply & Update] [Cancel]            │
└──────────────────────────────────────┘
```

---

## **DETAILED IMPLEMENTATION GUIDE**

### **Step 1: Node Structure**
Each node object should have:
```javascript
{
  id: "n3",
  title: "SACAgent.train",
  group: "m1",  // m1 or m2
  x: 600, y: 118,  // position
  
  // Inputs schema
  inputSchema: [
    {
      column: "batch",
      dataType: "TensorBatch",
      sample: "(64, 42)",
      source: "n2 (Replay buffer)",
      description: "..."
    }
  ],
  
  // Outputs schema
  outputSchema: [
    {
      column: "policy_loss",
      dataType: "float",
      sample: "0.021",
      source: "n3",
      usedIn: "training loss tracking",
      notes: "..."
    }
  ],
  
  // Configuration parameters
  configRows: [
    {
      name: "batch_size",
      value: "64",
      type: "int",
      source: "config.yaml"
    },
    {
      name: "gamma",
      value: "0.99",
      type: "float",
      source: "config.yaml"
    },
    {
      name: "tau",
      value: "0.005",
      type: "float",
      source: "config.yaml"
    }
  ],
  
  // Execution tracking
  execution: {
    status: "idle",  // idle, running, success, error
    lastMs: null,
    schemaIssues: []
  },
  
  // For code display
  code: {
    path: "drl/sac_agent.py",
    language: "python",
    content: "..."
  }
}
```

### **Step 2: Data Explorer Implementation**

```javascript
function openDataExplorer(nodeId) {
  const node = nodes.find(n => n.id === nodeId);
  
  // Build inputs table
  const inputsHtml = buildTable(node.inputSchema, 
    ['column', 'dataType', 'sample', 'source']);
  
  // Build outputs table
  const outputsHtml = buildTable(node.outputSchema,
    ['column', 'dataType', 'sample', 'usedIn']);
  
  // Build config table
  const configHtml = buildTable(node.configRows,
    ['name', 'value', 'type', 'source']);
  
  // Show modal with tabs
  showModal('data-explorer', {
    inputs: inputsHtml,
    outputs: outputsHtml,
    config: configHtml
  });
}
```

### **Step 3: Parameter Editor Implementation**

```javascript
function openParamEditor(nodeId) {
  const node = nodes.find(n => n.id === nodeId);
  
  // Build form from configRows
  let formHtml = '';
  node.configRows.forEach((param, idx) => {
    const inputType = getInputType(param.type);
    formHtml += `
      <div class="param-field">
        <label>${param.name}</label>
        <input type="${inputType}" 
               data-idx="${idx}" 
               value="${param.value}" />
        <span class="hint">Source: ${param.source}</span>
      </div>
    `;
  });
  
  // Show modal with form
  showModal('param-editor', formHtml);
}

function applyParameterChanges(nodeId) {
  const node = nodes.find(n => n.id === nodeId);
  
  // Collect form values
  document.querySelectorAll('[data-idx]').forEach(input => {
    const idx = input.dataset.idx;
    node.configRows[idx].value = input.value;
  });
  
  // Close modal and refresh
  closeModal();
  refreshInspector();
}
```

### **Step 4: Run Node Implementation**

```javascript
function runNode(nodeId) {
  const node = nodes.find(n => n.id === nodeId);
  
  // Set status
  node.execution.status = 'running';
  refreshInspector();
  
  // Simulate execution
  setTimeout(() => {
    node.execution.status = 'success';
    node.execution.lastMs = 50 + Math.random() * 200;
    setStatus(`✅ ${node.title} completed in ${node.execution.lastMs}ms`);
    refreshInspector();
  }, 400 + Math.random() * 300);
}
```

---

## **CSS STYLING**

### **Key Classes:**
```css
/* Modals */
.modal { display: none; position: fixed; z-index: 1000; }
.modal.open { display: flex; }
.modal-content { background: white; border-radius: 8px; max-width: 1000px; }

/* Data tabs */
.data-tab-btn { padding: 8px 16px; border: none; background: transparent; }
.data-tab-btn.active { border-bottom: 3px solid #217346; color: #217346; }

/* Tables */
.modal-table { width: 100%; border-collapse: collapse; }
.modal-table th { background: #f0f0f0; padding: 8px; }
.modal-table td { padding: 8px; border: 1px solid #ddd; }

/* Forms */
.param-field { display: flex; flex-direction: column; margin-bottom: 12px; }
.param-field label { font-weight: 600; margin-bottom: 4px; }
.param-field input { padding: 6px 8px; border: 1px solid #ddd; border-radius: 4px; }

/* Buttons */
.btn-data { color: #217346; border: 1px solid #217346; }
.btn-params { color: #d84315; border: 1px solid #d84315; }
.btn-run { background: #4caf50; color: white; }
```

---

## **HTML STRUCTURE**

```html
<!-- Inspector Panel: Action Buttons -->
<div class="node-action-row">
  <button class="btn-action data" onclick="openDataExplorer(selectedId)">
    📊 View Data
  </button>
  <button class="btn-action params" onclick="openParamEditor(selectedId)">
    ⚙️ Edit Params
  </button>
  <button class="btn-action run" onclick="runNode(selectedId)">
    ▶ Run Node
  </button>
</div>

<!-- Data Explorer Modal -->
<div id="data-explorer-modal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      📊 Data Explorer: <span id="data-node-name"></span>
      <span class="close" onclick="closeDataExplorer()">×</span>
    </div>
    <div class="modal-body">
      <div class="data-tabs">
        <button class="data-tab-btn active" onclick="switchTab('inputs')">
          📥 Inputs
        </button>
        <button class="data-tab-btn" onclick="switchTab('outputs')">
          📤 Outputs
        </button>
        <button class="data-tab-btn" onclick="switchTab('config')">
          ⚙️ Config
        </button>
      </div>
      <div id="tab-inputs" class="tab-content active"></div>
      <div id="tab-outputs" class="tab-content"></div>
      <div id="tab-config" class="tab-content"></div>
    </div>
  </div>
</div>

<!-- Parameter Editor Modal -->
<div id="param-editor-modal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      ⚙️ Edit Parameters: <span id="param-node-name"></span>
      <span class="close" onclick="closeParamEditor()">×</span>
    </div>
    <div class="modal-body">
      <div class="param-form" id="param-form"></div>
      <div class="modal-buttons">
        <button class="btn-apply" onclick="applyParameterChanges()">
          Apply & Update
        </button>
        <button class="btn-cancel" onclick="closeParamEditor()">
          Cancel
        </button>
      </div>
    </div>
  </div>
</div>
```

---

## **USER WORKFLOW (Step by Step)**

1. **User opens dashboard** → Sees workflow canvas with 7 nodes (n1-n7)
2. **User clicks node** (e.g., "SACAgent.train") → Inspector panel shows on right
3. **Inspector shows:**
   - Node status (idle/running/success)
   - Last execution time
   - Three action buttons
4. **User clicks "📊 View Data"** → Modal opens showing:
   - Inputs: What data this node needs (from n2)
   - Outputs: What this node produces (policy loss)
   - Config: Parameters used (batch_size=64, gamma=0.99, etc.)
5. **User closes modal, clicks "⚙️ Edit Params"** → Modal opens with editable form
6. **User changes** `batch_size` from 64 to 128
7. **User clicks "Apply & Update"** → Change saved, modal closes
8. **User clicks "▶ Run Node"** → Node executes
9. **Status bar shows:** "✅ SACAgent.train completed in 145ms"
10. **User can now run next nodes** or click "Run all" to execute full pipeline

---

## **ACCEPTANCE CRITERIA**

✅ **Each node displays input/output schema** (Field, Type, Sample, Source/UsedIn)
✅ **Parameters are editable** from the UI (all types: number, text, bool, select)
✅ **Individual nodes can run** (not just "Run All")
✅ **Data Explorer shows** inputs, outputs, and configuration in separate tabs
✅ **Parameter changes persist** and show in node when re-selected
✅ **Execution status tracked** (idle → running → success, with timing)
✅ **Modals close properly** (X button, ESC key, click outside)
✅ **Responsive design** (works on desktop and tablets)
✅ **All buttons are functionally linked** to their respective modals/functions

---

## **TECH STACK**

- **HTML5** for structure
- **CSS3** for styling (Grid, Flexbox)
- **Vanilla JavaScript** (no frameworks required)
- **Canvas API** optional for drawing edges (or use SVG)

---

## **START BUILDING**

1. **Parse node data** and render workflow canvas (grid layout, two sections)
2. **Build Inspector panel** with node details
3. **Add action buttons** (View Data, Edit Params, Run Node)
4. **Create Data Explorer modal** with tabs (Inputs/Outputs/Config)
5. **Create Parameter Editor modal** with form inputs
6. **Wire up all onclick handlers** to open/close modals
7. **Test all interactions** (click node, view data, edit params, run)
8. **Style to match** existing DRL Workflow Studio design

---

**This is everything Pinchas asked for. Build it and you'll have a fully functional QA/Process UI! 🚀**
