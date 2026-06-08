<div align="center">

# 🧬 CT-COEVO

### **C**ontext-**T**ool **Co**-**Evo**lution

**An Autonomous Agent for Recommender System Design**

<br>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-28A745.svg?style=for-the-badge)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/xxxx.xxxxx)

<br>

<img src="docs/framework.png" alt="CT-COEVO Framework" width="90%">

<br>

**CT-COEVO** is an autonomous LLM-based agent that tackles recommendation competitions end-to-end.
<br>
It uniquely **co-evolves** its contextual memory and algorithmic toolkit across tasks, achieving continuous improvement without human intervention.

</div>

---

## 📖 Table of Contents

- [Why CT-COEVO?](#-why-ct-coevo)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture Deep Dive](#-architecture-deep-dive)
- [Benchmark](#-benchmark)
- [Advanced](#-advanced)
- [License](#-license)

---

## 🎯 Why CT-COEVO?

Existing automated ML agents (AIDE, DS-Agent, MLE-Agent) treat each task in isolation — they start from scratch every time. This means:

- ❌ No transfer of knowledge across tasks
- ❌ Repeatedly making the same mistakes
- ❌ No growth in algorithmic capabilities

**CT-COEVO** solves this with **co-evolution**:

| | Traditional Agents | CT-COEVO |
|---|---|---|
| **Memory** | None or session-only | Persistent, hierarchical (Exper. → Expt. → Exec.) |
| **Toolkit** | Fixed or ad-hoc | Scalable 4-tier system (Base → Meta → Global → Temp) |
| **Learning** | None | Cross-task transfer via memory distillation |
| **Improvement** | Resets each task | Cumulative — gets better over time |

---

## ⚙️ How It Works

The agent operates in a loop inspired by reinforcement learning principles. At each step:

### Step 1: Extract Context

```
E_t = Extract(q, o_{t-1}; M)
```

The agent queries its hierarchical memory **M** to retrieve the **K** most relevant experiences — past successes, failures, and transferable heuristics.

### Step 2: Select & Configure Tool

```
(τ_t, θ_t) = π(q, E_t; K)
```

Based on the task and retrieved context, the LLM selects a tool from toolkit **K** and generates configuration (e.g., PyTorch training code).

### Step 3: Execute

```
o_t = Experiment(τ_t, θ_t; D)
```

The tool runs in an isolated workspace. Results (logs, submission files) are captured.

### Step 4: Distill

```
M ← M ∪ {Distill(q, τ_t, θ_t, o_t)}
```

The LLM synthesizes the execution into a new memory entry — what was tried, what happened, and the key lesson.

### Post-Task Consolidation

After each task completes:

```
┌─────────────────────────────────────────────────────────┐
│  1. Distill Experimental → Experiential memory          │
│     (transferable heuristics for future tasks)          │
│                                                         │
│  2. Promote effective Temporary tools → Global          │
│     (reusable algorithm pipelines)                      │
│                                                         │
│  3. Prune underperforming tools                         │
│     (prevent toolkit bloat)                             │
│                                                         │
│  4. Merge local state → global state                    │
│     (share across datasets)                             │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
CT-COEVO/
│
├── ct_coevo/                   # Core framework
│   ├── agent.py                #   Main agent loop (Eq. 1–4)
│   ├── memory.py               #   HierarchicalContextualMemory (M)
│   ├── toolkit.py              #   ScalableAlgorithmicToolkit (K)
│   ├── prompts.py              #   11 prompt templates (paper Listings 1–11)
│   ├── grader.py               #   Grading interface (loads metric.py)
│   ├── runner.py               #   CLI entry point
│   └── evolution_loop.py       #   Long-run evolution engine
│
├── docs/
│   └── framework.png           #   Architecture diagram
│
├── requirements.txt            #   Python dependencies
├── LICENSE                     #   MIT License
└── README.md                   #   This file
```

### Key Components

| File | Lines | Description |
|------|-------|-------------|
| `agent.py` | ~1100 | Core loop: extract → select → execute → distill |
| `memory.py` | ~300 | Three-tier memory with markdown file storage |
| `toolkit.py` | ~200 | Four-tier toolkit with promotion/pruning |
| `prompts.py` | ~570 | All 11 prompt templates aligned with paper |
| `grader.py` | ~100 | Loads `metric.py` from benchmark to grade submissions |
| `runner.py` | ~160 | CLI orchestrating 83 datasets (34 Evo + 49 Eval) |

---

## 🔧 Installation

### Prerequisites

- **Python** 3.10 or higher
- **CUDA** 11.8+ (for GPU training)
- **GPU** — recommended: 4× NVIDIA RTX 3090 (24GB each)
- **API** — an OpenAI-compatible LLM API endpoint

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/YYTbit/CT-COEVO.git
cd CT-COEVO

# 2. Create conda environment
conda create -n CT-COEVO python=3.10 -y
conda activate CT-COEVO

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set PYTHONPATH (parent directory of ct_coevo/)
export PYTHONPATH=$(cd .. && pwd):$PYTHONPATH
```

<details>
<summary><b>📦 What's in requirements.txt?</b></summary>

**Core (required)**:
```
openai>=1.0.0
pandas>=1.5.0
numpy>=1.21.0
```

**Optional (for agent-generated training code)**:
```
torch>=2.0.0
lightgbm>=3.3.0
xgboost>=1.6.0
implicit>=0.6.0
scikit-learn>=1.0.0
```

The core framework only needs `openai`, `pandas`, `numpy`. The optional packages are used by the agent's generated training scripts — install them if you want the agent to create GPU-based models.

</details>

---

## 🚀 Usage

### Single Dataset

```bash
python -m ct_coevo.runner \
    --mode evo \
    --dataset ml_1m \
    --api-key YOUR_API_KEY \
    --api-url https://api.example.com/v1 \
    --model deepseek-ai/DeepSeek-V3.2 \
    --timeout 86400
```

### All Datasets

```bash
python -m ct_coevo.runner \
    --mode evo \
    --dataset all \
    --api-key YOUR_API_KEY \
    --api-url https://api.example.com/v1 \
    --model deepseek-ai/DeepSeek-V3.2
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mode` | ✅ | — | `evo` (evolution) or `eval` (evaluation) |
| `--dataset` | ✅ | — | Dataset name, or `all` |
| `--api-key` | ✅ | — | LLM API key |
| `--api-url` | — | `https://api.siliconflow.cn/v1` | API base URL |
| `--model` | — | `deepseek-ai/DeepSeek-V3.2` | Model ID |
| `--timeout` | — | `86400` | Total time limit in seconds (24h) |

### Running Long Experiments

```bash
# Start in tmux (persists across disconnects)
tmux new-session -d -s ctcoevo \
    "conda activate CT-COEVO && python -m ct_coevo.runner \
     --mode evo --dataset ml_1m \
     --api-key KEY --api-url URL --model MODEL"

# Detach: Ctrl+B, then D
# Reattach:
tmux attach -t ctcoevo
```

### Python API

```python
from ct_coevo import CTCoEvoAgent

agent = CTCoEvoAgent(
    dataset_name="ml_1m",
    data_dir="/path/to/data/public",
    api_key="YOUR_API_KEY",
    model="deepseek-ai/DeepSeek-V3.2",
    base_url="https://api.example.com/v1",
    timeout_sec=86400,
    evolve=True,   # True = evo mode, False = eval mode
)
result = agent.run()
print(f"Best score: {result['best_score']}")
```

### Checkpoint & Resume

The agent automatically saves checkpoints. To resume a crashed run:

```python
agent = CTCoEvoAgent(
    dataset_name="ml_1m",
    data_dir="/path/to/data/public",
    api_key="YOUR_API_KEY",
    model="your-model",
    base_url="https://api.example.com/v1",
    workspace_dir="/path/to/existing/workspace",  # ← Resume
)
agent.run()
```

---

## 🏗️ Architecture Deep Dive

### Hierarchical Contextual Memory (M)

Three memory types with increasing specificity:

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Hierarchy                         │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Experiential (Exper.)                                │  │
│  │  "Use pairwise loss for ranking metrics"              │  │
│  │  → Transferable across tasks                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ▲                                  │
│                     Distill                                 │
│                          │                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Experimental (Expt.)                                 │  │
│  │  "Tried LightGBM on sparse IDs → failed"             │  │
│  │  → Per-task observations                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ▲                                  │
│                     Observe                                 │
│                          │                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Execution (Exec.)                                    │  │
│  │  "Trained DeepFM 100 epochs, loss=0.45"              │  │
│  │  → Per-tool traces (attached to tool records)         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Storage**: Each memory item is a markdown file with title, summary, and body.

```markdown
---
label: Exper.
title: Use pairwise loss for ranking metrics
summary: When the evaluation metric is NDCG or Recall, prefer BPR loss over BCE.
---

## Body

Detailed conditions, actions, exceptions, and evidence...
```

### Scalable Algorithmic Toolkit (K)

```
┌─────────────────────────────────────────────────────────┐
│                    Toolkit Tiers                         │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │
│  │  Base   │  │  Meta   │  │ Global  │  │   Temp    │  │
│  │ python  │  │ create  │  │ DeepFM  │  │ v1_trial  │  │
│  │ bash    │  │ _tool   │  │ LightFM │  │ v2_trial  │  │
│  │         │  │ edit    │  │ SASRec  │  │           │  │
│  │         │  │ _tool   │  │  ...    │  │           │  │
│  └─────────┘  └─────────┘  └─────────┘  └───────────┘  │
│                                                         │
│  Immutable    Creates new    Reviewed &    Experimental  │
│  primitives   tools          reusable      variants      │
│                                                         │
│                            ▲           │                │
│                            └───────────┘                │
│                           Promote effective              │
│                           Prune underperforming          │
└─────────────────────────────────────────────────────────┘
```

### Agent-Tool Interaction

The agent communicates with tools via JSON:

```json
[
  {
    "tool_id": "tool_1780763698112_2",
    "name": "deepfm_v1",
    "code": "import torch\nimport torch.nn as nn\n...",
    "description": "DeepFM model for CTR prediction with GPU training",
    "review_time": 1800
  }
]
```

| Field | Description |
|-------|-------------|
| `tool_id` | Target tool identifier |
| `name` | Tool name (for `create_tool`) |
| `code` | Complete Python training script |
| `description` | What the tool does |
| `review_time` | Seconds before returning logs. `-1` = wait forever. Default: `-1`. Recommended: `1800` (30 min) for training tasks. |

---

## 📊 Benchmark

The agent is evaluated on **83 recommendation competition datasets** spanning 5 task categories.

| Category | Task Type | Evaluation Metric | #EvoSet | #EvalSet |
|----------|-----------|-------------------|---------|----------|
| **R&S** | Rating & Scoring | RMSE | 12 | 5 |
| **CTR** | Click-Through Rate | AUC | 7 | 12 |
| **Rank** | Ranking | NDCG / Recall | 9 | 10 |
| **MLC** | Multi-Label Classification | F1 | 3 | 2 |
| **Seq** | Sequential Recommendation | MRR | 3 | 2 |

- **EvoSet** (34 datasets, 1997–2010): Classical recommendation datasets for agent evolution
- **EvalSet** (49 datasets, 2012–2025): Recent competitions for transfer evaluation

> 📥 Datasets will be available on Google Drive (link TBD).

### Data Format

Each dataset follows a uniform structure:

```
{dataset_id}/
├── data/
│   ├── public/
│   │   ├── train.csv (or .json)
│   │   ├── test.csv (or .json)
│   │   ├── sample_submission.csv
│   │   └── description.md
│   └── private/
│       └── answers.csv          # Ground truth (hidden from agent)
└── utils/
    ├── metric.py                # Grading function
    └── prepare.py               # Data preparation script
```

---

## 🔬 Advanced

### Two Operating Modes

| Mode | Purpose | Memory & Toolkit | Datasets |
|------|---------|------------------|----------|
| **Evo** | Build capabilities | Evolve (grow & improve) | EvoSet (34) |
| **Eval** | Test transfer | Frozen from Evo phase | EvalSet (49) |

### State Persistence

Evolved state is stored at `ct_coevo/state/global/`:

```
state/global/
├── memory/                    # Experiential memory files
│   ├── Exper._use_pairwise_loss.md
│   ├── Exper._gpu_for_training.md
│   └── ...
└── toolkit/                   # Global tool definitions
    ├── toolkit_items.json     # Tool metadata
    ├── deepfm_v1.py           # Tool source code
    └── ...
```

### Checking Results

```bash
# Run results
cat ct_coevo/log/{dataset}-{timestamp}/results.json

# Agent reasoning (full message log)
python3 -m json.tool ct_coevo/log/{dataset}-{timestamp}/message_log.jsonl

# Checkpoint state
python3 -m json.tool ct_coevo/workspace/{dataset}-{timestamp}/checkpoint.json

# Evolved memories
ls ct_coevo/state/global/memory/

# Evolved toolkit
python3 -m json.tool ct_coevo/state/global/toolkit/toolkit_items.json
```

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**[Paper](https://arxiv.org/abs/xxxx.xxxxx)** ·
**[Code](https://github.com/YYTbit/CT-COEVO)**

</div>
