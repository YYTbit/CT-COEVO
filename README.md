<div align="center">

# 🧬 CT-COEVO

**An Autonomous Agent for Recommender System Design via Context-Tool Co-Evolution**

<br>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-28A745.svg)](LICENSE)

<br>

<img src="docs/framework.png" alt="CT-COEVO Framework" width="85%">

</div>

---

## Overview

CT-COEVO is an autonomous LLM-based agent that solves recommender system competitions end-to-end. Its key innovation is **co-evolving** two components across tasks:

- **Contextual Memory (M)** — accumulates transferable heuristics from past experiments
- **Algorithmic Toolkit (K)** — grows a library of reusable algorithm tools

This enables the agent to improve over time: lessons learned on early tasks directly boost performance on later ones.

### Core Loop (Paper Eq. 1–4)

| Step | Equation | Description |
|------|----------|-------------|
| **Extract** | `E_t = Extract(q, o_{t-1}; M)` | Retrieve relevant memories |
| **Select** | `(τ_t, θ_t) = π(q, E_t; K)` | Choose tool + configuration |
| **Execute** | `o_t = Experiment(τ_t, θ_t; D)` | Run tool in isolated workspace |
| **Distill** | `M ← M ∪ {Distill(...)}` | Store results as new memories |

After each task, effective tools are **promoted** to the global toolkit; failed experiments generate **negative heuristics** to prevent repeating mistakes.

---

## Installation

```bash
# Clone
git clone https://github.com/YYTbit/CT-COEVO.git
cd CT-COEVO

# Environment
conda create -n CT-COEVO python=3.10
conda activate CT-COEVO

# Dependencies
pip install -r requirements.txt

# Set PYTHONPATH (parent directory of ct_coevo/)
export PYTHONPATH=/path/to/parent:$PYTHONPATH
```

> **Note**: Core framework requires only `openai`, `pandas`, `numpy`. Install optional deps (`torch`, `lightgbm`, etc.) if you want the agent to generate GPU training code.

---

## Quick Start

### Evolution Mode (EvoSet — 34 datasets)

Build memory and toolkit by running the agent on classical recommendation datasets:

```bash
python -m ct_coevo.runner \
    --mode evo \
    --dataset ml_1m \
    --api-key YOUR_API_KEY \
    --api-url https://api.example.com/v1 \
    --model your-model-name
```

### Evaluation Mode (EvalSet — 49 datasets)

Test transfer with **frozen** memory and toolkit on recent competitions:

```bash
python -m ct_coevo.runner \
    --mode eval \
    --dataset recsys_2018_spotify \
    --api-key YOUR_API_KEY \
    --api-url https://api.example.com/v1 \
    --model your-model-name
```

### Run All Datasets

```bash
python -m ct_coevo.runner --mode evo --dataset all \
    --api-key YOUR_API_KEY --api-url https://api.example.com/v1 --model your-model-name
```

---

## Project Structure

```
ct_coevo/
├── agent.py              # Core agent loop (Eq. 1–4)
├── memory.py             # HierarchicalContextualMemory (M)
├── toolkit.py            # ScalableAlgorithmicToolkit (K)
├── prompts.py            # 11 prompt templates (paper Listings 1–11)
├── grader.py             # Grading interface (loads metric.py from benchmarks)
├── runner.py             # CLI entry point for all 83 datasets
└── evolution_loop.py     # Long-run evolution engine
```

### Memory Types

| Type | Symbol | Role |
|------|--------|------|
| Experiential | `Exper.` | Transferable heuristics distilled across tasks |
| Experimental | `Expt.` | Per-task observations (what was tried, what happened) |
| Execution | `Exec.` | Per-tool performance traces (appended to tool records) |

### Toolkit Tiers

| Tier | Symbol | Role |
|------|--------|------|
| Base | `K_base` | Immutable primitives (`python`, `bash`) |
| Meta | `K_meta` | Tool-creating operations (`create_tool`, `edit_tool`) |
| Global | `K_global` | Reviewed, reusable pipelines |
| Temporary | `K_temp` | Task-scoped experimental variants |

---

## Benchmark

The agent is evaluated on **83 recommendation competition datasets** spanning 5 task categories:

| Category | Task | Metric | EvoSet | EvalSet |
|----------|------|--------|--------|---------|
| **R&S** | Rating & Scoring | RMSE | 12 | 5 |
| **CTR** | Click-Through Rate | AUC | 7 | 12 |
| **Rank** | Ranking | NDCG / Recall | 9 | 10 |
| **MLC** | Multi-Label Classification | F1 | 3 | 2 |
| **Seq** | Sequential Recommendation | MRR | 3 | 2 |

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
    └── prepare.py               # Data preparation
```

Datasets will be available on Google Drive (link TBD).

---

## Advanced Usage

### Checkpoint & Resume

The agent saves checkpoints after each step. To resume:

```python
from ct_coevo import CTCoEvoAgent

agent = CTCoEvoAgent(
    dataset_name="ml_1m",
    data_dir="/path/to/data/public",
    api_key="YOUR_API_KEY",
    model="your-model",
    base_url="https://api.example.com/v1",
    workspace_dir="/path/to/existing/workspace",  # Resume from checkpoint
)
result = agent.run()
```

### Long Runs with tmux

```bash
tmux new-session -d -s ctcoevo \
    "conda activate CT-COEVO && python -m ct_coevo.runner --mode evo --dataset ml_1m \
     --api-key KEY --api-url URL --model MODEL"

# Monitor
tmux attach -t ctcoevo
```

### Tool Call Format

The agent outputs JSON arrays for tool selection:

```json
[{
  "tool_id": "tool_xxx",
  "config": {},
  "name": "deepfm_v1",
  "code": "import torch\n...",
  "description": "DeepFM for CTR prediction",
  "review_time": 1800
}]
```

| Field | Description |
|-------|-------------|
| `tool_id` | Target tool ID |
| `config` | Configuration parameters |
| `name` | Tool name (for `create_tool`) |
| `code` | Python source code (for `create_tool` / `edit_tool`) |
| `description` | Tool description (for `create_tool`) |
| `review_time` | Seconds before returning logs. `-1` = wait forever (default). Recommended: `1800` for training. |

---

## License

MIT License. See [LICENSE](LICENSE).
