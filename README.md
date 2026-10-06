# ARC-AGI-3 Agent & Development Kit

[![Competition](https://img.shields.io/badge/Competition-ARC--Prize--2026-blue)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
[![Track](https://img.shields.io/badge/Track-ARC--AGI--3-orange)](https://docs.arcprize.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

An end-to-end framework and autonomous object-centric agent engineered for the **ARC-AGI-3 (ARC Prize 2026)** competition. This kit contains the core autonomous reasoning agent, an offline simulation engine, interactive terminal visualization, and an automated Kaggle submission pipeline.

---

## 🎯 Purpose of This Build

In **ARC-AGI-3**, an autonomous AI agent is dropped into an unlabelled $64 \times 64$, 16-color interactive environment with **zero prior instructions, rules, or text prompts**.

1. **The Challenge**:
   - The agent can only execute 8 discrete actions:
     - `RESET` (0)
     - `ACTION1` - `ACTION5` (1–5, typically directional movement/interaction)
     - `ACTION6` (6, coordinate clicks: `[y, x]`)
     - `ACTION7` (7, undo)
   - The agent receives only raw grid frames and a single sparse reward signal: whether a level has been completed.
   - The agent is evaluated on **Relative Human Action Efficiency (RHAE)**:
     $$\text{RHAE} = \left(\frac{\text{human\_actions}}{\text{agent\_actions}}\right)^2$$
     Solving a puzzle isn't enough—the agent must discover rules and reach the goal with minimal steps.

2. **The Goal of This Build**:
   - Deliver a **self-contained, internet-independent agent** that autonomously learns environment physics, infers goals, navigates obstacles, and solves unseen games without human supervision.
   - Provide a complete local development suite for offline rapid testing and one-command Kaggle competition packaging.

---

## 🚀 What Was Built

```
arc_agi3_kit/
├── agent/
│   └── my_agent.py             # Core autonomous object-centric agent (v3 Hybrid)
├── arc_agi3_kit/               # Lightweight Python SDK for ARC-AGI-3
│   ├── client.py               # REST API client matching official OpenAPI spec
│   ├── models.py               # Frame, FrameResponse, GameAction, GameState models
│   ├── world_model.py          # Object extraction, tracking, & causal state model
│   ├── causal_memory.py        # Cross-step causal transition memory
│   ├── planner.py              # Priority-based exploration & exploitation planner
│   └── agents/                 # Reference agents (Explorer, Random)
├── notebooks/
│   ├── kernel-metadata.json    # Kaggle notebook configuration
│   └── submission.ipynb        # Standalone competition submission notebook
├── scripts/
│   ├── build_notebook.py       # Inlines code into self-contained submission notebook
│   ├── evaluate_local.py       # Procedural offline benchmark simulator (100+ puzzles)
│   ├── play_local.py           # Interactive terminal game viewer with ANSI colors
│   └── run.py                  # CLI runner for ARC REST API
├── Makefile                    # Developer shortcuts (build, test, run, submit)
├── play-local.bat              # One-click Windows local player
├── status.bat                  # One-click Kaggle status monitor
├── submit.bat                  # One-click Kaggle build & push script
└── .gitignore                  # Exclusion for keys, virtualenvs, logs, & cache
```

---

## 🧠 Core Features & Architecture

### 1. Hybrid Empirical Object-Centric Agent (`agent/my_agent.py`)
Rather than treating frames as raw pixel tensors or doing brute-force trial-and-error, the agent decomposes each puzzle using cognitive inductive priors:
- **Object Segmentation**: Flood-fills connected components by color to extract discrete sprites, bounding boxes, sizes, and centroids.
- **Majority-Voting Kinematics**: Tracks object movements across consecutive frames to deduce which action corresponds to which physical direction (e.g., discovering `ACTION1 = UP`, `ACTION2 = RIGHT`) across dynamic level geometries.
- **Goal & Landmark Inference**: Identifies static targets, exit doors, color-changing flags, or interactable keys by tracking state changes associated with level completions.
- **Relaxed BFS Pathfinding**: Once agent and target centroids are established, computes the shortest valid path to the goal, ignoring irrelevant noise while respecting detected obstacles and border walls.
- **Loop Breaking & Anti-Stuck Memory**: Detects cyclic transitions and suppresses no-op actions, automatically triggering undo (`ACTION7`) or fallback exploration to break out of dead ends.
- **Saliency-Ranked Coordinate Clicks**: Intelligently ranks `ACTION6` click coordinates by color rarity, object centroids, and non-zero raster scans instead of guessing randomly across 4,096 cells.

### 2. ARC-AGI-3 Python SDK (`arc_agi3_kit/`)
- Pure Python REST client implementing the official `docs.arcprize.org/arc3v1.yaml` specification.
- Resilient session-affinity handling, cookie persistence, and automatic backoff/retry.

### 3. Procedural Offline Testbench (`scripts/evaluate_local.py`)
- Simulates procedurally generated ARC-AGI-3 puzzles locally (mazes, sokoban-like push mechanics, color-matching gates).
- Calculates win rates, average actions, and RHAE metrics completely offline without an API key or internet access.

### 4. Zero-Dependency Kaggle Pipeline (`scripts/build_notebook.py` & `submit.bat`)
- Bundles all agent code directly into `notebooks/submission.ipynb`.
- Includes sidecar discovery and health-check retry loops for the Kaggle competition execution sandbox.
- Produces valid `submission.parquet` records.

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/PARTH0777-O/ARC_AGI3_kit.git
cd ARC_AGI3_kit

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r arc_agi3_kit/requirements.txt
```

### 2. Run Local Offline Evaluation
Benchmark the agent locally across procedural puzzles:
```bash
python scripts/evaluate_local.py
```

### 3. Interactive Terminal Player
Play puzzles manually or inspect agent decisions with real-time ANSI colored grids:
```bash
# Windows:
.\play-local.bat --game ls20 --offline

# Or via Python directly:
python arc_agi3_kit/scripts/play_local.py --game ls20 --offline
```

### 4. Build & Submit to Kaggle
To build the notebook and submit to the Kaggle competition:
```bash
# Windows one-click:
.\submit.bat

# Or using Make:
make submit
```

To monitor submission status:
```bash
.\status.bat
# or
make status
```

---

## 🔒 Security & Privacy
- Sensitive credentials (`.env`, `ARC_API_KEY`) and personal data are strictly ignored via `.gitignore` and omitted from version control.
- Dummy placeholders (`your-username/arc-agi-3-agent`) are used in metadata templates so users can plug in their own competition handles safely.
