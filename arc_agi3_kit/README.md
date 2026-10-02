# arc_agi3_kit

A small, dependency-light Python SDK + agent framework for the **ARC-AGI-3**
track of ARC Prize 2026 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).

Built directly against the official REST spec (`docs.arcprize.org/arc3v1.yaml`),
so `client.py` matches the real API's request/response shapes exactly.

## The problem, in one paragraph

You control an agent that gets dropped into a 64x64, 16-colour grid game with
**no instructions**. Each turn you send one of 8 actions (`RESET`, `ACTION1`-`ACTION5`,
the coordinate action `ACTION6`, and `ACTION7`/undo) and get back the next frame(s),
whether you've completed a level, and whether the run ended in `WIN`/`GAME_OVER`.
You're scored on **RHAE**: `(human_actions / your_actions)^2` per level — so
correctness alone isn't enough, you need to solve levels *efficiently*.

## Why this architecture

Four things are being tested (exploration, modeling, goal-setting, planning),
so the code is split to match:

```
arc_agi3_kit/
  client.py          # REST client: auth, session-affinity cookies, retries
  models.py           # GameAction/GameState enums, Frame, FrameResponse
  world_model.py       # turns raw grids into structure the agent can reason about
  planner.py           # decides what to do next, given the world model
  agent.py             # Agent ABC + Runner (the episode loop)
  agents/
    random_agent.py     # baseline
    explorer_agent.py    # the real agent: world_model + planner wired together
scripts/run.py         # CLI
tests/test_world_model.py  # offline unit tests (no API key needed)
```

**`world_model.py`** is the core idea. Instead of treating each frame as an
opaque 64x64 array:

- `extract_objects` does a flood-fill over same-colour regions and returns
  connected components (position, size, colour) — most ARC-AGI-3 games are
  built from a handful of sprites moving/colliding, not literal pixel noise.
- `match_objects` tracks those components across two frames by nearest
  centroid, so the agent can reason about "the object moved" instead of
  "63 cells changed."
- `WorldModel` is the agent's memory for one game: which actions are no-ops
  (skip them), a `(state, action) -> next_state` transition table (for loop
  detection), and — since `levels_completed` is the *only* reward signal the
  API gives you — which `(state, action)` pairs were immediately followed by
  progress, so they can be replayed instead of re-discovered.
- `click_candidates` ranks likely `ACTION6` targets by rarity/size instead of
  guessing from 4096 coordinates blind.

**`planner.py`** then just applies a priority order every turn:
**exploit** a known-progress action → **break loops** (undo/least-tried) →
**explore** the least-tried action (curiosity) → fallback random. This is a
strong, cheap-to-run baseline; it's meant to be a scaffold you improve on
(swap in RL, MCTS over the transition graph, etc.) rather than a finished
solution — 100% is a $700K unsolved grand prize for a reason.

## Quick start

```bash
pip install -r requirements.txt
export ARC_API_KEY="your_key_here"   # get one at docs.arcprize.org/api-keys

# play interactively in your terminal (with live colored grid graphics)
python scripts/play_local.py --game ls20
# or on Windows:
.\play-local.bat --game ls20
# or with make (Linux / macOS / WSL):
make play-local GAME=ls20

# run local offline simulator (no network required)
python scripts/play_local.py --game ls20 --offline

# see what games you have access to
python scripts/run.py --list-games

# run the explorer agent on one game
python scripts/run.py --game ls20 --agent explorer --max-actions 500 -v

# run the offline unit tests (no API key / network needed)
python tests/test_world_model.py
```

## Extending it

- **Plug in a real learner:** `Agent.choose_action` gets the full frame
  history and the shared `WorldModel` — swap `ExplorationPlanner` for a
  policy trained with RL (e.g. PPO over a CNN encoding of the grid), or add
  MCTS over `WorldModel.transitions` once you have a few episodes of data.
- **Persist the world model across runs:** `Agent.on_episode_end` is the
  hook — pickle `world_model` there and reload it in `on_episode_start` if
  you want cross-session memory for a given game.
- **Vision/LLM agents:** ARC Prize also publishes reference LLM-based agents
  (see `docs.arcprize.org/llm_agents`) if you want to compare a
  hypothesis-driven, language-model-in-the-loop approach against this
  model-free one. Note: **no internet access is allowed during Kaggle
  evaluation**, so any API-based agent (GPT/Claude/etc.) only works for local
  development — your final submission needs to be self-contained.


## Reference docs

- REST overview: https://docs.arcprize.org/rest_overview
- OpenAPI spec: https://docs.arcprize.org/arc3v1.yaml
- Scoring methodology: https://docs.arcprize.org/methodology
- Local vs online (for the no-internet Kaggle submission): https://docs.arcprize.org/local-vs-online
