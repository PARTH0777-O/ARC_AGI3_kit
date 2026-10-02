#!/usr/bin/env python3
"""Evaluates MyAgent locally against ARC-AGI-3 offline games and computes scores."""

import sys
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Import agent directly from agent/my_agent.py
spec = importlib.util.spec_from_file_location("my_agent", ROOT / "agent" / "my_agent.py")
my_agent_module = importlib.util.module_from_spec(spec)
sys.modules["my_agent"] = my_agent_module
spec.loader.exec_module(my_agent_module)
MyAgent = my_agent_module.MyAgent

# Import models
models_path = ROOT / "arc_agi3_kit" / "arc_agi3_kit" / "models.py"
spec_m = importlib.util.spec_from_file_location("models", models_path)
models_module = importlib.util.module_from_spec(spec_m)
sys.modules["models"] = models_module
spec_m.loader.exec_module(models_module)
GameAction = models_module.GameAction
GameState = models_module.GameState
Frame = models_module.Frame
FrameResponse = models_module.FrameResponse


# =============================================================================
# Procedural Engine Generators for 100 Diverse ARC-AGI-3 Puzzles
# =============================================================================

class ProceduralMazeEngine:
    """Procedural Maze & Labyrinth Corridor Puzzle Generator."""
    def __init__(self, puzzle_id: int, seed: int):
        self.game_id = f"proc-maze-{puzzle_id}"
        self.guid = f"guid-m-{puzzle_id}"
        self.rng = np.random.RandomState(seed)
        self.win_levels = 2
        self.levels_completed = 0
        self.state = GameState.NOT_FINISHED
        self.size = self.rng.randint(14, 22)
        self.wall_color = int(self.rng.choice([4, 5, 6]))
        self.corridor_color = int(self.rng.choice([0, 3]))
        self.goal_color = int(self.rng.choice([11, 12, 13]))
        self.player_color = 1
        self._init_level()

    def _init_level(self):
        self.grid = np.full((self.size, self.size), self.wall_color, dtype=np.uint8)
        # Generate random connected corridor path from (1, 1) to (size-2, size-2)
        r, c = 1, 1
        self.grid[r, c] = self.corridor_color
        path = [(r, c)]
        gr, gc = self.size - 2, self.size - 2

        curr_r, curr_c = r, c
        while (curr_r, curr_c) != (gr, gc):
            moves = []
            if curr_r < gr: moves.append((1, 0))
            if curr_c < gc: moves.append((0, 1))
            if curr_r > 1 and self.rng.rand() < 0.2: moves.append((-1, 0))
            if curr_c > 1 and self.rng.rand() < 0.2: moves.append((0, -1))
            if not moves: moves = [(1, 0), (0, 1)]
            dr, dc = moves[self.rng.randint(0, len(moves))]
            curr_r, curr_c = curr_r + dr, curr_c + dc
            self.grid[curr_r, curr_c] = self.corridor_color
            path.append((curr_r, curr_c))

        # Add random dead-end branches
        for _ in range(self.size):
            br_r, br_c = path[self.rng.randint(0, len(path))]
            for _ in range(3):
                dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][self.rng.randint(0, 4)]
                nr, nc = br_r + dr, br_c + dc
                if 1 <= nr < self.size - 1 and 1 <= nc < self.size - 1:
                    self.grid[nr, nc] = self.corridor_color
                    br_r, br_c = nr, nc

        self.player_r, self.player_c = 1, 1
        self.goal_r, self.goal_c = gr, gc
        self.grid[self.goal_r, self.goal_c] = self.goal_color
        self.grid[self.player_r, self.player_c] = self.player_color

    def step(self, action: GameAction, x=None, y=None) -> FrameResponse:
        dr, dc = 0, 0
        if action == GameAction.ACTION1: dr = -1
        elif action == GameAction.ACTION2: dr = 1
        elif action == GameAction.ACTION3: dc = -1
        elif action == GameAction.ACTION4: dc = 1
        elif action == GameAction.RESET:
            self._init_level()
            return self._make_resp()

        nr, nc = self.player_r + dr, self.player_c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            target_col = self.grid[nr, nc]
            if target_col in (self.corridor_color, self.goal_color):
                self.grid[self.player_r, self.player_c] = self.corridor_color
                self.player_r, self.player_c = nr, nc
                self.grid[self.player_r, self.player_c] = self.player_color

        if self.player_r == self.goal_r and self.player_c == self.goal_c:
            self.levels_completed += 1
            if self.levels_completed >= self.win_levels:
                self.state = GameState.WIN
            else:
                self._init_level()

        return self._make_resp()

    def _make_resp(self) -> FrameResponse:
        return FrameResponse(
            game_id=self.game_id,
            guid=self.guid,
            frames=[Frame(self.grid.copy())],
            state=self.state,
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4],
        )


class ProceduralKeyLockEngine:
    """Procedural Key-Switch & Gate Door Unlocking Generator."""
    def __init__(self, puzzle_id: int, seed: int):
        self.game_id = f"proc-keylock-{puzzle_id}"
        self.guid = f"guid-k-{puzzle_id}"
        self.rng = np.random.RandomState(seed)
        self.win_levels = 2
        self.levels_completed = 0
        self.state = GameState.NOT_FINISHED
        self.size = 16
        self.switch_color = 7
        self.gate_color = 9
        self.goal_color = 2
        self.gate_open = False
        self._init_level()

    def _init_level(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        self.grid[0, :] = 5; self.grid[-1, :] = 5; self.grid[:, 0] = 5; self.grid[:, -1] = 5
        mid = self.size // 2
        self.grid[:, mid] = 5
        self.gate_r, self.gate_c = mid, mid
        self.grid[self.gate_r, self.gate_c] = self.gate_color

        self.switch_r = self.rng.randint(2, self.size - 2)
        self.switch_c = self.rng.randint(1, mid - 1)
        self.grid[self.switch_r, self.switch_c] = self.switch_color

        self.goal_r = self.rng.randint(2, self.size - 2)
        self.goal_c = self.rng.randint(mid + 1, self.size - 2)
        self.grid[self.goal_r, self.goal_c] = self.goal_color

        self.player_r, self.player_c = 2, 2
        self.grid[self.player_r, self.player_c] = 1
        self.gate_open = False

    def step(self, action: GameAction, x=None, y=None) -> FrameResponse:
        dr, dc = 0, 0
        if action == GameAction.ACTION1: dr = -1
        elif action == GameAction.ACTION2: dr = 1
        elif action == GameAction.ACTION3: dc = -1
        elif action == GameAction.ACTION4: dc = 1

        nr, nc = self.player_r + dr, self.player_c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            cell = self.grid[nr, nc]
            if cell != 5 and (cell != self.gate_color or self.gate_open):
                self.grid[self.player_r, self.player_c] = 0
                self.player_r, self.player_c = nr, nc
                self.grid[self.player_r, self.player_c] = 1
                if (nr, nc) == (self.switch_r, self.switch_c):
                    self.gate_open = True
                    self.grid[self.gate_r, self.gate_c] = 0

        if self.player_r == self.goal_r and self.player_c == self.goal_c:
            self.levels_completed += 1
            if self.levels_completed >= self.win_levels:
                self.state = GameState.WIN
            else:
                self._init_level()

        return FrameResponse(
            game_id=self.game_id,
            guid=self.guid,
            frames=[Frame(self.grid.copy())],
            state=self.state,
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4],
        )


class ProceduralCollectibleEngine:
    """Procedural Multi-Item Collectible Puzzle Generator."""
    def __init__(self, puzzle_id: int, seed: int):
        self.game_id = f"proc-collect-{puzzle_id}"
        self.guid = f"guid-c-{puzzle_id}"
        self.rng = np.random.RandomState(seed)
        self.win_levels = 2
        self.levels_completed = 0
        self.state = GameState.NOT_FINISHED
        self.size = 14
        self.num_gems = self.rng.randint(2, 4)
        self.gem_color = 8
        self._init_level()

    def _init_level(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        self.grid[0, :] = 5; self.grid[-1, :] = 5; self.grid[:, 0] = 5; self.grid[:, -1] = 5
        self.player_r, self.player_c = self.size // 2, self.size // 2
        self.grid[self.player_r, self.player_c] = 1
        self.gems = set()
        while len(self.gems) < self.num_gems:
            gr = self.rng.randint(1, self.size - 1)
            gc = self.rng.randint(1, self.size - 1)
            if (gr, gc) != (self.player_r, self.player_c):
                self.gems.add((gr, gc))
                self.grid[gr, gc] = self.gem_color

    def step(self, action: GameAction, x=None, y=None) -> FrameResponse:
        dr, dc = 0, 0
        if action == GameAction.ACTION1: dr = -1
        elif action == GameAction.ACTION2: dr = 1
        elif action == GameAction.ACTION3: dc = -1
        elif action == GameAction.ACTION4: dc = 1

        nr, nc = self.player_r + dr, self.player_c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            if self.grid[nr, nc] != 5:
                self.grid[self.player_r, self.player_c] = 0
                self.player_r, self.player_c = nr, nc
                self.grid[self.player_r, self.player_c] = 1
                if (nr, nc) in self.gems:
                    self.gems.remove((nr, nc))

        if not self.gems:
            self.levels_completed += 1
            if self.levels_completed >= self.win_levels:
                self.state = GameState.WIN
            else:
                self._init_level()

        return FrameResponse(
            game_id=self.game_id,
            guid=self.guid,
            frames=[Frame(self.grid.copy())],
            state=self.state,
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4],
        )


class ProceduralClickEngine:
    """Procedural Interactive Coordinate Click Puzzle Generator."""
    def __init__(self, puzzle_id: int, seed: int):
        self.game_id = f"proc-click-{puzzle_id}"
        self.guid = f"guid-cl-{puzzle_id}"
        self.rng = np.random.RandomState(seed)
        self.win_levels = 2
        self.levels_completed = 0
        self.state = GameState.NOT_FINISHED
        self.size = 16
        self.target_color = int(self.rng.choice([11, 12, 13, 14]))
        self._init_level()

    def _init_level(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        self.target_r = self.rng.randint(2, self.size - 2)
        self.target_c = self.rng.randint(2, self.size - 2)
        self.grid[self.target_r, self.target_c] = self.target_color

    def step(self, action: GameAction, x=None, y=None) -> FrameResponse:
        if action == GameAction.ACTION6 and x is not None and y is not None:
            if x == self.target_c and y == self.target_r:
                self.levels_completed += 1
                if self.levels_completed >= self.win_levels:
                    self.state = GameState.WIN
                else:
                    self._init_level()

        return FrameResponse(
            game_id=self.game_id,
            guid=self.guid,
            frames=[Frame(self.grid.copy())],
            state=self.state,
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            available_actions=[GameAction.ACTION6],
        )


def run_100_puzzle_benchmark():
    print("====================================================================")
    print("  ARC-AGI-3 GRANDMASTER 100-PUZZLE STRESS TEST BENCHMARK SUITE")
    print("====================================================================")

    puzzles = []
    # 25 Mazes
    for i in range(1, 26):
        puzzles.append(("Maze Labyrinth", ProceduralMazeEngine(i, seed=1000 + i), 100))
    # 25 Key-Locks
    for i in range(26, 51):
        puzzles.append(("Key & Gate Lock", ProceduralKeyLockEngine(i, seed=2000 + i), 90))
    # 25 Collectibles
    for i in range(51, 76):
        puzzles.append(("Multi-Collectible", ProceduralCollectibleEngine(i, seed=3000 + i), 80))
    # 25 Click Puzzles
    for i in range(76, 101):
        puzzles.append(("Interactive Click", ProceduralClickEngine(i, seed=4000 + i), 40))

    total_cleared_levels = 0
    total_target_levels = 0
    total_wins = 0
    total_actions = 0
    category_results = {}

    for idx, (cat_name, engine, max_acts) in enumerate(puzzles, 1):
        agent = MyAgent(game_id=engine.game_id)
        resp = engine.step(GameAction.RESET) if hasattr(engine, "_init_level") else engine._make_resp()
        history = [resp]
        actions_taken = 0

        while not resp.state.is_terminal and actions_taken < max_acts:
            act = agent.choose_action(history, resp)
            actions_taken += 1
            x, y = None, None
            if hasattr(act, "data") and isinstance(act.data, dict):
                x = act.data.get("x")
                y = act.data.get("y")
            resp = engine.step(act, x=x, y=y)
            history.append(resp)

        is_win = (resp.state == GameState.WIN)
        if is_win:
            total_wins += 1
        total_cleared_levels += resp.levels_completed
        total_target_levels += engine.win_levels
        total_actions += actions_taken

        if cat_name not in category_results:
            category_results[cat_name] = {"won": 0, "total": 0, "levels_won": 0, "levels_total": 0}
        category_results[cat_name]["total"] += 1
        category_results[cat_name]["levels_total"] += engine.win_levels
        category_results[cat_name]["levels_won"] += resp.levels_completed
        if is_win:
            category_results[cat_name]["won"] += 1

        if idx % 10 == 0 or idx == 100:
            pct = (total_cleared_levels / max(1, total_target_levels)) * 100.0
            print(f"  [Progress {idx:3d}/100] Puzzles Solved: {total_wins}/{idx} | Levels Solved: {total_cleared_levels}/{total_target_levels} ({pct:.1f}%)")

    print("\n====================================================================")
    print("                100-PUZZLE BENCHMARK CATEGORY BREAKDOWN")
    print("====================================================================")
    for cat, stats in category_results.items():
        cat_pct = (stats["levels_won"] / max(1, stats["levels_total"])) * 100.0
        print(f"  {cat:26s}: Won {stats['won']:2d}/{stats['total']:2d} Games | Levels: {stats['levels_won']:2d}/{stats['levels_total']:2d} ({cat_pct:.1f}%)")

    overall_accuracy = (total_cleared_levels / max(1, total_target_levels)) * 100.0
    print("\n====================================================================")
    print("                     FINAL EVALUATION METRICS")
    print("====================================================================")
    print(f"  Total Puzzles Evaluated  : 100")
    print(f"  Full 100% Puzzle Wins    : {total_wins}/100 ({total_wins:.1f}%)")
    print(f"  Total Levels Solved      : {total_cleared_levels}/{total_target_levels} ({overall_accuracy:.1f}%)")
    print(f"  Average Actions Per Game : {total_actions / 100.0:.1f}")
    print(f"  Grandmaster Score Index  : {total_cleared_levels / max(1, total_target_levels):.4f}")
    print("====================================================================\n")


if __name__ == "__main__":
    run_100_puzzle_benchmark()
