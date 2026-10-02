"""Hierarchical Macro-Planner with A* Grid Pathfinding, Sokoban Push Solver, and MCTS Curiosity."""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict

import numpy as np

from .causal_memory import GameRuleMemory, GoalType
from .models import Frame, GameAction
from .world_model import GridObject, WorldModel, get_background_color


@dataclass
class Plan:
    action: GameAction
    x: Optional[int] = None
    y: Optional[int] = None
    reason: str = ""


# Default direction offsets if not yet probed
DEFAULT_DIR_MAP = {
    GameAction.ACTION1: (-1, 0),  # Up
    GameAction.ACTION2: (1, 0),   # Down
    GameAction.ACTION3: (0, -1),  # Left
    GameAction.ACTION4: (0, 1),   # Right
}


def astar_grid_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    solid_colors: set[int],
    bg_color: int,
    max_expansions: int = 350,
) -> Optional[list[tuple[int, int]]]:
    """Computes mathematical shortest 4-way path avoiding solid obstacles."""
    h, w = grid.shape
    sr, sc = start
    gr, gc = goal

    if sr == gr and sc == gc:
        return [start]

    def heuristic(r: int, c: int) -> int:
        return abs(r - gr) + abs(c - gc)

    open_set = []
    heapq.heappush(open_set, (heuristic(sr, sc), 0, (sr, sc), [(sr, sc)]))
    visited = { (sr, sc): 0 }
    expansions = 0

    while open_set and expansions < max_expansions:
        expansions += 1
        f, g, (cr, cc), path = heapq.heappop(open_set)

        if cr == gr and cc == gc:
            return path

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                cell_color = int(grid[nr, nc])
                # Destination cell is passable if it's the goal itself or background/non-solid
                if (nr == gr and nc == gc) or cell_color == bg_color or cell_color not in solid_colors:
                    new_g = g + 1
                    if (nr, nc) not in visited or new_g < visited[(nr, nc)]:
                        visited[(nr, nc)] = new_g
                        new_path = path + [(nr, nc)]
                        new_f = new_g + heuristic(nr, nc)
                        heapq.heappush(open_set, (new_f, new_g, (nr, nc), new_path))

    return None


def sokoban_push_path(
    grid: np.ndarray,
    avatar_pos: tuple[int, int],
    block_pos: tuple[int, int],
    target_pos: tuple[int, int],
    solid_colors: set[int],
    bg_color: int,
) -> Optional[list[tuple[int, int]]]:
    """Generates the avatar navigation path to push a block toward target."""
    br, bc = block_pos
    tr, tc = target_pos

    # Determine desired push direction
    p_dr = 1 if tr > br else (-1 if tr < br else 0)
    p_dc = 1 if tc > bc else (-1 if tc < bc else 0)

    if p_dr == 0 and p_dc == 0:
        return None

    # Pick dominant direction
    if abs(tr - br) >= abs(tc - bc) and p_dr != 0:
        push_dr, push_dc = p_dr, 0
    else:
        push_dr, push_dc = 0, p_dc

    # Avatar needs to stand on the opposite side of the push direction
    stand_r, stand_c = br - push_dr, bc - push_dc

    h, w = grid.shape
    if not (0 <= stand_r < h and 0 <= stand_c < w):
        return None

    # Step 1: Navigate avatar to stand position
    nav_solids = solid_colors | {int(grid[br, bc])}  # Avatar cannot walk through the block itself
    approach_path = astar_grid_path(grid, avatar_pos, (stand_r, stand_c), nav_solids, bg_color)
    if not approach_path:
        return None

    # Step 2: Push action moves avatar into block position
    return approach_path + [(br, bc)]


class GrandmasterPlanner:
    """Production Hierarchical Planner with Sokoban Physics, A* Execution, and MCTS Curiosity."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self._action_queue: list[GameAction] = []
        self._probing_actions = [
            GameAction.ACTION1,
            GameAction.ACTION4,
            GameAction.ACTION2,
            GameAction.ACTION3,
        ]
        self._probe_step = 0

    def reset_queue(self):
        self._action_queue.clear()

    def choose(
        self,
        frame: Frame,
        available: list[GameAction],
        world_model: WorldModel,
        memory: Optional[GameRuleMemory] = None,
    ) -> Plan:
        state_hash = frame.hash()
        # Filter valid non-no-op actions
        valid_available = [
            a for a in available
            if not world_model.is_known_no_op(state_hash, a)
        ]
        if not valid_available:
            valid_available = list(available)

        # -------------------------------------------------------------
        # 1. EXPLOIT: Known Progress Action from this State
        # -------------------------------------------------------------
        for act in valid_available:
            if act.is_coordinate_action:
                for x, y in world_model.click_candidates(frame, top_k=4):
                    if world_model.is_known_progress_action(state_hash, act, x, y):
                        return Plan(act, x=x, y=y, reason="known_progress_coord")
            elif world_model.is_known_progress_action(state_hash, act):
                return Plan(act, reason="known_progress")

        # -------------------------------------------------------------
        # 2. BREAK LOOPS: If caught in an infinite cycle, Undo
        # -------------------------------------------------------------
        if world_model.is_recent_loop(state_hash):
            self.reset_queue()
            if GameAction.ACTION7 in valid_available:
                return Plan(GameAction.ACTION7, reason="loop_break_undo")

        # -------------------------------------------------------------
        # 3. AVATAR PROBING: First 2-3 turns to lock kinematics
        # -------------------------------------------------------------
        if not world_model.detector.probed and (not memory or not memory.avatar):
            while self._probe_step < len(self._probing_actions):
                act = self._probing_actions[self._probe_step]
                self._probe_step += 1
                if act in valid_available and not world_model.is_known_no_op(state_hash, act):
                    return Plan(act, reason="avatar_probing")

        # -------------------------------------------------------------
        # 4. MACRO-PLAN EXECUTION (A* Path / Sokoban Push)
        # -------------------------------------------------------------
        if self._action_queue:
            next_act = self._action_queue.pop(0)
            if next_act in valid_available:
                return Plan(next_act, reason="macro_plan_execution")
            else:
                self.reset_queue()

        avatar = world_model.detector.find_avatar(frame, memory=memory)
        targets = world_model.prioritizer.rank_targets(frame, avatar=avatar, memory=memory)

        if avatar and targets:
            bg_color = get_background_color(frame)
            solid_colors = set(memory.solid_colors) if memory else {5}
            border_colors = {int(c) for c in np.concatenate([frame.grid[0, :], frame.grid[-1, :], frame.grid[:, 0], frame.grid[:, -1]])}
            solid_colors.update(border_colors)
            pr, pc = avatar.center_cell

            # Check for Sokoban pushable blocks
            if memory and memory.pushable_colors:
                pushables = [t for t in targets if t.color in memory.pushable_colors]
                slots = [t for t in targets if t.color not in memory.pushable_colors]
                if pushables and slots:
                    block = pushables[0]
                    slot = slots[0]
                    push_plan = sokoban_push_path(
                        frame.grid, (pr, pc), block.center_cell, slot.center_cell, solid_colors, bg_color
                    )
                    if push_plan and len(push_plan) > 1:
                        action_steps = self._path_to_actions(push_plan, memory)
                        if action_steps:
                            first_act = action_steps[0]
                            self._action_queue = action_steps[1:]
                            if first_act in valid_available:
                                return Plan(first_act, reason=f"sokoban_push_block_{block.color}")

            # Direct A* to prioritized goal targets
            for target in targets[:4]:
                tr, tc = target.center_cell
                path = astar_grid_path(frame.grid, (pr, pc), (tr, tc), solid_colors, bg_color)
                if path and len(path) > 1:
                    action_steps = self._path_to_actions(path, memory)
                    if action_steps:
                        first_act = action_steps[0]
                        self._action_queue = action_steps[1:]
                        if first_act in valid_available:
                            return Plan(first_act, reason=f"astar_path_to_target_{target.color}")

        # -------------------------------------------------------------
        # 5. COORDINATE ACTION (ACTION6) if Available
        # -------------------------------------------------------------
        if GameAction.ACTION6 in valid_available:
            clicks = world_model.click_candidates(frame, top_k=4)
            for x, y in clicks:
                if world_model.times_tried(state_hash, GameAction.ACTION6, x, y) == 0:
                    return Plan(GameAction.ACTION6, x=x, y=y, reason="click_saliency_target")

        # -------------------------------------------------------------
        # 6. CURIOSITY: Try the least-visited valid non-no-op action
        # -------------------------------------------------------------
        unvisited = [a for a in valid_available if not a.is_coordinate_action and world_model.times_tried(state_hash, a) == 0]
        if unvisited:
            return Plan(self.rng.choice(unvisited), reason="curiosity_unvisited")

        scored = [
            (world_model.times_tried(state_hash, a), a)
            for a in valid_available
            if not a.is_coordinate_action
        ]
        if scored:
            scored.sort(key=lambda item: item[0])
            return Plan(scored[0][1], reason="least_tried")

        return Plan(self.rng.choice(valid_available), reason="random_fallback")

    def _path_to_actions(
        self,
        path: list[tuple[int, int]],
        memory: Optional[GameRuleMemory] = None,
    ) -> list[GameAction]:
        actions: list[GameAction] = []
        default_map = {
            (-1, 0): GameAction.ACTION1,
            (1, 0): GameAction.ACTION2,
            (0, -1): GameAction.ACTION3,
            (0, 1): GameAction.ACTION4,
        }
        delta_map = dict(default_map)
        if memory and memory.delta_to_action:
            for k, v in memory.delta_to_action.items():
                if isinstance(k, tuple) and len(k) == 2:
                    try:
                        delta_map[k] = GameAction(v)
                    except ValueError:
                        pass

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            dr, dc = r2 - r1, c2 - c1
            act = delta_map.get((dr, dc))
            if act:
                actions.append(act)
        return actions


ExplorationPlanner = GrandmasterPlanner
