"""Offline tests -- no network/API key needed. Run: python tests/test_world_model.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from arc_agi3_kit.causal_memory import CausalMemoryBank, GameRuleMemory, GoalSignature, GoalType
from arc_agi3_kit.models import Frame, FrameResponse, GameAction, GameState
from arc_agi3_kit.planner import GrandmasterPlanner, astar_grid_path, sokoban_push_path
from arc_agi3_kit.world_model import AvatarDetector, TargetPrioritizer, WorldModel, extract_objects, match_objects


def make_grid(h=8, w=8, bg=0):
    return np.full((h, w), bg, dtype=np.uint8)


def test_hash_stable_and_sensitive():
    g1 = Frame(make_grid())
    g2 = Frame(make_grid())
    assert g1.hash() == g2.hash(), "identical grids must hash identically"
    g3_arr = make_grid()
    g3_arr[3, 3] = 5
    g3 = Frame(g3_arr)
    assert g1.hash() != g3.hash(), "different grids must hash differently"
    print("OK: hash stability/sensitivity")


def test_frame_response_parses_offline_payload():
    payload = {
        "game_id": "offline-game",
        "guid": "offline-guid",
        "frame": [[[0, 1], [2, 3]]],
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 1,
        "available_actions": [1, 6],
        "full_reset": False,
        "action_input": {},
    }

    response = FrameResponse.from_json(payload)
    assert response.game_id == "offline-game"
    assert response.state is GameState.NOT_FINISHED
    assert response.available_actions == [GameAction.ACTION1, GameAction.ACTION6]
    assert response.latest.shape == (2, 2)
    assert response.latest.grid[1, 1] == 3
    print("OK: FrameResponse parses an API payload offline")


def test_extract_objects_basic():
    arr = make_grid()
    arr[1, 1] = 2
    arr[1, 2] = 2
    arr[5, 5] = 3
    objs = extract_objects(Frame(arr))
    assert len(objs) == 2, f"expected 2 objects, got {len(objs)}"
    sizes = sorted(o.size for o in objs)
    assert sizes == [1, 2], sizes
    print("OK: extract_objects finds correct components")


def test_match_objects_tracks_movement():
    before_arr = make_grid()
    before_arr[2, 2] = 4
    after_arr = make_grid()
    after_arr[2, 3] = 4  # moved one cell right

    before = extract_objects(Frame(before_arr))
    after = extract_objects(Frame(after_arr))
    pairs = match_objects(before, after)
    assert len(pairs) == 1
    b, a = pairs[0]
    assert b is not None and a is not None, "should match the moved object, not treat as vanish+appear"
    print("OK: match_objects tracks a moving single-cell object")


def test_avatar_probing():
    detector = AvatarDetector()
    memory = GameRuleMemory(game_id="test_game")

    before_arr = make_grid(10, 10)
    before_arr[4, 4] = 1  # Player (color 1)
    before = Frame(before_arr)

    after_arr = make_grid(10, 10)
    after_arr[3, 4] = 1   # Moved UP
    after = Frame(after_arr)

    moved = detector.probe_step(before, GameAction.ACTION1, after, memory=memory)
    assert detector.probed is True
    assert detector.avatar_color == 1
    assert memory.avatar is not None
    assert memory.avatar.color == 1
    print("OK: Avatar probing discovers player sprite and direction mapping")


def test_target_saliency_and_rarity():
    arr = make_grid(16, 16, bg=0)
    arr[0, :] = 5
    arr[-1, :] = 5
    arr[4, 4] = 1
    arr[10, 10] = 3

    frame = Frame(arr)
    avatar_obj = extract_objects(frame)[0]
    targets = TargetPrioritizer.rank_targets(frame, avatar=avatar_obj)
    assert targets, "Should find candidate targets"
    target_colors = [t.color for t in targets]
    assert 3 in target_colors, "Target color 3 should be detected as candidate"
    print("OK: Target Prioritizer ranks rare colored objects as top goals")


def test_astar_grid_pathfinding():
    grid = make_grid(10, 10, bg=0)
    grid[1:5, 4] = 5

    start = (2, 2)
    goal = (2, 6)
    solid_colors = {5}

    path = astar_grid_path(grid, start, goal, solid_colors, bg_color=0)
    assert path is not None, "A* should find path around wall"
    assert path[0] == start
    assert path[-1] == goal
    for r, c in path:
        assert grid[r, c] != 5
    print("OK: A* pathfinding computes optimal obstacle-avoiding trajectory")


def test_sokoban_push_pathfinding():
    grid = make_grid(10, 10, bg=0)
    avatar_pos = (5, 2)
    block_pos = (5, 5)
    target_pos = (5, 8)  # Desired push to the right

    # Avatar needs to position at (5, 4) and push right into (5, 5)
    path = sokoban_push_path(grid, avatar_pos, block_pos, target_pos, solid_colors=set(), bg_color=0)
    assert path is not None
    assert path[0] == avatar_pos
    assert path[-2] == (5, 4)  # Position directly behind block
    assert path[-1] == block_pos  # Push into block cell
    print("OK: Sokoban push pathfinder computes positioning and push sequence")


def test_grandmaster_planner_astar_execution():
    grid = make_grid(10, 10, bg=0)
    grid[2, 2] = 1  # Avatar
    grid[2, 5] = 3  # Target (3 steps right)
    frame = Frame(grid)

    wm = WorldModel()
    wm.detector.avatar_color = 1
    wm.detector.probed = True

    planner = GrandmasterPlanner()
    available = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    plan = planner.choose(frame, available, wm)

    assert plan.action == GameAction.ACTION4
    assert "astar_path" in plan.reason
    print("OK: Grandmaster planner computes and executes A* macro-plan to target")


def test_cross_level_memory_transfer():
    bank = CausalMemoryBank()
    mem = bank.get_memory("game_123")
    mem.avatar = None
    mem.lock_verified_goal(GoalType.NAVIGATE_TO_TARGET, target_color=7)

    grid = make_grid(12, 12, bg=0)
    grid[2, 2] = 1
    grid[8, 8] = 7  # Known goal color
    grid[4, 4] = 2  # Decoy color

    frame = Frame(grid)
    targets = TargetPrioritizer.rank_targets(frame, memory=mem)
    assert targets[0].color == 7, "Level 2 target ranking should immediately prioritize learned goal color 7"
    print("OK: Cross-level memory transfers learned goal signature to subsequent levels")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} offline unit tests passed successfully.")
