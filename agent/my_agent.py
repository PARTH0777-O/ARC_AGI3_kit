"""Universal Empirical Object-Centric ARC-AGI-3 Agent (v3 hybrid).

Merges proven v1 mechanics with v2 intelligence:
- From v1: Click handling (raster-scan non-zero), border-wall detection, relaxed BFS
- From v2: Persistent kinematics across levels, goal-colour memory, plan caching,
           no-op suppression per (state, action), majority-voting kinematics,
           saliency-ranked targets
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from arc_agi3_kit.models import Frame, FrameResponse, GameAction, GameState
except ImportError:
    try:
        from arcengine import FrameData, GameAction, GameState
    except ImportError:
        class GameAction(IntEnum):
            RESET = 0
            ACTION1 = 1
            ACTION2 = 2
            ACTION3 = 3
            ACTION4 = 4
            ACTION5 = 5
            ACTION6 = 6
            ACTION7 = 7

            @property
            def is_coordinate_action(self) -> bool:
                return self.value == 6

            @classmethod
            def from_id(cls, aid: int) -> "GameAction":
                return cls(aid)

        class GameState(str, Enum):
            NOT_PLAYED = "NOT_PLAYED"
            NOT_FINISHED = "NOT_FINISHED"
            WIN = "WIN"
            GAME_OVER = "GAME_OVER"

            @property
            def is_terminal(self) -> bool:
                return self in (GameState.WIN, GameState.GAME_OVER)

if not hasattr(GameAction, "is_coordinate_action"):
    setattr(GameAction, "is_coordinate_action", property(lambda self: self.value == 6))

if not hasattr(GameAction, "data"):
    setattr(GameAction, "data", None)

try:
    from arc_agi3_kit.agent import Agent
except Exception:
    try:
        from ..agent import Agent
    except Exception:
        class Agent:
            def choose_action(self, *a, **k):
                raise NotImplementedError


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def to_game_action(v: Union[int, str, GameAction]) -> GameAction:
    if isinstance(v, GameAction):
        return v
    if isinstance(v, int):
        try:
            return GameAction(v)
        except Exception:
            return GameAction.ACTION1
    if isinstance(v, str):
        try:
            return GameAction[v]
        except Exception:
            return GameAction.ACTION1
    return GameAction.ACTION1


# ---------------------------------------------------------------- perception

@dataclass
class GridObject:
    color: int
    cells: frozenset

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def center_cell(self) -> Tuple[int, int]:
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return (int(round(sum(rows) / len(rows))), int(round(sum(cols) / len(cols))))


def border_wall_colors(grid: np.ndarray) -> Set[int]:
    """Identifies wall colors from the grid perimeter (v1 proven approach)."""
    h, w = grid.shape
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    counts = np.bincount(border, minlength=16)
    return {int(c) for c, n in enumerate(counts) if n >= (h + w) * 0.3}


def extract_objects(grid: np.ndarray, exclude: Set[int]) -> List[GridObject]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: List[GridObject] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(grid[r, c])
            if col in exclude:
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(GridObject(color=col, cells=frozenset(cells)))
    return out


# ---------------------------------------------------------------- kinematics (v2: majority voting, persistent across levels)

UNIT_DELTAS = {(-1, 0), (1, 0), (0, -1), (0, 1)}


class Kinematics:
    """Learned action -> (dr, dc). Persists across levels; majority voting."""

    def __init__(self):
        self.votes: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
        self.noop: Dict[int, int] = defaultdict(int)

    @property
    def vec(self) -> Dict[int, Tuple[int, int]]:
        out = {}
        for aid, tally in self.votes.items():
            if tally:
                best = max(tally.items(), key=lambda kv: kv[1])
                if best[1] >= 1:
                    out[aid] = best[0]
        return out

    @property
    def confirmed(self) -> Set[int]:
        done = set()
        for aid, tally in self.votes.items():
            if not tally:
                continue
            ranked = sorted(tally.values(), reverse=True)
            if ranked[0] >= 2 and (len(ranked) == 1 or ranked[0] > ranked[1]):
                done.add(aid)
        return done

    def record(self, action: GameAction, prev, curr):
        aid = int(action)
        if prev is None or curr is None:
            return
        delta = (curr[0] - prev[0], curr[1] - prev[1])
        if delta == (0, 0):
            self.noop[aid] += 1
        elif delta in UNIT_DELTAS:
            self.votes[aid][delta] += 1
            self.noop[aid] = 0

    def action_for(self, dr: int, dc: int) -> Optional[GameAction]:
        for aid, v in self.vec.items():
            if v == (dr, dc):
                return to_game_action(aid)
        return None

    def action_for_direction(self, dr: int, dc: int) -> GameAction:
        """Exact match first, then fallback to sign-match, then default mapping."""
        # Exact unit delta match
        exact = self.action_for(dr, dc)
        if exact is not None:
            return exact
        # Sign-based match from learned vectors
        for aid, (vdr, vdc) in self.vec.items():
            if dr != 0 and dc == 0 and vdc == 0:
                if (vdr > 0 and dr > 0) or (vdr < 0 and dr < 0):
                    return to_game_action(aid)
            if dc != 0 and dr == 0 and vdr == 0:
                if (vdc > 0 and dc > 0) or (vdc < 0 and dc < 0):
                    return to_game_action(aid)
        # Default fallback
        if abs(dr) >= abs(dc):
            return GameAction.ACTION1 if dr < 0 else GameAction.ACTION2
        else:
            return GameAction.ACTION3 if dc < 0 else GameAction.ACTION4

    def unprobed(self, available: List[GameAction]) -> List[GameAction]:
        conf = self.confirmed
        known = set(self.vec.values())
        out = []
        for a in available:
            aid = int(a)
            if aid in (0, 6) or aid in conf or self.noop[aid] >= 2:
                continue
            if len(known) >= 4 and aid not in self.votes:
                continue
            out.append(a)
        return out


# ---------------------------------------------------------------- spatial (v1 proven BFS)

class SpatialMap:
    """Per-level walkability learned from actual movement outcomes."""

    def __init__(self):
        self.blocked: Set[Tuple[int, int]] = set()
        self.walked: Set[Tuple[int, int]] = set()

    def reset_level(self):
        self.blocked.clear()
        self.walked.clear()

    def record(self, prev, curr, action: GameAction, kin: Kinematics):
        if prev is None:
            return
        self.walked.add(prev)
        if curr is not None and curr != prev:
            self.walked.add(curr)
            return
        vec = kin.vec.get(int(action))
        if vec:
            self.blocked.add((prev[0] + vec[0], prev[1] + vec[1]))

    def path(self, start, targets: Set[Tuple[int, int]],
             grid: np.ndarray, walls: Set[int]) -> Optional[List[Tuple[int, int]]]:
        """BFS pathfinding: all non-wall cells are traversable (v1 proven relaxed BFS)."""
        if start in targets:
            return [start]
        h, w = grid.shape
        q = deque([start])
        prevmap = {start: None}
        while q:
            cur = q.popleft()
            if cur in targets:
                path = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = prevmap[node]
                return path[::-1]
            cr, cc = cur
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                nb = (nr, nc)
                if not (0 <= nr < h and 0 <= nc < w) or nb in prevmap:
                    continue
                if nb in self.blocked and nb not in targets:
                    continue
                if int(grid[nr, nc]) in walls and nb not in targets and nb not in self.walked:
                    continue
                prevmap[nb] = cur
                q.append(nb)
        return None


# ---------------------------------------------------------------- agent (v3 hybrid)

class UniversalAgentV3(Agent):
    def __init__(self, seed: Optional[int] = None, game_id: Optional[str] = None):
        self.game_id = game_id
        self.rng = random.Random(seed)
        self.kin = Kinematics()         # PERSISTS across levels (v2 idea)
        self.space = SpatialMap()

        self.avatar_color: Optional[int] = None
        self.avatar_pos: Optional[Tuple[int, int]] = None
        self.avatar_votes: Dict[int, float] = defaultdict(float)

        # v2: cross-level goal learning
        self.goal_colors: Dict[int, float] = defaultdict(float)

        self.last_grid: Optional[np.ndarray] = None
        self.last_action: Optional[GameAction] = None
        self.last_pos: Optional[Tuple[int, int]] = None
        self.last_level = 0
        self.last_extra: dict = {}

        # v2: plan caching
        self.plan: List[Tuple[int, int]] = []

        # v2: no-op suppression per state
        self.state_counts: Dict[str, int] = defaultdict(int)
        self.noop_state_actions: Set[Tuple[str, int]] = set()

        # v1: click tracking
        self.clicked_coords: Set[Tuple[int, int]] = set()
        self.step_count = 0
        self.available_actions: List[GameAction] = []

    def on_episode_start(self, game_id: str = ""):
        self.kin = Kinematics()
        self.space.reset_level()
        self.avatar_color = None
        self.avatar_pos = None
        self.avatar_votes.clear()
        self.goal_colors.clear()
        self.last_grid = None
        self.last_action = None
        self.last_pos = None
        self.last_level = 0
        self.plan.clear()
        self.state_counts.clear()
        self.noop_state_actions.clear()
        self.clicked_coords.clear()
        self.step_count = 0

    def on_episode_end(self, result: Any = None):
        pass

    # ---- grid extraction (v1 proven robust)
    def _extract_grid(self, obj: Any) -> np.ndarray:
        if isinstance(obj, np.ndarray):
            return obj.astype(np.uint8)
        if hasattr(obj, "latest") and hasattr(obj.latest, "grid"):
            return np.array(obj.latest.grid, dtype=np.uint8)
        if hasattr(obj, "grid"):
            return np.array(obj.grid, dtype=np.uint8)
        if hasattr(obj, "frames") and obj.frames:
            return np.array(obj.frames[-1].grid, dtype=np.uint8)
        if isinstance(obj, list):
            if obj:
                last = obj[-1]
                if hasattr(last, "grid"):
                    return np.array(last.grid, dtype=np.uint8)
                if hasattr(last, "latest") and hasattr(last.latest, "grid"):
                    return np.array(last.latest.grid, dtype=np.uint8)
                if hasattr(last, "frames") and last.frames:
                    return np.array(last.frames[-1].grid, dtype=np.uint8)
            return np.array(obj, dtype=np.uint8)
        return np.array(obj, dtype=np.uint8)

    # ---- avatar tracking (v2 improved diff-based with continuity)
    def _track_avatar(self, prev: np.ndarray, curr: np.ndarray,
                      action: GameAction) -> Optional[Tuple[int, int]]:
        if prev.shape != curr.shape:
            return self.avatar_pos
        # Once identified, just locate directly
        if self.avatar_color is not None:
            return self._locate(curr)

        diff = np.argwhere(prev != curr)
        if len(diff) == 0 or len(diff) > 64:
            return self._locate(curr)

        walls = border_wall_colors(prev)
        # Find the color that moved in a direction-consistent way
        vanished: Dict[int, list] = {}
        appeared: Dict[int, list] = {}
        for r, c in diff:
            pc, cc = int(prev[r, c]), int(curr[r, c])
            if pc not in walls and pc != 0:
                vanished.setdefault(pc, []).append((int(r), int(c)))
            if cc not in walls and cc != 0:
                appeared.setdefault(cc, []).append((int(r), int(c)))

        best = None
        kin_vec = self.kin.vec
        for col, newcells in appeared.items():
            if col not in vanished:
                continue
            for (nr, nc) in newcells:
                for (orr, occ) in vanished[col]:
                    delta = (nr - orr, nc - occ)
                    if delta not in UNIT_DELTAS:
                        continue
                    known = kin_vec.get(int(action))
                    weight = 2.0 if known == delta else 1.0
                    # Also boost if action ID matches common convention
                    aid = int(action)
                    if (aid == 1 and delta[0] < 0) or (aid == 2 and delta[0] > 0) or \
                       (aid == 3 and delta[1] < 0) or (aid == 4 and delta[1] > 0):
                        weight += 1.0
                    if best is None or weight > best[0]:
                        best = (weight, col, (nr, nc))

        if best is not None:
            _, col, pos = best
            self.avatar_votes[col] += 1.0
            self.avatar_color = col
            self.avatar_pos = pos
            return self.avatar_pos
        return self._locate(curr)

    def _locate(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        if self.avatar_color is not None:
            cells = np.argwhere(grid == self.avatar_color)
            if len(cells):
                if self.avatar_pos is not None:
                    pr, pc = self.avatar_pos
                    best = min(cells, key=lambda x: abs(x[0] - pr) + abs(x[1] - pc))
                else:
                    best = cells[0]
                self.avatar_pos = (int(best[0]), int(best[1]))
                return self.avatar_pos
        walls = border_wall_colors(grid)
        unq, cnt = np.unique(grid, return_counts=True)
        freq = dict(zip(unq.tolist(), cnt.tolist()))
        cands = [int(c) for c in unq if int(c) != 0 and int(c) not in walls]
        cands.sort(key=lambda c: freq[c])
        for c in cands:
            cells = np.argwhere(grid == c)
            if len(cells) == 1:
                self.avatar_pos = (int(cells[0][0]), int(cells[0][1]))
                return self.avatar_pos
        if cands:
            cells = np.argwhere(grid == cands[0])
            self.avatar_pos = (int(cells[0][0]), int(cells[0][1]))
            return self.avatar_pos
        return self.avatar_pos

    # ---- v2: learn what colour triggers level completion
    def _learn_goal(self, prev: np.ndarray):
        import sys
        print(f"[_learn_goal] last_pos={self.last_pos} last_action={self.last_action}", file=sys.stderr)
        if self.last_pos is None:
            print(f"[_learn_goal] SKIP: last_pos is None", file=sys.stderr)
            return
        vec = self.kin.vec.get(int(self.last_action)) if self.last_action else None
        print(f"[_learn_goal] vec={vec} kin.vec={self.kin.vec}", file=sys.stderr)
        if vec:
            tr, tc = self.last_pos[0] + vec[0], self.last_pos[1] + vec[1]
            print(f"[_learn_goal] target=({tr},{tc}) prev.shape={prev.shape}", file=sys.stderr)
            if 0 <= tr < prev.shape[0] and 0 <= tc < prev.shape[1]:
                col = int(prev[tr, tc])
                walls = border_wall_colors(prev)
                print(f"[_learn_goal] color={col} walls={walls}", file=sys.stderr)
                if col != 0 and col not in walls:
                    self.goal_colors[col] += 5.0
                    print(f"[_learn_goal] LEARNED color={col}!", file=sys.stderr)
                    return
                else:
                    print(f"[_learn_goal] REJECTED: col==0 or col in walls", file=sys.stderr)
            else:
                print(f"[_learn_goal] REJECTED: out of bounds", file=sys.stderr)
        if self.last_extra.get("x") is not None:
            x, y = int(self.last_extra["x"]), int(self.last_extra["y"])
            if 0 <= y < prev.shape[0] and 0 <= x < prev.shape[1]:
                self.goal_colors[int(prev[y, x])] += 5.0

    # ---- v2: rank targets by goal-memory + rarity + proximity
    def _rank_targets(self, grid: np.ndarray, objs: List[GridObject],
                      avatar: Tuple[int, int]) -> List[GridObject]:
        h, w = grid.shape
        hist = np.bincount(grid.ravel(), minlength=16)
        total = float(grid.size)

        def score(o: GridObject) -> float:
            s = 0.0
            if o.color in self.goal_colors:
                s += 1000.0 + self.goal_colors[o.color]
            s += (1.0 - hist[o.color] / total) * 40.0      # rarity
            s += 20.0 / (1.0 + o.size)                      # compactness
            if o.size == 1:
                s += 30.0                                    # singleton bonus
            r, c = o.center_cell
            s -= 0.5 * (abs(r - avatar[0]) + abs(c - avatar[1]))  # proximity
            return s

        cand = [o for o in objs
                if avatar not in o.cells
                and o.color != self.avatar_color
                and o.color != 0                             # skip floor
                and o.size <= max(16, (h * w) // 20)]        # skip huge objects
        return sorted(cand, key=score, reverse=True)

    # ---- main entry
    def choose_action(self, frame_or_history: Any, actions_or_world_model: Any = None,
                      *args, **kwargs):
        runner_call = (actions_or_world_model is not None
                       and type(actions_or_world_model).__name__ == "WorldModel")
        target = frame_or_history
        if isinstance(frame_or_history, list) and frame_or_history:
            target = frame_or_history[-1]
        action, extra = self._decide(target, actions_or_world_model, **kwargs)
        return (action, extra) if runner_call else action

    def _decide(self, resp: Any, actions_or_available: Any = None,
                **kwargs) -> Tuple[GameAction, dict]:
        self.step_count += 1
        grid = self._extract_grid(resp)
        h, w = grid.shape

        # Resolve available actions (v1 robust method)
        avail = None
        if isinstance(actions_or_available, (list, tuple, set)):
            avail = actions_or_available
        elif hasattr(resp, "available_actions") and resp.available_actions:
            avail = resp.available_actions
        if avail:
            self.available_actions = [to_game_action(a) for a in avail]
        else:
            self.available_actions = [GameAction(i) for i in (1, 2, 3, 4, 5, 6, 7)]

        simple = [a for a in self.available_actions if int(a) not in (0, 6)]
        supports_coord = any(int(a) == 6 for a in self.available_actions)

        level = getattr(resp, "levels_completed", 0) or 0

        # Learn from previous transition
        if self.last_grid is not None and self.last_action is not None:
            if level > self.last_level:
                self._learn_goal(self.last_grid)
            curr_pos = self._track_avatar(self.last_grid, grid, self.last_action)
            self.kin.record(self.last_action, self.last_pos, curr_pos)
            self.space.record(self.last_pos, curr_pos, self.last_action, self.kin)
            # v2: no-op suppression
            if np.array_equal(self.last_grid, grid):
                sh = hashlib.md5(self.last_grid.tobytes()).hexdigest()
                self.noop_state_actions.add((sh, int(self.last_action)))
                self.plan.clear()
        else:
            self._locate(grid)

        # Level transition: reset spatial, KEEP kinematics + goal colors (v2 idea)
        if level > self.last_level:
            self.space.reset_level()
            self.plan.clear()
            self.clicked_coords.clear()
            self.last_level = level

        walls = border_wall_colors(grid)
        objs = extract_objects(grid, exclude=walls)
        avatar = self._locate(grid) or (h // 2, w // 2)

        state_h = hashlib.md5(grid.tobytes()).hexdigest()
        self.state_counts[state_h] += 1
        is_looping = self.state_counts[state_h] > 3

        def allowed(a: GameAction) -> bool:
            return (state_h, int(a)) not in self.noop_state_actions

        def pack(a: GameAction, ex: Optional[dict] = None):
            ex = ex or {}
            self.last_grid = grid.copy()
            self.last_action = a
            self.last_pos = avatar
            self.last_extra = ex
            try:
                a.data = ex
            except Exception:
                pass
            return a, ex

        # 1. ACTION6 Click Targeting (v1 proven approach: raster-scan non-zero)
        #    Activates when click is the ONLY or primary action type
        if supports_coord and len(simple) == 0:
            non_zeros = np.argwhere(grid != 0)
            for r, c in non_zeros:
                rc = (int(r), int(c))
                if rc not in self.clicked_coords:
                    self.clicked_coords.add(rc)
                    return pack(GameAction(6), {"x": int(c), "y": int(r)})

        # 2. Probe unknown controls early (v2: uses majority voting probing)
        if self.step_count <= 8:
            if self.avatar_color is None and self.step_count <= 4:
                # v1 approach: probe in fixed order to discover avatar
                probe_actions = [GameAction.ACTION2, GameAction.ACTION4,
                                 GameAction.ACTION1, GameAction.ACTION3]
                act = probe_actions[(self.step_count - 1) % len(probe_actions)]
                if act in self.available_actions:
                    return pack(act)
            else:
                # v2 approach: probe unconfirmed actions
                todo = [a for a in self.kin.unprobed(simple) if allowed(a)]
                if todo:
                    return pack(todo[0])

        # 3. Follow a committed plan (v2 idea: plan caching)
        if self.plan and len(self.plan) > 1 and self.plan[0] == avatar:
            nxt = self.plan[1]
            act = self.kin.action_for(nxt[0] - avatar[0], nxt[1] - avatar[1])
            if act is not None and act in self.available_actions and allowed(act):
                self.plan.pop(0)
                return pack(act)
            self.plan.clear()

        # 4. Loop Breaking (v1 proven)
        if is_looping:
            dir_choices = [a for a in simple
                           if allowed(a) and self.kin.noop.get(int(a), 0) == 0]
            if dir_choices:
                return pack(self.rng.choice(dir_choices))
            pool = [a for a in simple if allowed(a)] or simple
            if pool:
                return pack(self.rng.choice(pool))

        # 5. Navigate to best-ranked target (v2: goal-memory + saliency ranking)
        if simple and self.kin.vec:
            for tgt in self._rank_targets(grid, objs, avatar):
                target_cells = set(tgt.cells)
                path = self.space.path(avatar, target_cells, grid, walls)
                if path and len(path) > 1:
                    dr = path[1][0] - avatar[0]
                    dc = path[1][1] - avatar[1]
                    act = self.kin.action_for_direction(dr, dc)
                    if act in self.available_actions and allowed(act):
                        self.plan = path
                        self.plan.pop(0)
                        return pack(act)

        # 6. Click targets by saliency (v2 idea, for games with mixed actions)
        if supports_coord:
            for tgt in self._rank_targets(grid, objs, avatar):
                rc = tgt.center_cell
                if rc not in self.clicked_coords:
                    self.clicked_coords.add(rc)
                    return pack(GameAction(6), {"x": int(rc[1]), "y": int(rc[0])})

        # 7. Explore unvisited reachable floor (v1+v2 hybrid)
        if simple and self.kin.vec:
            unseen = {(r, c) for r in range(h) for c in range(w)
                      if (r, c) not in self.space.walked
                      and (r, c) not in self.space.blocked
                      and int(grid[r, c]) not in walls}
            if unseen:
                path = self.space.path(avatar, unseen, grid, walls)
                if path and len(path) > 1:
                    dr = path[1][0] - avatar[0]
                    dc = path[1][1] - avatar[1]
                    act = self.kin.action_for_direction(dr, dc)
                    if act in self.available_actions and allowed(act):
                        self.plan = path
                        self.plan.pop(0)
                        return pack(act)

        # 8. Minimum-blocked direction fallback (v1+v2)
        pool = [a for a in simple if allowed(a)] or simple
        if pool:
            if is_looping:
                return pack(self.rng.choice(pool))
            return pack(min(pool, key=lambda a: self.kin.noop.get(int(a), 0)))

        if supports_coord:
            return pack(GameAction(6),
                        {"x": self.rng.randrange(w), "y": self.rng.randrange(h)})
        return pack(self.available_actions[0] if self.available_actions else GameAction(1))

    def step(self, observation: Any) -> Any:
        return self.choose_action(observation)


MyAgent = UniversalAgentV3
ExplorerAgent = UniversalAgentV3
