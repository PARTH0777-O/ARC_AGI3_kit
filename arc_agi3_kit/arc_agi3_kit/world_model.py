"""World Model, Multi-Scale Perception Engine, and Saliency for ARC-AGI-3."""
from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set, Dict

import numpy as np

from .models import Frame, GameAction, GameState
from .causal_memory import AvatarProfile, GameRuleMemory, GoalType, GoalHypothesis, InteractionType


@dataclass
class GridObject:
    color: int
    cells: frozenset[tuple[int, int]]  # (row, col)
    is_composite: bool = False

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def centroid(self) -> tuple[float, float]:
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return (sum(rows) / len(rows), sum(cols) / len(cols))

    @property
    def center_cell(self) -> tuple[int, int]:
        r, c = self.centroid
        return (int(round(r)), int(round(c)))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return (min(rows), min(cols), max(rows), max(cols))

    @property
    def shape_dim(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0 + 1, c1 - c0 + 1)


def get_background_color(frame: Frame) -> int:
    """Detects background color by combining border flood-fill and histogram frequency."""
    grid = frame.grid
    h, w = grid.shape
    border_pixels = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    border_counts = np.bincount(border_pixels, minlength=16)
    if border_counts.max() > len(border_pixels) * 0.4:
        return int(np.argmax(border_counts))
    return int(np.argmax(frame.color_histogram()))


def extract_objects(frame: Frame, ignore_color: Optional[int] = None) -> list[GridObject]:
    """4-connected flood fill over same-colour regions."""
    grid = frame.grid
    h, w = grid.shape
    if ignore_color is None:
        ignore_color = get_background_color(frame)

    visited = np.zeros_like(grid, dtype=bool)
    objects: list[GridObject] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == ignore_color:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            objects.append(GridObject(color=color, cells=frozenset(cells)))

    return objects


def match_objects(
    before: list[GridObject], after: list[GridObject], max_distance: float = 8.0
) -> list[tuple[Optional[GridObject], Optional[GridObject]]]:
    """Nearest-centroid matching of same-colour objects across frames."""
    pairs: list[tuple[Optional[GridObject], Optional[GridObject]]] = []
    remaining_after = list(after)

    for b in before:
        best_j, best_d = None, max_distance
        for j, a in enumerate(remaining_after):
            if a.color != b.color:
                continue
            br, bc = b.centroid
            ar, ac = a.centroid
            d = ((br - ar) ** 2 + (bc - ac) ** 2) ** 0.5
            if d <= best_d:
                best_j, best_d = j, d
        if best_j is not None:
            pairs.append((b, remaining_after.pop(best_j)))
        else:
            pairs.append((b, None))

    pairs.extend((None, a) for a in remaining_after)
    return pairs


class AvatarDetector:
    """Probes and isolates the player-controlled sprite in the first few actions."""

    def __init__(self):
        self.probed = False
        self.avatar_color: Optional[int] = None
        self.avatar_shape: Optional[tuple[int, int]] = None
        self.avatar_size: Optional[int] = None

    def probe_step(
        self,
        before: Frame,
        action: GameAction,
        after: Frame,
        memory: Optional[GameRuleMemory] = None,
    ) -> Optional[GridObject]:
        bg = get_background_color(before)
        before_objs = extract_objects(before, ignore_color=bg)
        after_objs = extract_objects(after, ignore_color=bg)

        pairs = match_objects(before_objs, after_objs)
        for b, a in pairs:
            if b is not None and a is not None:
                br, bc = b.centroid
                ar, ac = a.centroid
                dr, dc = int(round(ar - br)), int(round(ac - bc))
                if dr != 0 or dc != 0:
                    self.avatar_color = b.color
                    self.avatar_size = b.size
                    self.avatar_shape = b.shape_dim
                    self.probed = True
                    if memory:
                        memory.avatar = AvatarProfile(
                            color=b.color,
                            size=b.size,
                            shape_signature=b.shape_dim,
                            is_multicell=b.size > 1,
                        )
                        memory.register_action_movement(action, dr, dc)
                    return a
        return None

    def find_avatar(self, frame: Frame, memory: Optional[GameRuleMemory] = None) -> Optional[GridObject]:
        bg = get_background_color(frame)
        border_colors = {int(c) for c in np.concatenate([frame.grid[0, :], frame.grid[-1, :], frame.grid[:, 0], frame.grid[:, -1]])}
        objs = extract_objects(frame, ignore_color=bg)
        if not objs:
            return None

        target_color = self.avatar_color or (memory.avatar.color if memory and memory.avatar else None)
        target_size = self.avatar_size or (memory.avatar.size if memory and memory.avatar else None)

        if target_color is not None:
            matches = [o for o in objs if o.color == target_color]
            if target_size is not None:
                size_matches = [o for o in matches if o.size == target_size]
                if size_matches:
                    return size_matches[0]
            if matches:
                return matches[0]

        # Fallback heuristic: small non-border entity (size 1 to 9)
        candidates = [o for o in objs if 1 <= o.size <= 9 and o.color not in border_colors and o.color != bg]
        if candidates:
            # Sort by compactness and rarity
            hist = frame.color_histogram()
            total_cells = float(frame.grid.size)
            candidates.sort(key=lambda o: (o.size, hist[o.color] / total_cells))
            return candidates[0]

        return None


class TargetPrioritizer:
    """Multi-Hypothesis Information Saliency Engine."""

    @staticmethod
    def rank_targets(
        frame: Frame,
        avatar: Optional[GridObject] = None,
        memory: Optional[GameRuleMemory] = None,
    ) -> list[GridObject]:
        bg = get_background_color(frame)
        objs = extract_objects(frame, ignore_color=bg)
        if not objs:
            return []

        non_avatar_objs = []
        for o in objs:
            if avatar and o.color == avatar.color and o.size == avatar.size:
                continue
            if o.size > 300:  # Giant wall/container
                continue
            non_avatar_objs.append(o)

        if not non_avatar_objs:
            return objs

        hist = frame.color_histogram()
        total_cells = float(frame.grid.size)
        pr, pc = avatar.center_cell if avatar else (frame.shape[0] // 2, frame.shape[1] // 2)

        def score_target(o: GridObject) -> float:
            # 1. Verified Goal from previous levels
            if memory and memory.verified_goal and o.color == memory.verified_goal.target_color:
                return 5000.0 - (abs(o.center_cell[0] - pr) + abs(o.center_cell[1] - pc))

            # 2. Color Rarity
            rarity = 1.0 - (float(hist[o.color]) / total_cells)

            # 3. Compactness
            compactness = 1.0 / (1.0 + o.size)

            # 4. Proximity penalty
            dist = abs(o.center_cell[0] - pr) + abs(o.center_cell[1] - pc)
            dist_penalty = 0.001 * dist

            # 5. Pushable block / Sokoban slots bonus
            sokoban_bonus = 1.5 if (memory and o.color in memory.pushable_colors) else 0.0

            return (rarity * 4.0) + (compactness * 2.5) + sokoban_bonus - dist_penalty

        return sorted(non_avatar_objs, key=score_target, reverse=True)

    @staticmethod
    def identify_sokoban_slots(frame: Frame, bg_color: int) -> list[GridObject]:
        """Identifies target slots/receptacles for pushable blocks."""
        objs = extract_objects(frame, ignore_color=bg_color)
        return [o for o in objs if o.size <= 4]


@dataclass
class WorldModel:
    """Online World Model with Causal Rules, Saliency, and State Transitions."""

    transitions: dict[tuple[str, tuple], str] = field(default_factory=dict)
    visit_counts: dict[tuple[str, tuple], int] = field(default_factory=lambda: defaultdict(int))
    progress_actions: dict[tuple[str, tuple], bool] = field(default_factory=dict)
    no_op_actions: set[tuple[str, tuple]] = field(default_factory=set)
    action_effect_rate: dict[int, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    state_visit_history: deque = field(default_factory=lambda: deque(maxlen=60))
    detector: AvatarDetector = field(default_factory=AvatarDetector)
    prioritizer: TargetPrioritizer = field(default_factory=TargetPrioritizer)

    @staticmethod
    def action_key(action: GameAction, x: Optional[int] = None, y: Optional[int] = None) -> tuple:
        return (int(action), x, y)

    def record_transition(
        self,
        before: Frame,
        action: GameAction,
        after: Frame,
        levels_completed_before: int,
        levels_completed_after: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        memory: Optional[GameRuleMemory] = None,
    ) -> None:
        key = self.action_key(action, x, y)
        before_hash, after_hash = before.hash(), after.hash()

        self.transitions[(before_hash, key)] = after_hash
        self.visit_counts[(before_hash, key)] += 1
        self.state_visit_history.append(before_hash)

        changed = int(after_hash != before_hash)
        stats = self.action_effect_rate[int(action)]
        stats[0] += changed
        stats[1] += 1

        # Track strict no-ops (wall collisions or unhandled clicks)
        if not changed and levels_completed_after == levels_completed_before:
            self.no_op_actions.add((before_hash, key))

        if not self.detector.probed:
            self.detector.probe_step(before, action, after, memory=memory)

        # Detect collision outcome
        if memory and self.detector.probed:
            if not changed and not action.is_coordinate_action:
                dr, dc = memory.action_to_delta.get(int(action), (0, 0))
                avatar = self.detector.find_avatar(before, memory=memory)
                if avatar and (dr != 0 or dc != 0):
                    ar, ac = avatar.center_cell
                    tr, tc = ar + dr, ac + dc
                    if 0 <= tr < before.shape[0] and 0 <= tc < before.shape[1]:
                        obs_color = int(before.grid[tr, tc])
                        bg = get_background_color(before)
                        if obs_color != bg:
                            memory.record_collision(obs_color, InteractionType.SOLID)

        # Progress reward detected
        if levels_completed_after > levels_completed_before:
            self.progress_actions[(before_hash, key)] = True
            if memory:
                avatar = self.detector.find_avatar(before, memory=memory)
                targets = self.prioritizer.rank_targets(before, avatar=avatar, memory=memory)
                if targets:
                    top = targets[0]
                    memory.lock_verified_goal(GoalType.NAVIGATE_TO_TARGET, top.color)

    def is_known_progress_action(self, state_hash: str, action: GameAction, x=None, y=None) -> bool:
        return self.progress_actions.get((state_hash, self.action_key(action, x, y)), False)

    def is_known_no_op(self, state_hash: str, action: GameAction, x=None, y=None) -> bool:
        return (state_hash, self.action_key(action, x, y)) in self.no_op_actions

    def times_tried(self, state_hash: str, action: GameAction, x=None, y=None) -> int:
        return self.visit_counts.get((state_hash, self.action_key(action, x, y)), 0)

    def is_recent_loop(self, state_hash: str, window: int = 8, min_repeats: int = 3) -> bool:
        recent = list(self.state_visit_history)[-window:]
        return recent.count(state_hash) >= min_repeats

    def click_candidates(self, frame: Frame, top_k: int = 8) -> list[tuple[int, int]]:
        avatar = self.detector.find_avatar(frame)
        targets = self.prioritizer.rank_targets(frame, avatar=avatar)
        if not targets:
            h, w = frame.shape
            return [(w // 2, h // 2)]

        candidates = []
        for obj in targets[:top_k]:
            r, c = obj.center_cell
            candidates.append((c, r))  # (x, y) = (col, row)
        return candidates
