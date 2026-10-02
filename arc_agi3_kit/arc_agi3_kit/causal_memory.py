"""Causal physics learning, multi-hypothesis goal engine, and cross-level memory for ARC-AGI-3."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict, List, Set, Tuple

import numpy as np

from .models import Frame, GameAction, GameState


class InteractionType(str, Enum):
    UNKNOWN = "UNKNOWN"
    SOLID = "SOLID"              # Impassable obstacle / wall
    LETHAL = "LETHAL"            # Causes reset / game over / health loss
    WIN_TRIGGER = "WIN_TRIGGER"  # Causes level completion
    PUSHABLE = "PUSHABLE"        # Displaces by delta when avatar moves into it (Sokoban)
    COLLECTIBLE = "COLLECTIBLE"  # Disappears upon avatar contact
    SWITCH = "SWITCH"            # Triggers state transformation on another entity
    PASSABLE = "PASSABLE"        # Walkable floor / background


class GoalType(str, Enum):
    NAVIGATE_TO_TARGET = "NAVIGATE_TO_TARGET"  # Reach the rare/exit entity
    COLLECT_ALL = "COLLECT_ALL"                # Touch & clear all instances of a collectible color
    PUSH_BLOCK_TO_SLOT = "PUSH_BLOCK_TO_SLOT"  # Push block of color A onto slot of color B
    ACTIVATE_SWITCHES = "ACTIVATE_SWITCHES"    # Step on switches / buttons to unlock path
    INTERACTIVE_CLICK = "INTERACTIVE_CLICK"    # Click (ACTION6) interactive hotspots


@dataclass
class AvatarProfile:
    """Kinematics and visual signature of the player avatar."""
    color: int
    size: int
    shape_signature: tuple[int, int]  # (height, width)
    action_deltas: dict[int, tuple[int, int]] = field(default_factory=dict)  # action -> (dr, dc)
    is_multicell: bool = False

    def is_match(self, color: int, size: int, shape: tuple[int, int]) -> bool:
        return self.color == color and self.size == size and self.shape_signature == shape


@dataclass
class GoalSignature:
    """Verified signature of the winning goal condition."""
    goal_type: GoalType
    target_color: int
    target_size: int = 1
    slot_color: Optional[int] = None
    trigger_action: Optional[GameAction] = None


@dataclass
class GoalHypothesis:
    """A candidate objective with Bayesian confidence score."""
    goal_type: GoalType
    target_color: int
    confidence: float = 1.0
    slot_color: Optional[int] = None


@dataclass
class GameRuleMemory:
    """Persistent physics, kinematics, and goal memory across levels of the same game."""

    game_id: str
    avatar: Optional[AvatarProfile] = None
    verified_goal: Optional[GoalSignature] = None
    hypotheses: list[GoalHypothesis] = field(default_factory=list)
    solid_colors: set[int] = field(default_factory=set)
    lethal_colors: set[int] = field(default_factory=set)
    collectible_colors: set[int] = field(default_factory=set)
    pushable_colors: set[int] = field(default_factory=set)
    switch_colors: set[int] = field(default_factory=set)
    action_to_delta: dict[int, tuple[int, int]] = field(default_factory=dict)  # action -> (dr, dc)
    delta_to_action: dict[tuple[int, int], int] = field(default_factory=dict)  # (dr, dc) -> action
    no_op_actions: set[int] = field(default_factory=set)

    def register_action_movement(self, action: GameAction, dr: int, dc: int) -> None:
        if dr != 0 or dc != 0:
            act_val = int(action)
            self.action_to_delta[act_val] = (dr, dc)
            self.delta_to_action[(dr, dc)] = act_val
            if act_val in self.no_op_actions:
                self.no_op_actions.remove(act_val)

    def record_collision(self, obstacle_color: int, outcome: InteractionType) -> None:
        if outcome == InteractionType.SOLID:
            self.solid_colors.add(obstacle_color)
        elif outcome == InteractionType.LETHAL:
            self.lethal_colors.add(obstacle_color)
        elif outcome == InteractionType.COLLECTIBLE:
            self.collectible_colors.add(obstacle_color)
        elif outcome == InteractionType.PUSHABLE:
            self.pushable_colors.add(obstacle_color)
        elif outcome == InteractionType.SWITCH:
            self.switch_colors.add(obstacle_color)

    def is_passable(self, color: int, background_color: int) -> bool:
        if color == background_color:
            return True
        if color in self.solid_colors or color in self.lethal_colors:
            return False
        if color in self.collectible_colors or (self.verified_goal and color == self.verified_goal.target_color):
            return True
        return True

    def lock_verified_goal(self, goal_type: GoalType, target_color: int, slot_color: Optional[int] = None) -> None:
        self.verified_goal = GoalSignature(
            goal_type=goal_type,
            target_color=target_color,
            slot_color=slot_color,
        )
        # Update hypothesis confidence
        for h in self.hypotheses:
            if h.goal_type == goal_type and h.target_color == target_color:
                h.confidence = 10.0
            else:
                h.confidence *= 0.1


class CausalMemoryBank:
    """Global repository of cross-game and cross-level learned rules."""

    def __init__(self):
        self._memories: dict[str, GameRuleMemory] = {}

    def get_memory(self, game_id: str) -> GameRuleMemory:
        clean_id = game_id.split("-")[0] if "-" in game_id else game_id
        if clean_id not in self._memories:
            self._memories[clean_id] = GameRuleMemory(game_id=clean_id)
        return self._memories[clean_id]
