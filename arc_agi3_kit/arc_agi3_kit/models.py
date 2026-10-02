"""Typed data structures mirroring the ARC-AGI-3 REST API schema.

Reference: https://docs.arcprize.org/arc3v1.yaml
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional

import numpy as np


class GameAction(IntEnum):
    """The 8 standardized commands every ARC-AGI-3 game accepts."""

    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6  # coordinate action: requires integer (x, y) in [0, 63]
    ACTION7 = 7  # undo -- only meaningful on games that support it

    @property
    def is_coordinate_action(self) -> bool:
        return self is GameAction.ACTION6

    @classmethod
    def simple_pool(cls) -> list["GameAction"]:
        """Every action except RESET and the coordinate action."""
        return [cls.ACTION1, cls.ACTION2, cls.ACTION3, cls.ACTION4, cls.ACTION5, cls.ACTION7]


class GameState(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"

    @property
    def is_terminal(self) -> bool:
        return self in (GameState.WIN, GameState.GAME_OVER)


@dataclass
class Frame:
    """A single 64x64 grid of 4-bit colour indices (0-15)."""

    grid: np.ndarray  # shape (H, W), dtype uint8

    @classmethod
    def from_raw(cls, raw: list[list[int]]) -> "Frame":
        return cls(grid=np.array(raw, dtype=np.uint8))

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def hash(self) -> str:
        """Stable content hash used to detect repeated / looping states."""
        return hashlib.blake2b(self.grid.tobytes(), digest_size=12).hexdigest()

    def diff_mask(self, other: "Frame") -> np.ndarray:
        """Boolean mask of cells that changed, `other` (before) -> `self` (after)."""
        if self.grid.shape != other.grid.shape:
            return np.ones_like(self.grid, dtype=bool)
        return self.grid != other.grid

    def changed_fraction(self, other: "Frame") -> float:
        mask = self.diff_mask(other)
        return float(mask.sum()) / mask.size

    def color_histogram(self) -> np.ndarray:
        """Count of each of the 16 colours present in the grid."""
        return np.bincount(self.grid.ravel(), minlength=16)


@dataclass
class FrameResponse:
    """Snapshot returned after every RESET or ACTION command."""

    game_id: str
    guid: str
    frames: list[Frame]
    state: GameState
    levels_completed: int
    win_levels: int
    available_actions: list[GameAction]
    full_reset: bool = False
    action_input: dict[str, Any] = field(default_factory=dict)

    @property
    def latest(self) -> Frame:
        return self.frames[-1]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FrameResponse":
        return cls(
            game_id=payload["game_id"],
            guid=payload["guid"],
            frames=[Frame.from_raw(f) for f in payload["frame"]],
            state=GameState(payload["state"]),
            levels_completed=payload["levels_completed"],
            win_levels=payload["win_levels"],
            available_actions=[GameAction(a) for a in payload.get("available_actions", [])],
            full_reset=payload.get("full_reset", False),
            action_input=payload.get("action_input", {}),
        )


@dataclass
class GameInfo:
    game_id: str
    title: Optional[str] = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GameInfo":
        return cls(game_id=payload["game_id"], title=payload.get("title"))
