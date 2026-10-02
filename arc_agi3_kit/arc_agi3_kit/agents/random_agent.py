"""Trivial baseline: uniformly random legal action every turn.

Useful as a sanity check that the client/runner plumbing works, and as the
denominator you should always beat -- RHAE scoring means "more efficient
than a human" is the bar, but "better than random" is the first checkpoint.
"""
from __future__ import annotations

import random

from ..agent import Agent
from ..models import FrameResponse, GameAction
from ..world_model import WorldModel


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_action(
        self, history: list[FrameResponse], world_model: WorldModel
    ) -> tuple[GameAction, dict]:
        latest = history[-1]
        candidates = [a for a in latest.available_actions if a != GameAction.RESET] or [GameAction.RESET]
        action = self.rng.choice(candidates)
        extra = {}
        if action.is_coordinate_action:
            h, w = latest.latest.shape
            extra = {"x": self.rng.randrange(w), "y": self.rng.randrange(h)}
        return action, extra
