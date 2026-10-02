"""arc_agi3_kit - a small, dependency-light SDK + agent framework for ARC-AGI-3.

Built against the official REST spec: https://docs.arcprize.org/arc3v1.yaml
"""

from .models import Frame, FrameResponse, GameAction, GameInfo, GameState
from .client import ArcClient, ArcApiError
from .agent import Agent, Runner, RunResult
from .world_model import WorldModel

__all__ = [
    "Frame",
    "FrameResponse",
    "GameAction",
    "GameInfo",
    "GameState",
    "ArcClient",
    "ArcApiError",
    "Agent",
    "Runner",
    "RunResult",
    "WorldModel",
]

__version__ = "0.1.0"
