"""Agent interface and the loop that drives one game against the API."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .client import ArcClient
from .models import Frame, FrameResponse, GameAction, GameState
from .world_model import WorldModel

logger = logging.getLogger("arc_agi3_kit")


@dataclass
class RunResult:
    game_id: str
    guid: str
    final_state: GameState
    levels_completed: int
    win_levels: int
    actions_taken: int


class Agent(ABC):
    """Subclass this and implement `choose_action`.

    `choose_action` receives the full frame history for the run so far (in
    case your strategy wants temporal context beyond the single latest
    frame) plus the shared `WorldModel`, and must return a `(GameAction,
    extra)` pair where `extra` is `{"x": int, "y": int}` for ACTION6 and
    `{}` otherwise.
    """

    @abstractmethod
    def choose_action(
        self,
        history: list[FrameResponse],
        world_model: WorldModel,
    ) -> tuple[GameAction, dict]:
        raise NotImplementedError

    def on_episode_start(self, game_id: str) -> None:
        """Optional hook, e.g. to reset per-game internal state."""

    def on_episode_end(self, result: RunResult) -> None:
        """Optional hook, e.g. to log/persist learned state."""


class Runner:
    """Drives `Agent` against `ArcClient` for one or more games."""

    def __init__(
        self,
        client: ArcClient,
        agent: Agent,
        max_actions_per_game: int = 500,
        share_world_model_across_games: bool = False,
    ):
        self.client = client
        self.agent = agent
        self.max_actions_per_game = max_actions_per_game
        self._shared_world_model: WorldModel | None = (
            WorldModel() if share_world_model_across_games else None
        )

    def play_game(self, game_id: str, card_id: str) -> RunResult:
        game_id = self.client.resolve_game_id(game_id)
        world_model = self._shared_world_model or WorldModel()
        self.agent.on_episode_start(game_id)

        response = self.client.reset(game_id, card_id)
        active_game_id = response.game_id
        history: list[FrameResponse] = [response]
        actions_taken = 0

        logger.info("Starting %s (guid=%s)", active_game_id, response.guid)

        while response.state != GameState.WIN and actions_taken < self.max_actions_per_game:
            action, extra = self.agent.choose_action(history, world_model)
            before_frame: Frame = response.latest
            levels_before = response.levels_completed

            # Level reset on death or explicit reset
            if response.state == GameState.GAME_OVER or action is GameAction.RESET:
                response = self.client.reset(active_game_id, card_id, guid=response.guid)
            else:
                response = self.client.step(
                    action,
                    active_game_id,
                    response.guid,
                    x=extra.get("x"),
                    y=extra.get("y"),
                )
                world_model.record_transition(
                    before=before_frame,
                    action=action,
                    after=response.latest,
                    levels_completed_before=levels_before,
                    levels_completed_after=response.levels_completed,
                    x=extra.get("x"),
                    y=extra.get("y"),
                )

            history.append(response)
            actions_taken += 1

            logger.debug(
                "step=%d action=%s state=%s levels=%d/%d",
                actions_taken,
                action.name,
                response.state.value,
                response.levels_completed,
                response.win_levels,
            )

        result = RunResult(
            game_id=game_id,
            guid=response.guid,
            final_state=response.state,
            levels_completed=response.levels_completed,
            win_levels=response.win_levels,
            actions_taken=actions_taken,
        )
        self.agent.on_episode_end(result)
        logger.info(
            "Finished %s: state=%s levels=%d/%d actions=%d",
            game_id,
            result.final_state.value,
            result.levels_completed,
            result.win_levels,
            result.actions_taken,
        )
        return result

    def play_games(self, game_ids: list[str], card_id: str) -> list[RunResult]:
        return [self.play_game(gid, card_id) for gid in game_ids]
