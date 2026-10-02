#!/usr/bin/env python3
"""Interactive Terminal Player & Local Runner for ARC-AGI-3.

Play games manually in your terminal with live colored graphics, or step/auto-run agents.

Usage:
    python scripts/play_local.py --game ls20
    python scripts/play_local.py --game ls20 --offline
    python scripts/play_local.py --list-games
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi3_kit import ArcClient, Runner
from arc_agi3_kit.agents import ExplorerAgent, RandomAgent
from arc_agi3_kit.models import Frame, FrameResponse, GameAction, GameState
from arc_agi3_kit.world_model import WorldModel

# ARC 16-color RGB palette
PALETTE = [
    (16, 16, 16),       # 0: Black / Background
    (30, 119, 180),     # 1: Blue
    (214, 39, 40),      # 2: Red
    (44, 160, 44),      # 3: Green
    (255, 215, 0),      # 4: Yellow
    (128, 128, 128),    # 5: Grey
    (227, 119, 194),    # 6: Magenta/Pink
    (255, 127, 14),     # 7: Orange
    (23, 190, 207),     # 8: Cyan/Teal
    (140, 86, 75),      # 9: Maroon/Brown
    (31, 78, 121),      # 10: Navy/Dark Blue
    (27, 77, 62),       # 11: Dark Green
    (124, 252, 0),      # 12: Lime
    (148, 103, 189),    # 13: Purple
    (255, 187, 120),    # 14: Salmon/Peach
    (245, 245, 245),    # 15: White
]


def rgb_to_ansi_bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def rgb_to_ansi_fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


ANSI_RESET = "\033[0m"


def render_grid(frame: Frame, max_dim: int = 32, show_coords: bool = True) -> str:
    """Render a 2D ARC-AGI grid as ANSI colored blocks."""
    grid = frame.grid
    h, w = grid.shape

    # Downscale for terminal display if larger than max_dim
    step = max(1, max(h, w) // max_dim)
    sub_h = h // step
    sub_w = w // step

    lines = []
    if show_coords:
        header = "    " + "".join(f"{c % 10} " if c % 5 == 0 else ". " for c in range(0, w, step))
        lines.append("\033[90m" + header + ANSI_RESET)

    for r_idx in range(sub_h):
        orig_r = r_idx * step
        row_str = f"\033[90m{orig_r:2d} |{ANSI_RESET}" if show_coords else ""
        for c_idx in range(sub_w):
            orig_c = c_idx * step
            color_idx = int(grid[orig_r, orig_c]) % len(PALETTE)
            r, g, b = PALETTE[color_idx]
            bg = rgb_to_ansi_bg(r, g, b)
            # Display cell as two shaded spaces
            row_str += f"{bg}  {ANSI_RESET}"
        lines.append(row_str)

    return "\n".join(lines)


class LocalMockEngine:
    """Offline game simulator for testing without network/API keys."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.guid = "local-mock-guid"
        self.levels_completed = 0
        self.win_levels = 3
        self.state = GameState.NOT_FINISHED
        self.size = 16
        self.player_x = 2
        self.player_y = 2
        self.goal_x = 13
        self.goal_y = 13
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        self._init_level()

    def _init_level(self):
        self.grid.fill(0)
        # Border
        self.grid[0, :] = 5
        self.grid[-1, :] = 5
        self.grid[:, 0] = 5
        self.grid[:, -1] = 5
        # Goal & Player
        self.player_x, self.player_y = 2, 2
        self.goal_x, self.goal_y = 13 - self.levels_completed, 13
        self.grid[self.goal_y, self.goal_x] = 3  # Green Goal
        self.grid[self.player_y, self.player_x] = 1  # Blue Player

    def reset(self) -> FrameResponse:
        self.levels_completed = 0
        self.state = GameState.NOT_FINISHED
        self._init_level()
        return self._make_response()

    def step(self, action: GameAction, x: Optional[int] = None, y: Optional[int] = None) -> FrameResponse:
        if self.state.is_terminal:
            return self._make_response()

        dx, dy = 0, 0
        if action == GameAction.ACTION1:  # Up
            dy = -1
        elif action == GameAction.ACTION2:  # Down
            dy = 1
        elif action == GameAction.ACTION3:  # Left
            dx = -1
        elif action == GameAction.ACTION4:  # Right
            dx = 1
        elif action == GameAction.ACTION6 and x is not None and y is not None:
            if 0 <= x < self.size and 0 <= y < self.size and self.grid[y, x] == 0:
                self.grid[y, x] = 4  # Yellow obstacle
        elif action == GameAction.RESET:
            self._init_level()

        nx, ny = self.player_x + dx, self.player_y + dy
        if 0 < nx < self.size - 1 and 0 < ny < self.size - 1:
            if self.grid[ny, nx] != 5 and self.grid[ny, nx] != 4:
                self.grid[self.player_y, self.player_x] = 0
                self.player_x, self.player_y = nx, ny
                self.grid[self.player_y, self.player_x] = 1

        if self.player_x == self.goal_x and self.player_y == self.goal_y:
            self.levels_completed += 1
            if self.levels_completed >= self.win_levels:
                self.state = GameState.WIN
            else:
                self._init_level()

        return self._make_response()

    def _make_response(self) -> FrameResponse:
        return FrameResponse(
            game_id=self.game_id,
            guid=self.guid,
            frames=[Frame(self.grid.copy())],
            state=self.state,
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            available_actions=[
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.ACTION5,
                GameAction.ACTION6,
                GameAction.ACTION7,
            ],
        )


def print_banner(resp: FrameResponse, step_num: int, last_action: Optional[str] = None):
    # Clear screen on ANSI terminals
    print("\033[2J\033[H", end="")
    print("=" * 64)
    print(f" ARC-AGI-3 Play Local | Game: \033[1;36m{resp.game_id}\033[0m | Step: \033[1;33m{step_num}\033[0m")
    state_color = "\033[1;32m" if resp.state == GameState.WIN else ("\033[1;31m" if resp.state == GameState.GAME_OVER else "\033[1;37m")
    print(f" State: {state_color}{resp.state.value}\033[0m | Level: \033[1;35m{resp.levels_completed}/{resp.win_levels}\033[0m")
    if last_action:
        print(f" Last Action: \033[1;32m{last_action}\033[0m")
    print("-" * 64)


def print_controls(available: list[GameAction]):
    avail_names = [a.name for a in available]
    print("\n\033[1mControls:\033[0m")
    print("  \033[1;33m1\033[0m or \033[1;33mw/up\033[0m: ACTION1   \033[1;33m2\033[0m or \033[1;33ms/down\033[0m: ACTION2   \033[1;33m3\033[0m or \033[1;33ma/left\033[0m: ACTION3   \033[1;33m4\033[0m or \033[1;33md/right\033[0m: ACTION4")
    print("  \033[1;33m5\033[0m / space: ACTION5       \033[1;33m6 x y\033[0m: ACTION6 (click coord)  \033[1;33m7\033[0m or \033[1;33mu\033[0m: ACTION7 (undo)")
    print("  \033[1;33mr\033[0m: RESET                \033[1;33mauto [N]\033[0m: AI Agent Auto Play     \033[1;33mq\033[0m: Quit")
    print(f"  Available actions: \033[90m{', '.join(avail_names)}\033[0m")


def play_game_interactive(
    game_id: str,
    client: Optional[ArcClient] = None,
    card_id: Optional[str] = None,
    offline: bool = False,
):
    if offline or client is None:
        mock = LocalMockEngine(game_id)
        resp = mock.reset()
    else:
        resolved_id = client.resolve_game_id(game_id)
        resp = client.reset(resolved_id, card_id)

    step_num = 0
    last_act = None
    world_model = WorldModel()
    agent = ExplorerAgent()
    history = [resp]

    while True:
        print_banner(resp, step_num, last_act)
        print(render_grid(resp.latest))
        print_controls(resp.available_actions)

        if resp.state.is_terminal:
            print(f"\n\033[1;32mGame ended with state: {resp.state.value} (levels completed: {resp.levels_completed}/{resp.win_levels})\033[0m")
            cmd = input("\nPress [r] to reset and play again, or [q] to quit: ").strip().lower()
            if cmd == "r":
                step_num = 0
                if offline or client is None:
                    resp = mock.reset()
                else:
                    resp = client.reset(resp.game_id, card_id)
                history = [resp]
                continue
            else:
                break

        try:
            raw_cmd = input("\nAction > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not raw_cmd:
            continue

        cmd = raw_cmd.lower()
        if cmd in ("q", "quit", "exit"):
            break

        action: Optional[GameAction] = None
        extra: dict = {}

        # Directional and shortcut mappings
        if cmd in ("1", "w", "up"):
            action = GameAction.ACTION1
        elif cmd in ("2", "s", "down"):
            action = GameAction.ACTION2
        elif cmd in ("3", "a", "left"):
            action = GameAction.ACTION3
        elif cmd in ("4", "d", "right"):
            action = GameAction.ACTION4
        elif cmd in ("5", "space"):
            action = GameAction.ACTION5
        elif cmd in ("7", "u", "undo"):
            action = GameAction.ACTION7
        elif cmd in ("0", "r", "reset"):
            action = GameAction.RESET
        elif cmd.startswith("6") or cmd.startswith("c"):
            parts = cmd.split()
            if len(parts) >= 3:
                try:
                    x, y = int(parts[1]), int(parts[2])
                    action = GameAction.ACTION6
                    extra = {"x": x, "y": y}
                except ValueError:
                    print("Invalid coordinates! Usage: 6 <x> <y>")
                    time.sleep(1)
                    continue
            else:
                print("Missing coordinates! Usage: 6 <x> <y>")
                time.sleep(1)
                continue
        elif cmd.startswith("auto") or cmd.startswith("agent"):
            # Auto play with agent
            tokens = cmd.split()
            count = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 20
            print(f"Running agent for up to {count} steps...")
            for _ in range(count):
                if resp.state.is_terminal:
                    break
                act, act_extra = agent.choose_action(history, world_model)
                before_frame = resp.latest
                levels_before = resp.levels_completed
                if act == GameAction.RESET:
                    resp = mock.reset() if (offline or client is None) else client.reset(resp.game_id, card_id, guid=resp.guid)
                else:
                    if offline or client is None:
                        resp = mock.step(act, **act_extra)
                    else:
                        resp = client.step(act, resp.game_id, resp.guid, **act_extra)
                    world_model.record_transition(
                        before=before_frame,
                        action=act,
                        after=resp.latest,
                        levels_completed_before=levels_before,
                        levels_completed_after=resp.levels_completed,
                        **act_extra,
                    )
                step_num += 1
                last_act = f"Agent {act.name} {act_extra if act_extra else ''}"
                history.append(resp)
                print_banner(resp, step_num, last_act)
                print(render_grid(resp.latest))
                time.sleep(0.1)
            continue
        else:
            print(f"Unknown command: {raw_cmd}")
            time.sleep(0.8)
            continue

        # Execute single action
        before_frame = resp.latest
        levels_before = resp.levels_completed
        if action == GameAction.RESET:
            resp = mock.reset() if (offline or client is None) else client.reset(resp.game_id, card_id, guid=resp.guid)
            last_act = "RESET"
        else:
            if offline or client is None:
                resp = mock.step(action, **extra)
            else:
                resp = client.step(action, resp.game_id, resp.guid, **extra)
            world_model.record_transition(
                before=before_frame,
                action=action,
                after=resp.latest,
                levels_completed_before=levels_before,
                levels_completed_after=resp.levels_completed,
                **extra,
            )
            last_act = f"{action.name} {extra if extra else ''}"

        step_num += 1
        history.append(resp)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Play ARC-AGI-3 games interactively or offline.")
    parser.add_argument("--game", default="ls20", help="Game ID or prefix (default: ls20)")
    parser.add_argument("--offline", action="store_true", help="Run local offline mock environment")
    parser.add_argument("--list-games", action="store_true", help="List available games online")
    parser.add_argument("--api-key", default=None, help="ARC API Key (defaults to $ARC_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ARC_API_KEY")

    if args.list_games:
        if not api_key:
            print("Error: ARC_API_KEY required for --list-games. Provide --api-key or set ARC_API_KEY in .env")
            sys.exit(1)
        client = ArcClient(api_key=api_key)
        for g in client.list_games():
            print(f"{g.game_id}\t{g.title or ''}")
        return

    if args.offline or not api_key:
        print(f"Starting Local/Offline player for {args.game}...")
        play_game_interactive(args.game, offline=True)
    else:
        client = ArcClient(api_key=api_key)
        card_id = client.open_scorecard(tags=["interactive", "play-local"])
        try:
            play_game_interactive(args.game, client=client, card_id=card_id, offline=False)
        finally:
            try:
                summary = client.close_scorecard(card_id)
                print(f"\nClosed scorecard {card_id}: score = {summary.get('score')}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
