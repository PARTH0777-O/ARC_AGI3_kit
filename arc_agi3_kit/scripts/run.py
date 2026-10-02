#!/usr/bin/env python3
"""CLI entry point.

Examples:
    export ARC_API_KEY=your_key_here

    # list games available on your account
    python scripts/run.py --list-games

    # run the explorer agent against one game
    python scripts/run.py --game ls20 --agent explorer --max-actions 500

    # run against several games under one scorecard, with tags
    python scripts/run.py --game ls20 ft09 --agent explorer --tags baseline v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi3_kit import ArcClient, Runner
from arc_agi3_kit.agents import ExplorerAgent, RandomAgent

AGENTS = {"random": RandomAgent, "explorer": ExplorerAgent}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run an ARC-AGI-3 agent.")
    parser.add_argument("--game", nargs="+", help="game_id(s) or stable prefixes, e.g. ls20")
    parser.add_argument("--agent", choices=AGENTS.keys(), default="explorer")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--api-key", default=None, help="defaults to $ARC_API_KEY")
    parser.add_argument("--list-games", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    client = ArcClient(api_key=args.api_key)

    if args.list_games:
        for g in client.list_games():
            print(f"{g.game_id}\t{g.title or ''}")
        return

    if not args.game:
        parser.error("--game is required unless --list-games is passed")

    card_id = client.open_scorecard(source_url=args.source_url, tags=args.tags)
    agent = AGENTS[args.agent](seed=args.seed)
    runner = Runner(client, agent, max_actions_per_game=args.max_actions)

    try:
        results = runner.play_games(args.game, card_id)
    finally:
        summary = client.close_scorecard(card_id)
        print(f"\nScorecard {card_id}: overall score = {summary.get('score')}")

    for r in results:
        print(f"{r.game_id}: {r.final_state.value} levels={r.levels_completed}/{r.win_levels} actions={r.actions_taken}")


if __name__ == "__main__":
    main()
