"""Thin, dependency-light HTTP client for the ARC-AGI-3 REST API.

Handles auth, session-affinity cookies, retries, and JSON <-> dataclass
translation. API reference: https://docs.arcprize.org/rest_overview
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests

from .models import FrameResponse, GameAction, GameInfo

DEFAULT_ROOT_URL = "https://three.arcprize.org"


class ArcApiError(RuntimeError):
    """Raised for non-2xx responses from the ARC-AGI-3 API."""

    def __init__(self, status_code: int, message: str, payload: Any = None):
        super().__init__(f"[{status_code}] {message}: {payload}")
        self.status_code = status_code
        self.payload = payload


class ArcClient:
    """Talks to the ARC-AGI-3 REST API.

    A `requests.Session` is used deliberately (not one-off `requests.post`
    calls) because the API relies on session-affinity cookies (AWSALB*) to
    route every ACTION for a given `guid` back to the same backend instance.
    Losing those cookies mid-run silently desyncs the agent from the real
    game state, so we let `requests` manage them automatically via the
    session's cookie jar.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        root_url: str = DEFAULT_ROOT_URL,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("ARC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Pass api_key=... or set the ARC_API_KEY "
                "environment variable (get one from the ARC-AGI-3 web console: "
                "https://docs.arcprize.org/api-keys)."
            )
        self.root_url = root_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key, "Content-Type": "application/json"})

    # ------------------------------------------------------------------ #
    # low-level request helper
    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> Any:
        url = f"{self.root_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(method, url, json=json_body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(self.backoff_seconds * attempt)
                continue

            if resp.status_code == 429:  # rate limited; see docs.arcprize.org/rate_limits
                wait = float(resp.headers.get("Retry-After", self.backoff_seconds * attempt))
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except ValueError:
                    detail = resp.text
                raise ArcApiError(resp.status_code, f"{method} {path} failed", detail)

            content_type = resp.headers.get("content-type", "")
            if resp.text.strip() == "" or content_type.startswith("text/plain"):
                return {"raw": resp.text}
            return resp.json()

        raise ArcApiError(0, f"{method} {path} failed after {self.max_retries} retries", last_error)

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    def health(self) -> str:
        return self._request("GET", "/api/healthcheck")["raw"]

    def list_games(self) -> list[GameInfo]:
        return [GameInfo.from_json(g) for g in self._request("GET", "/api/games")]

    def get_game(self, game_id: str) -> GameInfo:
        return GameInfo.from_json(self._request("GET", f"/api/games/{game_id}"))

    def resolve_game_id(self, game_id_or_prefix: str) -> str:
        """Resolve a full game ID or prefix (e.g. 'ls20') to the matching game ID."""
        try:
            games = self.list_games()
            for g in games:
                if g.game_id == game_id_or_prefix:
                    return g.game_id
            prefix_matches = [
                g.game_id
                for g in games
                if g.game_id.startswith(game_id_or_prefix)
                or (g.title and g.title.lower() == game_id_or_prefix.lower())
            ]
            if len(prefix_matches) == 1:
                return prefix_matches[0]
            elif len(prefix_matches) > 1:
                raise ValueError(f"Ambiguous game prefix '{game_id_or_prefix}': matches {prefix_matches}")
        except Exception:
            pass
        return game_id_or_prefix

    # ------------------------------------------------------------------ #
    # scorecards
    # ------------------------------------------------------------------ #
    def open_scorecard(
        self,
        source_url: Optional[str] = None,
        tags: Optional[list[str]] = None,
        opaque: Optional[Any] = None,
        competition_mode: bool = False,
    ) -> str:
        body = {
            k: v
            for k, v in dict(
                source_url=source_url, tags=tags, opaque=opaque, competition_mode=competition_mode
            ).items()
            if v not in (None, False)
        }
        return self._request("POST", "/api/scorecard/open", body)["card_id"]

    def close_scorecard(self, card_id: str) -> dict:
        return self._request("POST", "/api/scorecard/close", {"card_id": card_id})

    def get_scorecard(self, card_id: str, game_id: Optional[str] = None) -> dict:
        path = f"/api/scorecard/{card_id}" if game_id is None else f"/api/scorecard/{card_id}/{game_id}"
        return self._request("GET", path)

    # ------------------------------------------------------------------ #
    # gameplay
    # ------------------------------------------------------------------ #
    def reset(self, game_id: str, card_id: str, guid: Optional[str] = None) -> FrameResponse:
        body: dict[str, Any] = {"game_id": game_id, "card_id": card_id}
        if guid:
            body["guid"] = guid
        return FrameResponse.from_json(self._request("POST", "/api/cmd/RESET", body))

    def step(
        self,
        action: GameAction,
        game_id: str,
        guid: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reasoning: Optional[Any] = None,
    ) -> FrameResponse:
        if action is GameAction.RESET:
            raise ValueError("Use .reset() for RESET -- it needs a card_id, not x/y.")

        body: dict[str, Any] = {"game_id": game_id, "guid": guid}
        if reasoning is not None:
            body["reasoning"] = reasoning
        if action.is_coordinate_action:
            if x is None or y is None:
                raise ValueError("ACTION6 requires integer x, y in [0, 63].")
            body["x"], body["y"] = int(x), int(y)

        return FrameResponse.from_json(self._request("POST", f"/api/cmd/{action.name}", body))
