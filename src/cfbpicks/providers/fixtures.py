"""Offline sample data.

Lets the engine be demonstrated, developed and tested end to end with no
API keys and no network. The fixtures are a synthetic week — realistic in
shape, not real games.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..models import Game, MarketQuote, Rating
from ..util.teams import canonical_team, game_key
from .base import Provider

#: Sample data shipped with the package.
PACKAGE_FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


class FixturesProvider(Provider):
    name = "fixtures"
    description = "Bundled offline sample data (no key, no network)"
    requires_key = False
    provides_games = True
    provides_ratings = True
    provides_odds = True

    @property
    def dir(self) -> Path:
        """Prefer a project-local fixtures dir, else the packaged copy.

        The bundled sample data ships inside the package so ``cfbpicks
        demo`` works from any working directory, including an installed
        wheel with no repository checked out.
        """
        configured = self.config.path(self.config.fixtures_dir)
        if configured.exists():
            return configured
        return PACKAGE_FIXTURES

    def _load(self, filename: str) -> Any:
        path = self.dir / filename
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def fetch_games(self, season: int, week: Optional[int] = None, **kwargs) -> list[Game]:
        out = []
        for row in self._load("games.json"):
            if row.get("season") != season:
                continue
            if week is not None and row.get("week") != week:
                continue
            home = canonical_team(row["home_team"])
            away = canonical_team(row["away_team"])
            out.append(
                Game(
                    game_id=game_key(row["season"], row["week"], home, away),
                    season=row["season"], week=row["week"],
                    kickoff=_parse(row.get("kickoff")),
                    home_team=home, away_team=away,
                    neutral_site=bool(row.get("neutral_site", False)),
                    conference_game=bool(row.get("conference_game", False)),
                    venue=row.get("venue"),
                    home_score=row.get("home_score"), away_score=row.get("away_score"),
                    source=self.name,
                )
            )
        return out

    def fetch_ratings(self, season: int, week: Optional[int] = None, **kwargs) -> list[Rating]:
        return [
            Rating(
                source=row["source"], season=row["season"], week=row.get("week", 0),
                team=canonical_team(row["team"]), rating=float(row["rating"]),
                offense=row.get("offense"), defense=row.get("defense"),
            )
            for row in self._load("ratings.json")
            if row.get("season") == season
        ]

    def fetch_odds(self, season: int, week: Optional[int] = None, **kwargs) -> list[MarketQuote]:
        fetched = datetime.now(timezone.utc)
        out = []
        for row in self._load("odds.json"):
            if row.get("season") != season:
                continue
            if week is not None and row.get("week") != week:
                continue
            home = canonical_team(row["home_team"])
            away = canonical_team(row["away_team"])
            gid = game_key(row["season"], row["week"], home, away)
            for book, markets in row["books"].items():
                for market, sides in markets.items():
                    for side, quote in sides.items():
                        out.append(
                            MarketQuote(
                                game_id=gid, book=book, market=market, side=side,
                                line=quote.get("line"), price=float(quote["price"]),
                                fetched_at=fetched, source=self.name,
                            )
                        )
        return out


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
