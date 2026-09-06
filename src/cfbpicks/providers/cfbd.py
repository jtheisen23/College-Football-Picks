"""CollegeFootballData.com provider.

The single richest free source for this project: schedules and results,
SP+/SRS/Elo power ratings, and historical betting lines from several
books. An API key (free at collegefootballdata.com/key) is required and
is read from ``CFBD_API_KEY``.

CFBD renamed its response fields from ``snake_case`` to ``camelCase``
partway through its life, so every field read here accepts both.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from ..models import Game, MarketQuote, Rating
from ..util.teams import canonical_team, game_key
from .base import Provider

#: Books CFBD relays, mapped to the names used elsewhere in the app.
BOOK_ALIASES = {
    "consensus": "consensus",
    "teamrankings": "TeamRankings",
    "numberfire": "numberFire",
    "bovada": "Bovada",
    "draftkings": "DraftKings",
    "espn bet": "ESPN Bet",
    "william hill (new jersey)": "Caesars",
}


def pick(data: dict, *names: str, default: Any = None) -> Any:
    """Read the first present key, tolerating snake_case/camelCase drift."""
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class CfbdProvider(Provider):
    name = "cfbd"
    description = "CollegeFootballData.com — schedules, results, SP+/SRS/Elo, betting lines"
    requires_key = True
    provides_games = True
    provides_ratings = True
    provides_odds = True

    @property
    def base_url(self) -> str:
        return (self.settings.base_url or "https://api.collegefootballdata.com").rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.require_key()}", "Accept": "application/json"}

    def _get(self, path: str, params: dict, *, ttl: Optional[int] = None) -> Any:
        return self.client.get_json(
            f"{self.base_url}{path}",
            params={k: v for k, v in params.items() if v is not None},
            headers=self._headers(),
            ttl=ttl,
        )

    def _ttl_for(self, season: int) -> Optional[int]:
        """Completed seasons never change, so cache them indefinitely."""
        current_year = datetime.now(timezone.utc).year
        return 0 if season < current_year else None

    # -- games -------------------------------------------------------------
    def fetch_games(
        self,
        season: int,
        week: Optional[int] = None,
        *,
        season_type: str = "regular",
        **kwargs,
    ) -> list[Game]:
        payload = self._get(
            "/games",
            {
                "year": season,
                "week": week,
                "seasonType": season_type,
                "division": self.option("division", "fbs"),
            },
            ttl=self._ttl_for(season),
        )
        return [g for g in (self._to_game(row, season) for row in payload or []) if g]

    def _to_game(self, row: dict, season: int) -> Optional[Game]:
        home = canonical_team(str(pick(row, "homeTeam", "home_team", default="")))
        away = canonical_team(str(pick(row, "awayTeam", "away_team", default="")))
        if not home or not away:
            return None
        week = int(pick(row, "week", default=0))
        return Game(
            game_id=game_key(season, week, home, away),
            season=int(pick(row, "season", default=season)),
            week=week,
            kickoff=_parse_dt(pick(row, "startDate", "start_date")),
            home_team=home,
            away_team=away,
            neutral_site=bool(pick(row, "neutralSite", "neutral_site", default=False)),
            conference_game=bool(pick(row, "conferenceGame", "conference_game", default=False)),
            venue=pick(row, "venue"),
            home_score=_as_int(pick(row, "homePoints", "home_points")),
            away_score=_as_int(pick(row, "awayPoints", "away_points")),
            season_type=str(pick(row, "seasonType", "season_type", default="regular")),
            source=self.name,
        )

    # -- ratings -----------------------------------------------------------
    def fetch_ratings(self, season: int, week: Optional[int] = None, **kwargs) -> list[Rating]:
        wanted = kwargs.get("systems") or self.option("ratings", ["sp_plus", "srs", "elo"])
        out: list[Rating] = []
        for system in wanted:
            try:
                out.extend(self._fetch_rating_system(system, season, week))
            except NotImplementedError:
                continue
        return out

    def _fetch_rating_system(self, system: str, season: int, week: Optional[int]) -> list[Rating]:
        ttl = self._ttl_for(season)
        if system in {"sp_plus", "sp", "sp+"}:
            # SP+ is a full-season rating; it has no per-week endpoint.
            rows = self._get("/ratings/sp", {"year": season}, ttl=ttl)
            return [
                Rating(
                    source="cfbd_sp_plus", season=season, week=week or 0,
                    team=canonical_team(str(pick(r, "team", default=""))),
                    rating=float(pick(r, "rating", default=0.0)),
                    offense=_nested(r, "offense", "rating"),
                    defense=_nested(r, "defense", "rating"),
                )
                for r in rows or []
                if pick(r, "team") and pick(r, "rating") is not None
            ]

        if system == "srs":
            rows = self._get("/ratings/srs", {"year": season}, ttl=ttl)
            return [
                Rating(
                    source="cfbd_srs", season=season, week=week or 0,
                    team=canonical_team(str(pick(r, "team", default=""))),
                    rating=float(pick(r, "rating", default=0.0)),
                )
                for r in rows or []
                if pick(r, "team") and pick(r, "rating") is not None
            ]

        if system == "elo":
            rows = self._get("/ratings/elo", {"year": season, "week": week}, ttl=ttl)
            elos = [
                (canonical_team(str(pick(r, "team", default=""))), float(pick(r, "elo", default=0.0)))
                for r in rows or []
                if pick(r, "team") and pick(r, "elo") is not None
            ]
            if not elos:
                return []
            # Elo is an abstract scale; ~25 Elo points is worth 1 point of
            # spread, so centre it and divide to reach points-above-average.
            mean_elo = sum(e for _, e in elos) / len(elos)
            divisor = float(self.option("elo_points_divisor", 25.0))
            return [
                Rating(
                    source="cfbd_elo", season=season, week=week or 0,
                    team=team, rating=(elo - mean_elo) / divisor,
                )
                for team, elo in elos
            ]

        raise NotImplementedError(f"Unknown CFBD rating system: {system}")

    # -- odds --------------------------------------------------------------
    def fetch_odds(
        self,
        season: int,
        week: Optional[int] = None,
        *,
        season_type: str = "regular",
        **kwargs,
    ) -> list[MarketQuote]:
        """Betting lines as relayed by CFBD.

        CFBD reports the *number* but not the juice on spreads and totals,
        so those are recorded at the standard price configured below and
        flagged as such. Moneylines come through with real prices.
        """
        payload = self._get(
            "/lines",
            {"year": season, "week": week, "seasonType": season_type},
            ttl=self._ttl_for(season),
        )
        assumed = float(self.option("assumed_price", -110))
        fetched = datetime.now(timezone.utc)
        quotes: list[MarketQuote] = []

        for row in payload or []:
            home = canonical_team(str(pick(row, "homeTeam", "home_team", default="")))
            away = canonical_team(str(pick(row, "awayTeam", "away_team", default="")))
            if not home or not away:
                continue
            gid = game_key(season, int(pick(row, "week", default=week or 0)), home, away)

            for line in pick(row, "lines", default=[]) or []:
                book = BOOK_ALIASES.get(
                    str(pick(line, "provider", default="unknown")).lower(),
                    str(pick(line, "provider", default="unknown")),
                )
                spread = _as_float(pick(line, "spread"))
                if spread is not None:
                    quotes.append(MarketQuote(gid, book, "spread", "home", spread, assumed, fetched, self.name))
                    quotes.append(MarketQuote(gid, book, "spread", "away", -spread, assumed, fetched, self.name))

                total = _as_float(pick(line, "overUnder", "over_under"))
                if total is not None:
                    quotes.append(MarketQuote(gid, book, "total", "over", total, assumed, fetched, self.name))
                    quotes.append(MarketQuote(gid, book, "total", "under", total, assumed, fetched, self.name))

                home_ml = _as_float(pick(line, "homeMoneyline", "home_moneyline"))
                away_ml = _as_float(pick(line, "awayMoneyline", "away_moneyline"))
                if home_ml and away_ml:
                    quotes.append(MarketQuote(gid, book, "moneyline", "home", None, home_ml, fetched, self.name))
                    quotes.append(MarketQuote(gid, book, "moneyline", "away", None, away_ml, fetched, self.name))

        return quotes

    # -- extras ------------------------------------------------------------
    def fetch_teams(self, season: int) -> list[dict]:
        """FBS team list, useful for validating the alias map."""
        return self._get("/teams/fbs", {"year": season}, ttl=0) or []

    def fetch_talent(self, season: int) -> dict[str, float]:
        """247-style composite talent, a useful early-season prior."""
        rows = self._get("/talent", {"year": season}, ttl=self._ttl_for(season)) or []
        return {
            canonical_team(str(pick(r, "school", "team", default=""))): float(pick(r, "talent", default=0.0))
            for r in rows
            if pick(r, "school", "team")
        }


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _nested(row: dict, *path: str) -> Optional[float]:
    node: Any = row
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return _as_float(node)
