"""The Odds API provider — live prices across many sportsbooks.

CFBD tells you what the number is; this tells you what you can actually
bet it at, and where. Both halves matter: taking -3 -105 instead of
-3.5 -115 is worth more over a season than most model improvements.

The free tier is roughly 500 requests a month, so responses are cached
and the remaining quota is surfaced after every call.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ..models import Game, MarketQuote
from ..util.teams import canonical_team
from .base import Provider

MARKET_KEYS = {"spread": "spreads", "total": "totals", "moneyline": "h2h"}


class OddsApiProvider(Provider):
    name = "odds_api"
    description = "The Odds API — live spreads, totals and moneylines across US books"
    requires_key = True
    provides_odds = True

    @property
    def base_url(self) -> str:
        return (self.settings.base_url or "https://api.the-odds-api.com/v4").rstrip("/")

    def fetch_odds(
        self,
        season: int,
        week: Optional[int] = None,
        *,
        games: Optional[list[Game]] = None,
        markets: Optional[list[str]] = None,
        **kwargs,
    ) -> list[MarketQuote]:
        """Pull the current board and map it onto known games.

        The Odds API has no concept of a college football *week*, so
        events are matched to scheduled games by team names and kickoff
        proximity. Unmatched events are skipped rather than guessed at.
        """
        sport = self.option("sport", "americanfootball_ncaaf")
        wanted = markets or self.config.betting.markets
        market_param = ",".join(MARKET_KEYS[m] for m in wanted if m in MARKET_KEYS)

        params: dict[str, Any] = {
            "apiKey": self.require_key(),
            "regions": self.option("regions", "us"),
            "markets": market_param,
            "oddsFormat": self.option("odds_format", "american"),
            "dateFormat": "iso",
        }
        books = self.option("books") or []
        if books:
            params["bookmakers"] = ",".join(books)

        payload = self.client.get_json(
            f"{self.base_url}/sports/{sport}/odds", params=params, ttl=self.config.cache_ttl_seconds
        )
        self.quota_remaining = self.client.quota_remaining

        index = _build_game_index(games or [])
        fetched = datetime.now(timezone.utc)
        quotes: list[MarketQuote] = []
        self.unmatched: list[str] = []

        for event in payload or []:
            home = canonical_team(str(event.get("home_team") or ""))
            away = canonical_team(str(event.get("away_team") or ""))
            kickoff = _parse_dt(event.get("commence_time"))
            game = _match_game(index, home, away, kickoff)
            if game is None:
                self.unmatched.append(f"{away} @ {home}")
                continue

            for bookmaker in event.get("bookmakers") or []:
                book = bookmaker.get("title") or bookmaker.get("key") or "unknown"
                for market in bookmaker.get("markets") or []:
                    quotes.extend(
                        self._market_quotes(game.game_id, book, market, home, away, fetched)
                    )
        return quotes

    def _market_quotes(
        self, game_id: str, book: str, market: dict, home: str, away: str, fetched: datetime
    ) -> list[MarketQuote]:
        key = market.get("key")
        out: list[MarketQuote] = []

        for outcome in market.get("outcomes") or []:
            price = _as_float(outcome.get("price"))
            if price is None:
                continue
            point = _as_float(outcome.get("point"))
            name = canonical_team(str(outcome.get("name") or ""))
            raw_name = str(outcome.get("name") or "").strip().lower()

            if key == "spreads":
                # The feed quotes each team's own spread; store both, with
                # the home number kept in home-perspective form.
                if name == home:
                    out.append(MarketQuote(game_id, book, "spread", "home", point, price, fetched, self.name))
                elif name == away:
                    out.append(MarketQuote(game_id, book, "spread", "away", point, price, fetched, self.name))
            elif key == "totals":
                if raw_name == "over":
                    out.append(MarketQuote(game_id, book, "total", "over", point, price, fetched, self.name))
                elif raw_name == "under":
                    out.append(MarketQuote(game_id, book, "total", "under", point, price, fetched, self.name))
            elif key == "h2h":
                if name == home:
                    out.append(MarketQuote(game_id, book, "moneyline", "home", None, price, fetched, self.name))
                elif name == away:
                    out.append(MarketQuote(game_id, book, "moneyline", "away", None, price, fetched, self.name))
        return out

    def fetch_scores(self, days_from: int = 3) -> list[dict]:
        """Recent final scores, handy for grading when CFBD lags."""
        sport = self.option("sport", "americanfootball_ncaaf")
        return self.client.get_json(
            f"{self.base_url}/sports/{sport}/scores",
            params={"apiKey": self.require_key(), "daysFrom": days_from},
            ttl=600,
        ) or []


def _build_game_index(games: list[Game]) -> dict[tuple[str, str], list[Game]]:
    index: dict[tuple[str, str], list[Game]] = {}
    for game in games:
        index.setdefault((game.home_team, game.away_team), []).append(game)
    return index


def _match_game(
    index: dict[tuple[str, str], list[Game]],
    home: str,
    away: str,
    kickoff: Optional[datetime],
) -> Optional[Game]:
    """Match an odds event to a scheduled game.

    Neutral-site games are the reason for the reversed lookup: the two
    feeds disagree about which side is nominally home often enough that
    ignoring it would drop real games.
    """
    candidates = index.get((home, away)) or index.get((away, home)) or []
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if kickoff is None:
        return candidates[0]

    def distance(game: Game) -> timedelta:
        if game.kickoff is None:
            return timedelta(days=999)
        return abs(game.kickoff - kickoff)

    best = min(candidates, key=distance)
    return best if distance(best) < timedelta(days=3) else None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
