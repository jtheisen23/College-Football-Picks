"""The Odds API provider — live prices across many sportsbooks.

CFBD tells you what the number is; this tells you what you can actually
bet it at, and where. Both halves matter: taking -3 at -105 instead of
-3.5 at -115 is worth more over a season than most model improvements.

Two things get careful treatment here.

**Quota.** The free tier is about 500 credits a month, and a request
costs one credit *per market per region* — so the default three markets
in one region is 3 credits a call, not 1. A twice-weekly snapshot across
a season is comfortably affordable, but only if nothing burns credits by
accident, so responses are cached, the remaining balance is read back
off every response, and the provider refuses to call at all once the
balance drops below a floor you set.

**Matching.** The Odds API has no concept of a college football week and
its own spelling of team names, so events are matched to the stored
schedule by canonical name and kickoff proximity. Anything that cannot
be matched confidently is reported rather than guessed at — a silently
dropped game is a game you never get a price on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from ..models import Game, MarketQuote
from ..util.http import ProviderError
from ..util.teams import canonical_team, match_teams
from .base import Provider

MARKET_KEYS = {"spread": "spreads", "total": "totals", "moneyline": "h2h"}

#: How far apart two records of the same game may be scheduled. Kickoffs
#: get moved, and the two feeds disagree about timezones now and then.
MATCH_WINDOW = timedelta(days=3)


@dataclass
class Quota:
    """What the API says about your remaining balance."""

    remaining: Optional[int] = None
    used: Optional[int] = None
    last_cost: Optional[int] = None

    def describe(self) -> str:
        if self.remaining is None:
            return "quota unknown"
        parts = [f"{self.remaining} credits left"]
        if self.last_cost:
            parts.append(f"this call cost {self.last_cost}")
        return ", ".join(parts)


@dataclass
class MatchReport:
    """Diagnostics from mapping the board onto the schedule."""

    matched: int = 0
    unmatched_events: list[str] = None  # type: ignore[assignment]
    unmatched_outcomes: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.unmatched_events is None:
            self.unmatched_events = []
        if self.unmatched_outcomes is None:
            self.unmatched_outcomes = set()


class OddsApiProvider(Provider):
    name = "odds_api"
    description = "The Odds API — live spreads, totals and moneylines across US books"
    requires_key = True
    provides_odds = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quota = Quota()
        self.unmatched: list[str] = []
        self.warnings: list[str] = []

    @property
    def base_url(self) -> str:
        return (self.settings.base_url or "https://api.the-odds-api.com/v4").rstrip("/")

    @property
    def quota_remaining(self) -> Optional[str]:
        """Surfaced by the pipeline in its fetch summary."""
        return None if self.quota.remaining is None else str(self.quota.remaining)

    # -- cost ---------------------------------------------------------------
    def estimate_cost(self, markets: Iterable[str]) -> int:
        """Credits one call will consume: markets x regions."""
        market_count = sum(1 for m in markets if m in MARKET_KEYS)
        regions = str(self.option("regions", "us")).split(",")
        return max(1, market_count) * max(1, len([r for r in regions if r.strip()]))

    # -- odds ---------------------------------------------------------------
    def fetch_odds(
        self,
        season: int,
        week: Optional[int] = None,
        *,
        games: Optional[list[Game]] = None,
        markets: Optional[list[str]] = None,
        **kwargs,
    ) -> list[MarketQuote]:
        """Pull the current board and map it onto known games."""
        self.unmatched = []
        self.warnings = []

        wanted = list(markets or self.config.betting.markets)
        market_param = ",".join(MARKET_KEYS[m] for m in wanted if m in MARKET_KEYS)
        if not market_param:
            return []

        self._guard_quota(wanted)

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

        sport = self.option("sport", "americanfootball_ncaaf")
        payload = self.client.get_json(
            f"{self.base_url}/sports/{sport}/odds",
            params=params,
            ttl=self.config.cache_ttl_seconds,
        )
        self._read_quota()

        quotes, report = self._map_events(payload or [], games or [])
        self.unmatched = report.unmatched_events
        if report.unmatched_outcomes:
            self.warnings.append(
                "outcome names not recognised as either team: "
                + ", ".join(sorted(report.unmatched_outcomes)[:8])
            )
        if not games:
            self.warnings.append(
                "no schedule to match against — fetch games before odds"
            )
        return quotes

    def _guard_quota(self, markets: Iterable[str]) -> None:
        """Refuse to spend the last credits without being told to.

        A scheduled snapshot that quietly exhausts the month's balance is
        worse than one that stops and says so, because the failure is
        invisible until you need a price and there isn't one.
        """
        floor = self.option("min_quota_remaining", 0)
        if not floor or self.quota.remaining is None:
            return
        cost = self.estimate_cost(markets)
        if self.quota.remaining - cost < int(floor):
            raise ProviderError(
                f"Refusing to call The Odds API: {self.quota.remaining} credits left, "
                f"this call costs {cost}, and the configured floor is {floor}. "
                f"Raise providers.odds_api.options.min_quota_remaining to override."
            )

    def _read_quota(self) -> None:
        def as_int(value: Optional[str]) -> Optional[int]:
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        self.quota = Quota(
            remaining=as_int(self.client.quota_remaining),
            used=as_int(self.client.quota_used),
            last_cost=as_int(self.client.quota_last_cost),
        )

    # -- matching -----------------------------------------------------------
    def _map_events(
        self, payload: list[dict], games: list[Game]
    ) -> tuple[list[MarketQuote], MatchReport]:
        index = _build_index(games)
        universe = {g.home_team for g in games} | {g.away_team for g in games}
        fetched = datetime.now(timezone.utc)

        quotes: list[MarketQuote] = []
        report = MatchReport()

        for event in payload:
            home = canonical_team(str(event.get("home_team") or ""))
            away = canonical_team(str(event.get("away_team") or ""))
            if not home or not away:
                continue
            kickoff = _parse_dt(event.get("commence_time"))

            game = _match_game(index, home, away, kickoff, universe)
            if game is None:
                report.unmatched_events.append(f"{away} @ {home}")
                continue
            report.matched += 1

            for bookmaker in event.get("bookmakers") or []:
                book = bookmaker.get("title") or bookmaker.get("key") or "unknown"
                for market in bookmaker.get("markets") or []:
                    quotes.extend(
                        self._market_quotes(game, book, market, fetched, report)
                    )
        return quotes, report

    def _market_quotes(
        self,
        game: Game,
        book: str,
        market: dict,
        fetched: datetime,
        report: MatchReport,
    ) -> list[MarketQuote]:
        key = market.get("key")
        out: list[MarketQuote] = []

        for outcome in market.get("outcomes") or []:
            price = _as_float(outcome.get("price"))
            if price is None:
                continue
            point = _as_float(outcome.get("point"))
            raw = str(outcome.get("name") or "").strip()
            name = canonical_team(raw)

            if key == "totals":
                lowered = raw.lower()
                if lowered == "over":
                    out.append(MarketQuote(game.game_id, book, "total", "over", point, price, fetched, self.name))
                elif lowered == "under":
                    out.append(MarketQuote(game.game_id, book, "total", "under", point, price, fetched, self.name))
                continue

            # Spreads and moneylines name a team; work out which one.
            side = _side_for(name, game)
            if side is None:
                report.unmatched_outcomes.add(raw)
                continue

            if key == "spreads":
                out.append(MarketQuote(game.game_id, book, "spread", side, point, price, fetched, self.name))
            elif key == "h2h":
                out.append(MarketQuote(game.game_id, book, "moneyline", side, None, price, fetched, self.name))
        return out

    # -- extras --------------------------------------------------------------
    def fetch_scores(self, days_from: int = 3) -> list[dict]:
        """Recent final scores, handy for grading when CFBD lags."""
        sport = self.option("sport", "americanfootball_ncaaf")
        result = self.client.get_json(
            f"{self.base_url}/sports/{sport}/scores",
            params={"apiKey": self.require_key(), "daysFrom": days_from},
            ttl=600,
        ) or []
        self._read_quota()
        return result


def _side_for(name: str, game: Game) -> Optional[str]:
    if name == game.home_team:
        return "home"
    if name == game.away_team:
        return "away"
    # One more pass, in case the feed's spelling normalises differently.
    resolved = match_teams(name, [game.home_team, game.away_team])
    if resolved == game.home_team:
        return "home"
    if resolved == game.away_team:
        return "away"
    return None


def _build_index(games: Iterable[Game]) -> dict[frozenset[str], list[Game]]:
    """Games keyed by the unordered pair, so venue disagreements still match."""
    index: dict[frozenset[str], list[Game]] = {}
    for game in games:
        index.setdefault(frozenset({game.home_team, game.away_team}), []).append(game)
    return index


def _match_game(
    index: dict[frozenset[str], list[Game]],
    home: str,
    away: str,
    kickoff: Optional[datetime],
    universe: set[str],
) -> Optional[Game]:
    """Find the scheduled game an odds event refers to.

    Keyed on the unordered pair because the two feeds regularly disagree
    about which side is nominally home at a neutral site. Falls back to
    fuzzy name resolution, then disambiguates by kickoff when a pair has
    played more than once in a season.
    """
    candidates = index.get(frozenset({home, away}))

    if not candidates and universe:
        resolved_home = match_teams(home, universe)
        resolved_away = match_teams(away, universe)
        if resolved_home and resolved_away and resolved_home != resolved_away:
            candidates = index.get(frozenset({resolved_home, resolved_away}))

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
    return best if distance(best) < MATCH_WINDOW else None


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
