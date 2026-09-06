"""Core domain objects shared by providers, the engine and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class Team:
    """A program, keyed by its canonical school name."""

    name: str
    conference: Optional[str] = None
    division: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


@dataclass
class Game:
    """A single scheduled or completed game."""

    game_id: str
    season: int
    week: int
    kickoff: Optional[datetime]
    home_team: str
    away_team: str
    neutral_site: bool = False
    conference_game: bool = False
    venue: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    season_type: str = "regular"
    source: str = "unknown"

    @property
    def completed(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def margin(self) -> Optional[int]:
        """Final ``home_score - away_score``, or ``None`` if unplayed."""
        if not self.completed:
            return None
        return int(self.home_score) - int(self.away_score)

    @property
    def total_points(self) -> Optional[int]:
        if not self.completed:
            return None
        return int(self.home_score) + int(self.away_score)

    @property
    def matchup(self) -> str:
        joiner = "vs" if self.neutral_site else "@"
        return f"{self.away_team} {joiner} {self.home_team}"


@dataclass
class Rating:
    """One source's power rating for one team, in points above average."""

    source: str
    season: int
    week: int
    team: str
    rating: float
    offense: Optional[float] = None
    defense: Optional[float] = None
    rank: Optional[int] = None


@dataclass
class MarketQuote:
    """One sportsbook's price on one side of one market for one game.

    ``line`` is the spread (home perspective) or the total, and is
    ``None`` for moneylines. ``price`` is American odds.
    """

    game_id: str
    book: str
    market: str  # spread | total | moneyline
    side: str    # home | away | over | under
    line: Optional[float]
    price: float
    fetched_at: Optional[datetime] = None
    source: str = "unknown"

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.game_id, self.book, self.market, self.side)


@dataclass
class BookLine:
    """One book's two-sided price on one market, with vig removed."""

    book: str
    line: Optional[float]
    home_price: float       # home / over price
    away_price: float       # away / under price
    fair_home_prob: float   # no-vig probability of the home / over side
    hold: float             # bookmaker overround on this pair
    implied_value: Optional[float] = None  # market's implied margin or total


@dataclass
class ConsensusMarket:
    """What every book collectively thinks about one market on one game.

    ``implied_value`` is the market's view of the *true* number — the
    projected margin for spreads and moneylines, the projected total for
    totals — recovered from each book's de-vigged price and then
    aggregated. Comparing the model against that, rather than against the
    raw posted line, correctly accounts for a book charging extra juice
    on one side.
    """

    game_id: str
    market: str
    consensus_line: Optional[float] = None
    implied_value: Optional[float] = None
    fair_home_prob: Optional[float] = None
    book_lines: list["BookLine"] = field(default_factory=list)
    median_hold: Optional[float] = None

    @property
    def book_count(self) -> int:
        return len(self.book_lines)

    def side_names(self) -> tuple[str, str]:
        return ("over", "under") if self.market == "total" else ("home", "away")


@dataclass
class ExpertProjection:
    """A published projection from an outside expert or rating service.

    Sagarin's "Predictions with Totals and Moneylines" table is the
    canonical example: a projected margin, total and moneyline per game.
    Treated as an *input* to the blend, never as a market price.
    """

    source: str
    season: int
    week: int
    game_id: str
    home_team: str
    away_team: str
    projected_margin: Optional[float] = None   # home - away
    projected_total: Optional[float] = None
    home_moneyline: Optional[float] = None
    away_moneyline: Optional[float] = None
    neutral_site: bool = False


@dataclass
class Prediction:
    """The model's view of a game, before it looks at any market price."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    projected_margin: float          # home - away
    projected_total: float
    home_win_prob: float
    fair_home_spread: float          # negative = home favoured
    home_rating: float
    away_rating: float
    hfa_applied: float
    components: dict[str, Any] = field(default_factory=dict)
    rating_sources: list[str] = field(default_factory=list)
    confidence: float = 1.0          # 0-1, shrinks when inputs are thin


@dataclass
class Recommendation:
    """A single actionable bet with its edge quantified."""

    game_id: str
    season: int
    week: int
    matchup: str
    market: str                # spread | total | moneyline
    side: str                  # home | away | over | under
    selection: str             # human-readable, e.g. "Georgia -7.5"
    line: Optional[float]
    price: float
    book: str
    model_prob: float
    market_prob: float
    prob_edge: float
    point_edge: float          # model line minus market line, in points
    expected_value: float      # per unit staked
    stake_units: float
    kelly_fraction: float
    push_prob: float = 0.0
    confidence: float = 1.0
    tier: str = "lean"         # pass | lean | play | strong
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


@dataclass
class BacktestBet:
    """A graded historical recommendation."""

    recommendation: Recommendation
    result: str                # win | loss | push | ungraded
    profit_units: float
    closing_line: Optional[float] = None
    clv_points: Optional[float] = None
