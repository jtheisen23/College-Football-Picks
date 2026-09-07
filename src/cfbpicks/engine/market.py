"""Read the market: de-vig every book and recover its true opinion.

Comparing a model number against the *posted* line is the classic
mistake. A book showing -3 at -125/+105 is not offering -3; after vig
removal it is really saying -3.4. Every book here is converted into the
margin (or total) it implies, and those are aggregated into one market
opinion the model can be measured against.

The maths, for a home spread ``L`` and no-vig home cover probability
``p``::

    P(cover) = Phi((M + L) / sigma)   =>   M = sigma * Phi^-1(p) - L

so a book's implied true margin falls straight out of its own prices.
"""

from __future__ import annotations

import statistics
from typing import Iterable, Mapping, Optional, Sequence

from ..config import Config
from ..models import BookLine, ConsensusMarket, MarketQuote
from ..util.odds import devig, median, normal_ppf, vig


#: Two-sided markets outside this hold band are treated as bad data.
#: Real US books run roughly 2%-8% on sides and totals, more on lopsided
#: moneylines; a genuine negative hold (arbitrage) is rare enough that
#: assuming a feed error is the safer default.
MIN_PLAUSIBLE_HOLD = -0.005
MAX_PLAUSIBLE_HOLD = 0.35


def _pair_quotes(
    quotes: Iterable[MarketQuote], market: str
) -> dict[str, dict[str, MarketQuote]]:
    """Group quotes by book, keeping only books that price both sides."""
    home_side, away_side = ("over", "under") if market == "total" else ("home", "away")
    by_book: dict[str, dict[str, MarketQuote]] = {}
    for quote in quotes:
        if quote.market != market:
            continue
        if quote.side not in (home_side, away_side):
            continue
        by_book.setdefault(quote.book, {})[quote.side] = quote
    return {
        book: sides
        for book, sides in by_book.items()
        if home_side in sides and away_side in sides
    }


def consensus_for_game(
    quotes: Sequence[MarketQuote],
    market: str,
    *,
    margin_sd: float,
    total_sd: float,
    devig_method: str = "proportional",
    moneyline_devig_method: str = "power",
) -> Optional[ConsensusMarket]:
    """Build the consensus view of one market for one game."""
    if not quotes:
        return None
    game_id = quotes[0].game_id
    paired = _pair_quotes(quotes, market)
    if not paired:
        return None

    home_side, away_side = ("over", "under") if market == "total" else ("home", "away")
    method = moneyline_devig_method if market == "moneyline" else devig_method
    sd = total_sd if market == "total" else margin_sd

    book_lines: list[BookLine] = []
    for book, sides in paired.items():
        home_q, away_q = sides[home_side], sides[away_side]
        try:
            hold = vig([home_q.price, away_q.price])
        except (ValueError, ZeroDivisionError):
            continue
        # A pair booked outside a plausible hold band is bad data, not an
        # opportunity: a negative hold means the two sides disagree about
        # what game this is, and a huge one means a stale or garbage feed.
        # De-vigging either produces a confident, completely wrong number.
        if not MIN_PLAUSIBLE_HOLD <= hold <= MAX_PLAUSIBLE_HOLD:
            continue
        try:
            fair_home, _ = devig([home_q.price, away_q.price], method)
        except (ValueError, ZeroDivisionError):
            continue
        # Guard the inverse-normal against a book quoting a certainty.
        fair_home = min(max(fair_home, 1e-6), 1 - 1e-6)

        if market == "moneyline":
            implied = sd * normal_ppf(fair_home)
            line = None
        elif market == "total":
            line = home_q.line
            if line is None:
                continue
            implied = sd * normal_ppf(fair_home) + line
        else:
            line = home_q.line
            if line is None:
                continue
            implied = sd * normal_ppf(fair_home) - line

        book_lines.append(
            BookLine(
                book=book, line=line,
                home_price=home_q.price, away_price=away_q.price,
                fair_home_prob=fair_home,
                hold=hold,
                implied_value=implied,
            )
        )

    if not book_lines:
        return None

    lines = [b.line for b in book_lines if b.line is not None]
    implied_values = [b.implied_value for b in book_lines if b.implied_value is not None]

    return ConsensusMarket(
        game_id=game_id,
        market=market,
        consensus_line=median(lines) if lines else None,
        implied_value=median(implied_values) if implied_values else None,
        fair_home_prob=median([b.fair_home_prob for b in book_lines]),
        book_lines=sorted(book_lines, key=lambda b: b.book),
        median_hold=median([b.hold for b in book_lines]),
    )


def build_consensus(
    quotes_by_game: Mapping[str, Sequence[MarketQuote]],
    config: Config,
    markets: Optional[Sequence[str]] = None,
) -> dict[str, dict[str, ConsensusMarket]]:
    """Consensus for every game and every requested market."""
    markets = list(markets or config.betting.markets)
    out: dict[str, dict[str, ConsensusMarket]] = {}
    for game_id, quotes in quotes_by_game.items():
        for market in markets:
            view = consensus_for_game(
                list(quotes), market,
                margin_sd=config.model.margin_sd,
                total_sd=config.model.total_sd,
                devig_method=config.betting.devig_method,
                moneyline_devig_method=config.betting.moneyline_devig_method,
            )
            if view is not None:
                out.setdefault(game_id, {})[market] = view
    return out


def best_quote_for_side(
    quotes: Iterable[MarketQuote], market: str, side: str
) -> Optional[MarketQuote]:
    """The most favourable posted quote for one side.

    "Most favourable" means the best line first and the best price as the
    tie-break — half a point is generally worth more than a few cents of
    juice on a spread, and the caller re-prices the winner properly
    anyway.
    """
    candidates = [q for q in quotes if q.market == market and q.side == side]
    if not candidates:
        return None

    def sort_key(q: MarketQuote) -> tuple[float, float]:
        if q.line is None:
            # Moneylines have no number to shop, only a payout.
            return (0.0, _price_rank(q.price))
        # A bettor wants the largest number for home/under and the
        # smallest for away/over... expressed in each side's own terms,
        # bigger is always better once signed correctly.
        directional = q.line if side in ("home", "away") else (
            -q.line if side == "over" else q.line
        )
        return (directional, _price_rank(q.price))

    return max(candidates, key=sort_key)


def line_movement(history: Sequence[MarketQuote]) -> Optional[float]:
    """Points the line has moved from the earliest to the latest snapshot."""
    dated = [q for q in history if q.fetched_at and q.line is not None]
    if len(dated) < 2:
        return None
    dated.sort(key=lambda q: q.fetched_at)  # type: ignore[arg-type]
    return dated[-1].line - dated[0].line  # type: ignore[operator]


def _price_rank(price: float) -> float:
    """Higher is better for the bettor, across the American odds discontinuity."""
    return price if price > 0 else 100.0 + price  # -105 -> -5, -120 -> -20
