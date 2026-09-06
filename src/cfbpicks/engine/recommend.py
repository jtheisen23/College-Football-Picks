"""Turn model projections plus market prices into actionable bets.

The comparison is always made **at the price you can actually get**: the
model's probability and the market's no-vig probability are both
evaluated at the specific line and juice of the best available quote,
not at some notional consensus number nobody is offering.

Two deliberate conservatisms:

* **Shrinkage.** The bet is priced on ``market + confidence * (model -
  market)`` rather than the raw model probability. When inputs are thin
  or the rating systems disagree, the model defers to the market instead
  of betting its own noise.
* **Push handling.** Whole-number spreads and totals carry a real chance
  of a stake-back push, which lifts EV slightly and is priced in rather
  than ignored.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from ..config import Config
from ..models import ConsensusMarket, Game, MarketQuote, Prediction, Recommendation
from ..util.odds import (
    cover_probability,
    expected_value,
    kelly_fraction,
    normal_cdf,
    over_probability,
    push_probability,
    stake_units,
    win_probability,
)
from .market import best_quote_for_side


class Recommender:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.betting = config.betting
        self.model = config.model

    def recommend_week(
        self,
        predictions: Iterable[Prediction],
        consensus: Mapping[str, Mapping[str, ConsensusMarket]],
        quotes_by_game: Mapping[str, Sequence[MarketQuote]],
        games: Optional[Mapping[str, Game]] = None,
    ) -> list[Recommendation]:
        games = games or {}
        out: list[Recommendation] = []
        for prediction in predictions:
            markets = consensus.get(prediction.game_id) or {}
            quotes = quotes_by_game.get(prediction.game_id) or []
            game = games.get(prediction.game_id)
            for market_name in self.betting.markets:
                view = markets.get(market_name)
                if view is None:
                    continue
                out.extend(self._evaluate(prediction, view, quotes, game))
        out.sort(key=lambda r: (r.prob_edge, r.expected_value), reverse=True)
        return out

    # -- per market --------------------------------------------------------
    def _evaluate(
        self,
        prediction: Prediction,
        view: ConsensusMarket,
        quotes: Sequence[MarketQuote],
        game: Optional[Game],
    ) -> list[Recommendation]:
        if view.book_count < self.betting.min_book_count:
            return []
        if view.implied_value is None:
            return []

        matchup = game.matchup if game else f"{prediction.away_team} @ {prediction.home_team}"
        results: list[Recommendation] = []

        for side in view.side_names():
            quote = best_quote_for_side(quotes, view.market, side)
            if quote is None:
                continue
            rec = self._evaluate_side(prediction, view, quote, side, matchup)
            if rec is not None:
                results.append(rec)

        # Both sides of the same market can never both be +EV once vig is
        # removed; keep the better one and drop any artefact.
        if len(results) == 2:
            results = [max(results, key=lambda r: r.prob_edge)]
        return [r for r in results if r.tier != "pass"]

    def _evaluate_side(
        self,
        prediction: Prediction,
        view: ConsensusMarket,
        quote: MarketQuote,
        side: str,
        matchup: str,
    ) -> Optional[Recommendation]:
        margin_sd = self.model.margin_sd
        total_sd = self.model.total_sd
        notes: list[str] = []

        if view.market == "spread":
            line = quote.line
            if line is None:
                return None
            if abs(line) > self.betting.max_spread_line:
                return None
            # Both probabilities are evaluated at this quote's own number.
            # `line` is always the quoted side's own number, and a side
            # covers when its margin plus its number is positive.
            sign = 1.0 if side == "home" else -1.0
            model_home = cover_probability(sign * prediction.projected_margin, line, margin_sd)
            market_home = cover_probability(sign * view.implied_value, line, margin_sd)
            push = push_probability(
                prediction.projected_margin if side == "home" else -prediction.projected_margin,
                line,
                margin_sd,
            )
            point_edge = (prediction.projected_margin - view.implied_value) * (1 if side == "home" else -1)
            selection = self._spread_selection(prediction, side, line)
            min_points = self.betting.min_point_edge_spread

        elif view.market == "total":
            line = quote.line
            if line is None:
                return None
            model_over = over_probability(prediction.projected_total, line, total_sd)
            market_over = over_probability(view.implied_value, line, total_sd)
            model_home = model_over if side == "over" else 1 - model_over
            market_home = market_over if side == "over" else 1 - market_over
            push = push_probability(prediction.projected_total, -line, total_sd)
            point_edge = (prediction.projected_total - view.implied_value) * (1 if side == "over" else -1)
            selection = f"{'Over' if side == 'over' else 'Under'} {line:g}"
            min_points = self.betting.min_point_edge_total

        elif view.market == "moneyline":
            line = None
            model_win = win_probability(prediction.projected_margin, margin_sd)
            market_win = win_probability(view.implied_value, margin_sd)
            model_home = model_win if side == "home" else 1 - model_win
            market_home = market_win if side == "home" else 1 - market_win
            push = 0.0
            point_edge = (prediction.projected_margin - view.implied_value) * (1 if side == "home" else -1)
            team = prediction.home_team if side == "home" else prediction.away_team
            selection = f"{team} ML"
            min_points = 0.0
            # A normal margin distribution understates how often a heavy
            # favourite simply wins, so the model reliably "finds" value on
            # big underdogs that is not there. Refuse to price those.
            if abs(view.implied_value) > self.betting.max_moneyline_margin:
                return None
        else:
            return None

        # -- shrink the model toward the market by our confidence ----------
        confidence = max(0.0, min(1.0, prediction.confidence))
        shrunk_prob = market_home + confidence * (model_home - market_home)
        shrunk_prob = min(max(shrunk_prob, 1e-6), 1 - 1e-6)

        prob_edge = shrunk_prob - market_home
        ev = expected_value(shrunk_prob, quote.price, push)
        kelly = kelly_fraction(shrunk_prob, quote.price, push)
        units = stake_units(
            shrunk_prob, quote.price, push_prob=push,
            kelly_multiplier=self.betting.kelly_multiplier,
            max_units=self.betting.max_units,
            unit_fraction=self.betting.unit_fraction,
        )

        if confidence < 1.0:
            notes.append(f"model shrunk {(1 - confidence) * 100:.0f}% toward market (confidence {confidence:.2f})")
        if push > 0.001:
            notes.append(f"push chance {push * 100:.1f}%")
        if view.median_hold is not None and view.median_hold > 0.06:
            notes.append(f"wide market: {view.median_hold * 100:.1f}% hold")
        if view.book_count == 1:
            notes.append("single book — no line shopping")

        tier = self._tier(prob_edge, ev, point_edge, min_points)

        return Recommendation(
            game_id=prediction.game_id, season=prediction.season, week=prediction.week,
            matchup=matchup, market=view.market, side=side, selection=selection,
            line=quote.line, price=quote.price, book=quote.book,
            model_prob=round(shrunk_prob, 5), market_prob=round(market_home, 5),
            prob_edge=round(prob_edge, 5), point_edge=round(point_edge, 3),
            expected_value=round(ev, 5), stake_units=round(units, 2),
            kelly_fraction=round(kelly, 5), push_prob=round(push, 5),
            confidence=round(confidence, 4), tier=tier, notes=notes,
        )

    def _spread_selection(self, prediction: Prediction, side: str, line: float) -> str:
        # `line` is stored from the quoted side's own perspective, so an
        # away quote of +16 is already "Iowa +16" and must not be flipped.
        team = prediction.home_team if side == "home" else prediction.away_team
        return f"{team} {line:+g}"

    def _tier(self, prob_edge: float, ev: float, point_edge: float, min_points: float) -> str:
        """Classify an edge, applying every threshold as a gate."""
        if prob_edge < self.betting.min_prob_edge:
            return "pass"
        if ev < self.betting.min_expected_value:
            return "pass"
        if min_points > 0 and abs(point_edge) < min_points:
            return "pass"

        thresholds = self.betting.tier_thresholds
        if prob_edge >= thresholds.get("strong", 0.055):
            return "strong"
        if prob_edge >= thresholds.get("play", 0.035):
            return "play"
        return "lean"
