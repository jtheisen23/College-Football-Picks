"""Grading and backtesting.

A predictions engine that is never graded is just an opinion generator.
This module settles bets against final scores and replays whole seasons
so the thresholds in ``config.yaml`` can be chosen from evidence.

The headline number to watch is not win rate — it is ROI per unit
staked, and closing line value. Beating the closing line is the only
leading indicator that an edge is real rather than variance.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .config import Config
from .engine.market import build_consensus
from .engine.predict import PredictionInputs, Predictor
from .engine.recommend import Recommender
from .models import Game, Recommendation
from .storage import Storage
from .util.odds import payout_multiple


def grade_recommendation(rec: Recommendation, game: Game) -> tuple[str, float]:
    """Settle one bet against a final score.

    Returns ``(result, profit_in_units)`` where result is win/loss/push.
    Profit is expressed in the same units as ``rec.stake_units``.
    """
    if not game.completed:
        return "ungraded", 0.0

    margin = float(game.margin)          # home - away
    total = float(game.total_points)
    stake = rec.stake_units
    win_profit = stake * payout_multiple(rec.price)

    if rec.market == "spread":
        if rec.line is None:
            return "ungraded", 0.0
        # The stored line is the bet side's own number.
        side_margin = margin if rec.side == "home" else -margin
        result_value = side_margin + rec.line
    elif rec.market == "total":
        if rec.line is None:
            return "ungraded", 0.0
        result_value = (total - rec.line) if rec.side == "over" else (rec.line - total)
    elif rec.market == "moneyline":
        result_value = margin if rec.side == "home" else -margin
    else:
        return "ungraded", 0.0

    if abs(result_value) < 1e-9:
        return "push", 0.0
    if result_value > 0:
        return "win", win_profit
    return "loss", -stake


@dataclass
class BacktestResult:
    bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    staked: float = 0.0
    profit: float = 0.0
    by_market: dict[str, dict[str, float]] = field(default_factory=dict)
    by_tier: dict[str, dict[str, float]] = field(default_factory=dict)
    by_week: dict[int, float] = field(default_factory=dict)
    closing_line_value: list[float] = field(default_factory=list)
    graded: list[tuple[Recommendation, str, float]] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.decided if self.decided else 0.0

    @property
    def roi(self) -> float:
        """Profit per unit staked — the number that actually matters."""
        return self.profit / self.staked if self.staked else 0.0

    @property
    def avg_clv(self) -> Optional[float]:
        return statistics.fmean(self.closing_line_value) if self.closing_line_value else None

    def add(self, rec: Recommendation, result: str, profit: float) -> None:
        self.bets += 1
        self.graded.append((rec, result, profit))
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        elif result == "push":
            self.pushes += 1
        if result != "push":
            self.staked += rec.stake_units
        self.profit += profit

        for bucket, key in ((self.by_market, rec.market), (self.by_tier, rec.tier)):
            entry = bucket.setdefault(key, {"bets": 0, "wins": 0, "staked": 0.0, "profit": 0.0})
            entry["bets"] += 1
            entry["wins"] += 1 if result == "win" else 0
            entry["staked"] += rec.stake_units if result != "push" else 0.0
            entry["profit"] += profit
        self.by_week[rec.week] = self.by_week.get(rec.week, 0.0) + profit

    def summary_lines(self) -> list[str]:
        if not self.bets:
            return ["No graded bets."]
        lines = [
            f"Bets:        {self.bets}  ({self.wins}-{self.losses}-{self.pushes})",
            f"Win rate:    {self.win_rate * 100:.1f}%  (break-even at -110 is 52.4%)",
            f"Staked:      {self.staked:.1f}u",
            f"Profit:      {self.profit:+.2f}u",
            f"ROI:         {self.roi * 100:+.2f}%",
        ]
        if self.avg_clv is not None:
            lines.append(f"Avg CLV:     {self.avg_clv:+.2f} pts")
        return lines


def grade_all(
    recs: Iterable[Recommendation], games: dict[str, Game]
) -> BacktestResult:
    """Grade a set of recommendations against final scores."""
    result = BacktestResult()
    for rec in recs:
        game = games.get(rec.game_id)
        if game is None or not game.completed:
            continue
        outcome, profit = grade_recommendation(rec, game)
        if outcome == "ungraded":
            continue
        result.add(rec, outcome, profit)
    return result


def backtest_season(
    storage: Storage,
    config: Config,
    season: int,
    weeks: Optional[Sequence[int]] = None,
) -> BacktestResult:
    """Replay a season week by week and grade every bet the engine makes.

    Ratings are taken as of the week being predicted, never later, so a
    week is priced only with what was knowable at the time. The one
    caveat worth stating plainly: stored *odds* are whatever snapshot was
    captured, so if lines were only ever pulled after kickoff the
    backtest is optimistic. Fetch odds before games start for honest
    numbers.
    """
    all_games = storage.games(season)
    by_id = {g.game_id: g for g in all_games}
    weeks = list(weeks) if weeks else sorted({g.week for g in all_games if g.completed})

    predictor = Predictor(config)
    recommender = Recommender(config)
    result = BacktestResult()

    for week in weeks:
        games = [g for g in all_games if g.week == week]
        if not games:
            continue
        ratings = storage.ratings(season, week)
        if not ratings:
            continue

        predictions = predictor.predict_week(
            PredictionInputs(games=games, ratings=ratings, schedule=all_games)
        )
        quotes_by_game = storage.quotes_for_games([g.game_id for g in games])
        if not quotes_by_game:
            continue
        consensus = build_consensus(quotes_by_game, config)
        recs = recommender.recommend_week(
            predictions, consensus, quotes_by_game, {g.game_id: g for g in games}
        )

        for rec in recs:
            game = by_id.get(rec.game_id)
            if game is None or not game.completed:
                continue
            outcome, profit = grade_recommendation(rec, game)
            if outcome == "ungraded":
                continue
            result.add(rec, outcome, profit)

            closing = storage.closing_quotes(rec.game_id, rec.market)
            clv_points = _clv_points(rec, closing)
            if clv_points is not None:
                result.closing_line_value.append(clv_points)

    return result


def _clv_points(rec: Recommendation, closing: Sequence) -> Optional[float]:
    """Points of closing line value, in the bet's favour."""
    if rec.line is None or not closing:
        return None
    same_side = [q.line for q in closing if q.side == rec.side and q.line is not None]
    if not same_side:
        return None
    close = statistics.median(same_side)
    # Getting a bigger number than the close is value on every side,
    # because the stored line is always the bet side's own number.
    return float(rec.line) - float(close)
