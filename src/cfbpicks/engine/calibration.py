"""Correcting the scale of predicted margins against what actually happened.

Every stage of this pipeline shrinks. Ridge regression pulls team
coefficients toward zero, blending several rating sources averages away
the places they disagree, and normalising each source onto a fixed
standard deviation caps how far apart two teams can finish. Each step is
defensible alone; together they produced margins about seventy per cent
of life size, which is not a small bias. It made every favourite look
overpriced and turned a whole slate into underdog bets.

The correction is a straight line fitted on completed games:

    actual_margin  =  intercept  +  slope * predicted_margin

and predictions are then pushed through that line. A slope above one
stretches the model back out to the scale the results say it should have
had. The intercept catches a systematic home-field error in the same
motion.

Two things this deliberately does not do.

It does not fit against the market. The market's number is a better
estimate of any single game than this model will ever produce, so
calibrating to it would pull the model toward the closing line and
quietly destroy the only thing worth having -- an opinion that differs
from the price for reasons of its own. Results are noisier and there are
far fewer of them, and they are the honest target.

It does not trust a weak fit. A slope measured on two hundred games
carries real uncertainty, and chasing it would be its own kind of
overfitting, so the estimate is shrunk toward "no correction at all" in
proportion to how well established the deviation is. One standard error
of evidence buys half the correction; five buys nearly all of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config
    from ..storage import Storage

#: Games needed before any correction is applied. Roughly three weeks of
#: a full FBS slate. Below this the slope is mostly noise, and the
#: uncorrected model is the safer default.
MIN_GAMES = 120
#: The furthest the correction may stretch a margin. Past this, the
#: predictions are so small next to real results that a units bug is
#: likelier than a real finding, and it is refused rather than obeyed.
#:
#: There is deliberately no floor to match it. The two directions are
#: not equally dangerous: shrinking costs bets, while stretching a model
#: that has no signal turns small nonsense into large nonsense and puts
#: money behind it. So a fit is allowed to shrink predictions as far as
#: the results say -- to nearly zero, which correctly produces no bets
#: at all -- and is only distrusted when it wants to stretch them.
MAX_SLOPE = 2.00
#: Points of systematic home-field correction allowed. The predictor
#: already applies a home-field advantage; this only nudges it.
MAX_INTERCEPT = 3.0


@dataclass(frozen=True)
class MarginCalibration:
    """A fitted line from predicted margin to actual margin."""

    games: int = 0
    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False
    #: Slope as measured, before shrinking and clamping. Kept so the
    #: correction can be explained rather than just applied.
    raw_slope: Optional[float] = None
    stderr: Optional[float] = None
    residual_sd: Optional[float] = None

    def apply(self, margin: float) -> float:
        if not self.fitted:
            return margin
        return self.intercept + self.slope * margin

    def describe(self) -> str:
        if not self.fitted:
            return f"uncalibrated ({self.games} completed games, need {MIN_GAMES})"
        raw = f"{self.raw_slope:.2f}" if self.raw_slope is not None else "?"
        return (
            f"margins x{self.slope:.2f} {self.intercept:+.1f} "
            f"(measured {raw} on {self.games} games)"
        )


#: What to use when nothing has been fitted: leave predictions alone.
IDENTITY = MarginCalibration()


def fit_margin_calibration(
    pairs: Sequence[tuple[float, float]]
) -> MarginCalibration:
    """Fit actual margin on predicted margin over completed games.

    ``pairs`` is ``(predicted, actual)``, both as home-team margins.
    """
    n = len(pairs)
    if n < 3:
        return MarginCalibration(games=n)

    mx = sum(p for p, _ in pairs) / n
    my = sum(a for _, a in pairs) / n
    sxx = sum((p - mx) ** 2 for p, _ in pairs)
    if sxx <= 1e-9:
        # The model said the same thing about every game; there is no
        # scale to correct, only a constant to argue about.
        return MarginCalibration(games=n)
    sxy = sum((p - mx) * (a - my) for p, a in pairs)

    raw_slope = sxy / sxx
    raw_intercept = my - raw_slope * mx

    resid = sum((a - raw_intercept - raw_slope * p) ** 2 for p, a in pairs)
    dof = n - 2
    residual_sd = math.sqrt(resid / dof) if dof > 0 else None
    stderr = math.sqrt(resid / dof / sxx) if dof > 0 and resid > 0 else None

    if n < MIN_GAMES:
        # Measured, reported, not yet acted on.
        return MarginCalibration(
            games=n, raw_slope=raw_slope, stderr=stderr, residual_sd=residual_sd
        )

    slope = _shrink(raw_slope, 1.0, stderr)
    intercept = _shrink(raw_intercept, 0.0, _intercept_stderr(pairs, residual_sd, mx, sxx))

    if slope > MAX_SLOPE or slope <= 0.0:
        # Too big to be a scale error, or the model is anti-correlated
        # with reality. Either is a broken measurement, not an
        # instruction, and neither is fixed by rescaling.
        return MarginCalibration(
            games=n, raw_slope=raw_slope, stderr=stderr, residual_sd=residual_sd
        )
    intercept = max(-MAX_INTERCEPT, min(MAX_INTERCEPT, intercept))

    return MarginCalibration(
        games=n, slope=slope, intercept=intercept, fitted=True,
        raw_slope=raw_slope, stderr=stderr, residual_sd=residual_sd,
    )


def _shrink(estimate: float, null: float, stderr: Optional[float]) -> float:
    """Pull an estimate back toward ``null`` by how uncertain it is.

    The weight is t^2/(1+t^2) for t standard errors of deviation, so a
    correction has to earn its size: one standard error of evidence buys
    half of it, three buys ninety per cent. Without a usable standard
    error the estimate is taken at face value, which is the right
    reading of "the residuals are zero".
    """
    if stderr is None or stderr <= 0:
        return estimate
    t = (estimate - null) / stderr
    weight = t * t / (1.0 + t * t)
    return null + (estimate - null) * weight


def _intercept_stderr(
    pairs: Sequence[tuple[float, float]],
    residual_sd: Optional[float],
    mean_x: float,
    sxx: float,
) -> Optional[float]:
    if residual_sd is None or sxx <= 0:
        return None
    n = len(pairs)
    return residual_sd * math.sqrt(1.0 / n + mean_x * mean_x / sxx)


def collect_margin_pairs(
    storage: "Storage",
    config: "Config",
    season: int,
    *,
    before_week: Optional[int] = None,
) -> list[tuple[float, float]]:
    """Replay completed weeks and pair each prediction with its result.

    The replay is the whole point, and it has to be honest about time.
    Each week is predicted from the ratings as they stood *that* week
    -- ``point_in_time_only`` drops the season-final numbers that
    already know how the season ended -- and ``before_week`` keeps a
    week's own result out of the calibration used to bet it.

    Predicting past games with today's ratings would be far easier and
    completely worthless: the regression is fitted on this season's
    results, so scoring it against those same results is in-sample. It
    would report a slope near one and conclude that nothing needs
    correcting, which is exactly the bias this exists to find.
    """
    from .predict import PredictionInputs, Predictor  # circular at module level

    all_games = storage.games(season)
    if not all_games:
        return []

    roster = storage.fbs_teams(season) if config.betting.fbs_only else set()
    predictor = Predictor(config)
    pairs: list[tuple[float, float]] = []

    weeks = sorted({g.week for g in all_games if g.completed})
    for week in weeks:
        if before_week is not None and week >= before_week:
            continue
        games = [g for g in all_games if g.week == week]
        ratings = storage.ratings(season, week, point_in_time_only=True)
        if not ratings:
            continue

        by_id = {g.game_id: g for g in games}
        predictions = predictor.predict_week(
            PredictionInputs(games=games, ratings=ratings, schedule=all_games)
        )
        for prediction in predictions:
            game = by_id.get(prediction.game_id)
            if game is None or not game.completed or game.margin is None:
                continue
            # Calibrate on the population actually bet. FCS opponents are
            # rated near average by a model built on FBS results, so
            # including them would fit the slope to games the engine
            # never prices.
            if roster and (game.home_team not in roster or game.away_team not in roster):
                continue
            pairs.append((prediction.projected_margin, float(game.margin)))

    return pairs
