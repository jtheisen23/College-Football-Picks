"""Sanity checks on a board before anyone is invited to bet it.

A model can be wrong in two very different ways. It can be wrong about
individual games, which is unavoidable and is what the edge thresholds
and Kelly staking exist to survive. Or it can be wrong about *scale* --
producing margins that are systematically too small or too large -- and
that failure does not look like noise. It looks like a board full of
confident bets that all lean the same way.

Compressed margins are the common case, because every stage of the
pipeline shrinks: ridge regression pulls coefficients toward zero,
blending several rating sources averages away disagreement, and
normalising each source onto a fixed scale caps how far apart two teams
can end up. Each step is individually defensible and they compound. The
symptom is unmistakable once you look for it: if every favourite is
smaller in the model than at the book, the model takes every dog and
every Over, and calls them all strong.

Nothing here changes a prediction. It measures the board against the
market and says, on the page itself, when the shape of the disagreement
is the kind that comes from a broken model rather than a soft line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .models import ConsensusMarket, Prediction, Recommendation

#: Below this, the model's margins are too compressed to bet: it will
#: prefer the underdog nearly everywhere for reasons that have nothing
#: to do with the individual game. 1.0 is perfect agreement on scale.
MIN_HEALTHY_SLOPE = 0.85
#: Above this the model is over-dispersed -- rarer, but the same defect
#: pointing the other way, and it backs favourites indiscriminately.
MAX_HEALTHY_SLOPE = 1.20
#: A slope fitted on a handful of games says nothing.
MIN_GAMES = 20
#: How many standard errors a market's split must sit from even before
#: it counts as evidence. A fixed share cannot work here: nine of twelve
#: is an ordinary Tuesday while twenty-one of twenty-eight is not, and
#: both are exactly 75%. Measuring against the binomial standard error
#: scales the bar with the number of picks, the way the slope check
#: already scales with its own.
ONE_SIDED_SIGMA = 2.0
#: ...but only once there are enough picks for the share to mean
#: anything. Four dogs out of five is a Tuesday.
MIN_PICKS_FOR_SPLIT = 12


@dataclass(frozen=True)
class ScaleCheck:
    """How the model's margin scale compares with the market's.

    Carries the sufficient statistics of the regression rather than its
    result, so checks from separate weeks can be added together. That
    matters because the defect being measured lives in the ratings, not
    in any one week: a slope fitted on Saturday's slate is evidence
    about next Saturday's board too, and next week's lines are usually
    too thin to judge on their own.
    """

    games: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0
    sum_yy: float = 0.0

    def __add__(self, other: "ScaleCheck") -> "ScaleCheck":
        if not isinstance(other, ScaleCheck):
            return NotImplemented
        return ScaleCheck(
            games=self.games + other.games,
            sum_x=self.sum_x + other.sum_x,
            sum_y=self.sum_y + other.sum_y,
            sum_xx=self.sum_xx + other.sum_xx,
            sum_xy=self.sum_xy + other.sum_xy,
            sum_yy=self.sum_yy + other.sum_yy,
        )

    __radd__ = __add__

    @property
    def _sxx(self) -> float:
        if self.games < 2:
            return 0.0
        return self.sum_xx - self.sum_x ** 2 / self.games

    @property
    def _sxy(self) -> float:
        if self.games < 2:
            return 0.0
        return self.sum_xy - self.sum_x * self.sum_y / self.games

    @property
    def _syy(self) -> float:
        if self.games < 2:
            return 0.0
        return self.sum_yy - self.sum_y ** 2 / self.games

    @property
    def slope(self) -> Optional[float]:
        if self._sxx <= 1e-9:
            return None
        return self._sxy / self._sxx

    @property
    def intercept(self) -> Optional[float]:
        slope = self.slope
        if slope is None:
            return None
        return (self.sum_y - slope * self.sum_x) / self.games

    @property
    def stderr(self) -> Optional[float]:
        """Standard error of the slope, or None when undefined."""
        slope = self.slope
        if slope is None or self.games <= 2:
            return None
        # Residual sum of squares, from the sufficient statistics.
        resid = max(self._syy - slope * self._sxy, 0.0)
        return math.sqrt(resid / (self.games - 2) / self._sxx)

    @property
    def model_sd(self) -> float:
        if self.games < 1:
            return 0.0
        return math.sqrt(max(self._syy, 0.0) / self.games)

    @property
    def market_sd(self) -> float:
        if self.games < 1:
            return 0.0
        return math.sqrt(max(self._sxx, 0.0) / self.games)

    @property
    def measurable(self) -> bool:
        return self.slope is not None and self.games >= MIN_GAMES

    @property
    def healthy(self) -> bool:
        """Unmeasurable counts as healthy: absence of evidence.

        A model that is merely noisy produces a noisy slope, and on one
        week's games the standard error is easily 0.1. Condemning a
        board on the point estimate alone would cry wolf on an honest
        model having a scattered week, so the healthy band has to be
        excluded by a couple of standard errors before this speaks.
        """
        if not self.measurable:
            return True
        slope = self.slope
        assert slope is not None
        if MIN_HEALTHY_SLOPE <= slope <= MAX_HEALTHY_SLOPE:
            return True
        stderr = self.stderr
        if stderr is None:
            return False
        margin = 2.0 * stderr
        if slope < MIN_HEALTHY_SLOPE:
            return slope + margin >= MIN_HEALTHY_SLOPE
        return slope - margin <= MAX_HEALTHY_SLOPE

    @property
    def correction(self) -> Optional[float]:
        """What the model's margins would need multiplying by."""
        slope = self.slope
        if not self.measurable or not slope:
            return None
        return 1.0 / slope

    def describe(self) -> str:
        if not self.measurable:
            return f"scale unmeasured ({self.games} priced games)"
        return (
            f"model margins are {self.slope:.2f}x the market's across "
            f"{self.games} games"
        )


def market_scale(
    predictions: Sequence[Prediction],
    consensus: Mapping[str, Mapping[str, ConsensusMarket]],
) -> ScaleCheck:
    """Regress the model's margin on the market's, over priced games.

    The market's number is not truth, and a model that matched it
    exactly would have no reason to exist. But it is an unbiased
    estimate of the margin, so across a full slate the *slope* between
    the two is a fair read on whether the model's scale is right. The
    residuals around that line are where any real edge lives; this only
    asks whether the line itself is at 45 degrees.
    """
    n = 0
    sx = sy = sxx = sxy = syy = 0.0
    for pred in predictions:
        view = consensus.get(pred.game_id, {}).get("spread")
        if view is None or view.implied_value is None:
            continue
        x = view.implied_value
        y = pred.projected_margin
        n += 1
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
        syy += y * y

    return ScaleCheck(games=n, sum_x=sx, sum_y=sy, sum_xx=sxx, sum_xy=sxy, sum_yy=syy)


def _one_sided(recs: Sequence[Recommendation]) -> list[tuple[str, str, int, int]]:
    """Where this board leans, per market.

    Each entry is ``(market, side, picks on that side, picks in that
    market)`` for a market that leans further than it should. Spreads
    and totals are counted separately and both are checked: a scale
    correction that fixes one says nothing about the other, and a board
    that is half right must not read as a board that is right.
    """
    out = []

    # The line is stored from the picked side's own perspective, so a
    # positive number is a pick taking points.
    spreads = [r for r in recs if r.market == "spread" and r.line is not None]
    if spreads:
        dogs = sum(1 for r in spreads if r.line > 0)
        out.append(("spread", "underdog", dogs, len(spreads)))

    totals = [r for r in recs if r.market == "total"]
    if totals:
        overs = sum(1 for r in totals if r.side == "over")
        out.append(("total", "Over", overs, len(totals)))

    leaning = []
    for market, side, picked, total in out:
        if total < MIN_PICKS_FOR_SPLIT:
            continue
        share = max(picked, total - picked) / total
        # Standard error of a fair coin over this many picks.
        stderr = 0.5 / math.sqrt(total)
        if share - 0.5 > ONE_SIDED_SIGMA * stderr:
            majority = side if picked * 2 > total else _opposite(market, side)
            leaning.append((market, majority, max(picked, total - picked), total))
    return leaning


def _opposite(market: str, side: str) -> str:
    return "Under" if market == "total" else "favourite"


def board_warning(
    recs: Sequence[Recommendation], scale: ScaleCheck
) -> Optional[str]:
    """The banner this board should carry, if any.

    Deliberately blunt. A page that hedges gets read as a page that is
    fine, and the failure this catches produces boards that look like
    the best week of the season.
    """
    if not recs:
        return None

    if not scale.healthy:
        assert scale.slope is not None
        direction = "too small" if scale.slope < 1 else "too large"
        return (
            f"Do not bet this board. The model's margins are {direction} "
            f"relative to every line on the slate ({scale.describe()}), so "
            "these are not real edges — they are the same scaling error "
            "repeated on every game."
        )

    leaning = _one_sided(recs)
    if leaning:
        parts = [
            f"{picked} of {n} {market} picks are on the {side}"
            for market, side, picked, n in leaning
        ]
        return (
            "Treat this board with suspicion: "
            + " and ".join(parts)
            + ". A model that disagrees with the market in one direction all "
            "week is usually miscalibrated rather than right."
        )

    # Checked last, because the two checks above need no market data and
    # are the backstop for exactly this case. Reaching here means the
    # board looks balanced but nothing has actually been verified, and
    # silence would read as a clean bill of health.
    if not scale.measurable:
        return (
            f"This board has not been checked for calibration \u2014 only "
            f"{scale.games} games on the slate are priced, too few to tell "
            "whether the model's numbers are on the right scale."
        )
    return None
