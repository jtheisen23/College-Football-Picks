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
import statistics
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
#: Share of spread picks landing on one side before the split itself is
#: evidence. A fair model lands near half; 0.75 is a long way out.
MAX_ONE_SIDED = 0.75
#: ...but only once there are enough picks for the share to mean
#: anything. Four dogs out of five is a Tuesday.
MIN_PICKS_FOR_SPLIT = 12


@dataclass(frozen=True)
class ScaleCheck:
    """How the model's margin scale compares with the market's.

    ``slope`` is from regressing the model's projected margin on the
    market's implied margin across every game priced this week. One
    means the two agree about how far apart teams are, which is the
    precondition for a disagreement on a single game meaning anything.
    """

    games: int = 0
    slope: Optional[float] = None
    intercept: Optional[float] = None
    stderr: Optional[float] = None
    model_sd: float = 0.0
    market_sd: float = 0.0

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
        assert self.slope is not None
        if MIN_HEALTHY_SLOPE <= self.slope <= MAX_HEALTHY_SLOPE:
            return True
        if self.stderr is None:
            return False
        margin = 2.0 * self.stderr
        if self.slope < MIN_HEALTHY_SLOPE:
            return self.slope + margin >= MIN_HEALTHY_SLOPE
        return self.slope - margin <= MAX_HEALTHY_SLOPE

    @property
    def correction(self) -> Optional[float]:
        """What the model's margins would need multiplying by."""
        if not self.measurable or not self.slope:
            return None
        return 1.0 / self.slope

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
    xs: list[float] = []
    ys: list[float] = []
    for pred in predictions:
        view = consensus.get(pred.game_id, {}).get("spread")
        if view is None or view.implied_value is None:
            continue
        xs.append(view.implied_value)
        ys.append(pred.projected_margin)

    n = len(xs)
    if n < 2:
        return ScaleCheck(games=n)

    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-9:
        # Every game priced identically: nothing to regress against.
        return ScaleCheck(games=n, model_sd=statistics.pstdev(ys))
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx

    stderr = None
    if n > 2:
        resid = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
        stderr = math.sqrt(resid / (n - 2) / sxx)

    return ScaleCheck(
        games=n,
        slope=slope,
        intercept=intercept,
        stderr=stderr,
        model_sd=statistics.pstdev(ys),
        market_sd=statistics.pstdev(xs),
    )


def _dog_share(recs: Sequence[Recommendation]) -> Optional[tuple[int, int]]:
    """(picks on the underdog, spread picks) for this board."""
    spreads = [r for r in recs if r.market == "spread" and r.line is not None]
    if not spreads:
        return None
    # The line is stored from the picked side's own perspective, so a
    # positive number is a pick taking points.
    return sum(1 for r in spreads if r.line > 0), len(spreads)


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

    split = _dog_share(recs)
    if split is not None:
        dogs, total = split
        if total >= MIN_PICKS_FOR_SPLIT:
            share = max(dogs, total - dogs) / total
            if share > MAX_ONE_SIDED:
                side = "underdog" if dogs * 2 > total else "favourite"
                return (
                    f"Treat this board with suspicion: {max(dogs, total - dogs)} "
                    f"of {total} spread picks are on the {side}. A model that "
                    "disagrees with the market in one direction all week is "
                    "usually miscalibrated rather than right."
                )
    return None
