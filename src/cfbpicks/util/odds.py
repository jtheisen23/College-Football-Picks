"""Odds conversion, de-vigging and probability math.

Conventions used everywhere in this package:

* A **spread** is always quoted from the home team's perspective.
  ``-7.5`` means the home team is favoured by 7.5 points.
* A **margin** is always ``home_score - away_score``.
* The home side covers when ``margin + spread > 0``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# American / decimal / probability conversions
# ---------------------------------------------------------------------------


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal (European) odds."""
    if american == 0:
        raise ValueError("American odds of 0 are not valid")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return round(-100.0 / (decimal - 1.0))


def american_to_prob(american: float) -> float:
    """Implied probability of American odds, vig included."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(prob: float) -> float:
    """Fair American odds for a probability (no vig)."""
    if not 0.0 < prob < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1")
    return decimal_to_american(1.0 / prob)


def payout_multiple(american: float) -> float:
    """Profit per unit staked, i.e. decimal odds minus the stake."""
    return american_to_decimal(american) - 1.0


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------


def vig(prices: Sequence[float]) -> float:
    """Bookmaker hold on a set of American prices covering one market.

    Returns the overround, e.g. ``0.045`` for a market booked to 104.5%.
    """
    return sum(american_to_prob(p) for p in prices) - 1.0


def devig_proportional(prices: Sequence[float]) -> list[float]:
    """Remove vig by scaling implied probabilities so they sum to 1.

    Simple, fast and unbiased for near-even two-way markets such as
    spreads and totals. It does over-price heavy favourites slightly,
    which is why :func:`devig_power` exists for lopsided moneylines.
    """
    raw = [american_to_prob(p) for p in prices]
    total = sum(raw)
    if total <= 0:
        raise ValueError("Cannot de-vig an empty or invalid market")
    return [r / total for r in raw]


def devig_power(prices: Sequence[float], tol: float = 1e-10) -> list[float]:
    """Remove vig with the power method: solve ``sum(p_i ** k) == 1``.

    The power method distributes the hold multiplicatively rather than
    additively, which fits the observed favourite-longshot bias on
    lopsided moneylines far better than proportional scaling.
    """
    raw = [american_to_prob(p) for p in prices]
    if any(r <= 0 for r in raw):
        raise ValueError("Cannot de-vig an empty or invalid market")
    if abs(sum(raw) - 1.0) < tol:
        return list(raw)

    lo, hi = 0.5, 3.0
    for _ in range(200):
        k = (lo + hi) / 2.0
        total = sum(r**k for r in raw)
        if abs(total - 1.0) < tol:
            break
        # sum(p**k) decreases as k grows because every p < 1.
        if total > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2.0
    fair = [r**k for r in raw]
    total = sum(fair)
    return [f / total for f in fair]


def devig(prices: Sequence[float], method: str = "proportional") -> list[float]:
    """De-vig a market using the named method."""
    if method == "proportional":
        return devig_proportional(prices)
    if method == "power":
        return devig_power(prices)
    raise ValueError(f"Unknown de-vig method: {method!r}")


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------


def normal_cdf(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    """Standard normal CDF via the error function (no SciPy dependency)."""
    if sd <= 0:
        raise ValueError("Standard deviation must be positive")
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def normal_ppf(p: float, mean: float = 0.0, sd: float = 1.0) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1")

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return mean + sd * x


# ---------------------------------------------------------------------------
# Game-level probability models
# ---------------------------------------------------------------------------


def cover_probability(margin: float, spread: float, sd: float) -> float:
    """Probability the home side covers ``spread`` given a projected ``margin``.

    ``spread`` is the home spread (negative = home favoured), so the home
    team covers when ``margin + spread > 0``.
    """
    return normal_cdf((margin + spread) / sd)


def push_probability(margin: float, spread: float, sd: float) -> float:
    """Probability of an exact push on a whole-number spread.

    Scoring margins are lumpy, so this is a coarse estimate: the normal
    density over a one-point band centred on the number. Half-point
    spreads always return 0.
    """
    if spread != round(spread):
        return 0.0
    needed = -spread
    lo = normal_cdf((needed - 0.5 - margin) / sd)
    hi = normal_cdf((needed + 0.5 - margin) / sd)
    return max(0.0, hi - lo)


def over_probability(total: float, line: float, sd: float) -> float:
    """Probability the game goes over ``line`` given a projected ``total``."""
    return normal_cdf((total - line) / sd)


def win_probability(margin: float, sd: float) -> float:
    """Straight-up win probability for the home team from a projected margin."""
    return normal_cdf(margin / sd)


def margin_from_win_probability(prob: float, sd: float) -> float:
    """Invert :func:`win_probability` back to an expected margin."""
    return normal_ppf(prob, sd=sd)


def spread_from_probability(prob: float, sd: float) -> float:
    """Fair home spread implied by a home cover/win probability."""
    return -normal_ppf(prob, sd=sd)


# ---------------------------------------------------------------------------
# Staking and expected value
# ---------------------------------------------------------------------------


def expected_value(prob: float, american: float, push_prob: float = 0.0) -> float:
    """Expected profit per unit staked.

    ``prob`` is the probability of winning outright; ``push_prob`` is the
    probability of a stake-back push. The loss probability is whatever is
    left over.
    """
    win_mult = payout_multiple(american)
    lose_prob = max(0.0, 1.0 - prob - push_prob)
    return prob * win_mult - lose_prob


def kelly_fraction(prob: float, american: float, push_prob: float = 0.0) -> float:
    """Full-Kelly stake as a fraction of bankroll. Negative means no bet."""
    b = payout_multiple(american)
    if b <= 0:
        return 0.0
    lose_prob = max(0.0, 1.0 - prob - push_prob)
    # Pushes return the stake, so they scale both sides of the wager down.
    live = prob + lose_prob
    if live <= 0:
        return 0.0
    p = prob / live
    q = lose_prob / live
    return (p * b - q) / b


def stake_units(
    prob: float,
    american: float,
    *,
    push_prob: float = 0.0,
    kelly_multiplier: float = 0.25,
    max_units: float = 3.0,
    unit_fraction: float = 0.01,
) -> float:
    """Recommended stake in *units* using fractional Kelly.

    One unit is ``unit_fraction`` of bankroll (1% by default), so a
    quarter-Kelly edge worth 2.4% of bankroll becomes a 2.4-unit play,
    capped at ``max_units``.
    """
    full = kelly_fraction(prob, american, push_prob)
    if full <= 0:
        return 0.0
    sized = full * kelly_multiplier / unit_fraction
    return min(sized, max_units)


def clv(closing_prob: float, bet_prob: float) -> float:
    """Closing line value as a probability delta.

    Positive means the bet was placed at a better price than the close.
    """
    return closing_prob - bet_prob


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median() of empty sequence")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
