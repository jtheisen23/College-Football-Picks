"""Power ratings the model computes for itself, from results only.

Every external rating has a provenance problem. SP+ and SRS are
season-level, so for a finished season they encode results we are trying
to predict; Sagarin and Massey publish only a current snapshot with no
history. That leaves nothing honest to backtest with.

This module removes the dependency. Given the games played *through* a
given week, it fits one rating per team plus a home-field term by
least squares on scoring margin::

    margin  ~=  rating[home] - rating[away] + home_field * (not neutral)

Because that fit is only ever shown games that finished before the week
being priced, the resulting rating is point-in-time by construction, and
a backtest built on it is measuring something real.

Three details make it behave on real data:

* **Ridge regularisation.** In September the system is wildly
  underdetermined: a 2-0 team has beaten two opponents nobody has a read
  on yet. An L2 penalty pulls thinly-evidenced teams toward average
  instead of letting them run to absurd numbers, and guarantees the
  normal equations are solvable even when the schedule graph is
  disconnected.
* **Margin capping.** A 63-0 result says little more than a 35-0 one.
  Margins are clamped so blowouts don't dominate the fit.
* **Recency weighting.** Optional exponential decay, for the fact that a
  team in November is not the team that played in August.

The solve is a Cholesky factorisation of a matrix whose size is the
number of teams plus one — a few hundred at most — so it runs in
milliseconds of pure Python and adds no dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from ..models import Game, Rating

#: Name this rating is stored under.
SOURCE = "cfbpicks_margin"

#: Margin capping is OFF by default, deliberately. Clamping blowouts is
#: standard for *ranking* systems, but it shrinks every fitted
#: coefficient toward zero, and this model's output is a point spread.
#: Measured on simulated seasons, a 21-point cap left predicted margins
#: 58% too small (calibration slope 1.58) and a 28-point cap 33% too
#: small — a bias that would tip the model onto underdogs across the
#: whole board. Set it only if a data source has scores you don't trust.
DEFAULT_MARGIN_CAP = None

#: L2 penalty pulling each team toward the prior, in units of "games of
#: evidence needed to move a team off it". Chosen by sweep rather than
#: taste: across simulated early- and late-season slates, 2.0 gave both
#: the best held-out accuracy and a calibration slope of ~1.02, while
#: 6.0 over-shrank to a slope of 1.64 in September.
DEFAULT_RIDGE = 2.0

#: Strength of the prior on the home-field term, in games of evidence.
HOME_FIELD_PRIOR_WEIGHT = 1.0


@dataclass
class FitResult:
    """Fitted ratings plus the diagnostics worth acting on."""

    ratings: dict[str, float] = field(default_factory=dict)
    home_field: float = 0.0
    games_used: int = 0
    #: Games each team contributed, for judging how much to trust it.
    appearances: dict[str, int] = field(default_factory=dict)
    #: RMSE of the fit's margin residuals. An empirical read on the
    #: `margin_sd` the rest of the engine assumes.
    residual_sd: Optional[float] = None

    @property
    def teams(self) -> int:
        return len(self.ratings)

    def to_ratings(self, season: int, week: int) -> list[Rating]:
        """Emit as stored ratings, valid for pricing ``week``."""
        return [
            Rating(
                source=SOURCE, season=season, week=week, team=team,
                rating=value, point_in_time=True,
            )
            for team, value in sorted(self.ratings.items())
        ]

    def summary(self) -> str:
        parts = [
            f"{self.teams} teams from {self.games_used} games",
            f"HFA {self.home_field:+.2f}",
        ]
        if self.residual_sd is not None:
            parts.append(f"residual sd {self.residual_sd:.1f}")
        return ", ".join(parts)


def fit_margin_ratings(
    games: Iterable[Game],
    *,
    ridge: float = DEFAULT_RIDGE,
    margin_cap: Optional[float] = DEFAULT_MARGIN_CAP,  # None = no capping
    prior: Optional[Mapping[str, float]] = None,
    estimate_home_field: bool = True,
    default_home_field: float = 2.2,
    recency_halflife: Optional[float] = None,
    as_of_week: Optional[int] = None,
) -> FitResult:
    """Fit one rating per team from completed games.

    ``prior`` supplies a per-team value to shrink toward — last season's
    final ratings are the natural choice, and are what stop Week 1 from
    being a coin flip. Teams absent from it shrink toward average.
    """
    completed = [g for g in games if g.completed and g.home_team and g.away_team]
    if not completed:
        return FitResult(ratings=dict(prior or {}), home_field=default_home_field)

    teams = sorted({g.home_team for g in completed} | {g.away_team for g in completed})
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    size = n + 1 if estimate_home_field else n

    # Normal equations, accumulated directly. The design matrix has one
    # +1 and one -1 per row, so X'X is assembled game by game without
    # ever building X itself.
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    appearances: dict[str, int] = {}
    rows: list[tuple[int, int, float, float, float]] = []

    for game in completed:
        i = index[game.home_team]
        j = index[game.away_team]
        margin = float(game.margin)
        if margin_cap is not None:
            margin = max(-margin_cap, min(margin_cap, margin))
        neutral = 0.0 if game.neutral_site else 1.0
        weight = _recency_weight(game, as_of_week, recency_halflife)
        if weight <= 0:
            continue

        target = margin if estimate_home_field else margin - default_home_field * neutral
        rows.append((i, j, neutral, target, weight))

        appearances[game.home_team] = appearances.get(game.home_team, 0) + 1
        appearances[game.away_team] = appearances.get(game.away_team, 0) + 1

        matrix[i][i] += weight
        matrix[j][j] += weight
        matrix[i][j] -= weight
        matrix[j][i] -= weight
        vector[i] += weight * target
        vector[j] -= weight * target

        if estimate_home_field:
            h = n
            matrix[i][h] += weight * neutral
            matrix[h][i] += weight * neutral
            matrix[j][h] -= weight * neutral
            matrix[h][j] -= weight * neutral
            matrix[h][h] += weight * neutral * neutral
            vector[h] += weight * neutral * target

    if not rows:
        return FitResult(ratings=dict(prior or {}), home_field=default_home_field)

    # Ridge toward the prior, on the team terms.
    lam = max(ridge, 1e-6)
    for team, i in index.items():
        matrix[i][i] += lam
        vector[i] += lam * float((prior or {}).get(team, 0.0))

    if estimate_home_field:
        # A weak prior on home field, worth about one game of evidence.
        # Real data overwhelms it immediately, but it keeps the system
        # solvable when the slate carries no information about venue --
        # an all-neutral schedule (bowls, a conference tournament) would
        # otherwise make the matrix singular and lose *every* rating,
        # not just the home-field term.
        matrix[n][n] += HOME_FIELD_PRIOR_WEIGHT
        vector[n] += HOME_FIELD_PRIOR_WEIGHT * default_home_field

    solution = _solve_spd(matrix, vector)
    if solution is None:  # pragma: no cover - ridge makes this unreachable
        return FitResult(ratings=dict(prior or {}), home_field=default_home_field)

    home_field = solution[n] if estimate_home_field else default_home_field
    ratings = {team: solution[i] for team, i in index.items()}

    # Express as points above average, matching every other source.
    mean = sum(ratings.values()) / len(ratings)
    ratings = {team: value - mean for team, value in ratings.items()}

    return FitResult(
        ratings=ratings,
        home_field=home_field,
        games_used=len(rows),
        appearances=appearances,
        residual_sd=_residual_sd(rows, ratings, index, home_field, estimate_home_field,
                                 default_home_field, teams),
    )


def _recency_weight(
    game: Game, as_of_week: Optional[int], halflife: Optional[float]
) -> float:
    if not halflife or as_of_week is None:
        return 1.0
    age = max(0, as_of_week - game.week)
    return 0.5 ** (age / halflife)


def _residual_sd(
    rows: Sequence[tuple[int, int, float, float, float]],
    ratings: Mapping[str, float],
    index: Mapping[str, int],
    home_field: float,
    estimate_home_field: bool,
    default_home_field: float,
    teams: Sequence[str],
) -> Optional[float]:
    """RMSE of the fitted margins, adjusted for parameters spent."""
    by_position = {i: ratings[team] for team, i in index.items()}
    hfa = home_field if estimate_home_field else default_home_field

    total = 0.0
    weight_total = 0.0
    for i, j, neutral, target, weight in rows:
        predicted = by_position[i] - by_position[j]
        if estimate_home_field:
            predicted += hfa * neutral
        total += weight * (target - predicted) ** 2
        weight_total += weight

    dof = weight_total - (len(teams) + (1 if estimate_home_field else 0))
    if dof <= 0:
        # Fewer games than parameters: the residual is meaningless rather
        # than impressive, so don't report one.
        return None
    return math.sqrt(total / dof)


def _solve_spd(matrix: list[list[float]], vector: list[float]) -> Optional[list[float]]:
    """Solve a symmetric positive-definite system by Cholesky."""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]

    for i in range(n):
        row_i = lower[i]
        for j in range(i + 1):
            row_j = lower[j]
            total = matrix[i][j] - sum(row_i[k] * row_j[k] for k in range(j))
            if i == j:
                if total <= 1e-12:
                    return None
                row_i[j] = math.sqrt(total)
            else:
                row_i[j] = total / row_j[j]

    # Forward substitution, then back substitution.
    y = [0.0] * n
    for i in range(n):
        y[i] = (vector[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]

    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return x
