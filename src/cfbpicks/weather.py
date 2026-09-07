"""How forecast conditions move a projected total.

Weather is one of the few public inputs a pure ratings model misses
entirely, and its effect is concentrated in one variable: **wind**. A
strong crosswind degrades the deep passing game and field goals, and the
effect on scoring is large enough to matter at the margins that decide a
total. Rain and cold are real but much weaker than folklore suggests --
a cold game between two run-heavy teams was usually going to be low
scoring anyway.

So the model here is deliberately conservative and mostly about wind:

* Below a threshold, wind does nothing. Every stadium has a breeze.
* Above it, each additional mph takes a fraction of a point off the
  total, up to a cap, because the relationship flattens -- a 40mph gale
  is not twice as bad as 20mph, both are simply "nobody is throwing".
* Precipitation and hard freezes take a small fixed amount off.
* Indoor games are untouched. This matters more than any coefficient:
  applying a wind penalty to a dome is a straightforward error.

Every number is configurable, and the total adjustment is capped so a
bad forecast can never dominate the projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import GameWeather


@dataclass
class WeatherModel:
    """Coefficients for turning a forecast into points off the total."""

    #: Wind below this is ordinary and does nothing.
    wind_threshold_mph: float = 10.0
    #: Points off the total per mph above the threshold.
    wind_points_per_mph: float = 0.25
    #: Most the wind term alone may take off.
    wind_max_points: float = 6.0
    #: Inches of precipitation that counts as a wet game, and its cost.
    precipitation_threshold_in: float = 0.05
    precipitation_points: float = 1.0
    heavy_precipitation_in: float = 0.25
    heavy_precipitation_points: float = 2.0
    #: A hard freeze, and its cost.
    freezing_point_f: float = 25.0
    freezing_points: float = 1.0
    #: Ceiling on the whole adjustment.
    max_total_points: float = 8.0


def total_adjustment(
    weather: Optional[GameWeather], model: Optional[WeatherModel] = None
) -> tuple[float, list[str]]:
    """Points to add to a projected total, with reasons.

    Returns ``(0.0, [])`` when there is no forecast or the game is
    indoors — the absence of data must never invent an adjustment.
    """
    model = model or WeatherModel()
    if weather is None or weather.indoor:
        return 0.0, []

    adjustment = 0.0
    reasons: list[str] = []

    wind = weather.wind_mph
    if wind is not None and wind > model.wind_threshold_mph:
        penalty = min(
            (wind - model.wind_threshold_mph) * model.wind_points_per_mph,
            model.wind_max_points,
        )
        if penalty >= 0.05:
            adjustment -= penalty
            reasons.append(f"{wind:.0f}mph wind ({-penalty:+.1f})")

    precip = weather.precipitation_in
    if precip is not None and precip >= model.precipitation_threshold_in:
        heavy = precip >= model.heavy_precipitation_in
        penalty = model.heavy_precipitation_points if heavy else model.precipitation_points
        adjustment -= penalty
        reasons.append(
            f"{'heavy ' if heavy else ''}precipitation {precip:.2f}in ({-penalty:+.1f})"
        )

    temp = weather.temperature_f
    if temp is not None and temp <= model.freezing_point_f:
        adjustment -= model.freezing_points
        reasons.append(f"{temp:.0f}°F ({-model.freezing_points:+.1f})")

    if adjustment < -model.max_total_points:
        adjustment = -model.max_total_points
        reasons.append(f"capped at {-model.max_total_points:+.1f}")

    return round(adjustment, 2), reasons
