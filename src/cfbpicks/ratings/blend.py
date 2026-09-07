"""Blend many rating systems into one power number per team.

Two things make this more than a weighted average:

* **Scale alignment.** SP+, SRS, a recentred Sagarin number and Elo/25
  are all nominally "points above average", but their spreads differ —
  SP+ ranges roughly twice as wide as SRS. Averaging them raw silently
  overweights the widest source, so each is rescaled to a common
  standard deviation first.
* **Disagreement as uncertainty.** When SP+ loves a team and Sagarin
  hates it, that spread is information. It is carried through as
  ``dispersion`` and ends up shrinking the stake on that game.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from ..models import Rating

#: Points of standard deviation to normalise every source onto. Close to
#: the historical spread of SP+ across FBS, so blended numbers stay on a
#: familiar scale.
DEFAULT_TARGET_STD = 11.0


@dataclass
class BlendedRating:
    team: str
    rating: float
    sources: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)
    dispersion: float = 0.0        # std of the normalised source ratings
    weight_total: float = 0.0

    @property
    def n_sources(self) -> int:
        return len(self.sources)


def _normalise_source(
    values: Mapping[str, float], target_std: float
) -> tuple[dict[str, float], float, float]:
    """Centre a source on zero and rescale it to ``target_std``."""
    numbers = list(values.values())
    if len(numbers) < 2:
        return dict(values), 0.0, 1.0
    mean = statistics.fmean(numbers)
    std = statistics.pstdev(numbers)
    if std <= 1e-9:
        return {t: 0.0 for t in values}, mean, 1.0
    scale = target_std / std
    return {t: (v - mean) * scale for t, v in values.items()}, mean, scale


def blend_ratings(
    ratings: Iterable[Rating],
    weights: Optional[Mapping[str, float]] = None,
    *,
    normalize: bool = True,
    target_std: float = DEFAULT_TARGET_STD,
    shrink_to_mean: bool = True,
    min_sources_for_full_confidence: int = 2,
) -> dict[str, BlendedRating]:
    """Combine per-source ratings into one rating per team.

    Unknown sources get a weight of 1.0 rather than being dropped, so a
    newly added feed contributes immediately without a config change.
    """
    weights = dict(weights or {})
    by_source: dict[str, dict[str, float]] = {}
    for rating in ratings:
        if not rating.team:
            continue
        by_source.setdefault(rating.source, {})[rating.team] = rating.rating

    if not by_source:
        return {}

    normalised: dict[str, dict[str, float]] = {}
    for source, values in by_source.items():
        if normalize:
            scaled, _, _ = _normalise_source(values, target_std)
            normalised[source] = scaled
        else:
            normalised[source] = dict(values)

    teams = {team for values in normalised.values() for team in values}
    out: dict[str, BlendedRating] = {}

    for team in teams:
        weighted_sum = 0.0
        weight_total = 0.0
        used: list[str] = []
        contributions: dict[str, float] = {}
        raw_values: list[float] = []

        for source, values in normalised.items():
            if team not in values:
                continue
            weight = float(weights.get(source, 1.0))
            if weight <= 0:
                continue
            value = values[team]
            weighted_sum += value * weight
            weight_total += weight
            used.append(source)
            contributions[source] = value
            raw_values.append(value)

        if weight_total <= 0:
            continue

        rating = weighted_sum / weight_total
        dispersion = statistics.pstdev(raw_values) if len(raw_values) > 1 else 0.0

        if shrink_to_mean and len(used) < min_sources_for_full_confidence:
            # One lone source is a weak signal; pull it toward average
            # rather than betting a full-confidence number on it.
            factor = len(used) / max(1, min_sources_for_full_confidence)
            rating *= factor

        out[team] = BlendedRating(
            team=team,
            rating=rating,
            sources=sorted(used),
            contributions=contributions,
            dispersion=dispersion,
            weight_total=weight_total,
        )

    return out


def confidence_from_blend(
    blend: BlendedRating, *, min_sources: int = 2, dispersion_scale: float = 8.0
) -> float:
    """Map source count and disagreement onto a 0-1 confidence multiplier."""
    coverage = min(1.0, blend.n_sources / max(1, min_sources))
    # Wide disagreement between systems erodes confidence smoothly.
    agreement = 1.0 / (1.0 + (blend.dispersion / dispersion_scale) ** 2)
    return max(0.15, coverage * agreement)


def describe_blend(blend: BlendedRating) -> str:
    """One-line explanation of where a team's number came from."""
    parts = ", ".join(
        f"{src}={val:+.1f}" for src, val in sorted(blend.contributions.items())
    )
    return f"{blend.team}: {blend.rating:+.2f} [{parts}] spread={blend.dispersion:.1f}"
