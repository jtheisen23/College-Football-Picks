"""Manual adjustments for things the ratings cannot see.

A power rating is a summary of games already played. It does not know
that a starting quarterback tore an ACL on Tuesday, that three offensive
linemen are suspended, or that a team is playing its third road game in
twelve days. The market prices all of that within minutes. This is the
channel for feeding it in.

The file is plain YAML, one entry per adjustment::

    - season: 2026
      week: 3
      team: Georgia
      out: [qb1]
      reason: starting QB out (shoulder)

    - season: 2026
      week: 3
      team: Alabama
      points: -2.5
      reason: three OL starters suspended

    - season: 2026
      week: 3
      game: Michigan @ Ohio State
      total: -4
      reason: 25mph crosswind forecast

``out`` uses the position table below for its point values; ``points``
sets a team's adjustment directly. Game-scoped entries shift the
projected ``margin`` or ``total`` for that one game instead.

Every applied adjustment is attached to the prediction and surfaced on
any bet it touches, because a number that moved for a reason you cannot
see is worse than no adjustment at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from .util.teams import canonical_team

#: What a missing player is worth, in points of team strength.
#:
#: The starting quarterback dominates: the widely cited range in college
#: football is 5-9 points, far above any other position, because backup
#: quality varies enormously and the position touches every snap. The
#: rest are deliberately modest -- a single missing lineman or receiver
#: is worth about a point, and stacking three of them should not move a
#: line by a touchdown.
POSITION_VALUES: dict[str, float] = {
    "qb1": -7.0,     # starting quarterback
    "qb2": -1.0,     # backup, i.e. the third-stringer is now next
    "rb1": -1.0,
    "wr1": -1.5,
    "te1": -0.75,
    "ol": -1.0,      # one offensive line starter
    "dl": -1.0,
    "lb": -0.75,
    "db": -0.75,
    "k": -0.5,
    "coach": -1.5,   # head coach or coordinator absent
}


@dataclass
class Override:
    """One manual adjustment, as written in the file."""

    season: int
    week: int
    reason: str = ""
    team: Optional[str] = None
    points: Optional[float] = None
    out: list[str] = field(default_factory=list)
    game: Optional[str] = None
    margin: Optional[float] = None
    total: Optional[float] = None

    @property
    def is_team_scoped(self) -> bool:
        return self.team is not None

    def team_points(self) -> float:
        """Total points this entry moves its team by."""
        if self.points is not None:
            return float(self.points)
        return sum(POSITION_VALUES.get(str(p).lower(), 0.0) for p in self.out)

    def describe(self) -> str:
        label = self.reason or ("; ".join(self.out) if self.out else "manual adjustment")
        if self.is_team_scoped:
            return f"{self.team} {self.team_points():+.1f}: {label}"
        parts = []
        if self.margin is not None:
            parts.append(f"margin {self.margin:+.1f}")
        if self.total is not None:
            parts.append(f"total {self.total:+.1f}")
        return f"{self.game} {', '.join(parts)}: {label}"


class OverrideError(ValueError):
    """Raised when the overrides file cannot be understood."""


@dataclass
class Adjustments:
    """Resolved adjustments for one week, ready for the predictor."""

    team_points: dict[str, float] = field(default_factory=dict)
    team_reasons: dict[str, list[str]] = field(default_factory=dict)
    game_margin: dict[str, float] = field(default_factory=dict)
    game_total: dict[str, float] = field(default_factory=dict)
    game_reasons: dict[str, list[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.team_points or self.game_margin or self.game_total)

    def for_team(self, team: str) -> float:
        return self.team_points.get(team, 0.0)

    def notes_for(self, home: str, away: str, game_id: str) -> list[str]:
        """Everything applied to one game, for display."""
        notes: list[str] = []
        for team in (home, away):
            notes.extend(self.team_reasons.get(team, []))
        notes.extend(self.game_reasons.get(game_id, []))
        return notes


def load_overrides(path: Path | str) -> list[Override]:
    """Read and validate the overrides file. Missing file means none."""
    path = Path(path)
    if not path.exists():
        return []

    try:
        raw = yaml.safe_load(path.read_text()) or []
    except yaml.YAMLError as exc:
        raise OverrideError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, list):
        raise OverrideError(f"{path} must contain a list of adjustments")

    return [_parse(entry, path, i) for i, entry in enumerate(raw, start=1)]


def _parse(entry: Any, path: Path, index: int) -> Override:
    where = f"{path} entry {index}"
    if not isinstance(entry, dict):
        raise OverrideError(f"{where}: expected a mapping, got {type(entry).__name__}")

    known = {"season", "week", "team", "points", "out", "reason", "game", "margin", "total"}
    unknown = set(entry) - known
    if unknown:
        raise OverrideError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")

    for required in ("season", "week"):
        if required not in entry:
            raise OverrideError(f"{where}: missing '{required}'")

    has_team = entry.get("team") is not None
    has_game = entry.get("game") is not None
    if has_team == has_game:
        raise OverrideError(f"{where}: set exactly one of 'team' or 'game'")

    out = entry.get("out") or []
    if isinstance(out, str):
        out = [out]
    unknown_positions = [p for p in out if str(p).lower() not in POSITION_VALUES]
    if unknown_positions:
        raise OverrideError(
            f"{where}: unknown position(s) {', '.join(unknown_positions)}. "
            f"Known: {', '.join(sorted(POSITION_VALUES))}"
        )

    if has_team and entry.get("points") is None and not out:
        raise OverrideError(f"{where}: a team entry needs 'points' or 'out'")
    if has_game and entry.get("margin") is None and entry.get("total") is None:
        raise OverrideError(f"{where}: a game entry needs 'margin' or 'total'")

    return Override(
        season=int(entry["season"]),
        week=int(entry["week"]),
        reason=str(entry.get("reason") or ""),
        team=canonical_team(str(entry["team"])) if has_team else None,
        points=_as_float(entry.get("points")),
        out=[str(p).lower() for p in out],
        game=str(entry["game"]) if has_game else None,
        margin=_as_float(entry.get("margin")),
        total=_as_float(entry.get("total")),
    )


def resolve(
    overrides: Iterable[Override], season: int, week: int, games: Iterable
) -> tuple[Adjustments, list[str]]:
    """Turn file entries into adjustments for one week's games.

    Returns the adjustments plus warnings for entries that matched
    nothing — a typo'd team name silently doing nothing is exactly the
    failure this whole feature exists to avoid.
    """
    adjustments = Adjustments()
    warnings: list[str] = []
    games = list(games)

    relevant = [o for o in overrides if o.season == season and o.week == week]
    if relevant and not games:
        # Every entry would look orphaned, which reads as four typos
        # rather than the one real problem: nothing has been fetched.
        return adjustments, [
            f"no games stored for {season} week {week}, so {len(relevant)} "
            "adjustment(s) could not be applied — fetch the week first"
        ]

    by_pair: dict[frozenset[str], Any] = {}
    teams: set[str] = set()
    for game in games:
        by_pair[frozenset({game.home_team, game.away_team})] = game
        teams.add(game.home_team)
        teams.add(game.away_team)

    for override in overrides:
        if override.season != season or override.week != week:
            continue

        if override.is_team_scoped:
            if override.team not in teams:
                warnings.append(
                    f"no game this week for {override.team!r} — adjustment ignored"
                )
                continue
            points = override.team_points()
            if points == 0.0:
                continue
            adjustments.team_points[override.team] = (
                adjustments.team_points.get(override.team, 0.0) + points
            )
            adjustments.team_reasons.setdefault(override.team, []).append(
                override.describe()
            )
            continue

        game = _find_game(override.game or "", by_pair)
        if game is None:
            warnings.append(f"no game matching {override.game!r} — adjustment ignored")
            continue
        if override.margin is not None:
            adjustments.game_margin[game.game_id] = (
                adjustments.game_margin.get(game.game_id, 0.0) + override.margin
            )
        if override.total is not None:
            adjustments.game_total[game.game_id] = (
                adjustments.game_total.get(game.game_id, 0.0) + override.total
            )
        adjustments.game_reasons.setdefault(game.game_id, []).append(override.describe())

    return adjustments, warnings


def _find_game(label: str, by_pair: dict[frozenset[str], Any]):
    """Match 'Away @ Home' (or 'A vs B') to a scheduled game."""
    for separator in (" @ ", " at ", " vs ", " v "):
        if separator in label:
            left, right = label.split(separator, 1)
            pair = frozenset({canonical_team(left), canonical_team(right)})
            return by_pair.get(pair)
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OverrideError(f"{value!r} is not a number") from exc
