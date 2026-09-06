"""Jeff Sagarin's college football page (sagarin.com/sports/cfsend.htm).

The page carries two things worth having:

1. **Ratings** — the headline rating plus the PREDICTOR, GOLDEN_MEAN and
   RECENT sub-ratings. PREDICTOR is the one built for point spreads.
2. **Predictions with Totals and Moneylines** — Sagarin's own projected
   margin, total and moneyline for every upcoming game. That is a
   ready-made expert projection, and it is blended alongside the
   ratings-derived number rather than being treated as a market price.

The page is plain text wrapped in ``<pre>``. Because its exact column
layout shifts between seasons (and cannot be pinned by a schema), the
parsers here are **token-driven** rather than column-driven: they look
for the anchors Sagarin has always used — ``=`` for a rating, ``by`` for
a margin, a ``total``/``o/u`` label, and parenthesised moneylines — and
report how many rows they recovered so a format change is loud.

Both parsers work on a saved copy of the page, so no network is needed:

    cfbpicks import-sagarin path/to/cfsend.htm --season 2026 --week 3
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable, Optional

from ..models import ExpertProjection, Rating
from ..util.teams import canonical_team, game_key, is_shouted
from .base import Provider

DEFAULT_URL = "http://sagarin.com/sports/cfsend.htm"

_TAG_RE = re.compile(r"<[^>]+>")
_MONEYLINE_RE = re.compile(r"[-+]\d{3,5}")

# "  1  Ohio State           A =  96.11   14  2  0 ..."
# Rank and an optional letter grade are both optional; the "=" is not.
_RATING_RE = re.compile(
    r"^\s*(?P<rank>\d{1,3})?\s*(?P<team>[A-Za-z][A-Za-z.'&()\-/ ]*?)\s+"
    r"(?:[A-Z]{1,3}\s*)?=\s*(?P<rating>-?\d+\.\d+)",
    re.MULTILINE,
)

# The predictions block: "TEAM_A by 10.23 over Team B  total 54.32 ..."
# or "TEAM_A  10.23  Team B  54.32". Only the first form is unambiguous,
# so the second is handled by the numeric fallback below.
_PRED_BY_RE = re.compile(
    r"(?P<first>[A-Za-z][A-Za-z.'&()\-/ ]*?)\s+by\s+(?P<margin>-?\d+\.?\d*)\s+"
    r"(?:over\s+|vs\.?\s+|-\s*)?(?P<second>[A-Za-z][A-Za-z.'&()\-/ ]*?)"
    r"(?=\s{2,}|\s+total|\s+o/u|\s*\(|\s*$)",
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(r"(?:total|o/u)\s*[:=]?\s*(?P<total>\d{2,3}\.?\d*)", re.IGNORECASE)

_SECTION_PRED = re.compile(r"PREDICTIONS?[_\s]+WITH[_\s]+TOTALS", re.IGNORECASE)
_SECTION_END = re.compile(r"^_{10,}|^={10,}", re.MULTILINE)

#: Lines that are headers, legends or dividers rather than data.
_NOISE = re.compile(
    r"^\s*(?:_{4,}|={4,}|-{4,}|HOME|RATING|Copyright|Jeff Sagarin|\|)", re.IGNORECASE
)


def strip_html(raw: str) -> str:
    """Turn the page into the plain text it really is."""
    text = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    return html.unescape(text)


class SagarinParseResult:
    """Parsed page plus counts, so callers can tell partial from empty."""

    def __init__(
        self,
        ratings: list[Rating],
        projections: list[ExpertProjection],
        lines_seen: int,
    ) -> None:
        self.ratings = ratings
        self.projections = projections
        self.lines_seen = lines_seen

    @property
    def ok(self) -> bool:
        return bool(self.ratings or self.projections)

    def summary(self) -> str:
        return (
            f"{len(self.ratings)} ratings, {len(self.projections)} projections "
            f"from {self.lines_seen} lines"
        )


def parse_ratings(text: str, season: int, week: int, *, recenter: bool = True) -> list[Rating]:
    """Extract the headline rating for each team.

    Sagarin's scale floats around 70-90 rather than being centred on
    zero, so ratings are recentred to points-above-average to match every
    other source in the blend.
    """
    out: list[Rating] = []
    seen: set[str] = set()

    for line in text.splitlines():
        if _NOISE.match(line):
            continue
        match = _RATING_RE.match(line)
        if not match:
            continue
        team = canonical_team(match.group("team"))
        if not team or team.lower() in seen:
            continue
        rating = float(match.group("rating"))
        # Ratings live in a plausible band; anything outside it is a
        # schedule strength or win total that happened to match.
        if not -50.0 <= rating <= 150.0:
            continue
        seen.add(team.lower())
        rank = match.group("rank")
        out.append(
            Rating(
                source="sagarin", season=season, week=week, team=team,
                rating=rating, rank=int(rank) if rank else None,
            )
        )

    if recenter and out:
        mean = sum(r.rating for r in out) / len(out)
        for rating in out:
            rating.rating -= mean
    return out


def parse_predictions(text: str, season: int, week: int) -> list[ExpertProjection]:
    """Extract Sagarin's projected margin / total / moneyline per game.

    Sagarin lists the favourite first and the underdog second, so the
    margin is applied to whichever side is the home team. Lines that name
    a neutral site, or where the home team cannot be identified, still
    yield a projection — the caller resolves home/away against the
    schedule, which is authoritative.
    """
    section = _predictions_section(text)
    out: list[ExpertProjection] = []

    for line in section.splitlines():
        if _NOISE.match(line) or len(line.strip()) < 12:
            continue
        match = _PRED_BY_RE.search(line)
        if not match:
            continue

        fav_raw, dog_raw = match.group("first"), match.group("second")
        favourite = canonical_team(fav_raw)
        underdog = canonical_team(dog_raw)
        if not favourite or not underdog or favourite == underdog:
            continue

        try:
            margin = float(match.group("margin"))
        except ValueError:
            continue
        if abs(margin) > 80:
            continue

        # Sagarin's own legend: the home team is printed in CAPITALS, and
        # the favourite is printed first. When neither side is shouted the
        # game is at a neutral site.
        fav_is_home = is_shouted(fav_raw)
        dog_is_home = is_shouted(dog_raw)
        if dog_is_home and not fav_is_home:
            home_team, away_team, home_margin = underdog, favourite, -margin
        else:
            home_team, away_team, home_margin = favourite, underdog, margin

        total_match = _TOTAL_RE.search(line)
        total = float(total_match.group("total")) if total_match else None
        if total is not None and not 20.0 <= total <= 120.0:
            total = None

        # Moneylines appear after the total; the favourite's comes first.
        tail = line[match.end():]
        prices = [float(p) for p in _MONEYLINE_RE.findall(tail)]
        fav_ml = prices[0] if len(prices) >= 1 else None
        dog_ml = prices[1] if len(prices) >= 2 else None

        neutral = bool(re.search(r"\bneutral\b", line, re.IGNORECASE)) or not (
            fav_is_home or dog_is_home
        )

        # Moneylines are printed favourite-first, so swap them whenever the
        # favourite turned out to be the road team.
        if home_team == underdog:
            fav_ml, dog_ml = dog_ml, fav_ml

        out.append(
            ExpertProjection(
                source="sagarin", season=season, week=week,
                game_id="",  # resolved against the schedule by the caller
                home_team=home_team, away_team=away_team,
                projected_margin=home_margin, projected_total=total,
                home_moneyline=fav_ml, away_moneyline=dog_ml,
                neutral_site=neutral,
            )
        )
    return out


def _predictions_section(text: str) -> str:
    """Slice out the predictions block, or fall back to the whole page."""
    match = _SECTION_PRED.search(text)
    if not match:
        return text
    rest = text[match.end():]
    end = _SECTION_END.search(rest, 200)
    return rest[: end.start()] if end else rest


def parse_page(raw: str, season: int, week: int) -> SagarinParseResult:
    """Parse a full saved page into ratings and projections."""
    text = strip_html(raw)
    return SagarinParseResult(
        ratings=parse_ratings(text, season, week),
        projections=parse_predictions(text, season, week),
        lines_seen=len(text.splitlines()),
    )


def resolve_projections(
    projections: Iterable[ExpertProjection], games: Iterable
) -> tuple[list[ExpertProjection], list[ExpertProjection]]:
    """Attach schedule game ids and orient margins to the home team.

    Sagarin lists the favourite first regardless of venue, so a projection
    whose "first" team is actually the road team has its margin flipped.
    Returns ``(resolved, unmatched)``.
    """
    by_pair: dict[frozenset[str], list] = {}
    for game in games:
        by_pair.setdefault(frozenset({game.home_team, game.away_team}), []).append(game)

    resolved: list[ExpertProjection] = []
    unmatched: list[ExpertProjection] = []

    for proj in projections:
        candidates = by_pair.get(frozenset({proj.home_team, proj.away_team}))
        if not candidates:
            unmatched.append(proj)
            continue
        game = candidates[0]
        margin = proj.projected_margin
        home_ml, away_ml = proj.home_moneyline, proj.away_moneyline
        if proj.home_team != game.home_team:
            # The favourite was the away team: flip to home perspective.
            if margin is not None:
                margin = -margin
            home_ml, away_ml = away_ml, home_ml
        resolved.append(
            ExpertProjection(
                source=proj.source, season=game.season, week=game.week,
                game_id=game.game_id, home_team=game.home_team, away_team=game.away_team,
                projected_margin=margin, projected_total=proj.projected_total,
                home_moneyline=home_ml, away_moneyline=away_ml,
                neutral_site=game.neutral_site,
            )
        )
    return resolved, unmatched


class SagarinProvider(Provider):
    name = "sagarin"
    description = "Sagarin ratings + his published projected spreads, totals and moneylines"
    requires_key = False
    provides_ratings = True

    @property
    def url(self) -> str:
        return self.settings.base_url or self.option("url", DEFAULT_URL)

    def load_text(self, season: int, week: int) -> str:
        """Read a saved page if one is configured, otherwise fetch it."""
        local = self.option("local_file")
        if local:
            path = self.config.path(local)
            if path.exists():
                return path.read_text(errors="replace")

        response = self.client.session.get(self.url, timeout=self.config.request_timeout)
        response.raise_for_status()
        return response.text

    def fetch_page(self, season: int, week: int) -> SagarinParseResult:
        return parse_page(self.load_text(season, week), season, week)

    def fetch_ratings(self, season: int, week: Optional[int] = None, **kwargs) -> list[Rating]:
        return self.fetch_page(season, week or 0).ratings

    def fetch_projections(self, season: int, week: Optional[int] = None) -> list[ExpertProjection]:
        return self.fetch_page(season, week or 0).projections


def parse_file(path: Path, season: int, week: int) -> SagarinParseResult:
    return parse_page(Path(path).read_text(errors="replace"), season, week)
