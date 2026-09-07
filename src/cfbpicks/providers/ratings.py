"""Third-party expert / computer power ratings (Massey, Sagarin, and friends).

These publishers have no supported API. Rather than hard-code a scraper
that silently rots, this provider is driven by configuration: each source
declares a URL and a format, and anything it cannot parse is reported
loudly instead of being dropped.

There is always a local escape hatch — drop a CSV or JSON file into
``data/ratings/`` and it is ingested with no network at all:

    data/ratings/massey-2026-week3.csv
    team,rating
    Georgia,24.1
    Alabama,21.7

The filename supplies the source, season and week when the file itself
does not carry them.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from ..models import Rating
from ..util.http import ProviderError
from ..util.htmltable import find_table_with, largest_table
from ..util.teams import canonical_team
from .base import Provider

#: Ratings whose scale is *not* points-above-average get normalised in
#: the blender; anything listed here is already on a points scale.
POINTS_SCALE_SOURCES = {"massey", "sagarin", "manual"}

_FILENAME_RE = re.compile(
    r"^(?P<source>[a-z0-9_+-]+?)[-_](?P<season>\d{4})(?:[-_]w(?:eek)?(?P<week>\d{1,2}))?",
    re.IGNORECASE,
)

# Sagarin's plain-text table: rank, team, an optional letter grade, then
# "=" and the rating. Team names contain spaces, hence the lazy capture.
_SAGARIN_RE = re.compile(
    r"^\s*(?P<rank>\d+)\s+(?P<team>.+?)\s+[A-Z]?\s*=\s*(?P<rating>-?\d+\.\d+)",
    re.MULTILINE,
)


class RatingsProvider(Provider):
    name = "ratings"
    description = "Expert & computer power ratings (Massey, Sagarin, custom feeds, local files)"
    requires_key = False
    provides_ratings = True

    def fetch_ratings(self, season: int, week: Optional[int] = None, **kwargs) -> list[Rating]:
        out: list[Rating] = []
        self.warnings: list[str] = []

        # Local files first: they are free, deterministic, and let a user
        # override a broken remote source without touching config.
        out.extend(self.load_local(season, week))

        for spec in self._source_specs():
            if not spec.get("url"):
                continue
            try:
                out.extend(self._fetch_remote(spec, season, week))
            except (ProviderError, ValueError) as exc:
                self.warnings.append(f"{spec.get('name', 'unknown')}: {exc}")
        return out

    # -- configuration ------------------------------------------------------
    def _source_specs(self) -> list[dict[str, Any]]:
        """Normalise the ``sources`` option into full specs.

        Accepts either a bare list of names (``["massey", "sagarin"]``) or
        a list of mappings with ``url``/``format``/``columns``.
        """
        raw = self.option("sources", []) or []
        specs: list[dict[str, Any]] = []
        for entry in raw:
            if isinstance(entry, str):
                specs.append({"name": entry, **_BUILTIN_SOURCES.get(entry, {})})
            elif isinstance(entry, dict):
                name = entry.get("name", "custom")
                merged = {"name": name, **_BUILTIN_SOURCES.get(name, {}), **entry}
                specs.append(merged)
        return specs

    # -- remote -------------------------------------------------------------
    def _fetch_remote(self, spec: dict[str, Any], season: int, week: Optional[int]) -> list[Rating]:
        url = str(spec["url"]).format(season=season, week=week or 0)
        fmt = str(spec.get("format", "json")).lower()
        source = spec.get("name", "custom")

        if fmt in {"html", "html_table", "csv", "text"}:
            rows = _rows_from_text(self._get_text(url), fmt, spec)
        else:
            payload = self.client.get_json(url, ttl=self.config.cache_ttl_seconds)
            rows = _rows_from_payload(payload, spec)
        return list(self._to_ratings(rows, source, season, week or 0, spec))

    def _get_text(self, url: str) -> str:
        """Fetch a non-JSON document, honouring offline mode."""
        if self.config.offline:
            raise ProviderError(f"offline mode: cannot fetch {url}")
        response = self.client.session.get(url, timeout=self.config.request_timeout)
        if response.status_code != 200:
            raise ProviderError(f"{url} returned {response.status_code}")
        return response.text

    def _to_ratings(
        self,
        rows: Iterable[dict[str, Any]],
        source: str,
        season: int,
        week: int,
        spec: dict[str, Any],
    ) -> Iterable[Rating]:
        columns = spec.get("columns") or {}
        team_key = columns.get("team", "team")
        rating_key = columns.get("rating", "rating")
        scale = float(spec.get("scale", 1.0))

        for row in rows:
            team_raw = row.get(team_key)
            rating_raw = row.get(rating_key)
            if team_raw is None or rating_raw is None:
                continue
            try:
                value = float(str(rating_raw).strip()) * scale
            except ValueError:
                continue
            team = canonical_team(str(team_raw))
            if not team:
                continue
            yield Rating(
                source=source,
                season=season,
                week=week,
                team=team,
                rating=value,
                offense=_maybe_float(row.get(columns.get("offense", "offense"))),
                defense=_maybe_float(row.get(columns.get("defense", "defense"))),
                rank=_maybe_int(row.get(columns.get("rank", "rank"))),
            )

    # -- local files --------------------------------------------------------
    def load_local(self, season: Optional[int] = None, week: Optional[int] = None) -> list[Rating]:
        """Ingest every CSV/JSON rating file in the configured input dir."""
        input_dir = self.config.path(self.option("input_dir", "data/ratings"))
        if not input_dir.exists():
            return []

        out: list[Rating] = []
        for path in sorted(input_dir.iterdir()):
            if path.suffix.lower() not in {".csv", ".json", ".html", ".htm"}:
                continue
            meta = _meta_from_filename(path)
            file_season = meta.get("season") or season
            if file_season is None:
                continue
            if season is not None and file_season != season:
                continue
            file_week = meta.get("week")
            if file_week is None:
                file_week = week or 0
            source = meta.get("source") or path.stem

            rows = _read_local_rows(path)
            spec = {"name": source, "columns": {}}
            out.extend(self._to_ratings(rows, source, int(file_season), int(file_week), spec))
        return out

    def parse_sagarin(self, text: str, season: int, week: int) -> list[Rating]:
        """Parse Sagarin's fixed-width text table into ratings.

        Exposed separately so a saved copy of the page can be ingested
        offline: ``cfbpicks import-ratings --format sagarin file.txt``.
        """
        out: list[Rating] = []
        for match in _SAGARIN_RE.finditer(text):
            team = canonical_team(match.group("team"))
            if not team:
                continue
            out.append(
                Rating(
                    source="sagarin", season=season, week=week, team=team,
                    rating=float(match.group("rating")), rank=int(match.group("rank")),
                )
            )
        # Sagarin's numbers sit around 70-90; recentre to points-above-average.
        if out:
            mean = sum(r.rating for r in out) / len(out)
            for rating in out:
                rating.rating -= mean
        return out


#: Known publishers. URLs are configurable because these pages move.
_BUILTIN_SOURCES: dict[str, dict[str, Any]] = {
    "massey": {
        # masseyratings.com/ranks renders its table client-side, so the
        # dependable path is to save the page and drop it in data/ratings/
        # (or point `url` at whichever JSON endpoint the page calls).
        "url": None,
        "format": "html_table",
        "columns": {"team": "team", "rating": "rating"},
        "note": "Save https://masseyratings.com/ranks as data/ratings/massey-<season>-w<week>.html",
    },
    "sagarin": {
        # Sagarin publishes HTML only; the practical path is to save the
        # page and run `cfbpicks import-ratings --format sagarin`.
        "url": None,
        "format": "text",
        "note": "No stable JSON endpoint. Use import-ratings with a saved page.",
    },
}


def _rows_from_payload(payload: Any, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce whatever a source returns into a list of row dicts."""
    root = spec.get("root")
    if isinstance(payload, dict) and root:
        payload = payload.get(root)
    if isinstance(payload, dict):
        # Pick the first list-of-things value, a common shape for these feeds.
        for value in payload.values():
            if isinstance(value, list) and value:
                payload = value
                break
    if not isinstance(payload, list):
        raise ValueError("expected a list of rating rows")

    if payload and isinstance(payload[0], dict):
        return payload
    if payload and isinstance(payload[0], list):
        headers = spec.get("headers")
        if not headers:
            raise ValueError("array-of-arrays payload needs a `headers` list in config")
        return [dict(zip(headers, row)) for row in payload]
    raise ValueError("unrecognised rating payload shape")


def _rows_from_text(text: str, fmt: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a text document (CSV or an HTML page) into row dicts."""
    if fmt == "csv" or (fmt == "text" and "," in text.split("\n", 1)[0]):
        return list(csv.DictReader(io.StringIO(text)))

    columns = spec.get("columns") or {}
    required = [v for v in (columns.get("team", "team"), columns.get("rating", "rating"))]
    rows = find_table_with(text, *required)
    if not rows:
        rows = largest_table(text)
    if not rows:
        raise ValueError("no parsable table found in document")
    return rows


def _read_local_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        rows = largest_table(text)
        if not rows:
            raise ValueError(f"no table found in {path.name}")
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return value
            return []
        return data if isinstance(data, list) else []
    return list(csv.DictReader(io.StringIO(text)))


def _meta_from_filename(path: Path) -> dict[str, Any]:
    match = _FILENAME_RE.match(path.stem)
    if not match:
        return {"source": path.stem}
    return {
        "source": match.group("source").lower(),
        "season": int(match.group("season")),
        "week": int(match.group("week")) if match.group("week") else None,
    }


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
