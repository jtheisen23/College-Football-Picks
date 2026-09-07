"""Kickoff forecasts, via Open-Meteo.

Two pieces are needed: where the stadium is, and what the weather will
do there. CFBD's venue list supplies coordinates and — importantly — a
dome flag, and Open-Meteo supplies the forecast. Open-Meteo needs no API
key and no account, which is why it is the default here.

Forecasts only reach about two weeks out, so this is a thing you run
during the week of the games, not in August.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from ..models import Game, GameWeather
from ..util.http import ProviderError
from .base import Provider
from .cfbd import CfbdProvider, pick

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

#: Hourly fields requested, in units a US bettor thinks in.
HOURLY_FIELDS = "temperature_2m,precipitation,wind_speed_10m"


class WeatherProvider(Provider):
    name = "weather"
    description = "Open-Meteo kickoff forecasts (no key), with CFBD venue coordinates"
    requires_key = False
    provides_weather = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warnings: list[str] = []
        self._venues: Optional[dict[str, dict]] = None

    # -- venues --------------------------------------------------------------
    def venue_index(self, season: int) -> dict[str, dict]:
        """Stadium name -> coordinates and dome flag, from CFBD.

        Cached for the life of the provider, and on disk indefinitely:
        stadiums do not move.
        """
        if self._venues is not None:
            return self._venues

        cfbd = CfbdProvider(self.config, self.config.provider("cfbd"))
        if not (cfbd.enabled and cfbd.configured):
            self.warnings.append(
                "no CFBD key, so stadium coordinates are unavailable — "
                "weather needs them to know where to look"
            )
            self._venues = {}
            return self._venues

        try:
            rows = cfbd.fetch_venues(season)
        except Exception as exc:  # noqa: BLE001 - any failure means no weather
            self.warnings.append(f"could not load venues: {exc}")
            self._venues = {}
            return self._venues

        index: dict[str, dict] = {}
        for row in rows or []:
            name = str(pick(row, "name", default="")).strip()
            if not name:
                continue
            latitude, longitude = _coordinates(row)
            index[name.lower()] = {
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "dome": bool(pick(row, "dome", default=False)),
            }
        self._venues = index
        return index

    # -- forecasts -------------------------------------------------------------
    def fetch_weather(
        self, season: int, week: Optional[int] = None, *, games: Iterable[Game] = (), **kwargs
    ) -> list[GameWeather]:
        self.warnings = []
        venues = self.venue_index(season)
        horizon = datetime.now(timezone.utc) + timedelta(
            days=int(self.option("forecast_horizon_days", 16))
        )

        out: list[GameWeather] = []
        missing_venues: set[str] = set()

        for game in games:
            if game.kickoff is None:
                continue
            if game.kickoff > horizon:
                # Beyond the forecast window there is nothing to fetch;
                # silently skipping is right, inventing a calm day is not.
                continue

            venue = venues.get(str(game.venue or "").lower())
            if venue is None:
                if game.venue:
                    missing_venues.add(game.venue)
                continue

            if venue["dome"]:
                out.append(GameWeather(
                    game_id=game.game_id, kickoff=game.kickoff, indoor=True,
                    venue=venue["name"], source=self.name,
                ))
                continue

            if venue["latitude"] is None or venue["longitude"] is None:
                missing_venues.add(venue["name"])
                continue

            try:
                conditions = self._forecast(venue, game.kickoff)
            except ProviderError as exc:
                self.warnings.append(f"{venue['name']}: {exc}")
                continue
            if conditions is None:
                continue

            out.append(GameWeather(
                game_id=game.game_id, kickoff=game.kickoff, venue=venue["name"],
                source=self.name, **conditions,
            ))

        if missing_venues:
            self.warnings.append(
                f"no coordinates for {len(missing_venues)} venue(s): "
                + ", ".join(sorted(missing_venues)[:5])
            )
        return out

    def _forecast(self, venue: dict, kickoff: datetime) -> Optional[dict[str, float]]:
        """Hourly forecast for the stadium, sampled at the kickoff hour."""
        day = kickoff.date().isoformat()
        payload = self.client.get_json(
            OPEN_METEO_URL,
            params={
                "latitude": round(float(venue["latitude"]), 3),
                "longitude": round(float(venue["longitude"]), 3),
                "hourly": HOURLY_FIELDS,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "UTC",
                "start_date": day,
                "end_date": day,
            },
            ttl=int(self.option("cache_ttl_seconds", 3600)),
        )
        return _sample_hour(payload, kickoff)


def _sample_hour(payload: Any, kickoff: datetime) -> Optional[dict[str, float]]:
    """Pick the forecast hour closest to kickoff."""
    hourly = (payload or {}).get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None

    target = kickoff.astimezone(timezone.utc).replace(tzinfo=None)
    best_index, best_gap = None, None
    for i, stamp in enumerate(times):
        try:
            when = datetime.fromisoformat(str(stamp))
        except ValueError:
            continue
        gap = abs((when - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best_index, best_gap = i, gap

    if best_index is None or (best_gap is not None and best_gap > 6 * 3600):
        return None

    def at(field: str) -> Optional[float]:
        values = hourly.get(field) or []
        if best_index >= len(values):
            return None
        value = values[best_index]
        return float(value) if value is not None else None

    return {
        "temperature_f": at("temperature_2m"),
        "wind_mph": at("wind_speed_10m"),
        "precipitation_in": at("precipitation"),
    }


def _coordinates(row: dict) -> tuple[Optional[float], Optional[float]]:
    """CFBD nests coordinates under `location` as x=longitude, y=latitude."""
    location = pick(row, "location", default={}) or {}
    latitude = pick(row, "latitude") or (location.get("y") if isinstance(location, dict) else None)
    longitude = pick(row, "longitude") or (location.get("x") if isinstance(location, dict) else None)
    try:
        return (
            float(latitude) if latitude is not None else None,
            float(longitude) if longitude is not None else None,
        )
    except (TypeError, ValueError):
        return None, None
