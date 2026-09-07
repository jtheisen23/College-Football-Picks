"""Forecast handling and its effect on totals."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cfbpicks.config import Config, ProviderConfig
from cfbpicks.models import Game, GameWeather
from cfbpicks.providers.weather import WeatherProvider, _coordinates, _sample_hour
from cfbpicks.weather import WeatherModel, total_adjustment


def w(**kwargs):
    return GameWeather("g1", **kwargs)


class TestEffectModel:
    def test_calm_weather_changes_nothing(self):
        assert total_adjustment(w(wind_mph=6, temperature_f=68))[0] == 0.0

    def test_no_forecast_invents_nothing(self):
        assert total_adjustment(None) == (0.0, [])

    def test_a_dome_is_never_adjusted(self):
        """Applying a wind penalty indoors is the obvious failure here."""
        indoors = w(wind_mph=40, precipitation_in=1.0, temperature_f=10, indoor=True)
        assert total_adjustment(indoors) == (0.0, [])

    def test_wind_lowers_the_total(self):
        adjustment, reasons = total_adjustment(w(wind_mph=25))
        assert adjustment < 0
        assert any("wind" in r for r in reasons)

    def test_stronger_wind_lowers_it_further(self):
        assert total_adjustment(w(wind_mph=30))[0] < total_adjustment(w(wind_mph=15))[0]

    def test_the_wind_term_is_capped(self):
        """A gale is not linearly worse than a stiff breeze."""
        model = WeatherModel()
        assert total_adjustment(w(wind_mph=80))[0] == pytest.approx(-model.wind_max_points)

    def test_precipitation_costs_more_when_heavy(self):
        light = total_adjustment(w(precipitation_in=0.08))[0]
        heavy = total_adjustment(w(precipitation_in=0.5))[0]
        assert heavy < light < 0

    def test_a_hard_freeze_costs_a_point(self):
        assert total_adjustment(w(temperature_f=15))[0] == -1.0
        assert total_adjustment(w(temperature_f=45))[0] == 0.0

    def test_effects_combine_but_stay_capped(self):
        model = WeatherModel()
        worst = w(wind_mph=60, precipitation_in=1.0, temperature_f=5)
        assert total_adjustment(worst, model)[0] == pytest.approx(-model.max_total_points)

    def test_reasons_explain_the_number(self):
        _, reasons = total_adjustment(w(wind_mph=22, precipitation_in=0.4, temperature_f=20))
        assert len(reasons) == 3

    def test_coefficients_are_configurable(self):
        gentle = WeatherModel(wind_points_per_mph=0.05)
        assert total_adjustment(w(wind_mph=30), gentle)[0] > total_adjustment(w(wind_mph=30))[0]


class TestForecastParsing:
    PAYLOAD = {
        "hourly": {
            "time": ["2026-09-19T21:00", "2026-09-19T22:00", "2026-09-19T23:00"],
            "temperature_2m": [72.0, 70.0, 68.0],
            "precipitation": [0.0, 0.0, 0.1],
            "wind_speed_10m": [8.0, 12.0, 18.0],
        }
    }

    def test_picks_the_hour_nearest_kickoff(self):
        kickoff = datetime(2026, 9, 19, 22, 10, tzinfo=timezone.utc)
        sample = _sample_hour(self.PAYLOAD, kickoff)
        assert sample["wind_mph"] == 12.0
        assert sample["temperature_f"] == 70.0

    def test_rounds_to_the_closer_hour(self):
        kickoff = datetime(2026, 9, 19, 22, 50, tzinfo=timezone.utc)
        assert _sample_hour(self.PAYLOAD, kickoff)["wind_mph"] == 18.0

    def test_a_kickoff_far_outside_the_forecast_returns_nothing(self):
        kickoff = datetime(2026, 9, 25, 22, 0, tzinfo=timezone.utc)
        assert _sample_hour(self.PAYLOAD, kickoff) is None

    def test_an_empty_payload_is_safe(self):
        kickoff = datetime(2026, 9, 19, 22, 0, tzinfo=timezone.utc)
        assert _sample_hour({}, kickoff) is None
        assert _sample_hour({"hourly": {"time": []}}, kickoff) is None


class TestVenueCoordinates:
    def test_cfbd_nests_them_under_location_as_x_and_y(self):
        # x is longitude and y is latitude, which is easy to get backwards.
        assert _coordinates({"location": {"x": -83.02, "y": 40.00}}) == (40.00, -83.02)

    def test_flat_keys_also_work(self):
        assert _coordinates({"latitude": 40.0, "longitude": -83.0}) == (40.0, -83.0)

    def test_missing_coordinates_are_none(self):
        assert _coordinates({}) == (None, None)

    def test_junk_coordinates_are_none(self):
        assert _coordinates({"latitude": "n/a", "longitude": "n/a"}) == (None, None)


@pytest.fixture()
def provider(tmp_path):
    config = Config(root=tmp_path)
    config.cache_dir = str(tmp_path / "cache")
    return WeatherProvider(config, ProviderConfig("weather"))


class TestProvider:
    def _venues(self, provider, dome=False):
        provider._venues = {
            "the horseshoe": {
                "name": "The Horseshoe", "latitude": 40.0,
                "longitude": -83.0, "dome": dome,
            }
        }

    def _game(self, hours_ahead=48, venue="The Horseshoe"):
        return Game("g1", 2026, 3, datetime.now(timezone.utc) + timedelta(hours=hours_ahead),
                    "Ohio State", "Michigan", venue=venue)

    def test_a_forecast_is_returned_for_a_known_venue(self, provider):
        self._venues(provider)
        provider.client.get_json = lambda *a, **k: {
            "hourly": {"time": [], "temperature_2m": [], "precipitation": [], "wind_speed_10m": []}
        }
        # Empty hourly data yields no forecast rather than a fabricated one.
        assert provider.fetch_weather(2026, 3, games=[self._game()]) == []

    def test_domes_are_recorded_without_a_lookup(self, provider):
        self._venues(provider, dome=True)
        called = []
        provider.client.get_json = lambda *a, **k: called.append(1) or {}
        forecasts = provider.fetch_weather(2026, 3, games=[self._game()])
        assert len(forecasts) == 1 and forecasts[0].indoor
        assert not called, "an indoor game needs no forecast at all"

    def test_unknown_venues_are_reported(self, provider):
        self._venues(provider)
        provider.client.get_json = lambda *a, **k: {}
        provider.fetch_weather(2026, 3, games=[self._game(venue="Somewhere Else")])
        assert any("no coordinates" in warning for warning in provider.warnings)

    def test_games_beyond_the_forecast_window_are_skipped(self, provider):
        self._venues(provider)
        called = []
        provider.client.get_json = lambda *a, **k: called.append(1) or {}
        assert provider.fetch_weather(2026, 3, games=[self._game(hours_ahead=24 * 40)]) == []
        assert not called

    def test_games_without_a_kickoff_are_skipped(self, provider):
        self._venues(provider)
        game = Game("g1", 2026, 3, None, "Ohio State", "Michigan", venue="The Horseshoe")
        assert provider.fetch_weather(2026, 3, games=[game]) == []

    def test_no_cfbd_key_means_no_coordinates_and_a_warning(self, provider):
        assert provider.venue_index(2026) == {}
        assert any("no CFBD key" in warning for warning in provider.warnings)


class TestEffectOnProjections:
    def test_wind_lowers_a_projected_total(self, tmp_path):
        from cfbpicks.providers.fixtures import PACKAGE_FIXTURES
        from cfbpicks.pipeline import Pipeline

        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.fixtures_dir = str(PACKAGE_FIXTURES)
        cfg.providers = {"fixtures": ProviderConfig("fixtures", enabled=True)}

        with Pipeline(cfg) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            before = {p.game_id: p.projected_total
                      for p in pipeline.predict(2026, 3, use_projections=False)}
            target = next(iter(before))
            pipeline.storage.upsert_weather([GameWeather(target, wind_mph=30.0)])
            after = {p.game_id: p.projected_total
                     for p in pipeline.predict(2026, 3, use_projections=False)}

        assert after[target] < before[target]
        # Everything else is untouched.
        assert all(after[g] == before[g] for g in before if g != target)
