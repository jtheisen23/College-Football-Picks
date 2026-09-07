"""Guards against hindsight leaking into a backtest.

A backtest that quietly uses end-of-season ratings will report a
spectacular, entirely fake edge. These tests exist because that exact
failure produced a 57.9% ATS win rate and +20% ROI before it was caught.
"""

from __future__ import annotations

import pytest

from cfbpicks.backtest import LookaheadError, backtest_season
from cfbpicks.config import Config, ProviderConfig
from cfbpicks.models import Game, MarketQuote, Rating
from cfbpicks.storage import Storage


@pytest.fixture()
def storage(tmp_path):
    store = Storage(tmp_path / "t.sqlite")
    yield store
    store.close()


class TestRatingProvenance:
    def test_flag_round_trips(self, storage):
        storage.upsert_ratings([
            Rating("cfbd_elo", 2025, 3, "Georgia", 10.0, point_in_time=True),
            Rating("cfbd_sp_plus", 2025, 0, "Georgia", 20.0, point_in_time=False),
        ])
        by_source = {r.source: r.point_in_time for r in storage.ratings(2025)}
        assert by_source == {"cfbd_elo": True, "cfbd_sp_plus": False}

    def test_season_final_ratings_can_be_filtered_out(self, storage):
        storage.upsert_ratings([
            Rating("cfbd_elo", 2025, 3, "Georgia", 10.0, point_in_time=True),
            Rating("cfbd_sp_plus", 2025, 0, "Georgia", 20.0, point_in_time=False),
        ])
        safe = storage.ratings(2025, point_in_time_only=True)
        assert [r.source for r in safe] == ["cfbd_elo"]

    def test_unfiltered_reads_still_see_everything(self, storage):
        """Live picks legitimately use the current season-level number."""
        storage.upsert_ratings([
            Rating("cfbd_sp_plus", 2026, 0, "Georgia", 20.0, point_in_time=False),
        ])
        assert len(storage.ratings(2026)) == 1

    def test_old_databases_are_migrated_to_the_unsafe_default(self, tmp_path):
        # A database created before the column existed can't prove its
        # ratings are point-in-time, so it must assume they are not.
        import sqlite3

        path = tmp_path / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE ratings (
                source TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
                team TEXT NOT NULL, rating REAL NOT NULL, offense REAL, defense REAL,
                rank INTEGER, updated_at TEXT NOT NULL,
                PRIMARY KEY (source, season, week, team)
            );
            INSERT INTO ratings VALUES ('cfbd_sp_plus',2025,0,'Georgia',20.0,NULL,NULL,NULL,'x');
            """
        )
        conn.commit()
        conn.close()

        store = Storage(path)
        assert store.ratings(2025, point_in_time_only=True) == []
        assert len(store.ratings(2025)) == 1
        store.close()


def _season_with_odds(storage, *, point_in_time: bool):
    """A tiny completed season with lines, priced by one rating source."""
    games, quotes, ratings = [], [], []
    for i in range(6):
        gid = f"g{i}"
        games.append(Game(gid, 2025, 1 + i % 3, None, f"Home{i}", f"Away{i}",
                          home_score=30, away_score=17))
        quotes += [
            MarketQuote(gid, "DK", "spread", "home", -3.0, -110),
            MarketQuote(gid, "DK", "spread", "away", 3.0, -110),
        ]
        ratings += [
            Rating("src", 2025, 0, f"Home{i}", 18.0, point_in_time=point_in_time),
            Rating("src", 2025, 0, f"Away{i}", -8.0, point_in_time=point_in_time),
        ]
    storage.upsert_games(games)
    storage.upsert_quotes(quotes)
    storage.upsert_ratings(ratings)


class TestBacktestRefusesHindsight:
    def test_refuses_when_only_season_final_ratings_exist(self, storage):
        _season_with_odds(storage, point_in_time=False)
        with pytest.raises(LookaheadError, match="season-final"):
            backtest_season(storage, Config(), 2025)

    def test_runs_on_point_in_time_ratings(self, storage):
        _season_with_odds(storage, point_in_time=True)
        result = backtest_season(storage, Config(), 2025)
        assert result.rating_sources == ["src"]
        assert result.excluded_sources == []

    def test_opt_in_override_runs_anyway(self, storage):
        _season_with_odds(storage, point_in_time=False)
        result = backtest_season(storage, Config(), 2025, allow_final_ratings=True)
        assert result.excluded_sources == ["src"]

    def test_excluded_sources_are_named(self, storage):
        _season_with_odds(storage, point_in_time=True)
        storage.upsert_ratings([
            Rating("cfbd_sp_plus", 2025, 0, "Home0", 25.0, point_in_time=False),
        ])
        result = backtest_season(storage, Config(), 2025)
        assert result.excluded_sources == ["cfbd_sp_plus"]
        assert "src" in result.rating_sources


class TestClvHonesty:
    def test_single_snapshot_reports_no_clv_rather_than_zero(self, storage):
        _season_with_odds(storage, point_in_time=True)
        result = backtest_season(storage, Config(), 2025)
        # Odds were captured once, so every bet's "closing" line is its own.
        assert result.avg_clv is None
        assert result.clv_unavailable == result.bets
        assert any("n/a" in line for line in result.summary_lines())

    def test_snapshot_count_tracks_captures(self, storage):
        q = MarketQuote("g1", "DK", "spread", "home", -3.0, -110)
        storage.upsert_quotes([q])
        assert storage.snapshot_count("g1", "spread") == 1
        storage.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -3.5, -110)])
        assert storage.snapshot_count("g1", "spread") == 2


class TestLiveOnlySources:
    def test_sagarin_refuses_a_past_season(self):
        from cfbpicks.providers.sagarin import (
            HistoricalDataUnavailable, SagarinProvider, current_season,
        )

        provider = SagarinProvider(Config(), ProviderConfig("sagarin"))
        with pytest.raises(HistoricalDataUnavailable, match="nothing for 2019"):
            provider.fetch_ratings(2019)
        # And it does not reach the network to find that out.
        assert current_season() >= 2024


class TestCfbdProvenanceLabels:
    """SP+ and SRS are season-level; Elo is week-indexed."""

    def _rows(self, system, payload, monkeypatch):
        from cfbpicks.providers.cfbd import CfbdProvider

        provider = CfbdProvider(Config(), ProviderConfig("cfbd", api_key="x"))
        monkeypatch.setattr(provider, "_get", lambda *a, **k: payload)
        return provider._fetch_rating_system(system, 2025, 3)

    def test_sp_plus_is_season_final(self, monkeypatch):
        rows = self._rows("sp_plus", [{"team": "Georgia", "rating": 20.0}], monkeypatch)
        assert rows[0].point_in_time is False

    def test_srs_is_season_final(self, monkeypatch):
        rows = self._rows("srs", [{"team": "Georgia", "rating": 12.0}], monkeypatch)
        assert rows[0].point_in_time is False

    def test_elo_is_point_in_time(self, monkeypatch):
        payload = [{"team": "Georgia", "elo": 1800}, {"team": "Vandy", "elo": 1400}]
        rows = self._rows("elo", payload, monkeypatch)
        assert all(r.point_in_time for r in rows)
