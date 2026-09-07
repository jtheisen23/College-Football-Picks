"""Closing line value: the earliest honest signal that an edge is real."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from cfbpicks.backtest import _clv_points, clv_direction
from cfbpicks.cli import main
from cfbpicks.config import Config
from cfbpicks.models import Game, MarketQuote, Recommendation
from cfbpicks.pipeline import Pipeline
from cfbpicks.storage import Storage

EARLY = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
LATE = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


def rec(market="spread", side="home", line=-6.5, placed_at=None):
    return Recommendation(
        game_id="g1", season=2026, week=3, matchup="Alabama @ Georgia",
        market=market, side=side, selection="x", line=line, price=-110, book="DK",
        model_prob=0.57, market_prob=0.52, prob_edge=0.05, point_edge=2.0,
        expected_value=0.04, stake_units=1.5, kelly_fraction=0.03,
        placed_at=placed_at,
    )


def close(market, side, line):
    return [MarketQuote("g1", "DK", market, side, line, -110, LATE)]


class TestClvSign:
    """Spreads and totals do not agree on which direction is good."""

    def test_spread_sides_both_want_a_bigger_number(self):
        assert clv_direction("home") == 1.0
        assert clv_direction("away") == 1.0

    def test_an_over_wants_a_smaller_number(self):
        assert clv_direction("over") == -1.0
        assert clv_direction("under") == 1.0

    @pytest.mark.parametrize("market,side,bet,closed,expected", [
        # Laying fewer points than the close is value.
        ("spread", "home", -7.0, -7.5, +0.5),
        ("spread", "home", -7.5, -7.0, -0.5),
        # Taking more points than the close is value.
        ("spread", "away", 7.5, 7.0, +0.5),
        ("spread", "away", 7.0, 7.5, -0.5),
        # An over bet at 52 is a good bet if the total closes at 53.
        ("total", "over", 52.0, 53.0, +1.0),
        ("total", "over", 52.0, 51.0, -1.0),
        # An under at 53 is good if the total closes at 52.
        ("total", "under", 53.0, 52.0, +1.0),
        ("total", "under", 52.0, 53.0, -1.0),
    ])
    def test_signs(self, market, side, bet, closed, expected):
        value = _clv_points(rec(market, side, bet), close(market, side, closed))
        assert value == pytest.approx(expected)

    def test_no_closing_line_means_no_clv(self):
        assert _clv_points(rec(), []) is None

    def test_moneyline_has_no_line_to_compare(self):
        assert _clv_points(rec("moneyline", "home", None), []) is None


@pytest.fixture()
def storage(tmp_path):
    store = Storage(tmp_path / "t.sqlite")
    yield store
    store.close()


class TestSnapshotHistory:
    def test_closing_quotes_ignores_captures_before_the_bet(self, storage):
        storage.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -6.5, -110, EARLY)])
        # The only capture predates the bet, so there is no closing line.
        assert storage.closing_quotes("g1", "spread", after=LATE) == []

    def test_closing_quotes_uses_a_later_capture(self, storage):
        storage.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -6.5, -110, EARLY)])
        storage.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -8.5, -110, LATE)])
        closing = storage.closing_quotes("g1", "spread", after=EARLY)
        assert [q.line for q in closing] == [-8.5]

    def test_line_history_is_ordered_and_medianed(self, storage):
        storage.upsert_quotes([
            MarketQuote("g1", "DK", "spread", "home", -7.0, -110, EARLY),
            MarketQuote("g1", "FD", "spread", "home", -6.0, -110, EARLY),
        ])
        storage.upsert_quotes([
            MarketQuote("g1", "DK", "spread", "home", -8.5, -110, LATE),
            MarketQuote("g1", "FD", "spread", "home", -8.5, -110, LATE),
        ])
        series = storage.line_history("g1", "spread", "home")
        assert [line for _, line, _ in series] == [-6.5, -8.5]
        assert series[0][0] < series[1][0]

    def test_single_capture_gives_no_series_to_compare(self, storage):
        storage.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -7.0, -110, EARLY)])
        assert len(storage.line_history("g1", "spread", "home")) == 1


class TestGradingRecordsClv:
    def _setup(self, tmp_path, second_capture: bool):
        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        store = Storage(cfg.database_path)
        store.upsert_games([
            Game("g1", 2026, 3, None, "Georgia", "Alabama", home_score=31, away_score=20)
        ])
        store.upsert_quotes([
            MarketQuote("g1", "DK", "spread", "home", -6.5, -110, EARLY),
            MarketQuote("g1", "DK", "spread", "away", 6.5, -110, EARLY),
        ])
        store.save_recommendations([rec()])
        if second_capture:
            later = datetime.now(timezone.utc) + timedelta(minutes=5)
            store.upsert_quotes([
                MarketQuote("g1", "DK", "spread", "home", -8.5, -110, later),
                MarketQuote("g1", "DK", "spread", "away", 8.5, -110, later),
            ])
        store.close()
        return cfg

    def _clv(self, cfg):
        with Pipeline(cfg) as pipeline:
            row = pipeline.storage.conn.execute(
                "SELECT result, profit_units, clv_points FROM recommendations"
            ).fetchone()
        return row

    def test_clv_is_recorded_when_the_market_moved(self, tmp_path):
        cfg = self._setup(tmp_path, second_capture=True)
        with Pipeline(cfg) as pipeline:
            results = pipeline.grade(2026, 3)
        assert results["with_clv"] == 1
        row = self._clv(cfg)
        # Bet -6.5, closed -8.5: two points better than the market's last word.
        assert row["clv_points"] == pytest.approx(2.0)
        assert row["result"] == "win"

    def test_clv_stays_null_with_only_one_capture(self, tmp_path):
        cfg = self._setup(tmp_path, second_capture=False)
        with Pipeline(cfg) as pipeline:
            results = pipeline.grade(2026, 3)
        assert results["with_clv"] == 0
        row = self._clv(cfg)
        assert row["clv_points"] is None
        assert row["result"] == "win"  # still graded normally

    def test_a_bet_can_win_with_negative_clv(self, tmp_path):
        """Winning and beating the close are different things."""
        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        store = Storage(cfg.database_path)
        store.upsert_games([
            Game("g1", 2026, 3, None, "Georgia", "Alabama", home_score=31, away_score=20)
        ])
        store.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -6.5, -110, EARLY),
                             MarketQuote("g1", "DK", "spread", "away", 6.5, -110, EARLY)])
        store.save_recommendations([rec()])
        later = datetime.now(timezone.utc) + timedelta(minutes=5)
        # Market moved against us: our -6.5 is worse than a -4.5 close.
        store.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -4.5, -110, later),
                             MarketQuote("g1", "DK", "spread", "away", 4.5, -110, later)])
        store.close()

        with Pipeline(cfg) as pipeline:
            pipeline.grade(2026, 3)
        row = self._clv(cfg)
        assert row["result"] == "win"
        assert row["clv_points"] == pytest.approx(-2.0)


class TestSnapshotCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_lines_explains_itself_with_nothing_to_compare(self, tmp_path):
        db = tmp_path / "t.sqlite"
        store = Storage(db)
        store.upsert_games([Game("g1", 2026, 3, None, "Georgia", "Alabama")])
        store.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -7.0, -110, EARLY)])
        store.close()
        result = self.runner.invoke(main, ["--db", str(db), "lines", "--week", "3"])
        assert result.exit_code == 0
        assert "two captures" in result.output

    def test_lines_reports_movement(self, tmp_path):
        db = tmp_path / "t.sqlite"
        store = Storage(db)
        store.upsert_games([Game("g1", 2026, 3, None, "Georgia", "Alabama")])
        store.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -6.5, -110, EARLY)])
        store.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -8.5, -110, LATE)])
        store.close()
        result = self.runner.invoke(main, ["--db", str(db), "lines", "--week", "3"])
        assert result.exit_code == 0
        assert "-2.0" in result.output

    def test_snapshot_quiet_is_silent_when_nothing_moved(self, tmp_path):
        db = tmp_path / "t.sqlite"
        Storage(db).close()
        result = self.runner.invoke(
            main, ["--db", str(db), "snapshot", "--season", "2026", "--quiet"]
        )
        assert result.exit_code == 0
        assert result.output.strip() == ""
