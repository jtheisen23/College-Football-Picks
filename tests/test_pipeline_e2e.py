"""End-to-end: fixtures in, graded backtest out — no network, no keys."""

from __future__ import annotations

import json
import random

import pytest

from cfbpicks.backtest import backtest_season, grade_all
from cfbpicks.config import Config, ProviderConfig
from cfbpicks.models import Game, Rating
from cfbpicks.pipeline import Pipeline
from cfbpicks.storage import Storage


@pytest.fixture()
def config(tmp_path):
    cfg = Config(root=tmp_path)
    cfg.database = str(tmp_path / "test.sqlite")
    cfg.cache_dir = str(tmp_path / "cache")
    # Point at the packaged sample data, the same copy an installed
    # wheel would use.
    from cfbpicks.providers.fixtures import PACKAGE_FIXTURES
    cfg.fixtures_dir = str(PACKAGE_FIXTURES)
    cfg.offline = True
    cfg.providers = {"fixtures": ProviderConfig("fixtures", enabled=True)}
    return cfg


class TestWeeklyFlow:
    def test_fetch_loads_everything(self, config):
        with Pipeline(config) as pipeline:
            report = pipeline.fetch(2026, 3, providers=["fixtures"])
        assert report.games == 12
        assert report.ratings > 0
        assert report.quotes > 0

    def test_predict_covers_every_game(self, config):
        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            predictions = pipeline.predict(2026, 3, use_projections=False)
        assert len(predictions) == 12
        assert all(p.confidence > 0.5 for p in predictions)
        assert all(-70 < p.projected_margin < 70 for p in predictions)
        assert all(20 < p.projected_total < 100 for p in predictions)

    def test_picks_are_produced_and_sane(self, config):
        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            recs = pipeline.picks(2026, 3)

        assert recs, "fixtures contain deliberate mispricings, so bets are expected"
        for rec in recs:
            assert rec.stake_units <= config.betting.max_units
            assert rec.prob_edge >= config.betting.min_prob_edge
            assert rec.expected_value > 0
            assert rec.book
            assert rec.tier in {"lean", "play", "strong"}

    def test_never_both_sides_of_one_market(self, config):
        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            recs = pipeline.picks(2026, 3)
        seen = {(r.game_id, r.market) for r in recs}
        assert len(seen) == len(recs)

    def test_predictions_are_persisted(self, config):
        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            pipeline.predict(2026, 3, use_projections=False)
            assert len(pipeline.storage.predictions(2026, 3)) == 12

    def test_rerunning_does_not_duplicate_picks(self, config):
        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            first = pipeline.picks(2026, 3)
            pipeline.picks(2026, 3)
            stored = pipeline.storage.recommendations(2026, 3)
        assert len(stored) == len(first)

    def test_empty_week_is_handled(self, config):
        with Pipeline(config) as pipeline:
            assert pipeline.predict(2026, 99) == []
            assert pipeline.picks(2026, 99) == []


class TestGradingFlow:
    def _play_out_the_week(self, pipeline, seed=7):
        """Attach plausible final scores so the week can be graded."""
        rng = random.Random(seed)
        played = []
        for game in pipeline.storage.games(2026, 3):
            home = rng.randint(10, 45)
            away = rng.randint(3, 42)
            played.append(
                Game(game.game_id, game.season, game.week, game.kickoff,
                     game.home_team, game.away_team, neutral_site=game.neutral_site,
                     home_score=home, away_score=away)
            )
        pipeline.storage.upsert_games(played)

    def test_grade_settles_every_bet(self, config):
        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            recs = pipeline.picks(2026, 3)
            self._play_out_the_week(pipeline)
            results = pipeline.grade(2026, 3)

        assert results["ungraded"] == 0
        assert results["win"] + results["loss"] + results["push"] == len(recs)

    def test_backtest_reports_consistent_arithmetic(self, config):
        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            pipeline.picks(2026, 3)
            self._play_out_the_week(pipeline)
            result = backtest_season(pipeline.storage, config, 2026, [3])

        assert result.bets > 0
        assert result.wins + result.losses + result.pushes == result.bets
        # Profit must equal the sum of the individual graded outcomes.
        assert result.profit == pytest.approx(sum(p for _, _, p in result.graded))
        assert set(result.by_market) <= {"spread", "total", "moneyline"}


class TestReportRendering:
    def test_every_format_renders(self, config):
        from cfbpicks.report import to_csv, to_json, to_markdown

        with Pipeline(config) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            recs = pipeline.picks(2026, 3)
            preds = pipeline.storage.predictions(2026, 3)

        markdown = to_markdown(recs, season=2026, week=3, predictions=preds)
        assert "Week 3" in markdown and "| Tier |" in markdown

        csv_text = to_csv(recs)
        assert csv_text.startswith("season,week,game_id")
        assert len(csv_text.strip().splitlines()) == len(recs) + 1

        parsed = json.loads(to_json(recs))
        assert len(parsed) == len(recs)

    def test_empty_week_renders_without_crashing(self):
        from cfbpicks.report import to_csv, to_json, to_markdown

        assert "No bets" in to_markdown([], season=2026, week=1)
        assert to_json([]) == "[]"
        assert to_csv([]).startswith("season,")


class TestCurrentWeek:
    """An unattended run has nobody to pass --week."""

    def _pipeline(self, tmp_path, games):
        from cfbpicks.storage import Storage

        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        store = Storage(cfg.database_path)
        store.upsert_games(games)
        store.close()
        return cfg

    def _season(self, weeks_ahead_of_now):
        """A season laid out around today, one week per entry."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        return [
            Game(f"g{i}", 2026, i + 1, now + timedelta(days=offset * 7),
                 f"H{i}", f"A{i}")
            for i, offset in enumerate(weeks_ahead_of_now)
        ]

    def test_the_current_week_is_the_next_kickoff(self, tmp_path):
        # Weeks 1-3 are in the past, 4 and 5 ahead.
        cfg = self._pipeline(tmp_path, self._season([-3, -2, -1, 1, 2]))
        with Pipeline(cfg) as pipeline:
            assert pipeline.current_week(2026) == 4

    def test_one_never_scored_early_game_does_not_pin_it_to_week_one(self, tmp_path):
        """The bug that published weeks 1 and 2 in late September.

        A cancelled or never-scored game in week 1 made "earliest
        unplayed week" answer 1 for the rest of the season.
        """
        games = self._season([-3, -2, -1, 1, 2])
        # Everything in the past is final except one stray week 1 game.
        played = []
        for game in games:
            if game.kickoff.timestamp() < __import__("time").time() and game.week != 1:
                game.home_score, game.away_score = 24, 17
            played.append(game)
        cfg = self._pipeline(tmp_path, played)
        with Pipeline(cfg) as pipeline:
            assert pipeline.current_week(2026) == 4

    def test_saturdays_slate_stays_current_while_it_is_played(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        cfg = self._pipeline(tmp_path, [
            Game("a", 2026, 5, now - timedelta(hours=6), "A", "B",
                 home_score=31, away_score=3),
            Game("b", 2026, 6, now + timedelta(days=6), "C", "D"),
        ])
        with Pipeline(cfg) as pipeline:
            assert pipeline.current_week(2026) == 5

    def test_after_the_season_it_is_the_last_week_played(self, tmp_path):
        cfg = self._pipeline(tmp_path, self._season([-14, -8, -1]))
        with Pipeline(cfg) as pipeline:
            assert pipeline.current_week(2026) == 3

    def test_an_empty_season_has_no_current_week(self, tmp_path):
        cfg = self._pipeline(tmp_path, [])
        with Pipeline(cfg) as pipeline:
            assert pipeline.current_week(2026) is None

    def test_without_kickoff_times_it_falls_back_to_unplayed(self, tmp_path):
        cfg = self._pipeline(tmp_path, [
            Game("a", 2026, 1, None, "A", "B", home_score=21, away_score=17),
            Game("b", 2026, 2, None, "C", "D"),
        ])
        with Pipeline(cfg) as pipeline:
            assert pipeline.current_week(2026) == 2


class TestBoardEligibility:
    """The 231-bet board: FCS opponents the model cannot rate.

    A model fitted on FBS results rates an FCS team near average. The
    market rates it four touchdowns worse. The difference reads as a
    enormous edge on every big underdog and is purely an artefact of
    betting outside the model's competence.
    """

    def _cfg(self, tmp_path):
        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        cfg.cache_dir = str(tmp_path / "cache")
        return cfg

    def _seed(self, cfg, *, roster):
        from cfbpicks.models import MarketQuote
        from cfbpicks.storage import Storage

        store = Storage(cfg.database_path)
        store.upsert_games([
            Game("fbs", 2026, 4, None, "Georgia", "Alabama"),
            Game("fcs", 2026, 4, None, "Georgia", "Mercer"),
        ])
        if roster:
            store.upsert_teams(2026, [{"team": t} for t in roster])
        for gid in ("fbs", "fcs"):
            store.upsert_quotes([
                MarketQuote(gid, "DK", "spread", "home", -7.0, -110),
                MarketQuote(gid, "DK", "spread", "away", 7.0, -110),
            ])
        store.upsert_ratings([
            Rating("a", 2026, 4, t, v) for t, v in
            [("Georgia", 22.0), ("Alabama", 18.0), ("Mercer", 0.0)]
        ] + [
            Rating("b", 2026, 4, t, v) for t, v in
            [("Georgia", 21.0), ("Alabama", 19.0), ("Mercer", 0.0)]
        ])
        store.close()

    def test_games_outside_fbs_are_not_priced(self, tmp_path):
        cfg = self._cfg(tmp_path)
        self._seed(cfg, roster=["Georgia", "Alabama"])
        with Pipeline(cfg) as pipeline:
            recs = pipeline.picks(2026, 4)
            assert pipeline.skipped_games["non_fbs"] == 1
        assert all(r.game_id != "fcs" for r in recs)

    def test_the_filter_can_be_turned_off(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg.betting.fbs_only = False
        self._seed(cfg, roster=["Georgia", "Alabama"])
        with Pipeline(cfg) as pipeline:
            pipeline.picks(2026, 4)
            assert pipeline.skipped_games["non_fbs"] == 0

    def test_no_roster_means_no_filtering(self, tmp_path):
        """A database that never fetched teams must not lose every game."""
        cfg = self._cfg(tmp_path)
        self._seed(cfg, roster=None)
        with Pipeline(cfg) as pipeline:
            pipeline.predict(2026, 4, use_projections=False)
            pipeline.picks(2026, 4, repredict=False)
            assert pipeline.skipped_games["non_fbs"] == 0

    def test_thin_coverage_is_skipped_not_shrunk(self, tmp_path):
        """One lone rating source is a reason not to bet, not to bet small."""
        from cfbpicks.models import MarketQuote
        from cfbpicks.storage import Storage

        cfg = self._cfg(tmp_path)
        cfg.betting.fbs_only = False
        cfg.betting.min_confidence = 0.6

        store = Storage(cfg.database_path)
        store.upsert_games([Game("thin", 2026, 4, None, "Georgia", "Mercer")])
        store.upsert_quotes([
            MarketQuote("thin", "DK", "spread", "home", -7.0, -110),
            MarketQuote("thin", "DK", "spread", "away", 7.0, -110),
        ])
        # Georgia is covered twice; Mercer only once, so the blend is thin.
        store.upsert_ratings([
            Rating("a", 2026, 4, "Georgia", 22.0),
            Rating("b", 2026, 4, "Georgia", 21.0),
            Rating("a", 2026, 4, "Mercer", -20.0),
        ])
        store.close()

        with Pipeline(cfg) as pipeline:
            recs = pipeline.picks(2026, 4)
            assert pipeline.skipped_games["low_confidence"] == 1
        assert recs == []
