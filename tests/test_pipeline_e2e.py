"""End-to-end: fixtures in, graded backtest out — no network, no keys."""

from __future__ import annotations

import json
import random

import pytest

from cfbpicks.backtest import backtest_season, grade_all
from cfbpicks.config import Config, ProviderConfig
from cfbpicks.models import Game
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
