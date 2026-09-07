"""The standalone HTML board."""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from cfbpicks.cli import main
from cfbpicks.models import Prediction, Recommendation
from cfbpicks.report import render, to_html


def rec(tier="play", selection="Georgia -6.5", matchup="Alabama @ Georgia", notes=None):
    return Recommendation(
        game_id="g1", season=2026, week=3, matchup=matchup, market="spread",
        side="home", selection=selection, line=-6.5, price=-110, book="DraftKings",
        model_prob=0.575, market_prob=0.523, prob_edge=0.052, point_edge=2.4,
        expected_value=0.043, stake_units=1.75, kelly_fraction=0.035, tier=tier,
        notes=notes or [],
    )


def pred():
    return Prediction(
        game_id="g1", season=2026, week=3, home_team="Georgia", away_team="Alabama",
        projected_margin=9.1, projected_total=52.4, home_win_prob=0.71,
        fair_home_spread=-9.1, home_rating=18.0, away_rating=11.1, hfa_applied=2.2,
    )


class TestStructure:
    def test_is_a_complete_document(self):
        page = to_html([rec()], season=2026, week=3)
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")

    def test_makes_no_external_requests(self):
        """It has to work from disk, offline, forever."""
        page = to_html([rec()], season=2026, week=3, predictions=[pred()])
        assert "http://" not in page and "https://" not in page
        assert "<script src" not in page and "<link" not in page

    def test_names_the_week(self):
        page = to_html([rec()], season=2026, week=7)
        assert "Week 7" in page and "2026" in page

    def test_renders_through_the_shared_entry_point(self):
        page = render([rec()], "html", season=2026, week=3)
        assert page.startswith("<!doctype html>")


class TestContent:
    def test_shows_the_bet_and_its_price(self):
        page = to_html([rec()], season=2026, week=3)
        assert "Georgia -6.5" in page
        assert "-110" in page
        assert "DraftKings" in page

    def test_summary_tiles_add_up(self):
        page = to_html([rec(), rec(tier="strong")], season=2026, week=3)
        assert "3.50u" in page          # two bets at 1.75 units
        assert ">2<" in page            # bet count

    def test_projections_are_included_when_given(self):
        page = to_html([rec()], season=2026, week=3, predictions=[pred()])
        assert "Alabama @ Georgia" in page
        assert "+9.1" in page

    def test_caveats_are_surfaced(self):
        page = to_html([rec(notes=["single book — no line shopping"])], season=2026, week=3)
        assert "single book" in page

    def test_an_empty_week_explains_itself(self):
        page = to_html([], season=2026, week=3)
        assert "No bets cleared" in page
        assert "not a failure" in page


class TestAccessibility:
    def test_tier_is_never_communicated_by_colour_alone(self):
        page = to_html([rec(tier="strong")], season=2026, week=3)
        # The badge carries its own text label.
        assert re.search(r'class="badge strong">strong<', page)

    def test_dark_mode_is_declared_for_both_scopes(self):
        """The OS setting and an explicit toggle must each win."""
        page = to_html([rec()], season=2026, week=3)
        assert "prefers-color-scheme: dark" in page
        assert ':root[data-theme="dark"]' in page

    def test_numeric_columns_are_tabular(self):
        page = to_html([rec()], season=2026, week=3)
        assert "tabular-nums" in page


class TestEscaping:
    def test_team_names_with_markup_are_escaped(self):
        nasty = to_html(
            [rec(matchup="<script>alert(1)</script> @ Georgia")], season=2026, week=3
        )
        assert "<script>alert(1)</script>" not in nasty
        assert "&lt;script&gt;" in nasty

    def test_ampersands_survive_as_text(self):
        page = to_html([rec(selection="Texas A&M -3")], season=2026, week=3)
        assert "Texas A&amp;M -3" in page


class TestCli:
    def setup_method(self):
        self.runner = CliRunner()

    def _seeded_db(self, tmp_path):
        from cfbpicks.config import Config, ProviderConfig
        from cfbpicks.pipeline import Pipeline
        from cfbpicks.providers.fixtures import PACKAGE_FIXTURES

        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.fixtures_dir = str(PACKAGE_FIXTURES)
        cfg.providers = {"fixtures": ProviderConfig("fixtures", enabled=True)}
        with Pipeline(cfg) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            pipeline.picks(2026, 3)
        return cfg.database

    def test_writes_a_board_file(self, tmp_path):
        db = self._seeded_db(tmp_path)
        out = tmp_path / "week3.html"
        result = self.runner.invoke(main, [
            "--db", db, "picks", "--week", "3", "--format", "html", "-o", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert out.read_text().startswith("<!doctype html>")

    def test_html_extension_implies_the_html_renderer(self, tmp_path):
        """`-o board.html` should do the obvious thing."""
        db = self._seeded_db(tmp_path)
        out = tmp_path / "board.html"
        result = self.runner.invoke(main, ["--db", db, "picks", "--week", "3", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.read_text().startswith("<!doctype html>")

    def test_markdown_is_still_the_default_for_other_extensions(self, tmp_path):
        db = self._seeded_db(tmp_path)
        out = tmp_path / "board.md"
        result = self.runner.invoke(main, ["--db", db, "picks", "--week", "3", "-o", str(out)])
        assert result.exit_code == 0
        assert out.read_text().startswith("# College Football Picks")
