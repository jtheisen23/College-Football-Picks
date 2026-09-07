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

    def test_the_only_external_resource_is_the_font(self):
        """It must still work from disk with no network.

        Webfonts are the one exception, and they are allowed to fail:
        nothing about the layout depends on them. No external script or
        stylesheet may carry anything the page needs to render.
        """
        import re

        page = to_html([rec()], season=2026, week=3, predictions=[pred()])
        urls = re.findall(r'https?://[^"\'\s>]+', page)
        assert urls, "the font link should be present"
        assert all("fonts.googleapis.com" in u or "fonts.gstatic.com" in u for u in urls), urls
        assert "<script src" not in page, "no external script"

    def test_every_font_stack_has_a_system_fallback(self):
        """So a failed font load costs polish, never legibility."""
        import re

        page = to_html([rec()], season=2026, week=3)
        stacks = re.findall(r"font-family:\s*([^;}]+)", page)
        stacks += [m for m in re.findall(r"font:\s*[^;}]*?\d+px[^;}]*", page)]
        assert stacks
        for stack in stacks:
            assert "system-ui" in stack or "sans-serif" in stack, stack

    def test_the_css_is_inline(self):
        page = to_html([rec()], season=2026, week=3)
        assert "<style>" in page
        assert 'rel="stylesheet" href="https://fonts.googleapis' in page

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

    def test_a_note_qualifies_the_whole_page(self):
        page = to_html([rec()], season=2026, week=3, note="Sample data — not real games")
        assert "Sample data — not real games" in page
        assert "note-bar" in page

    def test_no_note_means_no_banner(self):
        assert "note-bar\">" not in to_html([rec()], season=2026, week=3)

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


class TestIndexPage:
    def _entry(self, week=3, bets=12, staked=18.5):
        from cfbpicks.report import BoardEntry

        return BoardEntry(season=2026, week=week, filename=f"2026-week-{week:02d}.html",
                          bets=bets, staked=staked, generated="07 Sep 2026, 12:00 UTC")

    def test_lists_each_published_board(self):
        from cfbpicks.report import to_index_html

        page = to_index_html([self._entry(2), self._entry(3)])
        assert "2026-week-02.html" in page and "2026-week-03.html" in page

    def test_newest_week_comes_first(self):
        from cfbpicks.report import to_index_html

        page = to_index_html([self._entry(2), self._entry(5)])
        assert page.index("Week 5") < page.index("Week 2")

    def test_an_empty_site_says_what_to_run(self):
        from cfbpicks.report import to_index_html

        page = to_index_html([])
        assert "cfbpicks publish" in page

    def test_only_the_font_is_external(self):
        import re

        from cfbpicks.report import to_index_html

        page = to_index_html([self._entry()])
        urls = re.findall(r'https?://[^"\'\s>]+', page)
        assert all("fonts.g" in u for u in urls), urls


class TestPublishCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def _seeded(self, tmp_path):
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
        return cfg.database

    def test_writes_a_board_and_an_index(self, tmp_path, monkeypatch):
        db = self._seeded(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["--db", db, "publish", "--season", "2026", "--week", "3"])
        assert result.exit_code == 0, result.output
        docs = tmp_path / "docs"
        assert (docs / "2026-week-03.html").exists()
        assert (docs / "index.html").exists()
        assert (docs / ".nojekyll").exists(), "Pages must not run these through Jekyll"

    def test_republishing_keeps_earlier_weeks_listed(self, tmp_path, monkeypatch):
        import json

        db = self._seeded(tmp_path)
        monkeypatch.chdir(tmp_path)
        self.runner.invoke(main, ["--db", db, "publish", "--season", "2026", "--week", "3"])
        # Pretend an earlier week was published before this one.
        manifest = tmp_path / "docs" / "boards.json"
        data = json.loads(manifest.read_text())
        data["2026-week-01.html"] = {
            "season": 2026, "week": 1, "filename": "2026-week-01.html",
            "bets": 4, "staked": 5.0, "generated": "01 Sep 2026, 12:00 UTC",
        }
        manifest.write_text(json.dumps(data))
        self.runner.invoke(main, ["--db", db, "publish", "--season", "2026", "--week", "3"])
        index = (tmp_path / "docs" / "index.html").read_text()
        assert "2026-week-01.html" in index, "an earlier board must not be dropped"

    def test_an_unfetched_week_explains_itself(self, tmp_path, monkeypatch):
        db = self._seeded(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["--db", db, "publish", "--season", "2026", "--week", "9"])
        assert result.exit_code == 1
        assert "fetch" in result.output

    def test_it_does_not_touch_git_without_being_asked(self, tmp_path, monkeypatch):
        db = self._seeded(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["--db", db, "publish", "--season", "2026", "--week", "3"])
        assert "git add docs" in result.output, "it should print the command, not run it"


class TestWeekSwitcher:
    def _entry(self, week):
        from cfbpicks.report import BoardEntry

        return BoardEntry(season=2026, week=week, filename=f"2026-week-{week:02d}.html",
                          bets=5, staked=8.0, generated="07 Sep 2026, 12:00 UTC")

    def test_no_switcher_for_a_single_week(self):
        page = to_html([rec()], season=2026, week=3, nav=[self._entry(3)])
        assert '<nav class="weeknav"' not in page

    def test_switcher_appears_once_there_is_somewhere_to_go(self):
        page = to_html([rec()], season=2026, week=3,
                       nav=[self._entry(3), self._entry(4)])
        assert '<nav class="weeknav"' in page
        assert "2026-week-04.html" in page

    def test_the_page_you_are_on_is_marked_current(self):
        page = to_html([rec()], season=2026, week=4,
                       nav=[self._entry(3), self._entry(4)])
        assert 'aria-current="page">Week 4' in page
        assert 'aria-current="page">Week 3' not in page

    def test_weeks_are_listed_in_order(self):
        page = to_html([rec()], season=2026, week=3,
                       nav=[self._entry(4), self._entry(3)])
        assert page.index(">Week 3") < page.index(">Week 4")


class TestPublishesNextWeekToo:
    def setup_method(self):
        self.runner = CliRunner()

    def test_current_and_next_week_are_both_published(self, tmp_path, monkeypatch):
        from cfbpicks.config import Config, ProviderConfig
        from cfbpicks.models import Game
        from cfbpicks.pipeline import Pipeline
        from cfbpicks.providers.fixtures import PACKAGE_FIXTURES

        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.fixtures_dir = str(PACKAGE_FIXTURES)
        cfg.providers = {"fixtures": ProviderConfig("fixtures", enabled=True)}
        with Pipeline(cfg) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            # A week 4 slate so there is a "next" to look ahead to.
            pipeline.storage.upsert_games([
                Game(g.game_id.replace("2026-03", "2026-04"), 2026, 4, g.kickoff,
                     g.home_team, g.away_team)
                for g in pipeline.storage.games(2026, 3)
            ])

        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["--db", cfg.database, "publish", "--season", "2026"])
        assert result.exit_code == 0, result.output
        docs = tmp_path / "docs"
        assert (docs / "2026-week-03.html").exists()
        assert (docs / "2026-week-04.html").exists(), "next week should publish too"
        assert '<nav class="weeknav"' in (docs / "2026-week-03.html").read_text()

    def test_an_explicit_week_publishes_only_that_one(self, tmp_path, monkeypatch):
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

        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(
            main, ["--db", cfg.database, "publish", "--season", "2026", "--week", "3"]
        )
        assert result.exit_code == 0, result.output
        assert "week None" not in result.output
        assert (tmp_path / "docs" / "2026-week-03.html").exists()
