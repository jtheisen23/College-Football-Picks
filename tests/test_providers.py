"""Provider parsing: Sagarin's page, saved HTML ratings, CFBD field drift."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfbpicks.models import Game
from cfbpicks.providers.cfbd import CfbdProvider, pick
from cfbpicks.providers.sagarin import (
    parse_file, parse_predictions, parse_ratings, resolve_projections, strip_html,
)
from cfbpicks.util.htmltable import find_table_with, largest_table

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def sagarin():
    return parse_file(DATA / "sagarin_sample.htm", 2026, 3)


class TestSagarinRatings:
    def test_finds_every_team(self, sagarin):
        assert len(sagarin.ratings) == 6

    def test_names_are_canonicalised(self, sagarin):
        teams = {r.team for r in sagarin.ratings}
        assert "Mississippi" in teams   # listed as "Ole Miss"
        assert "Boise State" in teams   # listed as "Boise St"

    def test_ratings_are_recentred_on_zero(self, sagarin):
        assert sum(r.rating for r in sagarin.ratings) == pytest.approx(0.0, abs=1e-6)

    def test_order_is_preserved(self, sagarin):
        best = max(sagarin.ratings, key=lambda r: r.rating)
        assert best.team == "Ohio State"
        assert best.rank == 1

    def test_header_rows_are_not_mistaken_for_teams(self, sagarin):
        teams = {r.team.lower() for r in sagarin.ratings}
        assert "rating" not in teams and "copyright" not in teams


class TestSagarinPredictions:
    def test_finds_every_game(self, sagarin):
        assert len(sagarin.projections) == 4

    def test_capitalised_team_is_treated_as_home(self, sagarin):
        # "Georgia by 6.20 over TEXAS" -> Texas is home and is the dog.
        game = next(p for p in sagarin.projections if p.home_team == "Texas")
        assert game.away_team == "Georgia"
        assert game.projected_margin == pytest.approx(-6.20)

    def test_favourite_at_home_keeps_a_positive_margin(self, sagarin):
        game = next(p for p in sagarin.projections if p.home_team == "Ohio State")
        assert game.projected_margin == pytest.approx(17.45)

    def test_totals_are_captured(self, sagarin):
        game = next(p for p in sagarin.projections if p.home_team == "Ohio State")
        assert game.projected_total == pytest.approx(54.30)

    def test_moneylines_follow_the_teams_when_sides_swap(self, sagarin):
        # Georgia is the road favourite, so the negative price is Georgia's.
        game = next(p for p in sagarin.projections if p.home_team == "Texas")
        assert game.away_moneyline == pytest.approx(-235)
        assert game.home_moneyline == pytest.approx(195)

    def test_a_game_with_no_moneyline_still_parses(self, sagarin):
        game = next(p for p in sagarin.projections if p.home_team == "Auburn")
        assert game.projected_total == pytest.approx(51.0)
        assert game.home_moneyline is None

    def test_garbage_input_yields_nothing(self):
        assert parse_predictions("nothing to see here\n", 2026, 1) == []
        assert parse_ratings("nothing to see here\n", 2026, 1) == []


class TestProjectionAlignment:
    def test_margin_flips_when_the_schedule_disagrees(self):
        # Sagarin lists the favourite first; the schedule says otherwise.
        projections = parse_predictions(
            "     Georgia               by    6.20   over  TEXAS      total   48.75\n", 2026, 3
        )
        games = [Game("real-id", 2026, 3, None, "Texas", "Georgia")]
        resolved, unmatched = resolve_projections(projections, games)
        assert not unmatched
        assert resolved[0].game_id == "real-id"
        assert resolved[0].projected_margin == pytest.approx(-6.20)

    def test_unknown_games_are_reported_not_dropped_silently(self):
        projections = parse_predictions(
            "     ALABAMA               by    3.00   over  Auburn     total   50.0\n", 2026, 3
        )
        resolved, unmatched = resolve_projections(projections, [])
        assert not resolved and len(unmatched) == 1


class TestHtmlIngest:
    def test_saved_massey_page_parses(self):
        rows = largest_table((DATA / "massey_sample.html").read_text())
        assert len(rows) == 3
        assert rows[0]["team"] == "Ohio State"

    def test_layout_tables_are_skipped_by_header_match(self):
        rows = find_table_with((DATA / "massey_sample.html").read_text(), "team", "rating")
        assert len(rows) == 3

    def test_strip_html_recovers_plain_text(self):
        assert "Georgia" in strip_html("<pre><font>Georgia &amp; Co</font></pre>")
        assert "&amp;" not in strip_html("<pre>Georgia &amp; Co</pre>")


class TestCfbdFieldDrift:
    """CFBD renamed its fields to camelCase; both spellings must work."""

    def test_pick_prefers_the_first_present_key(self):
        assert pick({"homeTeam": "Georgia"}, "homeTeam", "home_team") == "Georgia"
        assert pick({"home_team": "Georgia"}, "homeTeam", "home_team") == "Georgia"
        assert pick({}, "homeTeam", "home_team", default="?") == "?"

    def test_null_values_fall_through_to_the_next_key(self):
        assert pick({"homeTeam": None, "home_team": "Georgia"}, "homeTeam", "home_team") == "Georgia"

    @pytest.mark.parametrize("row", [
        {"homeTeam": "Ole Miss", "awayTeam": "LSU", "week": 3, "homePoints": 28, "awayPoints": 21},
        {"home_team": "Ole Miss", "away_team": "LSU", "week": 3, "home_points": 28, "away_points": 21},
    ])
    def test_both_naming_styles_produce_the_same_game(self, row):
        from cfbpicks.config import Config, ProviderConfig

        provider = CfbdProvider(Config(), ProviderConfig("cfbd", api_key="x"))
        game = provider._to_game(row, 2026)
        assert (game.home_team, game.away_team) == ("Mississippi", "LSU")
        assert game.margin == 7
