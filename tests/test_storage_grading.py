"""Persistence and bet settlement."""

from __future__ import annotations

import pytest

from cfbpicks.backtest import grade_all, grade_recommendation
from cfbpicks.models import Game, MarketQuote, Rating, Recommendation
from cfbpicks.storage import Storage


@pytest.fixture()
def storage(tmp_path):
    store = Storage(tmp_path / "test.sqlite")
    yield store
    store.close()


def rec(market="spread", side="home", line=-7.5, price=-110, stake=1.0, week=1):
    return Recommendation(
        game_id="g1", season=2026, week=week, matchup="Alabama @ Georgia",
        market=market, side=side, selection="x", line=line, price=price, book="DK",
        model_prob=0.56, market_prob=0.52, prob_edge=0.04, point_edge=2.0,
        expected_value=0.05, stake_units=stake, kelly_fraction=0.02,
    )


class TestStorage:
    def test_games_round_trip(self, storage):
        storage.upsert_games([Game("g1", 2026, 1, None, "Georgia", "Alabama")])
        assert storage.game("g1").home_team == "Georgia"

    def test_a_schedule_refresh_never_erases_a_score(self, storage):
        storage.upsert_games([Game("g1", 2026, 1, None, "Georgia", "Alabama",
                                   home_score=31, away_score=24)])
        storage.upsert_games([Game("g1", 2026, 1, None, "Georgia", "Alabama")])
        assert storage.game("g1").margin == 7

    def test_filtering_by_week_and_completion(self, storage):
        storage.upsert_games([
            Game("g1", 2026, 1, None, "A", "B", home_score=20, away_score=10),
            Game("g2", 2026, 2, None, "C", "D"),
        ])
        assert len(storage.games(2026, 1)) == 1
        assert len(storage.games(2026, completed=True)) == 1
        assert len(storage.games(2026, completed=False)) == 1

    def test_ratings_use_the_latest_week_per_source(self, storage):
        storage.upsert_ratings([
            Rating("sp", 2026, 1, "Georgia", 10.0),
            Rating("sp", 2026, 3, "Georgia", 14.0),
            Rating("srs", 2026, 1, "Georgia", 8.0),
        ])
        latest = {(r.source, r.team): r.rating for r in storage.ratings(2026, week=3)}
        assert latest[("sp", "Georgia")] == 14.0
        assert latest[("srs", "Georgia")] == 8.0  # its only snapshot still counts

    def test_ratings_never_look_into_the_future(self, storage):
        storage.upsert_ratings([
            Rating("sp", 2026, 1, "Georgia", 10.0),
            Rating("sp", 2026, 9, "Georgia", 30.0),
        ])
        assert storage.ratings(2026, week=3)[0].rating == 10.0

    def test_quotes_upsert_and_append_history(self, storage):
        storage.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -7.0, -110)])
        storage.upsert_quotes([MarketQuote("g1", "DK", "spread", "home", -7.5, -110)])
        assert len(storage.quotes("g1")) == 1
        assert storage.quotes("g1")[0].line == -7.5
        assert len(storage.closing_quotes("g1", "spread")) >= 1

    def test_recommendations_round_trip(self, storage):
        storage.save_recommendations([rec()])
        assert len(storage.recommendations(2026, 1)) == 1

    def test_grading_marks_a_bet_settled(self, storage):
        storage.save_recommendations([rec()])
        rec_id, _ = storage.ungraded_recommendations(2026)[0]
        storage.grade_recommendation(rec_id, "win", 0.91)
        assert storage.ungraded_recommendations(2026) == []


class TestGrading:
    #  Georgia 31, Alabama 24: home by 7, total 55.
    game = Game("g1", 2026, 1, None, "Georgia", "Alabama", home_score=31, away_score=24)

    @pytest.mark.parametrize("side,line,expected", [
        ("home", -6.5, "win"),
        ("home", -7.0, "push"),
        ("home", -7.5, "loss"),
        ("away", 7.5, "win"),
        ("away", 7.0, "push"),
        ("away", 6.5, "loss"),
    ])
    def test_spreads(self, side, line, expected):
        assert grade_recommendation(rec(side=side, line=line), self.game)[0] == expected

    @pytest.mark.parametrize("side,line,expected", [
        ("over", 54.5, "win"), ("over", 55.0, "push"), ("over", 55.5, "loss"),
        ("under", 55.5, "win"), ("under", 55.0, "push"), ("under", 54.5, "loss"),
    ])
    def test_totals(self, side, line, expected):
        assert grade_recommendation(rec("total", side, line), self.game)[0] == expected

    def test_moneylines(self):
        assert grade_recommendation(rec("moneyline", "home", None, -250), self.game)[0] == "win"
        assert grade_recommendation(rec("moneyline", "away", None, 210), self.game)[0] == "loss"

    def test_profit_matches_the_price(self):
        _, profit = grade_recommendation(rec(side="home", line=-6.5, price=-110, stake=2.0), self.game)
        assert profit == pytest.approx(2.0 * 100 / 110)

    def test_a_loss_costs_the_stake(self):
        _, profit = grade_recommendation(rec(side="home", line=-7.5, stake=2.0), self.game)
        assert profit == pytest.approx(-2.0)

    def test_a_push_costs_nothing(self):
        assert grade_recommendation(rec(side="home", line=-7.0), self.game)[1] == 0.0

    def test_unplayed_games_are_ungraded(self):
        unplayed = Game("g1", 2026, 1, None, "Georgia", "Alabama")
        assert grade_recommendation(rec(), unplayed)[0] == "ungraded"

    def test_summary_totals(self):
        games = {"g1": self.game}
        result = grade_all([rec(side="home", line=-6.5), rec(side="home", line=-7.5)], games)
        assert (result.wins, result.losses) == (1, 1)
        assert result.staked == pytest.approx(2.0)
        assert result.roi < 0  # one win at -110 does not pay for one loss

    def test_pushes_are_excluded_from_staked(self):
        result = grade_all([rec(side="home", line=-7.0)], {"g1": self.game})
        assert result.pushes == 1
        assert result.staked == 0.0
        assert result.roi == 0.0
