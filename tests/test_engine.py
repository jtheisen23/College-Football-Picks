"""Blending, projection and recommendation behaviour."""

from __future__ import annotations

import pytest

from cfbpicks.config import Config
from cfbpicks.engine.market import consensus_for_game
from cfbpicks.engine.predict import PredictionInputs, Predictor
from cfbpicks.engine.recommend import Recommender
from cfbpicks.models import ExpertProjection, Game, MarketQuote, Rating
from cfbpicks.ratings.blend import blend_ratings, confidence_from_blend


def rating(source, team, value, week=1):
    return Rating(source, 2026, week, team, value)


def game(home="Georgia", away="Alabama", neutral=False, week=1):
    return Game("g1", 2026, week, None, home, away, neutral_site=neutral)


class TestBlend:
    def test_average_of_agreeing_sources(self):
        ratings = [
            rating("a", "X", 10.0), rating("a", "Y", -10.0),
            rating("b", "X", 10.0), rating("b", "Y", -10.0),
        ]
        blend = blend_ratings(ratings, normalize=False)
        assert blend["X"].rating == pytest.approx(10.0)
        assert blend["X"].n_sources == 2

    def test_weights_are_respected(self):
        ratings = [
            rating("a", "X", 20.0), rating("a", "Y", -20.0),
            rating("b", "X", 0.0), rating("b", "Y", 0.0),
        ]
        blend = blend_ratings(ratings, {"a": 3.0, "b": 1.0}, normalize=False)
        assert blend["X"].rating == pytest.approx(15.0)

    def test_scales_are_aligned_before_averaging(self):
        # Source "wide" spans twice as far as "narrow"; normalising must
        # stop it dominating the blend purely because of its units.
        ratings = []
        for team, value in (("X", 20.0), ("Y", 0.0), ("Z", -20.0)):
            ratings.append(rating("wide", team, value))
        for team, value in (("X", 10.0), ("Y", 0.0), ("Z", -10.0)):
            ratings.append(rating("narrow", team, value))
        blend = blend_ratings(ratings, normalize=True)
        assert blend["X"].contributions["wide"] == pytest.approx(
            blend["X"].contributions["narrow"], abs=1e-6
        )

    def test_disagreement_is_recorded(self):
        ratings = [
            rating("a", "X", 20.0), rating("a", "Y", 0.0), rating("a", "Z", -20.0),
            rating("b", "X", -20.0), rating("b", "Y", 0.0), rating("b", "Z", 20.0),
        ]
        blend = blend_ratings(ratings)
        assert blend["X"].dispersion > blend["Y"].dispersion

    def test_lone_source_is_shrunk_toward_average(self):
        ratings = [rating("a", "X", 20.0), rating("a", "Y", -20.0)]
        blend = blend_ratings(ratings, normalize=False, min_sources_for_full_confidence=2)
        assert abs(blend["X"].rating) < 20.0

    def test_confidence_falls_with_disagreement(self):
        # The sources flatly contradict each other on X but agree on Y.
        ratings = [
            rating("a", "X", 20.0), rating("a", "Y", 5.0), rating("a", "Z", -20.0),
            rating("b", "X", -20.0), rating("b", "Y", 5.0), rating("b", "Z", 20.0),
        ]
        blend = blend_ratings(ratings, normalize=False)
        assert blend["Y"].dispersion == pytest.approx(0.0)
        assert confidence_from_blend(blend["X"]) < confidence_from_blend(blend["Y"])

    def test_no_ratings_is_empty(self):
        assert blend_ratings([]) == {}


class TestPredictor:
    def setup_method(self):
        self.config = Config()
        self.predictor = Predictor(self.config)

    def _predict(self, g, ratings, projections=()):
        return self.predictor.predict_week(
            PredictionInputs(games=[g], ratings=list(ratings), expert_projections=list(projections))
        )[0]

    def test_home_field_is_applied(self):
        ratings = [rating("a", "Georgia", 0.0), rating("a", "Alabama", 0.0),
                   rating("b", "Georgia", 0.0), rating("b", "Alabama", 0.0)]
        pred = self._predict(game(), ratings)
        assert pred.projected_margin == pytest.approx(self.config.model.home_field_advantage)

    def test_neutral_site_removes_home_field(self):
        ratings = [rating("a", "Georgia", 0.0), rating("a", "Alabama", 0.0),
                   rating("b", "Georgia", 0.0), rating("b", "Alabama", 0.0)]
        pred = self._predict(game(neutral=True), ratings)
        assert pred.projected_margin == pytest.approx(0.0)
        assert pred.hfa_applied == 0.0

    def test_better_team_is_favoured(self):
        ratings = [rating("a", "Georgia", 14.0), rating("a", "Alabama", 0.0),
                   rating("b", "Georgia", 14.0), rating("b", "Alabama", 0.0)]
        pred = self._predict(game(), ratings)
        assert pred.projected_margin > 0
        assert pred.home_win_prob > 0.5
        assert pred.fair_home_spread < 0  # home laying points

    def test_fair_spread_mirrors_the_margin(self):
        ratings = [rating("a", "Georgia", 10.0), rating("a", "Alabama", 0.0),
                   rating("b", "Georgia", 10.0), rating("b", "Alabama", 0.0)]
        pred = self._predict(game(), ratings)
        assert pred.fair_home_spread == pytest.approx(-pred.projected_margin, abs=0.01)

    def test_unrated_team_collapses_confidence(self):
        ratings = [rating("a", "Georgia", 10.0), rating("a", "Someone", 0.0)]
        pred = self._predict(game(), ratings)
        assert pred.confidence <= 0.15

    def test_expert_projection_pulls_the_number(self):
        ratings = [rating("a", "Georgia", 14.0), rating("a", "Alabama", 0.0),
                   rating("b", "Georgia", 14.0), rating("b", "Alabama", 0.0)]
        base = self._predict(game(), ratings)
        proj = ExpertProjection("sagarin", 2026, 1, "g1", "Georgia", "Alabama", projected_margin=0.0)
        blended = self._predict(game(), ratings, [proj])
        assert blended.projected_margin < base.projected_margin

    def test_components_explain_the_number(self):
        ratings = [rating("a", "Georgia", 7.0), rating("a", "Alabama", 0.0),
                   rating("b", "Georgia", 7.0), rating("b", "Alabama", 0.0)]
        pred = self._predict(game(), ratings)
        assert "rating_margin" in pred.components
        assert "hfa" in pred.components


class TestRecommender:
    def setup_method(self):
        self.config = Config()
        self.recommender = Recommender(self.config)

    def _run(self, projected_margin, line, price=-110, confidence=1.0):
        from cfbpicks.models import Prediction

        pred = Prediction(
            game_id="g1", season=2026, week=1, home_team="Georgia", away_team="Alabama",
            projected_margin=projected_margin, projected_total=52.0,
            home_win_prob=0.6, fair_home_spread=-projected_margin,
            home_rating=10.0, away_rating=0.0, hfa_applied=2.2, confidence=confidence,
        )
        quotes = [
            MarketQuote("g1", "DK", "spread", "home", line, price),
            MarketQuote("g1", "DK", "spread", "away", -line, price),
        ]
        view = consensus_for_game(quotes, "spread", margin_sd=16.0, total_sd=13.5)
        self.config.betting.markets = ["spread"]
        return self.recommender.recommend_week([pred], {"g1": {"spread": view}}, {"g1": quotes})

    def test_no_edge_produces_no_bet(self):
        # Model agrees exactly with the market.
        assert self._run(projected_margin=7.0, line=-7.0) == []

    def test_model_liking_the_home_side_bets_the_home_side(self):
        recs = self._run(projected_margin=14.0, line=-7.0)
        assert len(recs) == 1
        assert recs[0].side == "home"
        assert recs[0].selection == "Georgia -7"
        assert recs[0].prob_edge > 0

    def test_model_liking_the_away_side_bets_the_away_side(self):
        recs = self._run(projected_margin=0.0, line=-7.0)
        assert len(recs) == 1
        assert recs[0].side == "away"
        # The away quote's own number is +7, and must not be flipped.
        assert recs[0].selection == "Alabama +7"

    def test_never_recommends_both_sides(self):
        recs = self._run(projected_margin=14.0, line=-7.0)
        assert len({r.side for r in recs}) == len(recs)

    def test_low_confidence_shrinks_the_edge(self):
        confident = self._run(projected_margin=14.0, line=-7.0, confidence=1.0)
        unsure = self._run(projected_margin=14.0, line=-7.0, confidence=0.3)
        assert unsure == [] or unsure[0].prob_edge < confident[0].prob_edge

    def test_stake_never_exceeds_the_cap(self):
        recs = self._run(projected_margin=30.0, line=-3.0)
        assert all(r.stake_units <= self.config.betting.max_units for r in recs)

    def test_absurd_spreads_are_skipped(self):
        self.config.betting.max_spread_line = 20.0
        assert self._run(projected_margin=60.0, line=-45.0) == []

    def test_tiers_escalate_with_edge(self):
        small = self._run(projected_margin=10.0, line=-7.0)
        large = self._run(projected_margin=20.0, line=-7.0)
        if small and large:
            from cfbpicks.report import TIER_ORDER
            assert TIER_ORDER[large[0].tier] <= TIER_ORDER[small[0].tier]
