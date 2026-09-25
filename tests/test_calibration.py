"""Rescaling predicted margins against what actually happened."""

from __future__ import annotations

import random

import pytest

from cfbpicks.engine.calibration import (
    IDENTITY,
    MAX_SLOPE,
    MIN_GAMES,
    MarginCalibration,
    fit_margin_calibration,
)


def pairs(compression: float = 1.0, n: int = 400, outcome_noise: float = 14.0,
          model_error: float = 0.0, intercept: float = 0.0, seed: int = 5
          ) -> list[tuple[float, float]]:
    """Games from a model whose margins are ``compression`` x life size.

    Where the randomness sits matters, and getting it wrong inverts the
    answer. ``outcome_noise`` is the game itself refusing to be
    predicted -- a two-touchdown residual is an ordinary Saturday -- and
    it belongs in the result. ``model_error`` is the model being wrong
    about a matchup, and it belongs in the prediction.

    They pull the fitted slope in opposite directions, which is the
    point: compression asks to be stretched back out, while a model
    that is mostly noise should be trusted less, not amplified. Fitting
    against results weighs the two automatically.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        signal = rng.gauss(0, 11.0)          # the genuinely predictable part
        actual = intercept + signal + rng.gauss(0, outcome_noise)
        predicted = compression * (signal + rng.gauss(0, model_error))
        out.append((predicted, actual))
    return out


class TestFit:
    def test_a_compressed_model_is_stretched_back_out(self):
        """The defect on the live board: margins at 0.7x life size."""
        cal = fit_margin_calibration(pairs(0.7))
        assert cal.fitted
        assert cal.slope > 1.15, "a 0.7x model must be corrected upward"
        assert cal.apply(10.0) > 11.5

    def test_a_calibrated_model_is_left_alone(self):
        cal = fit_margin_calibration(pairs(1.0))
        assert 0.9 < cal.slope < 1.1
        assert cal.apply(10.0) == pytest.approx(10.0, abs=1.5)

    def test_correcting_twice_is_not_better_than_once(self):
        """Feeding corrected margins back in must find nothing left."""
        first = fit_margin_calibration(pairs(0.7))
        corrected = [(first.apply(p), a) for p, a in pairs(0.7)]
        second = fit_margin_calibration(corrected)
        assert second.slope == pytest.approx(1.0, abs=0.12), second.describe()

    def test_an_over_dispersed_model_is_squashed(self):
        cal = fit_margin_calibration(pairs(1.5))
        assert cal.fitted and cal.slope < 0.95

    def test_a_home_field_error_is_absorbed_by_the_intercept(self):
        cal = fit_margin_calibration(pairs(1.0, intercept=4.0))
        assert cal.apply(0.0) > 1.0, "a real home bias should be corrected"


class TestSignalVersusNoise:
    """The property that makes fitting against results safe to do."""

    def test_a_model_that_is_mostly_noise_is_not_amplified(self):
        """Compression alone must not buy a stretch.

        Two models compressed identically to 0.7x. One has a real
        signal underneath; the other is mostly guessing. Stretching the
        guesser would turn small nonsense into large nonsense and stake
        real money on it, so the fit has to tell them apart.
        """
        real = fit_margin_calibration(pairs(0.7, model_error=1.0))
        noise = fit_margin_calibration(pairs(0.7, model_error=40.0))
        assert real.slope > 1.2, "a compressed but honest model gets stretched"
        assert noise.slope < real.slope
        assert noise.slope < 1.1, "a guesser must not be amplified"

    def test_a_pure_guesser_is_shrunk_toward_nothing(self):
        """Which correctly leaves it with no edges to bet."""
        cal = fit_margin_calibration(pairs(1.0, model_error=200.0))
        assert cal.fitted, "a model with no signal must still be corrected"
        assert cal.apply(20.0) < 5.0

    def test_an_anti_correlated_model_is_refused_not_flipped(self):
        """Negating a backwards model would be a finding, not a fix."""
        cal = fit_margin_calibration([(-p, a) for p, a in pairs(1.0)])
        assert not cal.fitted and cal.apply(7.0) == 7.0


class TestRefusingToGuess:
    def test_too_few_games_measures_but_does_not_correct(self):
        cal = fit_margin_calibration(pairs(0.6, n=MIN_GAMES - 1))
        assert not cal.fitted
        assert cal.apply(10.0) == 10.0, "nothing may change before the threshold"
        assert cal.raw_slope is not None, "it should still be measured and shown"

    def test_a_weak_signal_is_shrunk_toward_no_correction(self):
        """Noise must not buy a correction it has not earned.

        Same true compression, same sample size; only how predictable
        the games are differs. The scattered fit has to move less.
        """
        tight = fit_margin_calibration(pairs(0.75, outcome_noise=3.0))
        noisy = fit_margin_calibration(pairs(0.75, outcome_noise=45.0))
        assert tight.fitted and noisy.fitted
        assert abs(tight.slope - 1) > abs(noisy.slope - 1)

    def test_an_absurd_fit_is_disobeyed(self):
        """A slope outside the band is a broken measurement."""
        silly = [(p * 0.001, a) for p, a in pairs(1.0, outcome_noise=0.5)]
        cal = fit_margin_calibration(silly)
        assert not cal.fitted
        assert cal.raw_slope is not None and cal.raw_slope > MAX_SLOPE

    def test_a_flat_model_has_no_scale_to_correct(self):
        cal = fit_margin_calibration([(3.0, 7.0)] * 200)
        assert not cal.fitted and cal.apply(9.0) == 9.0

    def test_an_empty_season_is_the_identity(self):
        cal = fit_margin_calibration([])
        assert not cal.fitted and cal.apply(-4.5) == -4.5

    def test_the_intercept_cannot_run_away(self):
        cal = fit_margin_calibration(pairs(1.0, intercept=40.0, outcome_noise=2.0))
        assert abs(cal.intercept) <= 3.0

    def test_the_default_changes_nothing(self):
        assert IDENTITY.apply(13.7) == 13.7
        assert not IDENTITY.fitted

    def test_it_explains_itself_either_way(self):
        assert "need" in MarginCalibration(games=4).describe()
        assert "measured" in fit_margin_calibration(pairs(0.7)).describe()


class TestReplayHonesty:
    """Fitting on completed games is only safe if the replay is honest."""

    @pytest.fixture()
    def seeded(self, tmp_path):
        from cfbpicks.config import Config
        from cfbpicks.models import Game, Rating
        from cfbpicks.storage import Storage

        store = Storage(tmp_path / "t.sqlite")
        cfg = Config(root=tmp_path)
        cfg.betting.fbs_only = False

        games, ratings = [], []
        for week in (1, 2, 3):
            for i in range(30):
                home, away = f"H{i}", f"A{i}"
                games.append(Game(
                    game_id=f"w{week}g{i}", season=2026, week=week,
                    kickoff=None, home_team=home, away_team=away,
                    home_score=21 + i, away_score=14,
                ))
                # Point-in-time ratings for each week, plus a
                # season-final source that knows how the year ended.
                for source, pit in (("cfbd_elo", True), ("cfbd_sp_plus", False)):
                    ratings.append(Rating(source, 2026, week if pit else 0,
                                          home, 12.0, point_in_time=pit))
                    ratings.append(Rating(source, 2026, week if pit else 0,
                                          away, 4.0, point_in_time=pit))
        store.upsert_games(games)
        store.upsert_ratings(ratings)
        yield store, cfg
        store.close()

    def test_a_week_is_never_calibrated_on_its_own_results(self, seeded):
        from cfbpicks.engine.calibration import collect_margin_pairs

        store, cfg = seeded
        everything = collect_margin_pairs(store, cfg, 2026)
        before_three = collect_margin_pairs(store, cfg, 2026, before_week=3)
        assert len(everything) == 90
        assert len(before_three) == 60, "week 3 must not calibrate week 3"

    def test_week_one_has_nothing_to_learn_from(self, seeded):
        from cfbpicks.engine.calibration import collect_margin_pairs

        store, cfg = seeded
        assert collect_margin_pairs(store, cfg, 2026, before_week=1) == []

    def test_season_final_ratings_are_kept_out_of_the_replay(self, seeded):
        """The bug that once produced a fake +20% ROI.

        A season-final rating already knows how the season ended, so
        replaying with it makes the model look far better calibrated
        than it is and the correction comes out near one.
        """
        from cfbpicks.engine.calibration import collect_margin_pairs

        store, cfg = seeded
        store.upsert_ratings([])
        pairs_ = collect_margin_pairs(store, cfg, 2026)
        assert pairs_, "the point-in-time source should still produce pairs"

        # With only the contaminated source present there is nothing
        # safe to replay from, and the fit must come back empty rather
        # than quietly using it.
        from cfbpicks.models import Rating
        from cfbpicks.storage import Storage

        store2 = Storage(":memory:")
        store2.upsert_games(store.games(2026))
        store2.upsert_ratings([
            Rating("cfbd_sp_plus", 2026, 0, t, 10.0, point_in_time=False)
            for t in ("H1", "A1")
        ])
        assert collect_margin_pairs(store2, cfg, 2026) == []
        store2.close()

    def test_an_uncompleted_game_contributes_nothing(self, seeded):
        from cfbpicks.engine.calibration import collect_margin_pairs
        from cfbpicks.models import Game

        store, cfg = seeded
        store.upsert_games([Game(
            game_id="future", season=2026, week=2, kickoff=None,
            home_team="H1", away_team="A1",
        )])
        assert len(collect_margin_pairs(store, cfg, 2026)) == 90


class TestCli:
    def test_it_reports_when_there_is_nothing_to_replay(self, tmp_path):
        from click.testing import CliRunner

        from cfbpicks.cli import main

        result = CliRunner().invoke(
            main, ["--db", str(tmp_path / "t.sqlite"), "calibrate", "--season", "2026"]
        )
        assert result.exit_code == 0, result.output
        assert "No completed games" in result.output

    def test_it_shows_the_correction_it_would_apply(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from cfbpicks import pipeline as pipeline_mod
        from cfbpicks.cli import main
        from cfbpicks.engine.calibration import MarginCalibration

        monkeypatch.setattr(
            pipeline_mod.Pipeline, "margin_calibration",
            lambda self, season, before_week=None: MarginCalibration(
                games=240, slope=1.42, intercept=0.4, fitted=True,
                raw_slope=1.44, stderr=0.09, residual_sd=16.2,
            ),
        )
        result = CliRunner().invoke(
            main, ["--db", str(tmp_path / "t.sqlite"), "calibrate", "--season", "2026"]
        )
        assert result.exit_code == 0, result.output
        assert "240" in result.output
        assert "1.42" in result.output
        # The worked example is the part a reader actually understands.
        assert "+14.6" in result.output

    def test_an_unfitted_correction_says_why(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from cfbpicks import pipeline as pipeline_mod
        from cfbpicks.cli import main
        from cfbpicks.engine.calibration import MarginCalibration

        monkeypatch.setattr(
            pipeline_mod.Pipeline, "margin_calibration",
            lambda self, season, before_week=None: MarginCalibration(
                games=40, raw_slope=1.6, stderr=0.2,
            ),
        )
        result = CliRunner().invoke(
            main, ["--db", str(tmp_path / "t.sqlite"), "calibrate", "--season", "2026"]
        )
        assert "Not applied" in result.output
        assert "120" in result.output, "it should say how many games are needed"


class TestItReachesThePredictions:
    def test_a_fitted_correction_changes_the_margin(self):
        from cfbpicks.config import Config
        from cfbpicks.engine.predict import PredictionInputs, Predictor
        from cfbpicks.models import Game, Rating

        cfg = Config()
        game = Game(game_id="g", season=2026, week=4, kickoff=None,
                    home_team="Georgia", away_team="Auburn")
        ratings = [Rating("cfbd_elo", 2026, 4, "Georgia", 20.0, point_in_time=True),
                   Rating("cfbd_elo", 2026, 4, "Auburn", 5.0, point_in_time=True)]

        plain = Predictor(cfg).predict_week(
            PredictionInputs(games=[game], ratings=ratings)
        )[0]
        stretched = Predictor(cfg).predict_week(PredictionInputs(
            games=[game], ratings=ratings,
            calibration=MarginCalibration(games=300, slope=1.4, fitted=True),
        ))[0]

        assert stretched.projected_margin == pytest.approx(
            plain.projected_margin * 1.4, abs=0.02
        )
        # Everything downstream has to move with it, or the spread and
        # the moneyline would disagree about the same game.
        assert stretched.home_win_prob > plain.home_win_prob
        assert abs(stretched.fair_home_spread) > abs(plain.fair_home_spread)
        assert stretched.components["uncalibrated_margin"] == pytest.approx(
            plain.projected_margin, abs=0.02
        )

    def test_the_uncorrected_path_is_untouched(self):
        from cfbpicks.config import Config
        from cfbpicks.engine.predict import PredictionInputs, Predictor
        from cfbpicks.models import Game, Rating

        cfg = Config()
        game = Game(game_id="g", season=2026, week=4, kickoff=None,
                    home_team="Georgia", away_team="Auburn")
        ratings = [Rating("cfbd_elo", 2026, 4, "Georgia", 20.0, point_in_time=True)]
        pred = Predictor(cfg).predict_week(
            PredictionInputs(games=[game], ratings=ratings)
        )[0]
        assert "calibration" not in pred.components
