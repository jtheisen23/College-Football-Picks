"""The power ratings the engine fits for itself."""

from __future__ import annotations

import random
import statistics

import pytest

from cfbpicks.config import Config
from cfbpicks.models import Game, MarketQuote
from cfbpicks.pipeline import Pipeline
from cfbpicks.ratings.regression import SOURCE, fit_margin_ratings
from cfbpicks.storage import Storage


def game(home, away, home_score, away_score, week=1, neutral=False, gid=None):
    return Game(gid or f"{away}-at-{home}-{week}", 2025, week, None, home, away,
                neutral_site=neutral, home_score=home_score, away_score=away_score)


def simulate_season(seed=7, teams=40, weeks=12, hfa=2.2, noise=16.0):
    """A season with known team strengths, for measuring recovery."""
    rng = random.Random(seed)
    names = [f"T{i:02d}" for i in range(teams)]
    truth = {t: rng.uniform(-20, 20) for t in names}
    games, gid = [], 0
    for week in range(1, weeks + 1):
        pool = names[:]
        rng.shuffle(pool)
        for k in range(0, len(pool) - 1, 2):
            home, away = pool[k], pool[k + 1]
            gid += 1
            margin = truth[home] - truth[away] + hfa + rng.gauss(0, noise)
            hs = max(0, int(round(24 + margin / 2)))
            games.append(game(home, away, hs, max(0, int(round(hs - margin))),
                              week=week, gid=f"g{gid}"))
    return truth, games


class TestBasicFit:
    def test_better_team_rates_higher(self):
        games = [game("A", "B", 35, 10), game("B", "A", 20, 24, week=2)]
        fit = fit_margin_ratings(games)
        assert fit.ratings["A"] > fit.ratings["B"]

    def test_ratings_are_centred_on_average(self):
        _, games = simulate_season()
        fit = fit_margin_ratings(games)
        assert statistics.fmean(fit.ratings.values()) == pytest.approx(0.0, abs=1e-9)

    def test_no_games_is_safe(self):
        fit = fit_margin_ratings([])
        assert fit.ratings == {} and fit.games_used == 0

    def test_unplayed_games_are_ignored(self):
        fit = fit_margin_ratings([Game("g", 2025, 1, None, "A", "B")])
        assert fit.games_used == 0

    def test_appearances_are_counted(self):
        fit = fit_margin_ratings([game("A", "B", 30, 20), game("A", "C", 28, 21, week=2)])
        assert fit.appearances["A"] == 2 and fit.appearances["B"] == 1


class TestRecovery:
    """It must actually recover the strengths that generated the data."""

    def test_ordering_matches_the_truth(self):
        truth, games = simulate_season(teams=40, weeks=14)
        fit = fit_margin_ratings(games)
        ranked_truth = sorted(truth, key=lambda t: -truth[t])
        ranked_fit = sorted(fit.ratings, key=lambda t: -fit.ratings[t])
        # Rank correlation, not exact order: 16 points of per-game noise
        # means the ordering can never be recovered perfectly.
        pos = {t: i for i, t in enumerate(ranked_fit)}
        errors = [abs(pos[t] - i) for i, t in enumerate(ranked_truth)]
        assert statistics.fmean(errors) < len(truth) * 0.2

    def test_home_field_is_recovered(self):
        _, games = simulate_season(teams=60, weeks=14, hfa=3.0)
        fit = fit_margin_ratings(games)
        assert 1.5 < fit.home_field < 4.5

    def test_residual_sd_approximates_the_true_noise(self):
        _, games = simulate_season(teams=60, weeks=14, noise=16.0)
        fit = fit_margin_ratings(games)
        assert 13.0 < fit.residual_sd < 19.0

    def test_predictions_are_calibrated_not_shrunk(self):
        """Predicted margins must not be systematically too small.

        A model whose margins are uniformly undersized quietly bets every
        underdog on the board, so the regression slope of actual on
        predicted is checked directly.
        """
        truth, games = simulate_season(teams=60, weeks=16, hfa=2.2)
        train = [g for g in games if g.week <= 12]
        held = [g for g in games if g.week > 12]
        fit = fit_margin_ratings(train)

        xs, ys = [], []
        for g in held:
            if g.home_team in fit.ratings and g.away_team in fit.ratings:
                xs.append(fit.ratings[g.home_team] - fit.ratings[g.away_team] + fit.home_field)
                ys.append(truth[g.home_team] - truth[g.away_team] + 2.2)
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        assert 0.85 < slope < 1.25


class TestRegularisation:
    def test_thin_evidence_is_shrunk_toward_average(self):
        """One huge win must not make a team the best in the country."""
        _, games = simulate_season(teams=30, weeks=10)
        games.append(game("Newcomer", "T00", 70, 0, week=11))
        fit = fit_margin_ratings(games)
        best_established = max(v for t, v in fit.ratings.items() if t != "Newcomer")
        assert fit.ratings["Newcomer"] < best_established

    def test_more_ridge_shrinks_harder(self):
        _, games = simulate_season(teams=30, weeks=6)
        loose = fit_margin_ratings(games, ridge=1.0)
        tight = fit_margin_ratings(games, ridge=20.0)
        spread = lambda f: statistics.pstdev(f.ratings.values())
        assert spread(tight) < spread(loose)

    def test_capping_margins_shrinks_predictions(self):
        """Documents why capping is off by default."""
        _, games = simulate_season(teams=40, weeks=12)
        uncapped = fit_margin_ratings(games, margin_cap=None)
        capped = fit_margin_ratings(games, margin_cap=14.0)
        assert statistics.pstdev(capped.ratings.values()) < statistics.pstdev(uncapped.ratings.values())

    def test_prior_carries_when_there_are_no_games(self):
        fit = fit_margin_ratings([], prior={"A": 10.0})
        assert fit.ratings == {"A": 10.0}

    def test_prior_pulls_a_team_with_little_evidence(self):
        games = [game("A", "B", 21, 20)]
        high = fit_margin_ratings(games, prior={"A": 30.0}, ridge=5.0)
        none = fit_margin_ratings(games, ridge=5.0)
        assert high.ratings["A"] > none.ratings["A"]


class TestVenue:
    def test_neutral_games_get_no_home_field(self):
        _, games = simulate_season(teams=40, weeks=12, hfa=0.0)
        neutral = [
            Game(g.game_id, g.season, g.week, None, g.home_team, g.away_team,
                 neutral_site=True, home_score=g.home_score, away_score=g.away_score)
            for g in games
        ]
        fit = fit_margin_ratings(neutral, default_home_field=2.2)
        # Nothing in the slate says anything about venue, so the term
        # falls back to the configured default...
        assert fit.home_field == pytest.approx(2.2, abs=1e-6)
        # ...and, crucially, the ratings themselves still come out. An
        # unidentifiable home-field term used to sink the whole solve.
        assert len(fit.ratings) == 40

    def test_home_field_is_learned_from_the_non_neutral_games(self):
        """A mixed slate estimates venue from the games that have one."""
        _, games = simulate_season(teams=60, weeks=14, hfa=6.0)
        fit = fit_margin_ratings(games)
        assert fit.home_field > 4.0


class TestPointInTimeGuarantee:
    """The property that makes these ratings safe to backtest with."""

    @pytest.fixture()
    def storage(self, tmp_path):
        store = Storage(tmp_path / "t.sqlite")
        yield store
        store.close()

    def _config(self, tmp_path):
        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        return cfg

    def test_a_weeks_rating_ignores_that_week_and_later(self, tmp_path):
        cfg = self._config(tmp_path)
        early = [game("A", "B", 30, 10, week=1), game("C", "D", 24, 21, week=1)]
        later = [game("A", "B", 3, 60, week=2), game("C", "D", 55, 0, week=2)]

        with Pipeline(cfg) as pipeline:
            pipeline.storage.upsert_games(early + later)
            fits = pipeline.compute_ratings(2025, carry_prior=False)

        # Week 2's rating is fitted on week 1 only, so adding week 2's
        # blowout must not change it.
        assert 2 in fits
        with Pipeline(cfg) as pipeline:
            week2 = {r.team: r.rating for r in pipeline.storage.ratings(2025, 2, source=SOURCE)}
        assert week2["A"] > week2["B"], "week 2 must reflect week 1, not week 2's result"

    def test_stored_ratings_are_flagged_point_in_time(self, tmp_path):
        cfg = self._config(tmp_path)
        _, games = simulate_season(teams=20, weeks=4)
        with Pipeline(cfg) as pipeline:
            pipeline.storage.upsert_games(games)
            pipeline.compute_ratings(2025, carry_prior=False)
            stored = pipeline.storage.ratings(2025, source=SOURCE)
        assert stored and all(r.point_in_time for r in stored)

    def test_week_one_is_skipped_without_a_prior(self, tmp_path):
        cfg = self._config(tmp_path)
        with Pipeline(cfg) as pipeline:
            pipeline.storage.upsert_games([game("A", "B", 30, 10, week=1)])
            fits = pipeline.compute_ratings(2025, carry_prior=False)
        # Nothing was played before week 1, so there is nothing to fit.
        assert 1 not in fits

    def test_backtest_finds_no_edge_against_an_omniscient_market(self, tmp_path):
        """The decisive leakage test.

        The book is given the true margin of every game. A model with no
        hindsight cannot beat that, so anything better than break-even
        here means results are leaking into the ratings — which is
        exactly the bug this whole mechanism exists to prevent.
        """
        from cfbpicks.backtest import backtest_season

        cfg = self._config(tmp_path)
        cfg.betting.markets = ["spread"]
        rng = random.Random(3)
        names = [f"T{i:02d}" for i in range(50)]
        truth = {t: rng.uniform(-20, 20) for t in names}

        games, quotes, gid = [], [], 0
        for week in range(1, 13):
            pool = names[:]
            rng.shuffle(pool)
            for k in range(0, len(pool) - 1, 2):
                home, away = pool[k], pool[k + 1]
                gid += 1
                gkey = f"g{gid}"
                true_margin = truth[home] - truth[away] + 2.2
                m = true_margin + rng.gauss(0, 16.0)
                hs = max(0, int(round(24 + m / 2)))
                games.append(game(home, away, hs, max(0, int(round(hs - m))),
                                  week=week, gid=gkey))
                line = round(-true_margin * 2) / 2
                quotes += [
                    MarketQuote(gkey, "DK", "spread", "home", line, -110),
                    MarketQuote(gkey, "DK", "spread", "away", -line, -110),
                ]

        with Pipeline(cfg) as pipeline:
            pipeline.storage.upsert_games(games)
            pipeline.storage.upsert_quotes(quotes)
            pipeline.compute_ratings(2025, carry_prior=False)
            result = backtest_season(pipeline.storage, cfg, 2025)

        assert result.rating_sources == [SOURCE]
        assert result.bets > 30, "the test is meaningless without a real sample"
        assert result.roi < 0.02, (
            f"ROI {result.roi:.1%} against a market that knows every true "
            "margin indicates results are leaking into the ratings"
        )
