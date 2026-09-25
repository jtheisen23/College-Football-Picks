"""Catching a model that is wrong about scale rather than about games."""

from __future__ import annotations

import random

import pytest

from cfbpicks.diagnostics import (
    MIN_GAMES,
    ScaleCheck,
    board_warning,
    market_scale,
)
from cfbpicks.models import ConsensusMarket, Prediction, Recommendation


def prediction(game_id: str, margin: float) -> Prediction:
    return Prediction(
        game_id=game_id, season=2026, week=4, home_team="H", away_team="A",
        projected_margin=margin, projected_total=52.0,
        home_win_prob=0.6, fair_home_spread=-margin,
        home_rating=10.0, away_rating=5.0, hfa_applied=2.2,
    )


def consensus(game_id: str, implied: float) -> dict:
    return {"spread": ConsensusMarket(
        game_id=game_id, market="spread",
        consensus_line=-implied, implied_value=implied,
    )}


def slate(scale: float, n: int = 60, noise: float = 0.0, seed: int = 7, tag: str = "g"):
    """A slate where the model's margins are ``scale`` x the market's.

    ``tag`` keeps game ids distinct between slates, so two of them can
    be merged without one silently overwriting the other's lines.
    """
    rng = random.Random(seed)
    preds, cons = [], {}
    for i in range(n):
        market = rng.gauss(0, 14)
        gid = f"{tag}{i}"
        preds.append(prediction(gid, market * scale + rng.gauss(0, noise)))
        cons[gid] = consensus(gid, market)
    return preds, cons


def rec(line: float, market: str = "spread", side: str = "home") -> Recommendation:
    return Recommendation(
        game_id="g", season=2026, week=4, matchup="A @ H", market=market,
        side=side, selection=f"H {line:+.1f}", line=line, price=-110,
        book="DraftKings", model_prob=0.56, market_prob=0.52, prob_edge=0.04,
        point_edge=2.0, expected_value=0.03, stake_units=1.0,
        kelly_fraction=0.02, tier="play",
    )


class TestMarketScale:
    def test_an_honest_model_sits_at_one(self):
        check = market_scale(*slate(1.0, noise=6.0))
        assert 0.9 < check.slope < 1.1
        assert check.healthy

    def test_it_recovers_a_known_compression(self):
        """The defect this exists to find: margins two-thirds too small."""
        check = market_scale(*slate(0.65, noise=4.0))
        assert 0.6 < check.slope < 0.72
        assert not check.healthy
        assert 1.35 < check.correction < 1.7

    def test_over_dispersion_is_caught_too(self):
        check = market_scale(*slate(1.6, noise=4.0))
        assert not check.healthy

    def test_noise_alone_does_not_condemn_a_model(self):
        """Being wrong about games is normal; only the slope is judged."""
        check = market_scale(*slate(1.0, noise=13.0, seed=3))
        assert check.healthy, check.slope

    def test_the_same_slope_is_judged_on_its_evidence(self):
        """The whole point of carrying a standard error.

        Two slates, both fitting a slope near 0.8. The noisy one is an
        honest model having a scattered week and must not be condemned;
        the tight one cannot be explained by luck and must be.
        """
        noisy = market_scale(*slate(1.0, noise=13.0, seed=3))
        tight = market_scale(*slate(0.8, noise=1.0))
        assert noisy.slope < 0.85 and tight.slope < 0.85
        assert noisy.healthy
        assert not tight.healthy
        assert noisy.stderr > tight.stderr

    def test_a_short_slate_says_nothing(self):
        check = market_scale(*slate(0.4, n=MIN_GAMES - 1))
        assert not check.measurable
        assert check.healthy, "too little evidence must not raise an alarm"
        assert check.correction is None

    def test_unpriced_games_are_skipped_not_counted(self):
        preds, cons = slate(1.0, n=30)
        preds.append(prediction("nolines", 21.0))
        assert market_scale(preds, cons).games == 30

    def test_a_market_with_no_spread_of_its_own_is_not_a_crash(self):
        preds = [prediction(f"g{i}", 3.0) for i in range(30)]
        cons = {f"g{i}": consensus(f"g{i}", 7.0) for i in range(30)}
        check = market_scale(preds, cons)
        assert check.slope is None and check.healthy

    def test_an_empty_week_is_inert(self):
        check = market_scale([], {})
        assert check.games == 0 and check.healthy and check.correction is None

    def test_it_describes_itself_for_the_banner(self):
        assert "0.65x" in market_scale(*slate(0.65)).describe()


class TestPooling:
    """A thin week borrows its verdict from the weeks beside it."""

    def test_adding_two_checks_matches_measuring_them_together(self):
        a_preds, a_cons = slate(0.7, n=40, noise=5.0, seed=1, tag="a")
        b_preds, b_cons = slate(0.7, n=40, noise=5.0, seed=2, tag="b")
        pooled = market_scale(a_preds, a_cons) + market_scale(b_preds, b_cons)
        together = market_scale(a_preds + b_preds, {**a_cons, **b_cons})
        assert pooled.games == together.games == 80
        assert pooled.slope == pytest.approx(together.slope)
        assert pooled.stderr == pytest.approx(together.stderr)

    def test_a_thin_week_alone_cannot_be_judged(self):
        thin = market_scale(*slate(0.6, n=8, noise=3.0))
        assert not thin.measurable and thin.healthy

    def test_but_pooled_with_a_full_week_it_is(self):
        """The defect is in the ratings, so last week's evidence counts."""
        thin = market_scale(*slate(0.6, n=8, noise=3.0, seed=11))
        full = market_scale(*slate(0.6, n=55, noise=3.0, seed=12))
        assert (thin + full).measurable
        assert not (thin + full).healthy

    def test_summing_nothing_is_the_empty_check(self):
        assert sum([ScaleCheck(), ScaleCheck()], ScaleCheck()).games == 0

    def test_an_empty_check_is_the_identity(self):
        one = market_scale(*slate(0.9, n=30, noise=4.0))
        assert (one + ScaleCheck()).slope == pytest.approx(one.slope)


class TestBoardWarning:
    def test_a_compressed_model_is_told_not_to_be_bet(self):
        note = board_warning([rec(+6.5)], market_scale(*slate(0.6, noise=3.0)))
        assert note and note.startswith("Do not bet")
        assert "too small" in note

    def test_an_over_dispersed_model_is_named_correctly(self):
        note = board_warning([rec(-6.5)], market_scale(*slate(1.8, noise=3.0)))
        assert note and "too large" in note

    def test_a_healthy_board_carries_no_banner(self):
        recs = [rec(+3.5), rec(-3.5), rec(+7.0), rec(-2.5)] * 4
        assert board_warning(recs, market_scale(*slate(1.0, noise=6.0))) is None

    def test_a_one_sided_board_is_flagged_on_its_own(self):
        """Even with a defensible slope, all-dogs is its own evidence."""
        recs = [rec(+3.5) for _ in range(14)] + [rec(-3.0)]
        note = board_warning(recs, ScaleCheck())
        assert note and "underdog" in note
        assert "14 of 15" in note

    def test_all_favourites_reads_the_other_way(self):
        recs = [rec(-3.5) for _ in range(14)] + [rec(+3.0)]
        note = board_warning(recs, ScaleCheck())
        assert note and "favourite" in note

    def test_a_short_board_is_allowed_to_lean(self):
        """Five dogs in a thin week is a coincidence, not a diagnosis."""
        note = board_warning([rec(+3.5) for _ in range(5)], market_scale(*slate(1.0)))
        assert note is None

    def test_the_same_share_is_judged_on_its_sample_size(self):
        """Why a flat percentage cannot do this job.

        Both splits are exactly 75% on the Over. Nine of twelve is
        inside what a fair model throws off by chance; twenty-one of
        twenty-eight is not, and it is the one the live board hit.
        """
        def totals(over, under):
            return ([rec(48.5, market="total", side="over")] * over
                    + [rec(48.5, market="total", side="under")] * under)

        clean = market_scale(*slate(1.0, noise=6.0))
        assert board_warning(totals(9, 3), clean) is None
        assert board_warning(totals(21, 7), clean) is not None

    def test_an_unchecked_board_does_not_pass_itself_off_as_checked(self):
        """Silence would read as a clean bill of health."""
        note = board_warning([rec(+3.5), rec(-2.0)], market_scale(*slate(1.0, n=6)))
        assert note and "not been checked" in note

    def test_an_empty_board_says_nothing(self):
        assert board_warning([], market_scale(*slate(0.4))) is None

    def test_totals_are_judged_in_their_own_market(self):
        recs = [rec(48.5, market="total", side="over") for _ in range(20)]
        note = board_warning(recs, market_scale(*slate(1.0)))
        assert note and "20 of 20 total picks are on the Over" in note
        assert "spread" not in note, "a totals lean is not a spread lean"

    def test_a_fixed_spread_book_does_not_excuse_broken_totals(self):
        """The live failure this check was blind to.

        Calibrating margins fixed the spread split and silenced the
        scale banner, while the totals stayed 3:1 on the Over. A board
        that is half right must not read as a board that is right.
        """
        spreads = [rec(+3.5) for _ in range(12)] + [rec(-3.0) for _ in range(9)]
        totals = ([rec(48.5, market="total", side="over") for _ in range(21)]
                  + [rec(48.5, market="total", side="under") for _ in range(7)])
        note = board_warning(spreads + totals, market_scale(*slate(1.0, noise=6.0)))
        assert note, "a 3:1 Over lean must still be caught"
        assert "21 of 28 total picks are on the Over" in note

    def test_both_markets_leaning_are_both_named(self):
        recs = ([rec(+3.5) for _ in range(14)]
                + [rec(48.5, market="total", side="over") for _ in range(14)])
        note = board_warning(recs, market_scale(*slate(1.0, noise=6.0)))
        assert "spread picks are on the underdog" in note
        assert "total picks are on the Over" in note
