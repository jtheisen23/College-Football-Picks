"""The probability and staking maths everything else is built on."""

from __future__ import annotations

import math

import pytest

from cfbpicks.util.odds import (
    american_to_decimal, american_to_prob, cover_probability, decimal_to_american,
    devig_power, devig_proportional, expected_value, kelly_fraction, normal_cdf,
    normal_ppf, over_probability, prob_to_american, push_probability, stake_units, vig,
)


class TestConversions:
    @pytest.mark.parametrize("american,decimal", [(100, 2.0), (-110, 1 + 100 / 110), (150, 2.5), (-200, 1.5)])
    def test_american_to_decimal(self, american, decimal):
        assert american_to_decimal(american) == pytest.approx(decimal)

    @pytest.mark.parametrize("american", [-500, -110, -101, 105, 250, 1200])
    def test_round_trips(self, american):
        assert decimal_to_american(american_to_decimal(american)) == pytest.approx(american)
        assert prob_to_american(american_to_prob(american)) == pytest.approx(american)

    def test_standard_juice_implies_524(self):
        # -110 is the classic break-even number every bettor knows.
        assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError):
            american_to_decimal(0)


class TestDevig:
    def test_vig_measures_overround(self):
        assert vig([-110, -110]) == pytest.approx(0.0476, abs=1e-4)

    def test_proportional_sums_to_one(self):
        fair = devig_proportional([-140, 120])
        assert sum(fair) == pytest.approx(1.0)

    def test_symmetric_market_is_even(self):
        assert devig_proportional([-110, -110]) == pytest.approx([0.5, 0.5])

    def test_power_sums_to_one(self):
        fair = devig_power([-600, 425])
        assert sum(fair) == pytest.approx(1.0)

    def test_power_favours_the_favourite_more_than_proportional(self):
        # The favourite-longshot bias means longshot prices are inflated;
        # the power method is meant to correct further than proportional.
        prop = devig_proportional([-600, 425])
        power = devig_power([-600, 425])
        assert power[0] > prop[0]

    def test_already_fair_market_is_unchanged(self):
        assert devig_power([100, -100]) == pytest.approx([0.5, 0.5], abs=1e-6)


class TestNormal:
    def test_cdf_known_values(self):
        assert normal_cdf(0) == pytest.approx(0.5)
        assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)

    def test_ppf_inverts_cdf(self):
        for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
            assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-6)

    def test_ppf_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            normal_ppf(0.0)


class TestGameProbabilities:
    def test_pickem_is_a_coin_flip(self):
        assert cover_probability(0.0, 0.0, 16.0) == pytest.approx(0.5)

    def test_home_covers_when_projection_beats_the_number(self):
        # Projected to win by 10, laying only 7 -> better than even.
        assert cover_probability(10.0, -7.0, 16.0) > 0.5

    def test_home_fails_to_cover_a_bigger_number(self):
        assert cover_probability(10.0, -14.0, 16.0) < 0.5

    def test_the_two_sides_are_complementary(self):
        home = cover_probability(6.0, -3.5, 16.0)
        away = cover_probability(-6.0, 3.5, 16.0)
        assert home + away == pytest.approx(1.0)

    def test_push_only_on_whole_numbers(self):
        assert push_probability(7.0, -7.5, 16.0) == 0.0
        assert push_probability(7.0, -7.0, 16.0) > 0.0

    def test_push_is_likeliest_when_the_projection_sits_on_the_number(self):
        on = push_probability(7.0, -7.0, 16.0)
        off = push_probability(20.0, -7.0, 16.0)
        assert on > off

    def test_over_probability_tracks_the_projection(self):
        assert over_probability(60.0, 52.5, 13.5) > 0.5
        assert over_probability(45.0, 52.5, 13.5) < 0.5


class TestStaking:
    def test_no_edge_means_no_bet(self):
        # 52.38% at -110 is exactly break-even.
        assert kelly_fraction(american_to_prob(-110), -110) == pytest.approx(0.0, abs=1e-9)

    def test_negative_edge_is_negative_kelly(self):
        assert kelly_fraction(0.45, -110) < 0

    def test_edge_produces_a_positive_stake(self):
        assert kelly_fraction(0.58, -110) > 0

    def test_fractional_kelly_scales_the_stake(self):
        full = stake_units(0.58, -110, kelly_multiplier=1.0, max_units=99)
        quarter = stake_units(0.58, -110, kelly_multiplier=0.25, max_units=99)
        assert quarter == pytest.approx(full / 4)

    def test_stake_is_capped(self):
        assert stake_units(0.95, 200, kelly_multiplier=1.0, max_units=3.0) == 3.0

    def test_expected_value_is_zero_at_fair_odds(self):
        assert expected_value(0.5, 100) == pytest.approx(0.0)

    def test_pushes_lift_expected_value(self):
        without = expected_value(0.50, -110, push_prob=0.0)
        with_push = expected_value(0.50, -110, push_prob=0.08)
        assert with_push > without
