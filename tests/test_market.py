"""De-vigging, consensus and line shopping."""

from __future__ import annotations

import pytest

from cfbpicks.engine.market import (
    MAX_PLAUSIBLE_HOLD, best_quote_for_side, consensus_for_game, line_movement,
)
from cfbpicks.models import MarketQuote


def quote(book, market, side, line, price, fetched_at=None):
    return MarketQuote("g1", book, market, side, line, price, fetched_at)


def spread_market(*specs):
    out = []
    for book, line, home_price, away_price in specs:
        out.append(quote(book, "spread", "home", line, home_price))
        out.append(quote(book, "spread", "away", -line, away_price))
    return out


class TestConsensus:
    def test_even_juice_implies_the_posted_number(self):
        view = consensus_for_game(
            spread_market(("DK", -7.0, -110, -110)), "spread", margin_sd=16.0, total_sd=13.5
        )
        assert view.implied_value == pytest.approx(7.0)
        assert view.consensus_line == -7.0

    def test_extra_juice_on_the_favourite_implies_a_bigger_number(self):
        # -7 at -125/+105 is really more than a 7 point favourite.
        view = consensus_for_game(
            spread_market(("DK", -7.0, -125, 105)), "spread", margin_sd=16.0, total_sd=13.5
        )
        assert view.implied_value > 7.0

    def test_median_across_books(self):
        view = consensus_for_game(
            spread_market(("DK", -7.0, -110, -110), ("FD", -7.5, -110, -110), ("MGM", -6.5, -110, -110)),
            "spread", margin_sd=16.0, total_sd=13.5,
        )
        assert view.implied_value == pytest.approx(7.0)
        assert view.book_count == 3

    def test_hold_is_reported(self):
        view = consensus_for_game(
            spread_market(("DK", -3.0, -110, -110)), "spread", margin_sd=16.0, total_sd=13.5
        )
        assert view.median_hold == pytest.approx(0.0476, abs=1e-4)

    def test_one_sided_books_are_ignored(self):
        quotes = [quote("DK", "spread", "home", -7.0, -110)]
        assert consensus_for_game(quotes, "spread", margin_sd=16.0, total_sd=13.5) is None

    def test_implausible_hold_is_rejected_as_bad_data(self):
        # A pair booked to well over the plausible band is a broken feed,
        # and de-vigging it would produce a confident, wrong number.
        quotes = spread_market(("Bad", -7.0, -100000, -100000))
        assert consensus_for_game(quotes, "spread", margin_sd=16.0, total_sd=13.5) is None

    def test_totals_recover_the_projected_total(self):
        quotes = [
            quote("DK", "total", "over", 52.5, -110),
            quote("DK", "total", "under", 52.5, -110),
        ]
        view = consensus_for_game(quotes, "total", margin_sd=16.0, total_sd=13.5)
        assert view.implied_value == pytest.approx(52.5)

    def test_moneyline_implies_a_margin(self):
        quotes = [
            quote("DK", "moneyline", "home", None, -200),
            quote("DK", "moneyline", "away", None, 170),
        ]
        view = consensus_for_game(quotes, "moneyline", margin_sd=16.0, total_sd=13.5)
        assert view.implied_value > 0  # home is favoured
        assert view.consensus_line is None

    def test_empty_input(self):
        assert consensus_for_game([], "spread", margin_sd=16.0, total_sd=13.5) is None


class TestLineShopping:
    def test_picks_the_best_number_for_the_home_side(self):
        quotes = spread_market(("DK", -7.5, -110, -110), ("FD", -6.5, -110, -110))
        best = best_quote_for_side(quotes, "spread", "home")
        assert (best.book, best.line) == ("FD", -6.5)

    def test_picks_the_best_number_for_the_away_side(self):
        quotes = spread_market(("DK", -7.5, -110, -110), ("FD", -6.5, -110, -110))
        best = best_quote_for_side(quotes, "spread", "away")
        assert (best.book, best.line) == ("DK", 7.5)

    def test_price_breaks_a_tie_on_the_number(self):
        quotes = spread_market(("DK", -7.0, -115, -105), ("FD", -7.0, -105, -115))
        best = best_quote_for_side(quotes, "spread", "home")
        assert best.book == "FD"

    def test_best_moneyline_is_the_biggest_payout(self):
        quotes = [
            quote("DK", "moneyline", "home", None, -200),
            quote("FD", "moneyline", "home", None, -180),
        ]
        assert best_quote_for_side(quotes, "moneyline", "home").book == "FD"

    def test_missing_side_returns_none(self):
        assert best_quote_for_side([], "spread", "home") is None


class TestLineMovement:
    def test_reports_points_moved(self):
        from datetime import datetime, timezone
        early = datetime(2026, 9, 1, tzinfo=timezone.utc)
        late = datetime(2026, 9, 5, tzinfo=timezone.utc)
        history = [
            quote("DK", "spread", "home", -6.5, -110, early),
            quote("DK", "spread", "home", -7.5, -110, late),
        ]
        assert line_movement(history) == pytest.approx(-1.0)

    def test_needs_two_points(self):
        assert line_movement([]) is None
