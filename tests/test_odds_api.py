"""The Odds API: quota discipline and event matching."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cfbpicks.config import Config, ProviderConfig
from cfbpicks.models import Game
from cfbpicks.providers.odds_api import OddsApiProvider, Quota
from cfbpicks.util.http import ProviderError

PAYLOAD = json.loads((Path(__file__).parent / "data" / "odds_api_sample.json").read_text())


def dt(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


SCHEDULE = [
    Game("osu-mich", 2026, 3, dt("2026-09-19T23:30:00Z"), "Ohio State", "Michigan"),
    Game("tex-miss", 2026, 3, dt("2026-09-20T16:00:00Z"), "Texas", "Mississippi"),
]


@pytest.fixture()
def provider(tmp_path):
    config = Config(root=tmp_path)
    config.cache_dir = str(tmp_path / "cache")
    return OddsApiProvider(config, ProviderConfig("odds_api", api_key="test-key"))


def run(provider, payload=PAYLOAD, games=SCHEDULE, headers=None):
    """Drive fetch_odds against a canned payload, no network."""
    provider.client.get_json = lambda *a, **k: payload
    # `headers or {...}` would treat an empty dict as "use the defaults",
    # which is exactly the case one of these tests needs to exercise.
    provider.client.last_response_headers = {
        "x-requests-remaining": "417", "x-requests-used": "83", "x-requests-last": "3",
    } if headers is None else headers
    return provider.fetch_odds(2026, 3, games=games)


class TestMatching:
    def test_events_map_onto_scheduled_games(self, provider):
        quotes = run(provider)
        assert {q.game_id for q in quotes} == {"osu-mich", "tex-miss"}

    def test_provider_spellings_are_reconciled(self, provider):
        """"Ole Miss Rebels" and "Mississippi" are the same program."""
        quotes = run(provider)
        assert any(q.game_id == "tex-miss" for q in quotes)

    def test_unscheduled_events_are_reported_not_dropped_silently(self, provider):
        run(provider)
        assert provider.unmatched == ["Another Phantom @ Some Team Nobody Scheduled"]

    def test_neutral_site_venue_disagreement_still_matches(self, provider):
        """The feeds disagree about who is home at neutral sites."""
        flipped = [Game("osu-mich", 2026, 3, dt("2026-09-19T23:30:00Z"),
                        "Michigan", "Ohio State", neutral_site=True)]
        quotes = run(provider, games=flipped)
        assert any(q.game_id == "osu-mich" for q in quotes)

    def test_sides_follow_the_schedule_not_the_feed(self, provider):
        """A flipped venue must not flip which side a price belongs to."""
        flipped = [Game("osu-mich", 2026, 3, dt("2026-09-19T23:30:00Z"),
                        "Michigan", "Ohio State", neutral_site=True)]
        quotes = run(provider, games=flipped)
        spread = {q.side: q.line for q in quotes
                  if q.market == "spread" and q.book == "DraftKings"}
        # Michigan is home in this schedule, and Michigan is the +10.5 dog.
        assert spread["home"] == 10.5
        assert spread["away"] == -10.5

    def test_missing_schedule_is_flagged(self, provider):
        quotes = run(provider, games=[])
        assert quotes == []
        assert any("fetch games before odds" in w for w in provider.warnings)


class TestQuoteExtraction:
    def test_all_three_markets_are_read(self, provider):
        quotes = [q for q in run(provider) if q.game_id == "osu-mich"]
        assert {q.market for q in quotes} == {"spread", "total", "moneyline"}

    def test_spread_sides_keep_their_own_number(self, provider):
        quotes = run(provider)
        dk = {q.side: q for q in quotes
              if q.game_id == "osu-mich" and q.market == "spread" and q.book == "DraftKings"}
        assert dk["home"].line == -10.5 and dk["away"].line == 10.5

    def test_totals_carry_the_same_line_both_ways(self, provider):
        quotes = [q for q in run(provider) if q.market == "total"]
        assert {q.side for q in quotes} == {"over", "under"}
        assert {q.line for q in quotes} == {54.5}

    def test_juice_is_preserved_per_side(self, provider):
        quotes = {q.side: q.price for q in run(provider) if q.market == "total"}
        assert quotes["over"] == -105 and quotes["under"] == -115

    def test_moneylines_have_no_line(self, provider):
        quotes = [q for q in run(provider) if q.market == "moneyline"]
        assert all(q.line is None for q in quotes)
        assert {q.price for q in quotes} == {-450, 350}

    def test_multiple_books_are_kept_for_shopping(self, provider):
        books = {q.book for q in run(provider) if q.game_id == "osu-mich"}
        assert books == {"DraftKings", "FanDuel"}

    def test_unrecognised_outcome_names_are_reported(self, provider):
        payload = [{
            "commence_time": "2026-09-19T23:30:00Z",
            "home_team": "Ohio State Buckeyes", "away_team": "Michigan Wolverines",
            "bookmakers": [{"title": "DK", "markets": [{"key": "spreads", "outcomes": [
                {"name": "Ohio State Buckeyes", "price": -110, "point": -10.5},
                {"name": "Totally Different School", "price": -110, "point": 10.5},
            ]}]}],
        }]
        run(provider, payload=payload)
        assert any("Totally Different School" in w for w in provider.warnings)


class TestQuota:
    def test_balance_is_read_from_the_response(self, provider):
        run(provider)
        assert provider.quota.remaining == 417
        assert provider.quota.used == 83
        assert provider.quota.last_cost == 3

    def test_cost_is_markets_times_regions(self, provider):
        # The free tier bills per market per region, not per request.
        assert provider.estimate_cost(["spread", "total", "moneyline"]) == 3
        assert provider.estimate_cost(["spread"]) == 1
        provider.settings.options["regions"] = "us,uk"
        assert provider.estimate_cost(["spread", "total"]) == 4

    def test_missing_headers_are_tolerated(self, provider):
        run(provider, headers={})
        assert provider.quota.remaining is None
        assert "quota unknown" in provider.quota.describe()

    def test_a_floor_stops_the_call_before_it_spends(self, provider):
        provider.settings.options["min_quota_remaining"] = 100
        provider.quota = Quota(remaining=101)
        with pytest.raises(ProviderError, match="Refusing to call"):
            run(provider)

    def test_calls_proceed_while_above_the_floor(self, provider):
        provider.settings.options["min_quota_remaining"] = 100
        provider.quota = Quota(remaining=400)
        assert run(provider)

    def test_no_floor_means_no_guard(self, provider):
        provider.quota = Quota(remaining=1)
        assert run(provider)


class TestConfiguration:
    def test_books_can_be_restricted(self, provider):
        captured = {}

        def fake(url, params=None, **kwargs):
            captured.update(params or {})
            return PAYLOAD

        provider.settings.options["books"] = ["draftkings", "fanduel"]
        provider.client.get_json = fake
        provider.client.last_response_headers = {}
        provider.fetch_odds(2026, 3, games=SCHEDULE)
        assert captured["bookmakers"] == "draftkings,fanduel"

    def test_only_requested_markets_are_asked_for(self, provider):
        captured = {}

        def fake(url, params=None, **kwargs):
            captured.update(params or {})
            return []

        provider.client.get_json = fake
        provider.client.last_response_headers = {}
        provider.fetch_odds(2026, 3, games=SCHEDULE, markets=["spread"])
        assert captured["markets"] == "spreads"

    def test_no_valid_markets_makes_no_call(self, provider):
        called = []
        provider.client.get_json = lambda *a, **k: called.append(1) or []
        assert provider.fetch_odds(2026, 3, games=SCHEDULE, markets=["nonsense"]) == []
        assert not called

    def test_the_key_is_required(self, tmp_path):
        config = Config(root=tmp_path)
        bare = OddsApiProvider(config, ProviderConfig("odds_api"))
        assert not bare.configured
        with pytest.raises(Exception, match="API key"):
            bare.fetch_odds(2026, 3, games=SCHEDULE)
