"""Team name reconciliation across providers."""

from __future__ import annotations

import pytest

from cfbpicks.util.teams import canonical_team, game_key, is_shouted, match_teams


class TestCanonicalNames:
    @pytest.mark.parametrize("raw,expected", [
        ("Ole Miss", "Mississippi"),
        ("Mississippi Rebels", "Mississippi"),
        ("Miami (FL)", "Miami"),
        ("Miami Hurricanes", "Miami"),
        ("Miami (OH)", "Miami (OH)"),
        ("App State", "Appalachian State"),
        ("Texas A&M Aggies", "Texas A&M"),
        ("Boise St", "Boise State"),
        ("North Carolina State", "NC State"),
        ("Louisiana-Lafayette", "Louisiana"),
        ("Southern Cal", "USC"),
    ])
    def test_variants_collapse(self, raw, expected):
        assert canonical_team(raw) == expected

    def test_the_two_miamis_stay_apart(self):
        assert canonical_team("Miami (FL)") != canonical_team("Miami (OH)")

    @pytest.mark.parametrize("raw", ["LSU", "TCU", "BYU", "UCLA", "SMU"])
    def test_acronyms_keep_their_capitals(self, raw):
        assert canonical_team(raw) == raw

    def test_shouted_names_are_recased(self):
        # Sagarin prints the home team in capitals.
        assert canonical_team("TEXAS") == "Texas"
        assert canonical_team("OHIO STATE") == "Ohio State"

    def test_mascot_is_stripped_only_when_unambiguous(self):
        assert canonical_team("Ohio State Buckeyes") == "Ohio State"
        # "State" is part of the school, not a mascot.
        assert canonical_team("Ball State") == "Ball State"

    def test_empty_input_is_safe(self):
        assert canonical_team("") == ""

    def test_is_idempotent(self):
        for raw in ["Ole Miss", "TEXAS", "Boise St", "LSU"]:
            once = canonical_team(raw)
            assert canonical_team(once) == once


class TestShouting:
    def test_detects_capitals(self):
        assert is_shouted("OHIO STATE")
        assert not is_shouted("Ohio State")

    def test_short_acronyms_are_not_shouting(self):
        assert not is_shouted("LSU")
        assert not is_shouted("TCU")


class TestGameKeys:
    def test_key_is_stable_across_spellings(self):
        a = game_key(2026, 3, "Ole Miss", "Texas A&M Aggies")
        b = game_key(2026, 3, "Mississippi", "Texas A&M")
        assert a == b

    def test_home_and_away_are_distinguished(self):
        assert game_key(2026, 3, "Georgia", "Alabama") != game_key(2026, 3, "Alabama", "Georgia")


class TestMatching:
    def test_exact_match(self):
        assert match_teams("Ole Miss", ["Mississippi", "Alabama"]) == "Mississippi"

    def test_ambiguous_prefix_returns_nothing(self):
        assert match_teams("Louisiana", ["Louisiana Monroe", "Louisiana Tech"]) is None
