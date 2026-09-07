"""Manual injury / situational adjustments."""

from __future__ import annotations

import pytest

from cfbpicks.config import Config
from cfbpicks.models import Game
from cfbpicks.overrides import (
    POSITION_VALUES, OverrideError, load_overrides, resolve,
)
from cfbpicks.pipeline import Pipeline

GAMES = [
    Game("g1", 2026, 3, None, "Georgia", "Alabama"),
    Game("g2", 2026, 3, None, "Ohio State", "Michigan"),
]


def write(tmp_path, text):
    path = tmp_path / "overrides.yaml"
    path.write_text(text)
    return path


class TestParsing:
    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_overrides(tmp_path / "nope.yaml") == []

    def test_a_team_entry_parses(self, tmp_path):
        path = write(tmp_path, """
- season: 2026
  week: 3
  team: Ole Miss
  out: [qb1]
  reason: QB out
""")
        entry = load_overrides(path)[0]
        assert entry.team == "Mississippi"      # canonicalised
        assert entry.team_points() == -7.0

    def test_positions_stack(self, tmp_path):
        path = write(tmp_path, """
- season: 2026
  week: 3
  team: Georgia
  out: [ol, ol, wr1]
""")
        assert load_overrides(path)[0].team_points() == pytest.approx(-3.5)

    def test_explicit_points_win_over_positions(self, tmp_path):
        path = write(tmp_path, """
- season: 2026
  week: 3
  team: Georgia
  points: -1.5
  out: [qb1]
""")
        assert load_overrides(path)[0].team_points() == -1.5

    def test_a_single_position_may_be_a_bare_string(self, tmp_path):
        path = write(tmp_path, "- {season: 2026, week: 3, team: Georgia, out: qb1}")
        assert load_overrides(path)[0].team_points() == -7.0

    @pytest.mark.parametrize("text,message", [
        ("- {week: 3, team: Georgia, points: -1}", "missing 'season'"),
        ("- {season: 2026, team: Georgia, points: -1}", "missing 'week'"),
        ("- {season: 2026, week: 3, points: -1}", "exactly one of"),
        ("- {season: 2026, week: 3, team: A, game: B, points: -1}", "exactly one of"),
        ("- {season: 2026, week: 3, team: Georgia}", "needs 'points' or 'out'"),
        ("- {season: 2026, week: 3, game: A @ B}", "needs 'margin' or 'total'"),
        ("- {season: 2026, week: 3, team: Georgia, out: [punter]}", "unknown position"),
        ("- {season: 2026, week: 3, team: Georgia, points: -1, wat: 1}", "unknown key"),
        ("not a list", "must contain a list"),
    ])
    def test_bad_input_is_rejected_with_a_useful_message(self, tmp_path, text, message):
        with pytest.raises(OverrideError, match=message):
            load_overrides(write(tmp_path, text))

    def test_every_documented_position_has_a_value(self):
        assert all(v < 0 for v in POSITION_VALUES.values())
        # The quarterback must dominate; that is the whole point.
        assert POSITION_VALUES["qb1"] < min(
            v for k, v in POSITION_VALUES.items() if k != "qb1"
        )


class TestResolution:
    def _entries(self, tmp_path, text):
        return load_overrides(write(tmp_path, text))

    def test_a_team_adjustment_lands_on_its_team(self, tmp_path):
        entries = self._entries(tmp_path, """
- {season: 2026, week: 3, team: Georgia, out: [qb1], reason: QB out}
""")
        adjustments, warnings = resolve(entries, 2026, 3, GAMES)
        assert adjustments.for_team("Georgia") == -7.0
        assert adjustments.for_team("Alabama") == 0.0
        assert not warnings

    def test_other_weeks_are_ignored(self, tmp_path):
        entries = self._entries(tmp_path, "- {season: 2026, week: 9, team: Georgia, points: -3}")
        adjustments, _ = resolve(entries, 2026, 3, GAMES)
        assert not adjustments

    def test_a_game_entry_shifts_margin_and_total(self, tmp_path):
        entries = self._entries(tmp_path, """
- {season: 2026, week: 3, game: Michigan @ Ohio State, total: -4, margin: 1.5}
""")
        adjustments, warnings = resolve(entries, 2026, 3, GAMES)
        assert adjustments.game_total["g2"] == -4.0
        assert adjustments.game_margin["g2"] == 1.5
        assert not warnings

    def test_adjustments_accumulate(self, tmp_path):
        entries = self._entries(tmp_path, """
- {season: 2026, week: 3, team: Georgia, out: [qb1]}
- {season: 2026, week: 3, team: Georgia, out: [ol, ol]}
""")
        adjustments, _ = resolve(entries, 2026, 3, GAMES)
        assert adjustments.for_team("Georgia") == -9.0

    def test_a_typo_is_reported_rather_than_silently_doing_nothing(self, tmp_path):
        entries = self._entries(tmp_path, "- {season: 2026, week: 3, team: Notre Dam, points: -3}")
        adjustments, warnings = resolve(entries, 2026, 3, GAMES)
        assert not adjustments
        assert any("adjustment ignored" in w for w in warnings)

    def test_an_empty_schedule_gives_one_clear_message(self, tmp_path):
        """Not four 'orphaned entry' warnings when nothing is fetched."""
        entries = self._entries(tmp_path, """
- {season: 2026, week: 3, team: Georgia, points: -1}
- {season: 2026, week: 3, team: Alabama, points: -1}
""")
        _, warnings = resolve(entries, 2026, 3, [])
        assert len(warnings) == 1
        assert "fetch the week first" in warnings[0]

    def test_reasons_travel_with_the_game(self, tmp_path):
        entries = self._entries(tmp_path, """
- {season: 2026, week: 3, team: Georgia, out: [qb1], reason: Beck out}
""")
        adjustments, _ = resolve(entries, 2026, 3, GAMES)
        notes = adjustments.notes_for("Georgia", "Alabama", "g1")
        assert any("Beck out" in n for n in notes)


class TestEffectOnPredictions:
    def _pipeline(self, tmp_path):
        from cfbpicks.config import ProviderConfig
        from cfbpicks.providers.fixtures import PACKAGE_FIXTURES

        cfg = Config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.fixtures_dir = str(PACKAGE_FIXTURES)
        cfg.overrides_file = str(tmp_path / "overrides.yaml")
        cfg.providers = {"fixtures": ProviderConfig("fixtures", enabled=True)}
        return cfg

    def test_an_injury_moves_the_projection(self, tmp_path):
        cfg = self._pipeline(tmp_path)
        (tmp_path / "overrides.yaml").write_text(
            "- {season: 2026, week: 3, team: Ohio State, out: [qb1], reason: QB out}\n"
        )
        with Pipeline(cfg) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            without = {p.game_id: p for p in pipeline.predict(
                2026, 3, use_projections=False, use_overrides=False)}
            with_ = {p.game_id: p for p in pipeline.predict(
                2026, 3, use_projections=False, use_overrides=True)}

        moved = [g for g in without if
                 abs(with_[g].projected_margin - without[g].projected_margin) > 1e-6]
        assert len(moved) == 1
        game = moved[0]
        # Ohio State is home in the fixtures, so its margin drops by 7.
        assert with_[game].projected_margin == pytest.approx(
            without[game].projected_margin - 7.0
        )
        assert any("QB out" in n for n in with_[game].components["overrides"])

    def test_the_reason_reaches_the_bet(self, tmp_path):
        cfg = self._pipeline(tmp_path)
        (tmp_path / "overrides.yaml").write_text(
            "- {season: 2026, week: 3, team: Ohio State, out: [qb1], reason: QB out}\n"
        )
        with Pipeline(cfg) as pipeline:
            pipeline.fetch(2026, 3, providers=["fixtures"])
            recs = pipeline.picks(2026, 3)
        touched = [r for r in recs if "Ohio State" in r.matchup]
        assert touched, "the adjustment should create or change a bet"
        assert any("QB out" in note for r in touched for note in r.notes)

    def test_backtests_never_see_overrides(self, tmp_path):
        """A note added after kickoff would be pure hindsight."""
        import inspect

        from cfbpicks import backtest

        source = inspect.getsource(backtest.backtest_season)
        assert "adjustments" not in source
