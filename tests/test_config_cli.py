"""Configuration loading and CLI wiring."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cfbpicks.cli import _parse_weeks, default_season, main
from cfbpicks.config import Config, load_config


class TestConfig:
    def test_defaults_are_usable_with_no_file(self, tmp_path):
        cfg = load_config(root=tmp_path)
        assert cfg.model.home_field_advantage > 0
        assert "cfbd" in cfg.providers

    def test_yaml_overrides_defaults(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "model:\n  home_field_advantage: 3.5\nbetting:\n  kelly_multiplier: 0.5\n"
        )
        cfg = load_config(root=tmp_path)
        assert cfg.model.home_field_advantage == 3.5
        assert cfg.betting.kelly_multiplier == 0.5

    def test_local_config_wins_over_shared_config(self, tmp_path):
        (tmp_path / "config.yaml").write_text("model:\n  home_field_advantage: 2.0\n")
        (tmp_path / "config.local.yaml").write_text("model:\n  home_field_advantage: 4.0\n")
        assert load_config(root=tmp_path).model.home_field_advantage == 4.0

    def test_a_typo_is_rejected_rather_than_ignored(self, tmp_path):
        (tmp_path / "config.yaml").write_text("model:\n  home_feild_advantage: 3.0\n")
        with pytest.raises(ValueError, match="home_feild_advantage"):
            load_config(root=tmp_path)

    def test_environment_supplies_secrets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFBD_API_KEY", "secret-key")
        assert load_config(root=tmp_path).providers["cfbd"].api_key == "secret-key"

    def test_environment_beats_the_config_file(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text("providers:\n  cfbd:\n    api_key: from-file\n")
        monkeypatch.setenv("CFBD_API_KEY", "from-env")
        assert load_config(root=tmp_path).providers["cfbd"].api_key == "from-env"

    def test_offline_flag_from_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFBPICKS_OFFLINE", "1")
        assert load_config(root=tmp_path).offline is True


class TestWeekParsing:
    @pytest.mark.parametrize("spec,expected", [
        ("3", [3]),
        ("1-4", [1, 2, 3, 4]),
        ("3,5,7", [3, 5, 7]),
        ("1-3,7", [1, 2, 3, 7]),
        ("2,2,2", [2]),
    ])
    def test_parses(self, spec, expected):
        assert _parse_weeks(spec) == expected


class TestSeasonDefault:
    def test_is_a_plausible_year(self):
        assert 2020 <= default_season() <= 2100


class TestCli:
    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "predictions" in result.output.lower()

    def test_providers_lists_sources(self, tmp_path):
        result = self.runner.invoke(main, ["--db", str(tmp_path / "x.sqlite"), "providers"])
        assert result.exit_code == 0
        for name in ("cfbd", "odds_api", "sagarin"):
            assert name in result.output

    def test_init_creates_the_database(self, tmp_path):
        db = tmp_path / "new.sqlite"
        result = self.runner.invoke(main, ["--db", str(db), "init"])
        assert result.exit_code == 0
        assert db.exists()

    def test_demo_runs_the_whole_pipeline(self, tmp_path):
        result = self.runner.invoke(main, ["--db", str(tmp_path / "demo.sqlite"), "demo"])
        assert result.exit_code == 0
        assert "Loaded fixtures" in result.output

    def test_picks_on_an_empty_database_explains_itself(self, tmp_path):
        result = self.runner.invoke(
            main, ["--db", str(tmp_path / "empty.sqlite"), "picks", "--week", "3"]
        )
        assert result.exit_code == 1
        assert "fetch" in result.output

    def test_status_on_an_empty_database(self, tmp_path):
        result = self.runner.invoke(main, ["--db", str(tmp_path / "empty.sqlite"), "status"])
        assert result.exit_code == 0

    def test_import_ratings_from_csv(self, tmp_path):
        csv_path = tmp_path / "massey-2026-w3.csv"
        csv_path.write_text("team,rating\nOle Miss,18.4\nGeorgia,24.1\n")
        result = self.runner.invoke(main, [
            "--db", str(tmp_path / "x.sqlite"), "import-ratings", str(csv_path),
            "--source", "massey", "--season", "2026", "--week", "3",
        ])
        assert result.exit_code == 0
        assert "Saved 2" in result.output

    def test_import_ratings_reports_a_bad_file(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("school,power\nGeorgia,24.1\n")
        result = self.runner.invoke(main, [
            "--db", str(tmp_path / "x.sqlite"), "import-ratings", str(bad), "--source", "x",
        ])
        assert result.exit_code == 1
        assert "Columns found" in result.output

    def test_import_sagarin_dry_run(self, tmp_path):
        from pathlib import Path
        sample = Path(__file__).parent / "data" / "sagarin_sample.htm"
        result = self.runner.invoke(main, [
            "--db", str(tmp_path / "x.sqlite"), "import-sagarin", str(sample),
            "--season", "2026", "--week", "3", "--dry-run",
        ])
        assert result.exit_code == 0
        assert "6 ratings" in result.output
