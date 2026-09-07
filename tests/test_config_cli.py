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


class TestWarningHygiene:
    """A scheduled `snapshot` must not mail the user noise every run."""

    def test_package_root_does_not_import_urllib3(self):
        """The filter has to be registered before urllib3 loads.

        urllib3 emits its LibreSSL notice during import, so importing it
        from the package root to reference the exception class would
        trigger the very warning being suppressed.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, cfbpicks; print('urllib3' in sys.modules)"],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "False"

    def test_the_libressl_notice_is_filtered(self):
        import warnings

        import cfbpicks  # noqa: F401 -- registers the filter on import

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore", message=r"urllib3 v2 only supports OpenSSL", category=Warning
            )
            warnings.warn(
                "urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module "
                "is compiled with 'LibreSSL 2.8.3'.", Warning,
            )
            warnings.warn("a real problem", UserWarning)

        messages = [str(w.message) for w in caught]
        assert not any("LibreSSL" in m for m in messages)
        assert any("a real problem" in m for m in messages), "unrelated warnings must survive"


class TestSampleDataIsolation:
    """Invented games must never reach a live database.

    The fixtures pair real school names into matchups that do not exist.
    A `fetch` that quietly included them would publish fabricated games
    as real recommendations -- which is exactly what happened on the
    first cloud run before this was fixed.
    """

    def test_fixtures_are_off_by_default(self, tmp_path):
        cfg = load_config(root=tmp_path)
        assert cfg.providers["fixtures"].enabled is False

    def test_a_plain_fetch_stores_no_sample_games(self, tmp_path):
        from cfbpicks.pipeline import Pipeline

        cfg = load_config(root=tmp_path)
        cfg.database = str(tmp_path / "t.sqlite")
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.offline = True
        # Every network provider is unconfigured here, so anything that
        # lands in the database came from the bundled samples.
        with Pipeline(cfg) as pipeline:
            pipeline.fetch(2026, 3, want=("games",))
            assert pipeline.storage.games(2026) == []

    def test_demo_still_gets_its_sample_data(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "x.sqlite"), "demo"])
        assert result.exit_code == 0
        assert "Loaded fixtures: 12 games" in result.output


class TestRateWithoutResults:
    def test_a_season_with_no_completed_games_is_not_an_error(self, tmp_path):
        """Week 1 has nothing to fit on, and must not stop a scheduled run."""
        from cfbpicks.models import Game
        from cfbpicks.storage import Storage

        db = tmp_path / "t.sqlite"
        store = Storage(db)
        store.upsert_games([Game("g1", 2026, 1, None, "Georgia", "Alabama")])
        store.close()

        result = CliRunner().invoke(main, ["--db", str(db), "rate", "--season", "2026"])
        assert result.exit_code == 0, result.output
        assert "No completed games" in result.output

    def test_an_empty_season_is_still_an_error(self, tmp_path):
        result = CliRunner().invoke(
            main, ["--db", str(tmp_path / "empty.sqlite"), "rate", "--season", "2026"]
        )
        assert result.exit_code == 1
        assert "No games stored" in result.output
