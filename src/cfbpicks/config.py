"""Configuration loading: YAML file + environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_FILENAMES = ("config.local.yaml", "config.yaml")


@dataclass
class ModelConfig:
    """Tunables for the prediction model.

    Defaults reflect modern FBS behaviour: roughly 2.2 points of home
    advantage, a ~16 point standard deviation of actual margin around the
    closing spread, and ~13.5 points around the closing total.
    """

    home_field_advantage: float = 2.2
    margin_sd: float = 16.0
    total_sd: float = 13.5
    league_average_total: float = 54.0
    # Weight per rating source when blending into one power number.
    rating_weights: dict[str, float] = field(
        default_factory=lambda: {
            "cfbpicks_margin": 1.0,
            "cfbd_sp_plus": 1.0,
            "cfbd_srs": 0.5,
            "cfbd_elo": 0.5,
            "massey": 0.75,
            "sagarin": 0.75,
            "manual": 1.0,
        }
    )
    # Shrink a team's blended rating toward average when few sources agree.
    min_sources_for_full_confidence: int = 2
    # Extra points added for each day of rest advantage, capped.
    rest_advantage_per_day: float = 0.12
    max_rest_adjustment: float = 1.5
    # Regress early-season ratings toward the preseason number.
    early_season_regression_weeks: int = 4
    # Settings for the ratings the engine fits itself from results.
    # `ridge` is how much evidence it takes to move a team off the prior.
    # 2.0 was picked by sweeping simulated seasons: it minimised held-out
    # error and kept predicted margins calibrated (slope ~1.02). Raising
    # it shrinks predictions toward zero, which quietly biases the model
    # onto underdogs.
    regression_ridge: float = 2.0
    # Capping blowout margins biases every coefficient low; off by default.
    regression_margin_cap: Optional[float] = None
    #: Weeks for a game's weight to halve. None weights all games equally.
    regression_recency_halflife: Optional[float] = None
    #: Carry last season's final ratings in as the prior, shrunk by this.
    regression_carryover: float = 0.5


@dataclass
class BettingConfig:
    """Thresholds and staking rules for turning edges into bets."""

    devig_method: str = "proportional"
    moneyline_devig_method: str = "power"
    bankroll_units: float = 100.0
    unit_fraction: float = 0.01
    kelly_multiplier: float = 0.25
    max_units: float = 3.0
    min_prob_edge: float = 0.02
    min_point_edge_spread: float = 1.5
    min_point_edge_total: float = 2.5
    min_expected_value: float = 0.005
    max_spread_line: float = 35.0      # skip absurd blowout numbers
    # Beyond this projected margin, moneyline pricing from a normal model
    # is unreliable, so moneylines on those games are skipped entirely.
    max_moneyline_margin: float = 21.0
    min_book_count: int = 1
    markets: list[str] = field(default_factory=lambda: ["spread", "total", "moneyline"])
    # Tier boundaries measured in probability edge.
    tier_thresholds: dict[str, float] = field(
        default_factory=lambda: {"strong": 0.055, "play": 0.035, "lean": 0.02}
    )


@dataclass
class ProviderConfig:
    """Per-provider switches and credentials."""

    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    database: str = "cfbpicks.sqlite"
    cache_dir: str = "data/cache"
    fixtures_dir: str = "data/fixtures"  # falls back to the packaged copy
    reference_dir: str = "data/reference"
    reports_dir: str = "reports"
    overrides_file: str = "data/overrides.yaml"
    season: Optional[int] = None
    offline: bool = False
    request_timeout: float = 30.0
    cache_ttl_seconds: int = 900
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    weather: "WeatherModel" = field(default_factory=lambda: _weather_model())
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    root: Path = field(default_factory=Path.cwd)

    # -- path helpers ----------------------------------------------------
    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def database_path(self) -> Path:
        return self.path(self.database)

    def provider(self, name: str) -> ProviderConfig:
        return self.providers.get(name, ProviderConfig(name=name, enabled=False))


# Environment variables that fill in provider credentials.
ENV_KEYS = {
    "cfbd": ("CFBD_API_KEY", "COLLEGE_FOOTBALL_DATA_API_KEY"),
    "odds_api": ("ODDS_API_KEY", "THE_ODDS_API_KEY"),
}

def _weather_model():
    from .weather import WeatherModel

    return WeatherModel()


DEFAULT_PROVIDERS: dict[str, dict[str, Any]] = {
    "cfbd": {
        "enabled": True,
        "base_url": "https://api.collegefootballdata.com",
        "options": {"ratings": ["sp_plus", "srs", "elo"], "division": "fbs"},
    },
    "odds_api": {
        "enabled": True,
        "base_url": "https://api.the-odds-api.com/v4",
        "options": {
            "sport": "americanfootball_ncaaf",
            "regions": "us",
            "odds_format": "american",
            "books": [],
        },
    },
    "ratings": {
        "enabled": True,
        "options": {"sources": ["massey", "sagarin"], "input_dir": "data/ratings"},
    },
    "weather": {"enabled": True, "options": {"forecast_horizon_days": 16}},
    "fixtures": {"enabled": True, "options": {}},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def find_config_file(root: Path) -> Optional[Path]:
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def load_config(path: Optional[str] = None, root: Optional[Path] = None) -> Config:
    """Load configuration from YAML, then layer environment variables on top.

    Search order is ``config.local.yaml`` then ``config.yaml`` in the
    working directory. Missing files are fine — the defaults are usable.
    """
    root = Path(root or Path.cwd())
    raw: dict[str, Any] = {}

    config_path = Path(path) if path else find_config_file(root)
    if config_path and Path(config_path).exists():
        loaded = yaml.safe_load(Path(config_path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping")
        raw = loaded

    cfg = Config(root=root)
    for key in ("database", "cache_dir", "fixtures_dir", "reference_dir",
                "reports_dir", "season", "offline", "request_timeout",
                "cache_ttl_seconds"):
        if key in raw and raw[key] is not None:
            setattr(cfg, key, raw[key])

    if isinstance(raw.get("model"), dict):
        cfg.model = replace(ModelConfig(), **_valid_fields(ModelConfig, raw["model"]))
    if isinstance(raw.get("betting"), dict):
        cfg.betting = replace(BettingConfig(), **_valid_fields(BettingConfig, raw["betting"]))
    if isinstance(raw.get("weather"), dict):
        from .weather import WeatherModel

        cfg.weather = replace(
            WeatherModel(), **_valid_fields(WeatherModel, raw["weather"])
        )

    provider_raw = _deep_merge(DEFAULT_PROVIDERS, raw.get("providers") or {})
    for name, spec in provider_raw.items():
        spec = spec or {}
        cfg.providers[name] = ProviderConfig(
            name=name,
            enabled=bool(spec.get("enabled", True)),
            api_key=spec.get("api_key"),
            base_url=spec.get("base_url"),
            options=spec.get("options") or {},
        )

    # Environment always wins for secrets so keys stay out of the repo.
    for name, env_names in ENV_KEYS.items():
        provider = cfg.providers.get(name)
        if provider is None:
            continue
        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                provider.api_key = value
                break

    if os.environ.get("CFBPICKS_OFFLINE") in {"1", "true", "yes"}:
        cfg.offline = True
    if os.environ.get("CFBPICKS_DB"):
        cfg.database = os.environ["CFBPICKS_DB"]

    return cfg


def _valid_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that are real dataclass fields, so typos surface early."""
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} option(s): {', '.join(sorted(unknown))}"
        )
    return {k: v for k, v in data.items() if k in allowed}
