"""Provider contract.

A provider knows how to turn one upstream source into domain objects. It
never writes to storage and never decides anything about betting — that
keeps sources swappable and testable in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..config import Config, ProviderConfig
from ..models import Game, MarketQuote, Rating
from ..util.http import HttpClient, MissingCredentials


class Provider(ABC):
    """Base class for every data source."""

    #: Short identifier, matching the key under ``providers:`` in config.
    name: str = "provider"
    #: Human-readable description shown by ``cfbpicks providers``.
    description: str = ""
    #: Whether the provider needs an API key to do anything.
    requires_key: bool = False

    def __init__(self, config: Config, settings: ProviderConfig) -> None:
        self.config = config
        self.settings = settings
        self.client = HttpClient(
            cache_dir=config.path(config.cache_dir) / self.name,
            timeout=config.request_timeout,
            ttl_seconds=config.cache_ttl_seconds,
            offline=config.offline,
        )

    # -- capability flags -------------------------------------------------
    provides_games: bool = False
    provides_ratings: bool = False
    provides_odds: bool = False
    provides_weather: bool = False

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def configured(self) -> bool:
        """True when the provider has everything it needs to run."""
        return bool(self.settings.api_key) if self.requires_key else True

    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self.configured:
            return "missing key"
        return "ready"

    def require_key(self) -> str:
        if not self.settings.api_key:
            raise MissingCredentials(
                f"Provider {self.name!r} needs an API key. Set it in the environment "
                f"or in config.local.yaml under providers.{self.name}.api_key."
            )
        return self.settings.api_key

    def option(self, key: str, default=None):
        return self.settings.options.get(key, default)

    # -- data methods; override the ones the provider supports ------------
    def fetch_games(self, season: int, week: Optional[int] = None, **kwargs) -> list[Game]:
        raise NotImplementedError(f"{self.name} does not provide games")

    def fetch_ratings(self, season: int, week: Optional[int] = None, **kwargs) -> list[Rating]:
        raise NotImplementedError(f"{self.name} does not provide ratings")

    def fetch_odds(self, season: int, week: Optional[int] = None, **kwargs) -> list[MarketQuote]:
        raise NotImplementedError(f"{self.name} does not provide odds")

    def fetch_weather(self, season: int, week: Optional[int] = None, **kwargs) -> list:
        raise NotImplementedError(f"{self.name} does not provide weather")
