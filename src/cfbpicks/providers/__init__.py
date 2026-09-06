"""Provider registry.

Adding a source means writing a :class:`Provider` subclass and listing it
here; nothing else in the codebase needs to change.
"""

from __future__ import annotations

from typing import Iterable, Optional, Type

from ..config import Config, ProviderConfig
from .base import Provider
from .cfbd import CfbdProvider
from .fixtures import FixturesProvider
from .odds_api import OddsApiProvider
from .ratings import RatingsProvider
from .sagarin import SagarinProvider

PROVIDER_CLASSES: dict[str, Type[Provider]] = {
    CfbdProvider.name: CfbdProvider,
    OddsApiProvider.name: OddsApiProvider,
    RatingsProvider.name: RatingsProvider,
    SagarinProvider.name: SagarinProvider,
    FixturesProvider.name: FixturesProvider,
}

__all__ = [
    "Provider", "CfbdProvider", "OddsApiProvider", "RatingsProvider",
    "SagarinProvider", "FixturesProvider", "PROVIDER_CLASSES",
    "build_provider", "build_providers", "available",
]


def build_provider(name: str, config: Config) -> Provider:
    if name not in PROVIDER_CLASSES:
        raise KeyError(f"Unknown provider {name!r}. Known: {', '.join(sorted(PROVIDER_CLASSES))}")
    settings = config.providers.get(name) or ProviderConfig(name=name)
    return PROVIDER_CLASSES[name](config, settings)


def build_providers(config: Config, names: Optional[Iterable[str]] = None) -> list[Provider]:
    """Instantiate providers, defaulting to every one that is configured."""
    wanted = list(names) if names else list(PROVIDER_CLASSES)
    return [build_provider(name, config) for name in wanted if name in PROVIDER_CLASSES]


def available(config: Config, capability: str, names: Optional[Iterable[str]] = None) -> list[Provider]:
    """Providers that are enabled, credentialed and offer ``capability``.

    ``capability`` is one of ``games``, ``ratings`` or ``odds``.
    """
    attr = f"provides_{capability}"
    return [
        p for p in build_providers(config, names)
        if getattr(p, attr, False) and p.enabled and p.configured
    ]
