"""A small HTTP client with on-disk caching, retries and rate awareness.

Both upstream APIs meter requests (The Odds API's free tier is ~500 a
month), so every response is cached to disk and re-used until it goes
stale. That also makes reruns of a week's slate free and keeps tests
offline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


class ProviderError(RuntimeError):
    """Raised when an upstream provider cannot satisfy a request."""


class MissingCredentials(ProviderError):
    """Raised when a provider is enabled but has no API key configured."""


@dataclass
class HttpClient:
    cache_dir: Path
    timeout: float = 30.0
    ttl_seconds: int = 900
    offline: bool = False
    max_retries: int = 4
    user_agent: str = "cfbpicks/0.1 (+https://github.com/jtheisen23/College-Football-Picks)"

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.last_response_headers: dict[str, str] = {}

    # -- cache -----------------------------------------------------------
    def _cache_path(self, url: str, params: Optional[dict], headers_key: str) -> Path:
        payload = json.dumps(
            {"url": url, "params": params or {}, "auth": headers_key}, sort_keys=True
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
        host = url.split("//")[-1].split("/")[0].replace(":", "_")
        return self.cache_dir / host / f"{digest}.json"

    def _read_cache(self, path: Path, ttl: Optional[int]) -> Optional[Any]:
        if not path.exists():
            return None
        ttl = self.ttl_seconds if ttl is None else ttl
        age = time.time() - path.stat().st_mtime
        # ttl <= 0 means "cache forever", which is what historical
        # (already-final) seasons want.
        if ttl > 0 and age > ttl and not self.offline:
            return None
        try:
            return json.loads(path.read_text())["body"]
        except (json.JSONDecodeError, KeyError):
            return None

    def _write_cache(self, path: Path, body: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": time.time(), "body": body}))

    # -- requests --------------------------------------------------------
    def get_json(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        ttl: Optional[int] = None,
        force: bool = False,
    ) -> Any:
        """GET a JSON document, serving from cache when it is fresh."""
        headers = dict(headers or {})
        # The cache key must distinguish authenticated from anonymous calls
        # without ever writing the key itself to disk.
        auth_marker = "keyed" if any(
            k.lower() in {"authorization", "x-api-key"} for k in headers
        ) or (params or {}).get("apiKey") else "anon"
        path = self._cache_path(url, _cache_safe_params(params), auth_marker)

        if not force:
            cached = self._read_cache(path, ttl)
            if cached is not None:
                return cached

        if self.offline:
            raise ProviderError(
                f"Offline mode is on and no cached response exists for {url}. "
                "Run without --offline, or use the bundled fixtures."
            )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue

            self.last_response_headers = dict(response.headers)

            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise ProviderError(f"{url} returned non-JSON content") from exc
                self._write_cache(path, body)
                return body

            if response.status_code in {401, 403}:
                raise MissingCredentials(
                    f"{url} rejected the request ({response.status_code}). "
                    "Check that the provider's API key is set and still valid."
                )
            if response.status_code == 429:
                # Respect a Retry-After header when the API sends one.
                wait = float(response.headers.get("Retry-After", 2**attempt))
                last_error = ProviderError(f"{url} rate limited")
                time.sleep(min(wait, 30.0))
                continue
            if 500 <= response.status_code < 600:
                last_error = ProviderError(f"{url} returned {response.status_code}")
                time.sleep(2**attempt)
                continue

            raise ProviderError(
                f"{url} returned {response.status_code}: {response.text[:200]}"
            )

        # Every retry failed. A stale cache entry beats no data at all.
        stale = self._read_cache(path, ttl=0)
        if stale is not None:
            return stale
        raise ProviderError(f"Failed to fetch {url}: {last_error}")

    def _header(self, name: str) -> Optional[str]:
        """Case-insensitive lookup against the last response's headers."""
        lowered = {k.lower(): v for k, v in self.last_response_headers.items()}
        return lowered.get(name.lower())

    @property
    def quota_remaining(self) -> Optional[str]:
        """Requests left this period, when the provider reports it."""
        return self._header("x-requests-remaining")

    @property
    def quota_used(self) -> Optional[str]:
        return self._header("x-requests-used")

    @property
    def quota_last_cost(self) -> Optional[str]:
        """Credits the most recent call consumed."""
        return self._header("x-requests-last")


def _cache_safe_params(params: Optional[dict]) -> dict:
    """Strip credentials out of anything that gets hashed into a filename."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if k.lower() not in {"apikey", "api_key", "key"}}
