"""The weekly workflow, orchestrated in one place.

``fetch`` -> ``predict`` -> ``picks`` is the whole loop. Each stage reads
and writes storage, so any stage can be re-run on its own without
spending API calls again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .config import Config
from .engine.market import build_consensus
from .engine.predict import PredictionInputs, Predictor
from .engine.recommend import Recommender
from .models import ExpertProjection, Game, MarketQuote, Prediction, Rating, Recommendation
from .providers import available, build_provider
from .providers.sagarin import resolve_projections
from .storage import Storage
from .util.http import MissingCredentials, ProviderError


@dataclass
class FetchReport:
    """What each provider contributed, and what went wrong."""

    games: int = 0
    ratings: int = 0
    quotes: int = 0
    projections: int = 0
    by_provider: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def note(self, provider: str, message: str) -> None:
        self.by_provider[provider] = message

    def warn(self, message: str) -> None:
        self.warnings.append(message)


class Pipeline:
    def __init__(self, config: Config, storage: Optional[Storage] = None) -> None:
        self.config = config
        self.storage = storage or Storage(config.database_path)
        self._owns_storage = storage is None

    def close(self) -> None:
        if self._owns_storage:
            self.storage.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- fetch -------------------------------------------------------------
    def fetch(
        self,
        season: int,
        week: Optional[int] = None,
        *,
        providers: Optional[Sequence[str]] = None,
        want: Sequence[str] = ("games", "ratings", "odds"),
    ) -> FetchReport:
        report = FetchReport()

        if "games" in want:
            for provider in available(self.config, "games", providers):
                try:
                    games = provider.fetch_games(season, week)
                except (ProviderError, MissingCredentials) as exc:
                    report.warn(f"{provider.name} games: {exc}")
                    continue
                count = self.storage.upsert_games(games)
                report.games += count
                report.note(provider.name, f"{count} games")

        if "ratings" in want:
            for provider in available(self.config, "ratings", providers):
                try:
                    ratings = provider.fetch_ratings(season, week)
                except Exception as exc:  # noqa: BLE001
                    # Rating feeds are the flakiest input — saved pages,
                    # third-party layouts, hosts that block you. Never let
                    # one broken source stop the rest of the week's fetch.
                    report.warn(f"{provider.name} ratings: {exc}")
                    continue
                count = self.storage.upsert_ratings(ratings)
                report.ratings += count
                prior = report.by_provider.get(provider.name, "")
                report.note(provider.name, f"{prior + ', ' if prior else ''}{count} ratings")
                for warning in getattr(provider, "warnings", []) or []:
                    report.warn(f"{provider.name}: {warning}")

        if "odds" in want:
            known_games = self.storage.games(season, week)
            for provider in available(self.config, "odds", providers):
                try:
                    quotes = provider.fetch_odds(season, week, games=known_games)
                except (ProviderError, MissingCredentials) as exc:
                    report.warn(f"{provider.name} odds: {exc}")
                    continue
                count = self.storage.upsert_quotes(quotes)
                report.quotes += count
                prior = report.by_provider.get(provider.name, "")
                report.note(provider.name, f"{prior + ', ' if prior else ''}{count} quotes")
                for miss in getattr(provider, "unmatched", []) or []:
                    report.warn(f"{provider.name}: no scheduled game for {miss}")
                remaining = getattr(provider, "quota_remaining", None)
                if remaining:
                    report.note(provider.name, f"{report.by_provider[provider.name]} ({remaining} API calls left)")

        return report

    # -- expert projections -------------------------------------------------
    def load_projections(self, season: int, week: int) -> list[ExpertProjection]:
        """Fetch published game projections and align them to the schedule."""
        games = self.storage.games(season, week)
        out: list[ExpertProjection] = []
        provider = build_provider("sagarin", self.config)
        if not (provider.enabled and provider.configured):
            return out
        try:
            raw = provider.fetch_projections(season, week)
        except Exception:
            # Sagarin's page is fetched over plain HTTP from a host that
            # may be unreachable; projections are an enhancement, not a
            # prerequisite, so a failure here is silent by design.
            return out
        resolved, _ = resolve_projections(raw, games)
        out.extend(resolved)
        return out

    # -- predict -----------------------------------------------------------
    def predict(
        self, season: int, week: int, *, use_projections: bool = True
    ) -> list[Prediction]:
        games = self.storage.games(season, week)
        if not games:
            return []
        ratings = self.storage.ratings(season, week)
        schedule = self.storage.games(season)
        projections = self.load_projections(season, week) if use_projections else []

        predictor = Predictor(self.config)
        predictions = predictor.predict_week(
            PredictionInputs(
                games=games, ratings=ratings,
                expert_projections=projections, schedule=schedule,
            )
        )
        self.storage.upsert_predictions(predictions)
        return predictions

    # -- recommend ----------------------------------------------------------
    def picks(
        self, season: int, week: int, *, save: bool = True, repredict: bool = True
    ) -> list[Recommendation]:
        predictions = self.predict(season, week) if repredict else self.storage.predictions(season, week)
        if not predictions:
            return []

        game_ids = [p.game_id for p in predictions]
        quotes_by_game = self.storage.quotes_for_games(game_ids)
        consensus = build_consensus(quotes_by_game, self.config)
        games = {g.game_id: g for g in self.storage.games(season, week)}

        recs = Recommender(self.config).recommend_week(
            predictions, consensus, quotes_by_game, games
        )
        if save and recs:
            self.storage.save_recommendations(recs)
        return recs

    # -- grading -------------------------------------------------------------
    def grade(self, season: int, week: Optional[int] = None) -> dict[str, int]:
        """Settle any recommendation whose game has a final score."""
        from .backtest import grade_recommendation

        results = {"win": 0, "loss": 0, "push": 0, "ungraded": 0}
        for rec_id, rec in self.storage.ungraded_recommendations(season):
            if week is not None and rec.week != week:
                continue
            game = self.storage.game(rec.game_id)
            if game is None or not game.completed:
                results["ungraded"] += 1
                continue
            outcome, profit = grade_recommendation(rec, game)
            self.storage.grade_recommendation(rec_id, outcome, profit)
            results[outcome] = results.get(outcome, 0) + 1
        return results
