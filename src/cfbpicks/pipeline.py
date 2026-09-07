"""The weekly workflow, orchestrated in one place.

``fetch`` -> ``predict`` -> ``picks`` is the whole loop. Each stage reads
and writes storage, so any stage can be re-run on its own without
spending API calls again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .ratings.regression import FitResult

from .config import Config
from .engine.market import build_consensus
from .engine.predict import PredictionInputs, Predictor
from .engine.recommend import Recommender
from .models import ExpertProjection, Game, MarketQuote, Prediction, Rating, Recommendation
from .overrides import Adjustments, load_overrides, resolve as resolve_overrides
from .providers import available, build_provider
from .providers.sagarin import resolve_projections
from .ratings.regression import SOURCE
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
        want: Sequence[str] = ("games", "ratings", "weather", "odds"),
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

        if "weather" in want:
            known_games = self.storage.games(season, week)
            for provider in available(self.config, "weather", providers):
                try:
                    forecasts = provider.fetch_weather(season, week, games=known_games)
                except Exception as exc:  # noqa: BLE001 - never block a fetch
                    report.warn(f"{provider.name}: {exc}")
                    continue
                count = self.storage.upsert_weather(forecasts)
                report.note(provider.name, f"{count} forecasts")
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
                for warning in getattr(provider, "warnings", []) or []:
                    report.warn(f"{provider.name}: {warning}")
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

    # -- fitted ratings ------------------------------------------------------
    def compute_ratings(
        self,
        season: int,
        weeks: Optional[Sequence[int]] = None,
        *,
        carry_prior: bool = True,
    ) -> dict[int, "FitResult"]:
        """Fit the engine's own power ratings, one snapshot per week.

        Week ``W`` is fitted on games played *before* week ``W`` and
        stored stamped with week ``W``, which is exactly the snapshot the
        predictor picks up when pricing that week. That ordering is what
        makes these ratings safe to backtest with: the fit has never seen
        the game it is about to be asked about.
        """
        from .ratings.regression import fit_margin_ratings

        model = self.config.model
        all_games = self.storage.games(season)
        if not all_games:
            return {}

        played = [g for g in all_games if g.completed]
        target_weeks = list(weeks) if weeks else sorted({g.week for g in all_games})

        prior = self._carryover_prior(season) if carry_prior else None
        results: dict[int, FitResult] = {}

        for week in target_weeks:
            history = [g for g in played if g.week < week]
            if not history and not prior:
                continue
            fit = fit_margin_ratings(
                history,
                ridge=model.regression_ridge,
                margin_cap=model.regression_margin_cap,
                prior=prior,
                default_home_field=model.home_field_advantage,
                recency_halflife=model.regression_recency_halflife,
                as_of_week=week,
            )
            self.storage.upsert_ratings(fit.to_ratings(season, week))
            results[week] = fit
        return results

    def _carryover_prior(self, season: int) -> Optional[dict[str, float]]:
        """Last season's final fitted ratings, shrunk toward average.

        Week 1 has no games to fit on, so without a prior every team
        starts identical and the first weeks are noise. Carrying the
        previous season forward at partial strength is the standard
        remedy — teams change, but not completely.
        """
        weight = self.config.model.regression_carryover
        if weight <= 0:
            return None
        previous = self.storage.ratings(season - 1, source=SOURCE)
        if not previous:
            return None
        latest_week = max(r.week for r in previous)
        return {
            r.team: r.rating * weight for r in previous if r.week == latest_week
        }

    # -- predict -----------------------------------------------------------
    def load_adjustments(self, season: int, week: int) -> tuple[Adjustments, list[str]]:
        """Manual injury/situational adjustments for one week."""
        overrides = load_overrides(self.config.path(self.config.overrides_file))
        if not overrides:
            return Adjustments(), []
        return resolve_overrides(overrides, season, week, self.storage.games(season, week))

    def predict(
        self,
        season: int,
        week: int,
        *,
        use_projections: bool = True,
        use_overrides: bool = True,
    ) -> list[Prediction]:
        games = self.storage.games(season, week)
        if not games:
            return []
        ratings = self.storage.ratings(season, week)
        schedule = self.storage.games(season)
        projections = self.load_projections(season, week) if use_projections else []

        adjustments = Adjustments()
        self.override_warnings: list[str] = []
        if use_overrides:
            adjustments, self.override_warnings = self.load_adjustments(season, week)

        predictor = Predictor(self.config)
        predictions = predictor.predict_week(
            PredictionInputs(
                games=games, ratings=ratings,
                expert_projections=projections, schedule=schedule,
                adjustments=adjustments,
                weather=self.storage.weather([g.game_id for g in games]),
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
        """Settle any recommendation whose game has a final score.

        Also records closing line value where it can be established: the
        bet's number against the last odds capture taken *after* the bet
        was priced. With only one capture there is no later line and CLV
        stays null rather than being invented.
        """
        from .backtest import grade_recommendation
        from .backtest import _clv_points

        results = {"win": 0, "loss": 0, "push": 0, "ungraded": 0, "with_clv": 0}
        for rec_id, rec in self.storage.ungraded_recommendations(season):
            if week is not None and rec.week != week:
                continue
            game = self.storage.game(rec.game_id)
            if game is None or not game.completed:
                results["ungraded"] += 1
                continue
            outcome, profit = grade_recommendation(rec, game)

            closing = self.storage.closing_quotes(rec.game_id, rec.market, after=rec.placed_at)
            clv = _clv_points(rec, closing) if closing else None
            if clv is not None:
                results["with_clv"] += 1

            self.storage.grade_recommendation(rec_id, outcome, profit, clv)
            results[outcome] = results.get(outcome, 0) + 1
        return results

    # -- odds snapshots ------------------------------------------------------
    def snapshot_odds(
        self, season: int, week: Optional[int] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> tuple[FetchReport, dict[str, float]]:
        """Capture odds again and report what moved since the last capture.

        Repeated captures are what make closing line value computable at
        all, and CLV is the only early evidence that an edge is real.
        """
        before = self._current_lines(season, week)
        report = self.fetch(season, week, providers=providers, want=("odds",))
        after = self._current_lines(season, week)

        moves = {
            key: after[key] - before[key]
            for key in after
            if key in before and abs(after[key] - before[key]) > 1e-9
        }
        return report, moves

    def _current_lines(self, season: int, week: Optional[int]) -> dict[str, float]:
        """Median home/over line per game and market, as it stands now."""
        games = self.storage.games(season, week)
        quotes = self.storage.quotes_for_games([g.game_id for g in games])
        out: dict[str, float] = {}
        for game_id, rows in quotes.items():
            for market, side in (("spread", "home"), ("total", "over")):
                lines = sorted(
                    q.line for q in rows
                    if q.market == market and q.side == side and q.line is not None
                )
                if lines:
                    mid = len(lines) // 2
                    out[f"{game_id}|{market}"] = (
                        lines[mid] if len(lines) % 2 else (lines[mid - 1] + lines[mid]) / 2
                    )
        return out
