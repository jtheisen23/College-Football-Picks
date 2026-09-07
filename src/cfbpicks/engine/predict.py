"""Turn power ratings and expert projections into a number for each game.

The margin comes primarily from the blended power ratings, because that
is the most stable signal available. Published expert projections
(Sagarin's predictions table, Massey's game page) are blended in on top
with a configurable weight — they encode injury and situational
information a pure rating difference cannot see.

Totals are modelled separately from margins: a rating difference says
nothing about pace, so totals lean on offence/defence splits where a
source provides them and fall back to the league average otherwise.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable, Mapping, Optional

from ..config import Config
from ..models import ExpertProjection, Game, GameWeather, Prediction, Rating
from ..overrides import Adjustments
from ..weather import total_adjustment
from ..ratings.blend import BlendedRating, blend_ratings, confidence_from_blend
from ..util.odds import spread_from_probability, win_probability


@dataclass
class PredictionInputs:
    """Everything the predictor needs for one week."""

    games: list[Game]
    ratings: list[Rating] = field(default_factory=list)
    expert_projections: list[ExpertProjection] = field(default_factory=list)
    #: Full-season schedule, used to work out rest days. Falls back to games.
    schedule: Optional[list[Game]] = None
    #: Manual adjustments for injuries and anything else the ratings
    #: cannot see.
    adjustments: Optional[Adjustments] = None
    #: Kickoff forecasts, keyed by game id.
    weather: dict[str, GameWeather] = field(default_factory=dict)


class Predictor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = config.model

    # -- public ------------------------------------------------------------
    def predict_week(self, inputs: PredictionInputs) -> list[Prediction]:
        blend = blend_ratings(
            inputs.ratings,
            self.model.rating_weights,
            min_sources_for_full_confidence=self.model.min_sources_for_full_confidence,
        )
        offense, defense = _scoring_splits(inputs.ratings)
        rest = _rest_days(inputs.schedule or inputs.games)
        experts = _index_projections(inputs.expert_projections)

        adjustments = inputs.adjustments or Adjustments()
        return [
            self.predict_game(
                game, blend, offense, defense, rest,
                experts.get(game.game_id, []), adjustments,
                inputs.weather.get(game.game_id),
            )
            for game in inputs.games
        ]

    def predict_game(
        self,
        game: Game,
        blend: Mapping[str, BlendedRating],
        offense: Mapping[str, float],
        defense: Mapping[str, float],
        rest: Mapping[str, dict[int, int]],
        experts: Iterable[ExpertProjection] = (),
        adjustments: Optional[Adjustments] = None,
        weather: Optional[GameWeather] = None,
    ) -> Prediction:
        home_blend = blend.get(game.home_team)
        away_blend = blend.get(game.away_team)
        home_rating = home_blend.rating if home_blend else 0.0
        away_rating = away_blend.rating if away_blend else 0.0

        hfa = 0.0 if game.neutral_site else self.model.home_field_advantage
        rest_adj = self._rest_adjustment(game, rest)

        # Manual adjustments move a team's strength directly, so they
        # flow into the margin exactly as a rating difference would.
        adjustments = adjustments or Adjustments()
        home_adj = adjustments.for_team(game.home_team)
        away_adj = adjustments.for_team(game.away_team)
        manual_margin = adjustments.game_margin.get(game.game_id, 0.0)

        model_margin = (
            (home_rating + home_adj) - (away_rating + away_adj)
            + hfa + rest_adj + manual_margin
        )

        components: dict[str, object] = {
            "rating_margin": round(home_rating - away_rating, 3),
            "hfa": round(hfa, 3),
            "rest_adjustment": round(rest_adj, 3),
            "model_margin": round(model_margin, 3),
        }
        override_notes = adjustments.notes_for(game.home_team, game.away_team, game.game_id)
        if home_adj or away_adj or manual_margin:
            components["manual_margin_adjustment"] = round(
                home_adj - away_adj + manual_margin, 3
            )
        if override_notes:
            components["overrides"] = override_notes

        # -- expert blend --------------------------------------------------
        experts = [e for e in experts if e.projected_margin is not None]
        margin = model_margin
        if experts:
            weight = float(self.model.rating_weights.get("expert_projection", 0.0)) or \
                EXPERT_WEIGHT_DEFAULT
            expert_margin = statistics.fmean([e.projected_margin for e in experts])
            margin = (1.0 - weight) * model_margin + weight * expert_margin
            components["expert_margin"] = round(expert_margin, 3)
            components["expert_weight"] = round(weight, 3)
            components["expert_sources"] = sorted({e.source for e in experts})

        # -- total ----------------------------------------------------------
        total, total_note = self._project_total(game, offense, defense, experts)

        weather_points, weather_reasons = total_adjustment(weather, self.config.weather)
        if weather_points:
            total += weather_points
            components["weather_adjustment"] = weather_points
            components["weather"] = weather_reasons
            total_note = f"{total_note} + weather"
        elif weather is not None:
            components["weather"] = [weather.describe()]

        manual_total = adjustments.game_total.get(game.game_id, 0.0)
        if manual_total:
            total += manual_total
            components["manual_total_adjustment"] = round(manual_total, 3)
            total_note = f"{total_note} + manual"
        components["total_basis"] = total_note

        win_prob = win_probability(margin, self.model.margin_sd)
        fair_spread = spread_from_probability(win_prob, self.model.margin_sd)

        confidence = self._confidence(home_blend, away_blend, model_margin, experts)
        sources = sorted(
            set((home_blend.sources if home_blend else []))
            | set((away_blend.sources if away_blend else []))
        )

        return Prediction(
            game_id=game.game_id, season=game.season, week=game.week,
            home_team=game.home_team, away_team=game.away_team,
            projected_margin=round(margin, 3), projected_total=round(total, 3),
            home_win_prob=round(win_prob, 5), fair_home_spread=round(fair_spread, 3),
            home_rating=round(home_rating, 3), away_rating=round(away_rating, 3),
            hfa_applied=round(hfa, 3), components=components,
            rating_sources=sources, confidence=round(confidence, 4),
        )

    # -- internals ---------------------------------------------------------
    def _project_total(
        self,
        game: Game,
        offense: Mapping[str, float],
        defense: Mapping[str, float],
        experts: list[ExpertProjection],
    ) -> tuple[float, str]:
        base = self.model.league_average_total
        have_splits = (
            game.home_team in offense and game.away_team in offense
            and game.home_team in defense and game.away_team in defense
        )
        if have_splits:
            # Each side's scoring is its own offence against the other's
            # defence, both expressed relative to the league average.
            home_points = base / 2 + offense[game.home_team] + defense[game.away_team]
            away_points = base / 2 + offense[game.away_team] + defense[game.home_team]
            total = home_points + away_points
            note = "offense/defense splits"
        else:
            total = base
            note = "league average (no offence/defence splits available)"

        expert_totals = [e.projected_total for e in experts if e.projected_total is not None]
        if expert_totals:
            weight = EXPERT_WEIGHT_DEFAULT if not have_splits else EXPERT_WEIGHT_DEFAULT / 2
            # With no splits of our own, defer heavily to the expert number.
            weight = 0.8 if not have_splits else weight
            total = (1.0 - weight) * total + weight * statistics.fmean(expert_totals)
            note = f"{note} + expert totals"
        return total, note

    def _rest_adjustment(self, game: Game, rest: Mapping[str, dict[int, int]]) -> float:
        home_rest = rest.get(game.home_team, {}).get(game.week)
        away_rest = rest.get(game.away_team, {}).get(game.week)
        if home_rest is None or away_rest is None:
            return 0.0
        delta = (home_rest - away_rest) * self.model.rest_advantage_per_day
        cap = self.model.max_rest_adjustment
        return max(-cap, min(cap, delta))

    def _confidence(
        self,
        home_blend: Optional[BlendedRating],
        away_blend: Optional[BlendedRating],
        model_margin: float,
        experts: list[ExpertProjection],
    ) -> float:
        if home_blend is None or away_blend is None:
            # An unrated team (FCS opponent, or a name that failed to
            # match) is the single biggest source of bad bets. Say so.
            return 0.1

        min_sources = self.model.min_sources_for_full_confidence
        confidence = min(
            confidence_from_blend(home_blend, min_sources=min_sources),
            confidence_from_blend(away_blend, min_sources=min_sources),
        )

        if experts:
            gaps = [abs(model_margin - e.projected_margin) for e in experts]
            # A big model-vs-expert gap usually means one of them is
            # missing something; trim confidence rather than pick a winner.
            confidence *= 1.0 / (1.0 + (statistics.fmean(gaps) / 14.0) ** 2)
        return max(0.05, min(1.0, confidence))


#: Weight given to published expert projections when blending margins.
EXPERT_WEIGHT_DEFAULT = 0.35


def _scoring_splits(ratings: Iterable[Rating]) -> tuple[dict[str, float], dict[str, float]]:
    """Centre offence and defence ratings on the league average.

    CFBD reports SP+ offence and defence on absolute points scales (an
    offence near 30, a defence near 20), so both are recentred here.
    Defence keeps its "points allowed relative to average" orientation:
    positive means a defence that gives up more than the norm.
    """
    off_raw: dict[str, list[float]] = {}
    def_raw: dict[str, list[float]] = {}
    for rating in ratings:
        if rating.offense is not None:
            off_raw.setdefault(rating.team, []).append(rating.offense)
        if rating.defense is not None:
            def_raw.setdefault(rating.team, []).append(rating.defense)

    offense = {t: statistics.fmean(v) for t, v in off_raw.items()}
    defense = {t: statistics.fmean(v) for t, v in def_raw.items()}

    if len(offense) > 1:
        mean = statistics.fmean(offense.values())
        offense = {t: v - mean for t, v in offense.items()}
    if len(defense) > 1:
        mean = statistics.fmean(defense.values())
        defense = {t: v - mean for t, v in defense.items()}
    return offense, defense


def _rest_days(schedule: Iterable[Game]) -> dict[str, dict[int, int]]:
    """Days of rest before each team's game in each week."""
    by_team: dict[str, list[Game]] = {}
    for game in schedule:
        if game.kickoff is None:
            continue
        by_team.setdefault(game.home_team, []).append(game)
        by_team.setdefault(game.away_team, []).append(game)

    out: dict[str, dict[int, int]] = {}
    for team, games in by_team.items():
        games.sort(key=lambda g: g.kickoff)  # type: ignore[arg-type]
        for i, game in enumerate(games):
            if i == 0:
                continue
            delta: timedelta = game.kickoff - games[i - 1].kickoff  # type: ignore[operator]
            out.setdefault(team, {})[game.week] = int(delta.days)
    return out


def _index_projections(
    projections: Iterable[ExpertProjection],
) -> dict[str, list[ExpertProjection]]:
    out: dict[str, list[ExpertProjection]] = {}
    for proj in projections:
        if proj.game_id:
            out.setdefault(proj.game_id, []).append(proj)
    return out
