"""SQLite persistence for games, ratings, market quotes and picks.

Everything the engine reads is stored locally so a week can be re-priced,
re-graded or backtested without spending another API call.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from .models import Game, MarketQuote, Prediction, Rating, Recommendation

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    kickoff         TEXT,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    neutral_site    INTEGER NOT NULL DEFAULT 0,
    conference_game INTEGER NOT NULL DEFAULT 0,
    venue           TEXT,
    home_score      INTEGER,
    away_score      INTEGER,
    season_type     TEXT NOT NULL DEFAULT 'regular',
    source          TEXT,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_week ON games(season, week);

CREATE TABLE IF NOT EXISTS ratings (
    source   TEXT NOT NULL,
    season   INTEGER NOT NULL,
    week     INTEGER NOT NULL,
    team     TEXT NOT NULL,
    rating   REAL NOT NULL,
    offense  REAL,
    defense  REAL,
    rank     INTEGER,
    point_in_time INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source, season, week, team)
);
CREATE INDEX IF NOT EXISTS idx_ratings_week ON ratings(season, week);

CREATE TABLE IF NOT EXISTS market_quotes (
    game_id    TEXT NOT NULL,
    book       TEXT NOT NULL,
    market     TEXT NOT NULL,
    side       TEXT NOT NULL,
    line       REAL,
    price      REAL NOT NULL,
    fetched_at TEXT NOT NULL,
    source     TEXT,
    PRIMARY KEY (game_id, book, market, side)
);
CREATE INDEX IF NOT EXISTS idx_quotes_game ON market_quotes(game_id);

-- Every fetch also appends here, so line movement and closing lines
-- survive even though market_quotes only holds the latest snapshot.
CREATE TABLE IF NOT EXISTS quote_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id    TEXT NOT NULL,
    book       TEXT NOT NULL,
    market     TEXT NOT NULL,
    side       TEXT NOT NULL,
    line       REAL,
    price      REAL NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_game ON quote_history(game_id, market, fetched_at);

CREATE TABLE IF NOT EXISTS predictions (
    game_id          TEXT PRIMARY KEY,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    projected_margin REAL NOT NULL,
    projected_total  REAL NOT NULL,
    home_win_prob    REAL NOT NULL,
    fair_home_spread REAL NOT NULL,
    home_rating      REAL NOT NULL,
    away_rating      REAL NOT NULL,
    hfa_applied      REAL NOT NULL,
    confidence       REAL NOT NULL DEFAULT 1.0,
    components       TEXT,
    rating_sources   TEXT,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      TEXT NOT NULL,
    season       INTEGER NOT NULL,
    week         INTEGER NOT NULL,
    matchup      TEXT NOT NULL,
    market       TEXT NOT NULL,
    side         TEXT NOT NULL,
    selection    TEXT NOT NULL,
    line         REAL,
    price        REAL NOT NULL,
    book         TEXT NOT NULL,
    model_prob   REAL NOT NULL,
    market_prob  REAL NOT NULL,
    prob_edge    REAL NOT NULL,
    point_edge   REAL NOT NULL,
    expected_value REAL NOT NULL,
    stake_units  REAL NOT NULL,
    kelly_fraction REAL NOT NULL,
    push_prob    REAL NOT NULL DEFAULT 0,
    confidence   REAL NOT NULL DEFAULT 1,
    tier         TEXT NOT NULL,
    notes        TEXT,
    created_at   TEXT NOT NULL,
    result       TEXT,
    profit_units REAL,
    clv_points   REAL,
    UNIQUE (game_id, market, side, line, book, created_at)
);
CREATE INDEX IF NOT EXISTS idx_recs_week ON recommendations(season, week);

CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    # Microsecond precision, not seconds: quote_history keys snapshots by
    # this timestamp, and two captures inside the same second would
    # otherwise collapse into one and hide real line movement.
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class Storage:
    """Thin, explicit SQLite wrapper — no ORM, no magic."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(ratings)")}
        rec_columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(recommendations)")}
        if "clv_points" not in rec_columns:
            self.conn.execute("ALTER TABLE recommendations ADD COLUMN clv_points REAL")

        if "point_in_time" not in columns:
            # Existing rows predate the distinction. Assume the unsafe
            # case so an old database can't silently contaminate a
            # backtest; a re-fetch will label them correctly.
            self.conn.execute(
                "ALTER TABLE ratings ADD COLUMN point_in_time INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- games -----------------------------------------------------------
    def upsert_games(self, games: Iterable[Game]) -> int:
        rows = [
            (
                g.game_id, g.season, g.week,
                g.kickoff.isoformat() if g.kickoff else None,
                g.home_team, g.away_team,
                int(g.neutral_site), int(g.conference_game), g.venue,
                g.home_score, g.away_score, g.season_type, g.source, _now(),
            )
            for g in games
        ]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO games (game_id, season, week, kickoff, home_team, away_team,
                                   neutral_site, conference_game, venue, home_score,
                                   away_score, season_type, source, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(game_id) DO UPDATE SET
                    kickoff=excluded.kickoff,
                    neutral_site=excluded.neutral_site,
                    conference_game=excluded.conference_game,
                    venue=excluded.venue,
                    -- Never overwrite a known score with a NULL from a
                    -- schedule-only refresh.
                    home_score=COALESCE(excluded.home_score, games.home_score),
                    away_score=COALESCE(excluded.away_score, games.away_score),
                    season_type=excluded.season_type,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def games(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        *,
        completed: Optional[bool] = None,
    ) -> list[Game]:
        sql = "SELECT * FROM games WHERE 1=1"
        params: list[Any] = []
        if season is not None:
            sql += " AND season = ?"
            params.append(season)
        if week is not None:
            sql += " AND week = ?"
            params.append(week)
        if completed is True:
            sql += " AND home_score IS NOT NULL AND away_score IS NOT NULL"
        elif completed is False:
            sql += " AND (home_score IS NULL OR away_score IS NULL)"
        sql += " ORDER BY kickoff IS NULL, kickoff, home_team"
        return [_row_to_game(r) for r in self.conn.execute(sql, params)]

    def game(self, game_id: str) -> Optional[Game]:
        row = self.conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        return _row_to_game(row) if row else None

    # -- ratings ---------------------------------------------------------
    def upsert_ratings(self, ratings: Iterable[Rating]) -> int:
        rows = [
            (r.source, r.season, r.week, r.team, r.rating, r.offense, r.defense,
             r.rank, int(r.point_in_time), _now())
            for r in ratings
        ]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO ratings (source, season, week, team, rating, offense, defense,
                                     rank, point_in_time, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source, season, week, team) DO UPDATE SET
                    rating=excluded.rating, offense=excluded.offense,
                    defense=excluded.defense, rank=excluded.rank,
                    point_in_time=excluded.point_in_time,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def ratings(
        self,
        season: int,
        week: Optional[int] = None,
        source: Optional[str] = None,
        *,
        point_in_time_only: bool = False,
    ) -> list[Rating]:
        """Ratings for a week.

        When ``week`` is given, each source contributes its most recent
        snapshot at or before that week — sources publish on different
        cadences and some only post a preseason number.

        ``point_in_time_only`` drops season-final ratings. Backtests must
        set it: a season-final SP+ number already knows how the game you
        are about to "predict" turned out.
        """
        params: list[Any] = [season]
        sql = "SELECT * FROM ratings WHERE season = ?"
        if point_in_time_only:
            sql += " AND point_in_time = 1"
        if source:
            sql += " AND source = ?"
            params.append(source)
        if week is not None:
            sql += " AND week <= ?"
            params.append(week)
        rows = list(self.conn.execute(sql, params))

        latest: dict[tuple[str, str], sqlite3.Row] = {}
        for row in rows:
            key = (row["source"], row["team"])
            current = latest.get(key)
            if current is None or row["week"] > current["week"]:
                latest[key] = row
        return [
            Rating(
                source=r["source"], season=r["season"], week=r["week"], team=r["team"],
                rating=r["rating"], offense=r["offense"], defense=r["defense"], rank=r["rank"],
                point_in_time=bool(r["point_in_time"]),
            )
            for r in latest.values()
        ]

    def rating_sources(self, season: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT source FROM ratings WHERE season = ? ORDER BY source", (season,)
        )
        return [r["source"] for r in rows]

    # -- market ----------------------------------------------------------
    def upsert_quotes(self, quotes: Iterable[MarketQuote]) -> int:
        quotes = list(quotes)
        if not quotes:
            return 0
        now = _now()
        rows = [
            (
                q.game_id, q.book, q.market, q.side, q.line, q.price,
                q.fetched_at.isoformat() if q.fetched_at else now, q.source,
            )
            for q in quotes
        ]
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO market_quotes (game_id, book, market, side, line, price, fetched_at, source)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(game_id, book, market, side) DO UPDATE SET
                    line=excluded.line, price=excluded.price,
                    fetched_at=excluded.fetched_at, source=excluded.source
                """,
                rows,
            )
            conn.executemany(
                """
                INSERT INTO quote_history (game_id, book, market, side, line, price, fetched_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows],
            )
        return len(rows)

    def quotes(self, game_id: Optional[str] = None, market: Optional[str] = None) -> list[MarketQuote]:
        sql = "SELECT * FROM market_quotes WHERE 1=1"
        params: list[Any] = []
        if game_id:
            sql += " AND game_id = ?"
            params.append(game_id)
        if market:
            sql += " AND market = ?"
            params.append(market)
        return [_row_to_quote(r) for r in self.conn.execute(sql, params)]

    def quotes_for_games(self, game_ids: Sequence[str]) -> dict[str, list[MarketQuote]]:
        if not game_ids:
            return {}
        placeholders = ",".join("?" for _ in game_ids)
        rows = self.conn.execute(
            f"SELECT * FROM market_quotes WHERE game_id IN ({placeholders})", list(game_ids)
        )
        grouped: dict[str, list[MarketQuote]] = {}
        for row in rows:
            grouped.setdefault(row["game_id"], []).append(_row_to_quote(row))
        return grouped

    def line_history(
        self, game_id: str, market: str, side: str
    ) -> list[tuple[datetime, float, float]]:
        """Every capture of one side, as ``(when, median line, median price)``.

        Books disagree at any instant, so each capture is collapsed to a
        median across books; what matters here is how the market moved,
        not which book was a half point off.
        """
        rows = self.conn.execute(
            """SELECT fetched_at, line, price FROM quote_history
               WHERE game_id=? AND market=? AND side=? AND line IS NOT NULL
               ORDER BY fetched_at""",
            (game_id, market, side),
        )
        grouped: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            grouped.setdefault(row["fetched_at"], []).append((row["line"], row["price"]))

        series = []
        for stamp, values in sorted(grouped.items()):
            when = _parse_dt(stamp)
            if when is None:
                continue
            lines = sorted(v[0] for v in values)
            prices = sorted(v[1] for v in values)
            series.append((when, _median(lines), _median(prices)))
        return series

    def snapshot_count(self, game_id: str, market: str) -> int:
        """Distinct times this game/market was captured.

        One snapshot means the "closing" line is the same row the bet was
        priced from, so closing line value cannot be computed at all.
        """
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT fetched_at) AS n FROM quote_history WHERE game_id=? AND market=?",
            (game_id, market),
        ).fetchone()
        return int(row["n"]) if row else 0

    def closing_quotes(
        self, game_id: str, market: str, after: Optional[datetime] = None
    ) -> list[MarketQuote]:
        """The last snapshot recorded for a game/market.

        ``after`` restricts to captures strictly later than a moment —
        pass the time a bet was priced and this returns a genuine closing
        line, or nothing at all if the odds were never captured again.
        """
        sql = "SELECT MAX(fetched_at) AS last FROM quote_history WHERE game_id=? AND market=?"
        params: list[Any] = [game_id, market]
        if after is not None:
            sql += " AND fetched_at > ?"
            params.append(after.isoformat())
        row = self.conn.execute(sql, params).fetchone()
        if not row or not row["last"]:
            return []
        rows = self.conn.execute(
            "SELECT * FROM quote_history WHERE game_id=? AND market=? AND fetched_at=?",
            (game_id, market, row["last"]),
        )
        return [
            MarketQuote(
                game_id=r["game_id"], book=r["book"], market=r["market"], side=r["side"],
                line=r["line"], price=r["price"], fetched_at=_parse_dt(r["fetched_at"]),
            )
            for r in rows
        ]

    # -- predictions -----------------------------------------------------
    def upsert_predictions(self, predictions: Iterable[Prediction]) -> int:
        rows = [
            (
                p.game_id, p.season, p.week, p.home_team, p.away_team,
                p.projected_margin, p.projected_total, p.home_win_prob,
                p.fair_home_spread, p.home_rating, p.away_rating, p.hfa_applied,
                p.confidence, json.dumps(p.components), json.dumps(p.rating_sources), _now(),
            )
            for p in predictions
        ]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO predictions (game_id, season, week, home_team, away_team,
                    projected_margin, projected_total, home_win_prob, fair_home_spread,
                    home_rating, away_rating, hfa_applied, confidence, components,
                    rating_sources, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(game_id) DO UPDATE SET
                    projected_margin=excluded.projected_margin,
                    projected_total=excluded.projected_total,
                    home_win_prob=excluded.home_win_prob,
                    fair_home_spread=excluded.fair_home_spread,
                    home_rating=excluded.home_rating, away_rating=excluded.away_rating,
                    hfa_applied=excluded.hfa_applied, confidence=excluded.confidence,
                    components=excluded.components, rating_sources=excluded.rating_sources,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def predictions(self, season: int, week: Optional[int] = None) -> list[Prediction]:
        sql = "SELECT * FROM predictions WHERE season = ?"
        params: list[Any] = [season]
        if week is not None:
            sql += " AND week = ?"
            params.append(week)
        return [
            Prediction(
                game_id=r["game_id"], season=r["season"], week=r["week"],
                home_team=r["home_team"], away_team=r["away_team"],
                projected_margin=r["projected_margin"], projected_total=r["projected_total"],
                home_win_prob=r["home_win_prob"], fair_home_spread=r["fair_home_spread"],
                home_rating=r["home_rating"], away_rating=r["away_rating"],
                hfa_applied=r["hfa_applied"], confidence=r["confidence"],
                components=json.loads(r["components"] or "{}"),
                rating_sources=json.loads(r["rating_sources"] or "[]"),
            )
            for r in self.conn.execute(sql, params)
        ]

    # -- recommendations -------------------------------------------------
    def save_recommendations(self, recs: Iterable[Recommendation], *, replace_week: bool = True) -> int:
        recs = list(recs)
        if not recs:
            return 0
        created = _now()
        with self.transaction() as conn:
            if replace_week:
                weeks = {(r.season, r.week) for r in recs}
                for season, week in weeks:
                    conn.execute(
                        "DELETE FROM recommendations WHERE season=? AND week=? AND result IS NULL",
                        (season, week),
                    )
            conn.executemany(
                """
                INSERT OR IGNORE INTO recommendations (game_id, season, week, matchup, market, side,
                    selection, line, price, book, model_prob, market_prob, prob_edge, point_edge,
                    expected_value, stake_units, kelly_fraction, push_prob, confidence, tier,
                    notes, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r.game_id, r.season, r.week, r.matchup, r.market, r.side, r.selection,
                        r.line, r.price, r.book, r.model_prob, r.market_prob, r.prob_edge,
                        r.point_edge, r.expected_value, r.stake_units, r.kelly_fraction,
                        r.push_prob, r.confidence, r.tier, json.dumps(r.notes), created,
                    )
                    for r in recs
                ],
            )
        return len(recs)

    def recommendations(self, season: int, week: Optional[int] = None) -> list[Recommendation]:
        sql = "SELECT * FROM recommendations WHERE season = ?"
        params: list[Any] = [season]
        if week is not None:
            sql += " AND week = ?"
            params.append(week)
        sql += " ORDER BY prob_edge DESC"
        return [_row_to_rec(r) for r in self.conn.execute(sql, params)]

    def grade_recommendation(
        self,
        rec_id: int,
        result: str,
        profit_units: float,
        clv_points: Optional[float] = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE recommendations SET result=?, profit_units=?, clv_points=? WHERE id=?",
                (result, profit_units, clv_points, rec_id),
            )

    def ungraded_recommendations(self, season: Optional[int] = None) -> list[tuple[int, Recommendation]]:
        sql = "SELECT * FROM recommendations WHERE result IS NULL"
        params: list[Any] = []
        if season is not None:
            sql += " AND season = ?"
            params.append(season)
        return [(r["id"], _row_to_rec(r)) for r in self.conn.execute(sql, params)]

    # -- meta ------------------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO meta (key, value, updated_at) VALUES (?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key, value, _now()),
            )

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def counts(self) -> dict[str, int]:
        tables = ["games", "ratings", "market_quotes", "predictions", "recommendations"]
        return {
            t: self.conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] for t in tables
        }


def _median(values: list[float]) -> float:
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _row_to_game(row: sqlite3.Row) -> Game:
    return Game(
        game_id=row["game_id"], season=row["season"], week=row["week"],
        kickoff=_parse_dt(row["kickoff"]), home_team=row["home_team"],
        away_team=row["away_team"], neutral_site=bool(row["neutral_site"]),
        conference_game=bool(row["conference_game"]), venue=row["venue"],
        home_score=row["home_score"], away_score=row["away_score"],
        season_type=row["season_type"], source=row["source"] or "unknown",
    )


def _row_to_quote(row: sqlite3.Row) -> MarketQuote:
    return MarketQuote(
        game_id=row["game_id"], book=row["book"], market=row["market"], side=row["side"],
        line=row["line"], price=row["price"], fetched_at=_parse_dt(row["fetched_at"]),
        source=row["source"] or "unknown",
    )


def _row_to_rec(row: sqlite3.Row) -> Recommendation:
    return Recommendation(
        game_id=row["game_id"], season=row["season"], week=row["week"], matchup=row["matchup"],
        market=row["market"], side=row["side"], selection=row["selection"], line=row["line"],
        price=row["price"], book=row["book"], model_prob=row["model_prob"],
        market_prob=row["market_prob"], prob_edge=row["prob_edge"], point_edge=row["point_edge"],
        expected_value=row["expected_value"], stake_units=row["stake_units"],
        kelly_fraction=row["kelly_fraction"], push_prob=row["push_prob"],
        confidence=row["confidence"], tier=row["tier"], notes=json.loads(row["notes"] or "[]"),
        placed_at=_parse_dt(row["created_at"]) if "created_at" in row.keys() else None,
    )
