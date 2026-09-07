"""Command line interface.

A normal week is three commands::

    cfbpicks fetch --week 3
    cfbpicks picks --week 3
    cfbpicks grade            # after the games finish
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config, load_config
from .pipeline import Pipeline
from .providers import build_provider, build_providers
from .storage import Storage

console = Console()


def default_season() -> int:
    """College football seasons are named for the calendar year they start."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _config(ctx: click.Context) -> Config:
    return ctx.obj["config"]


def _season(ctx: click.Context, season: Optional[int]) -> int:
    return season or _config(ctx).season or default_season()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=click.Path(), help="Path to a config YAML file.")
@click.option("--db", type=click.Path(), help="Override the SQLite database path.")
@click.option("--offline", is_flag=True, help="Never hit the network; use cache and fixtures only.")
@click.version_option(__version__, prog_name="cfbpicks")
@click.pass_context
def main(ctx: click.Context, config_path: Optional[str], db: Optional[str], offline: bool) -> None:
    """College football predictions and betting-edge engine."""
    config = load_config(config_path)
    if db:
        config.database = db
    if offline:
        config.offline = True
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


# ---------------------------------------------------------------------------
@main.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Create the database and show where everything lives."""
    config = _config(ctx)
    storage = Storage(config.database_path)
    counts = storage.counts()
    storage.close()

    console.print(f"[green]Database ready:[/green] {config.database_path}")
    for name, count in counts.items():
        console.print(f"  {name:16} {count}")
    console.print(f"\nCache:    {config.path(config.cache_dir)}")
    console.print(f"Reports:  {config.path(config.reports_dir)}")
    console.print("\nNext: set CFBD_API_KEY, then run [bold]cfbpicks fetch --week 1[/bold]")


@main.command()
@click.pass_context
def providers(ctx: click.Context) -> None:
    """List data sources and whether each one is ready to use."""
    config = _config(ctx)
    table = Table(title="Data providers", header_style="bold")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Supplies")
    table.add_column("Description", overflow="fold")

    for provider in build_providers(config):
        supplies = ", ".join(
            name for name, flag in (
                ("games", provider.provides_games),
                ("ratings", provider.provides_ratings),
                ("odds", provider.provides_odds),
            ) if flag
        )
        status = provider.status()
        colour = {"ready": "green", "missing key": "yellow", "disabled": "dim"}.get(status, "")
        table.add_row(
            provider.name, f"[{colour}]{status}[/{colour}]", supplies, provider.description
        )
    console.print(table)

    missing = [p.name for p in build_providers(config) if p.enabled and not p.configured]
    if missing:
        console.print(
            "\n[yellow]Set an API key for:[/yellow] " + ", ".join(missing) +
            "\n  export CFBD_API_KEY=...    # collegefootballdata.com/key"
            "\n  export ODDS_API_KEY=...    # the-odds-api.com"
        )


# ---------------------------------------------------------------------------
@main.command()
@click.option("--season", type=int, help="Season year (defaults to the current one).")
@click.option("--week", type=int, help="Week number. Omit to fetch the whole season.")
@click.option("--provider", "only", multiple=True, help="Limit to specific providers.")
@click.option("--games/--no-games", default=True)
@click.option("--ratings/--no-ratings", default=True)
@click.option("--odds/--no-odds", default=True)
@click.pass_context
def fetch(ctx, season, week, only, games, ratings, odds) -> None:
    """Pull games, ratings and betting lines into local storage."""
    config = _config(ctx)
    season = _season(ctx, season)
    want = [n for n, on in (("games", games), ("ratings", ratings), ("odds", odds)) if on]

    with Pipeline(config) as pipeline:
        with console.status(f"Fetching {season} week {week or 'all'}…"):
            report = pipeline.fetch(season, week, providers=list(only) or None, want=want)

    console.print(
        f"[green]Fetched[/green] {report.games} games · {report.ratings} ratings · "
        f"{report.quotes} quotes"
    )
    for name, detail in sorted(report.by_provider.items()):
        console.print(f"  {name:10} {detail}")
    for warning in report.warnings[:15]:
        console.print(f"  [yellow]![/yellow] {warning}")
    if len(report.warnings) > 15:
        console.print(f"  [dim]… and {len(report.warnings) - 15} more warnings[/dim]")


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, required=True)
@click.option("--limit", type=int, help="Show only the N biggest mismatches.")
@click.pass_context
def predict(ctx, season, week, limit) -> None:
    """Project every game this week, before looking at any price."""
    from .report import predictions_table

    config = _config(ctx)
    season = _season(ctx, season)
    with Pipeline(config) as pipeline:
        predictions = pipeline.predict(season, week)

    if not predictions:
        console.print(
            f"[yellow]No games stored for {season} week {week}.[/yellow] "
            f"Run [bold]cfbpicks fetch --week {week}[/bold] first."
        )
        return
    console.print(predictions_table(predictions, limit))
    thin = [p for p in predictions if p.confidence < 0.3]
    if thin:
        console.print(
            f"\n[yellow]{len(thin)} game(s) have thin rating coverage[/yellow] "
            "(likely FCS opponents or unmatched team names)."
        )


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, required=True)
@click.option("--market", "markets", multiple=True,
              type=click.Choice(["spread", "total", "moneyline"]),
              help="Restrict to specific markets.")
@click.option("--min-edge", type=float, help="Minimum probability edge, e.g. 0.03 for 3%.")
@click.option("--max-units", type=float, help="Cap the stake on any single bet.")
@click.option("--format", "fmt", type=click.Choice(["table", "markdown", "csv", "json"]),
              default="table")
@click.option("--output", "-o", type=click.Path(), help="Write to a file instead of stdout.")
@click.option("--all", "show_all", is_flag=True, help="Include bets that fail the thresholds.")
@click.pass_context
def picks(ctx, season, week, markets, min_edge, max_units, fmt, output, show_all) -> None:
    """Recommend bets for a week, with the edge on each one quantified."""
    from .report import render, render_console

    config = _config(ctx)
    season = _season(ctx, season)
    if markets:
        config.betting.markets = list(markets)
    if min_edge is not None:
        config.betting.min_prob_edge = min_edge
    if max_units is not None:
        config.betting.max_units = max_units
    if show_all:
        config.betting.min_prob_edge = -1.0
        config.betting.min_expected_value = -1.0
        config.betting.min_point_edge_spread = 0.0
        config.betting.min_point_edge_total = 0.0

    with Pipeline(config) as pipeline:
        recommendations = pipeline.picks(season, week)
        predictions = pipeline.storage.predictions(season, week)

    if not predictions:
        console.print(
            f"[yellow]No games stored for {season} week {week}.[/yellow] "
            f"Run [bold]cfbpicks fetch --week {week}[/bold] first."
        )
        sys.exit(1)

    if fmt == "table" and not output:
        render_console(recommendations, console)
        return

    text = render(
        recommendations, "markdown" if fmt == "table" else fmt,
        season=season, week=week, predictions=predictions,
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        console.print(f"[green]Wrote[/green] {len(recommendations)} picks to {path}")
    else:
        click.echo(text)


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, help="Grade a single week (default: everything pending).")
@click.pass_context
def grade(ctx, season, week) -> None:
    """Settle past recommendations against final scores."""
    config = _config(ctx)
    season = _season(ctx, season)
    with Pipeline(config) as pipeline:
        results = pipeline.grade(season, week)
        recs = [r for r in pipeline.storage.recommendations(season, week)]
        games = {g.game_id: g for g in pipeline.storage.games(season)}

    from .backtest import grade_all
    summary = grade_all(recs, games)

    console.print(
        f"[green]Graded[/green] {results.get('win', 0)}W-{results.get('loss', 0)}L-"
        f"{results.get('push', 0)}P  ({results.get('ungraded', 0)} still pending)"
    )
    for line in summary.summary_lines():
        console.print(f"  {line}")


@main.command()
@click.option("--season", type=int, required=True)
@click.option("--weeks", help="Week range, e.g. 1-14 or 3,4,5. Defaults to every completed week.")
@click.option("--allow-final-ratings", is_flag=True,
              help="Include season-final ratings. Results will be inflated by hindsight.")
@click.pass_context
def backtest(ctx, season, weeks, allow_final_ratings) -> None:
    """Replay a season and report ROI, by market and by tier."""
    from .backtest import LookaheadError, backtest_season

    config = _config(ctx)
    week_list = _parse_weeks(weeks) if weeks else None

    try:
        with Pipeline(config) as pipeline:
            with console.status(f"Replaying {season}…"):
                result = backtest_season(
                    pipeline.storage, config, season, week_list,
                    allow_final_ratings=allow_final_ratings,
                )
    except LookaheadError as exc:
        console.print(f"[red]Backtest refused:[/red] {exc}")
        sys.exit(1)

    console.print(f"[bold]Backtest — {season}[/bold]")
    if allow_final_ratings:
        console.print(
            "  [red]HINDSIGHT ENABLED — these numbers are not real.[/red] "
            "Season-final ratings know how each game ended."
        )
    for line in result.summary_lines():
        console.print(f"  {line}")

    if result.rating_sources:
        console.print(f"  [dim]Ratings used: {', '.join(result.rating_sources)}[/dim]")
    if result.excluded_sources and not allow_final_ratings:
        console.print(
            f"  [dim]Excluded as season-final: {', '.join(result.excluded_sources)}[/dim]"
        )

    if result.by_market:
        table = Table(title="By market", header_style="bold")
        for column in ("Market", "Bets", "Win%", "Staked", "Profit", "ROI"):
            table.add_column(column, justify="right" if column != "Market" else "left")
        for market, stats in sorted(result.by_market.items()):
            decided = stats["bets"]
            roi = stats["profit"] / stats["staked"] if stats["staked"] else 0.0
            table.add_row(
                market, f"{int(stats['bets'])}",
                f"{stats['wins'] / decided * 100:.1f}%" if decided else "—",
                f"{stats['staked']:.1f}u", f"{stats['profit']:+.2f}u", f"{roi * 100:+.1f}%",
            )
        console.print(table)

    if result.by_tier:
        table = Table(title="By tier", header_style="bold")
        for column in ("Tier", "Bets", "Profit", "ROI"):
            table.add_column(column, justify="right" if column != "Tier" else "left")
        for tier in ("strong", "play", "lean"):
            stats = result.by_tier.get(tier)
            if not stats:
                continue
            roi = stats["profit"] / stats["staked"] if stats["staked"] else 0.0
            table.add_row(tier, f"{int(stats['bets'])}", f"{stats['profit']:+.2f}u", f"{roi * 100:+.1f}%")
        console.print(table)

    if result.bets < 100:
        console.print(
            "\n[yellow]Small sample.[/yellow] Under a few hundred bets, ROI is mostly noise — "
            "watch closing line value instead."
        )
    if result.clv_unavailable:
        console.print(
            f"\n[yellow]No closing line value available[/yellow] for "
            f"{result.clv_unavailable} bets: odds were captured only once, so "
            "there is no later line to measure against. Fetch odds during the "
            "week and again near kickoff to make CLV meaningful."
        )


# ---------------------------------------------------------------------------
@main.command("import-sagarin")
@click.argument("path", type=click.Path(exists=True))
@click.option("--season", type=int)
@click.option("--week", type=int, required=True)
@click.option("--dry-run", is_flag=True, help="Parse and report without saving.")
@click.pass_context
def import_sagarin(ctx, path, season, week, dry_run) -> None:
    """Ingest a saved copy of sagarin.com/sports/cfsend.htm.

    Grabs both the ratings and Sagarin's projected spreads, totals and
    moneylines. Save the page from a browser, then point this at it.
    """
    from .providers.sagarin import parse_file, resolve_projections

    config = _config(ctx)
    season = _season(ctx, season)
    result = parse_file(Path(path), season, week)
    console.print(f"Parsed {result.summary()}")

    if not result.ok:
        console.print(
            "[red]Nothing parsed.[/red] The page layout may have changed — "
            "check that this is the college football page and not an index."
        )
        sys.exit(1)

    with Pipeline(config) as pipeline:
        games = pipeline.storage.games(season, week)
        resolved, unmatched = resolve_projections(result.projections, games)
        if not dry_run:
            saved = pipeline.storage.upsert_ratings(result.ratings)
            console.print(f"[green]Saved[/green] {saved} Sagarin ratings")
        console.print(f"Matched {len(resolved)} projections to scheduled games")
        if unmatched:
            console.print(f"[yellow]{len(unmatched)} projections had no matching game[/yellow]")
            for proj in unmatched[:8]:
                console.print(f"  [dim]{proj.away_team} @ {proj.home_team}[/dim]")


@main.command("import-ratings")
@click.argument("path", type=click.Path(exists=True))
@click.option("--source", required=True, help="Name for this source, e.g. massey.")
@click.option("--season", type=int)
@click.option("--week", type=int, default=0)
@click.option("--team-column", default="team")
@click.option("--rating-column", default="rating")
@click.pass_context
def import_ratings(ctx, path, source, season, week, team_column, rating_column) -> None:
    """Ingest ratings from a saved CSV, JSON or HTML file.

    Works with a saved masseyratings.com page, an exported spreadsheet,
    or anything else with a team column and a rating column.
    """
    from .providers.ratings import _read_local_rows
    from .models import Rating
    from .util.teams import canonical_team

    config = _config(ctx)
    season = _season(ctx, season)
    rows = _read_local_rows(Path(path))

    ratings, skipped = [], 0
    for row in rows:
        team_raw = row.get(team_column)
        rating_raw = row.get(rating_column)
        if team_raw is None or rating_raw in (None, ""):
            skipped += 1
            continue
        try:
            value = float(str(rating_raw).strip())
        except ValueError:
            skipped += 1
            continue
        team = canonical_team(str(team_raw))
        if team:
            ratings.append(Rating(source, season, week, team, value))

    if not ratings:
        console.print(
            f"[red]No ratings parsed.[/red] Columns found: "
            f"{', '.join(sorted(rows[0])) if rows else 'none'}"
        )
        sys.exit(1)

    with Pipeline(config) as pipeline:
        saved = pipeline.storage.upsert_ratings(ratings)
    console.print(f"[green]Saved[/green] {saved} '{source}' ratings for {season} week {week}")
    if skipped:
        console.print(f"[dim]Skipped {skipped} unparsable rows[/dim]")


@main.command()
@click.option("--season", type=int)
@click.pass_context
def status(ctx, season) -> None:
    """Show what data is currently stored."""
    config = _config(ctx)
    season = _season(ctx, season)
    storage = Storage(config.database_path)

    counts = storage.counts()
    console.print(f"[bold]{config.database_path}[/bold]")
    for name, count in counts.items():
        console.print(f"  {name:16} {count}")

    games = storage.games(season)
    if games:
        weeks = sorted({g.week for g in games})
        done = sum(1 for g in games if g.completed)
        console.print(
            f"\n[bold]{season}[/bold]: {len(games)} games across weeks "
            f"{weeks[0]}–{weeks[-1]} ({done} final)"
        )
        console.print(f"  Rating sources: {', '.join(storage.rating_sources(season)) or 'none'}")
    else:
        console.print(f"\n[yellow]No games stored for {season}.[/yellow]")
    storage.close()


@main.command()
@click.pass_context
def demo(ctx: click.Context) -> None:
    """Run the whole pipeline on bundled sample data — no API key needed."""
    from .report import render_console

    config = _config(ctx)
    config.database = str(config.path("demo.sqlite"))
    Path(config.database).unlink(missing_ok=True)

    with Pipeline(config) as pipeline:
        report = pipeline.fetch(2026, 3, providers=["fixtures"])
        console.print(
            f"[green]Loaded fixtures:[/green] {report.games} games · "
            f"{report.ratings} ratings · {report.quotes} quotes\n"
        )
        predictions = pipeline.predict(2026, 3, use_projections=False)
        recommendations = pipeline.picks(2026, 3, repredict=False)

    from .report import predictions_table
    console.print(predictions_table(predictions))
    console.print()
    render_console(recommendations, console)
    console.print("\n[dim]Sample data. Set CFBD_API_KEY and run `cfbpicks fetch` for real games.[/dim]")


def _parse_weeks(spec: str) -> list[int]:
    """Parse ``1-14`` or ``3,4,5`` into a list of week numbers."""
    weeks: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            weeks.extend(range(int(start), int(end) + 1))
        elif part:
            weeks.append(int(part))
    return sorted(set(weeks))


if __name__ == "__main__":  # pragma: no cover
    main()
