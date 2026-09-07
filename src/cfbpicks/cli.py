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
@click.option("--weather/--no-weather", default=True)
@click.option("--odds/--no-odds", default=True)
@click.pass_context
def fetch(ctx, season, week, only, games, ratings, weather, odds) -> None:
    """Pull games, ratings, forecasts and betting lines into local storage."""
    config = _config(ctx)
    season = _season(ctx, season)
    want = [n for n, on in (("games", games), ("ratings", ratings),
                            ("weather", weather), ("odds", odds)) if on]

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
@click.option("--weeks", help="Week range, e.g. 1-14. Defaults to every week with games.")
@click.option("--no-carryover", is_flag=True, help="Ignore last season's ratings as a prior.")
@click.option("--top", type=int, default=15, help="How many teams to list.")
@click.pass_context
def rate(ctx, season, weeks, no_carryover, top) -> None:
    """Fit the engine's own power ratings from results.

    Unlike SP+ or Sagarin, these are computed per week from games played
    strictly beforehand, so they are safe to backtest with. Run this
    after `fetch` and before `predict`.
    """
    config = _config(ctx)
    season = _season(ctx, season)
    week_list = _parse_weeks(weeks) if weeks else None

    with Pipeline(config) as pipeline:
        with console.status(f"Fitting ratings for {season}…"):
            results = pipeline.compute_ratings(
                season, week_list, carry_prior=not no_carryover
            )

    if not results:
        console.print(
            f"[yellow]No games stored for {season}.[/yellow] "
            f"Run [bold]cfbpicks fetch --season {season}[/bold] first."
        )
        sys.exit(1)

    last_week = max(results)
    fit = results[last_week]
    console.print(
        f"[green]Fitted[/green] {len(results)} weekly snapshots · "
        f"latest is week {last_week}: {fit.summary()}"
    )

    if fit.residual_sd:
        console.print(
            f"  [dim]Fitted residual sd is {fit.residual_sd:.1f}; config uses "
            f"margin_sd {config.model.margin_sd}. Large gaps are worth "
            f"reconciling — that number drives every win probability.[/dim]"
        )

    table = Table(title=f"Power ratings — {season} week {last_week}", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Team")
    table.add_column("Rating", justify="right")
    table.add_column("Games", justify="right")
    for rank, (team, value) in enumerate(
        sorted(fit.ratings.items(), key=lambda kv: -kv[1])[:top], start=1
    ):
        table.add_row(str(rank), team, f"{value:+.2f}", str(fit.appearances.get(team, 0)))
    console.print(table)


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, required=True)
@click.pass_context
def weather(ctx, season, week) -> None:
    """Show kickoff forecasts and what they do to each total.

    Wind is the one that matters: a strong crosswind takes points off a
    total in a way no power rating can see. Domes are left alone.
    """
    from .weather import total_adjustment

    config = _config(ctx)
    season = _season(ctx, season)

    with Pipeline(config) as pipeline:
        games = {g.game_id: g for g in pipeline.storage.games(season, week)}
        forecasts = pipeline.storage.weather(list(games))

    if not forecasts:
        console.print(
            f"[yellow]No forecasts stored for {season} week {week}.[/yellow] "
            f"Run [bold]cfbpicks fetch --week {week}[/bold] — note that forecasts "
            "only reach about two weeks out."
        )
        return

    table = Table(title=f"Kickoff conditions — week {week}", header_style="bold")
    table.add_column("Matchup", overflow="fold")
    table.add_column("Conditions")
    table.add_column("Total", justify="right")

    rows = []
    for game_id, forecast in forecasts.items():
        game = games.get(game_id)
        adjustment, _ = total_adjustment(forecast, config.weather)
        rows.append((game.matchup if game else game_id, forecast, adjustment))

    for matchup, forecast, adjustment in sorted(rows, key=lambda r: r[2]):
        effect = f"{adjustment:+.1f}" if adjustment else "—"
        colour = "yellow" if adjustment <= -2 else ""
        table.add_row(
            matchup, forecast.describe(),
            f"[{colour}]{effect}[/{colour}]" if colour else effect,
        )
    console.print(table)

    moved = [r for r in rows if r[2]]
    console.print(
        f"\n{len(moved)} of {len(rows)} totals adjusted for weather."
        if moved else "\nNo forecast moves a total enough to matter this week."
    )


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, help="Show only one week.")
@click.pass_context
def overrides(ctx, season, week) -> None:
    """List and validate manual injury / situational adjustments.

    Ratings summarise games already played, so they cannot know a
    starting quarterback is out. This is where you tell the model.
    """
    from .overrides import POSITION_VALUES, OverrideError, load_overrides

    config = _config(ctx)
    season = _season(ctx, season)
    path = config.path(config.overrides_file)

    try:
        entries = load_overrides(path)
    except OverrideError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if not entries:
        console.print(f"[yellow]No overrides file at[/yellow] {path}")
        console.print(
            "\nCreate one to tell the model what the ratings cannot see:\n\n"
            "  [dim]- season: 2026\n"
            "    week: 3\n"
            "    team: Georgia\n"
            "    out: [qb1]\n"
            "    reason: starting QB out[/dim]\n\n"
            f"Positions: {', '.join(sorted(POSITION_VALUES))}"
        )
        return

    shown = [e for e in entries if e.season == season and (week is None or e.week == week)]
    table = Table(title=f"Overrides — {season}", header_style="bold")
    table.add_column("Week", justify="right")
    table.add_column("Scope")
    table.add_column("Effect", justify="right")
    table.add_column("Reason", overflow="fold")
    for entry in sorted(shown, key=lambda e: (e.week, e.team or e.game or "")):
        if entry.is_team_scoped:
            scope, effect = entry.team, f"{entry.team_points():+.1f}"
        else:
            scope = entry.game
            bits = []
            if entry.margin is not None:
                bits.append(f"margin {entry.margin:+.1f}")
            if entry.total is not None:
                bits.append(f"total {entry.total:+.1f}")
            effect = ", ".join(bits)
        table.add_row(str(entry.week), scope, effect, entry.reason or "—")
    console.print(table)
    console.print(f"[green]{len(entries)} entries parsed cleanly[/green] from {path}")

    # An entry that matches no scheduled game does nothing at all, which
    # is the failure mode worth catching before kickoff rather than after.
    if week is not None:
        with Pipeline(config) as pipeline:
            _, warnings = pipeline.load_adjustments(season, week)
        for warning in warnings:
            console.print(f"  [yellow]![/yellow] {warning}")


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

    for warning in getattr(pipeline, "override_warnings", []) or []:
        console.print(f"[yellow]Override ignored:[/yellow] {warning}")

    if not any("cfbpicks_margin" in p.rating_sources for p in predictions):
        console.print(
            "\n[yellow]The engine's own ratings aren't in this blend.[/yellow] "
            f"Run [bold]cfbpicks rate --season {season}[/bold] to fit them — they're "
            "the only rating source here that can also be backtested honestly."
        )

    thin = [p for p in predictions if p.confidence < 0.3]
    if thin:
        console.print(
            f"\n[yellow]{len(thin)} game(s) have thin rating coverage[/yellow] "
            "(likely FCS opponents or unmatched team names)."
        )


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, required=True)
@click.option("--push", is_flag=True, help="Commit and push, so GitHub Pages picks it up.")
@click.pass_context
def publish(ctx, season, week, push) -> None:
    """Write the week's board into docs/ for GitHub Pages.

    Produces one page per week plus an index listing them, so the board
    is readable from any browser instead of only from the machine that
    generated it.

    The site is served from whatever the repository's visibility allows.
    On a public repo that means the picks are public too.
    """
    import json
    import subprocess

    from .report import BoardEntry, to_html, to_index_html

    config = _config(ctx)
    season = _season(ctx, season)
    docs = config.path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    # Tell GitHub Pages not to run the files through Jekyll.
    (docs / ".nojekyll").write_text("")

    with Pipeline(config) as pipeline:
        recommendations = pipeline.picks(season, week)
        predictions = pipeline.storage.predictions(season, week)

    if not predictions:
        console.print(
            f"[yellow]No games stored for {season} week {week}.[/yellow] "
            f"Run [bold]cfbpicks fetch --week {week}[/bold] first."
        )
        sys.exit(1)

    filename = f"{season}-week-{week:02d}.html"
    (docs / filename).write_text(
        to_html(recommendations, season=season, week=week, predictions=predictions)
    )

    # A small manifest keeps the index accurate without re-opening every
    # page to work out what is in it.
    manifest_path = docs / "boards.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    manifest[filename] = {
        "season": season, "week": week, "filename": filename,
        "bets": len(recommendations),
        "staked": round(sum(r.stake_units for r in recommendations), 2),
        "generated": generated,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    entries = [BoardEntry(**row) for row in manifest.values()]
    (docs / "index.html").write_text(to_index_html(entries))

    console.print(
        f"[green]Wrote[/green] docs/{filename} and docs/index.html "
        f"({len(recommendations)} bets)"
    )

    if not push:
        console.print(
            "\nTo put it online:\n"
            "  [dim]git add docs && git commit -m 'Publish week "
            f"{week} board' && git push[/dim]\n"
            "  Then enable Pages once: Settings -> Pages -> Source: "
            "your branch, folder /docs"
        )
        return

    try:
        subprocess.run(["git", "add", "docs"], cwd=config.root, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Publish {season} week {week} board"],
            cwd=config.root, check=True, capture_output=True,
        )
        subprocess.run(["git", "push"], cwd=config.root, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]git failed:[/red] {exc}")
        sys.exit(1)
    console.print("[green]Pushed.[/green] GitHub Pages will update in a minute or two.")


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, required=True)
@click.option("--market", "markets", multiple=True,
              type=click.Choice(["spread", "total", "moneyline"]),
              help="Restrict to specific markets.")
@click.option("--min-edge", type=float, help="Minimum probability edge, e.g. 0.03 for 3%.")
@click.option("--max-units", type=float, help="Cap the stake on any single bet.")
@click.option("--format", "fmt",
              type=click.Choice(["table", "markdown", "html", "csv", "json"]),
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

    # Writing to a .html path implies the html renderer, so `-o board.html`
    # does the obvious thing without also passing --format.
    if output and fmt == "table" and str(output).lower().endswith((".html", ".htm")):
        fmt = "html"

    text = render(
        recommendations, "markdown" if fmt == "table" else fmt,
        season=season, week=week, predictions=predictions,
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        console.print(f"[green]Wrote[/green] {len(recommendations)} picks to {path}")
        if fmt == "html":
            console.print(f"  [dim]open it with:[/dim] open {path}")
    else:
        click.echo(text)


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, help="Restrict to one week.")
@click.option("--quiet", is_flag=True, help="Only print when something moved. Good for cron.")
@click.pass_context
def snapshot(ctx, season, week, quiet) -> None:
    """Capture current odds again, and report what moved.

    Closing line value needs a line recorded later than the bet. One
    capture a week gives you none, so run this on a schedule — midweek
    and again near kickoff is enough to make CLV a real measurement:

        0 12 * * 3,6  cd /path/to/repo && .venv/bin/cfbpicks snapshot --quiet
    """
    config = _config(ctx)
    season = _season(ctx, season)

    with Pipeline(config) as pipeline:
        report, moves = pipeline.snapshot_odds(season, week)

    if quiet and not moves:
        return

    console.print(f"[green]Captured[/green] {report.quotes} quotes")
    if not moves:
        console.print("[dim]No line movement since the last capture.[/dim]")
        return

    table = Table(title=f"Movement since last capture ({len(moves)} lines)", header_style="bold")
    table.add_column("Game", overflow="fold")
    table.add_column("Market")
    table.add_column("Moved", justify="right")
    for key, delta in sorted(moves.items(), key=lambda kv: -abs(kv[1]))[:25]:
        game_id, market = key.rsplit("|", 1)
        table.add_row(game_id, market, f"{delta:+.1f}")
    console.print(table)


@main.command()
@click.option("--season", type=int)
@click.option("--week", type=int, required=True)
@click.option("--market", default="spread", type=click.Choice(["spread", "total"]))
@click.pass_context
def lines(ctx, season, week, market) -> None:
    """Show how each line has moved across captures this week."""
    config = _config(ctx)
    season = _season(ctx, season)
    side = "home" if market == "spread" else "over"

    with Pipeline(config) as pipeline:
        games = pipeline.storage.games(season, week)
        rows = []
        for game in games:
            series = pipeline.storage.line_history(game.game_id, market, side)
            if len(series) < 2:
                continue
            opened, closed = series[0][1], series[-1][1]
            rows.append((game, opened, closed, closed - opened, len(series)))

    if not rows:
        console.print(
            f"[yellow]Nothing to compare for {season} week {week}.[/yellow] "
            "Line movement needs at least two captures — run "
            "[bold]cfbpicks snapshot[/bold] again later in the week."
        )
        return

    table = Table(title=f"{market.title()} movement — week {week}", header_style="bold")
    table.add_column("Matchup", overflow="fold")
    table.add_column("Open", justify="right")
    table.add_column("Now", justify="right")
    table.add_column("Move", justify="right")
    table.add_column("Caps", justify="right")
    for game, opened, closed, delta, count in sorted(rows, key=lambda r: -abs(r[3])):
        colour = "green" if delta > 0 else "red" if delta < 0 else ""
        move = f"[{colour}]{delta:+.1f}[/{colour}]" if colour else f"{delta:+.1f}"
        table.add_row(game.matchup, f"{opened:+g}", f"{closed:+g}", move, str(count))
    console.print(table)


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

    settled = results.get("win", 0) + results.get("loss", 0) + results.get("push", 0)
    with_clv = results.get("with_clv", 0)
    if settled and not with_clv:
        console.print(
            "\n[yellow]No closing line value recorded.[/yellow] Odds were never "
            "captured after these bets were priced, so there is no closing line "
            "to compare against. Run [bold]cfbpicks snapshot[/bold] later in the "
            "week — CLV is the earliest honest signal that an edge is real."
        )
    elif with_clv:
        console.print(f"  [dim]CLV recorded for {with_clv} of {settled} settled bets[/dim]")


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
