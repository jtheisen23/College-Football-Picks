"""Rendering: terminal tables, Markdown card sheets, CSV and JSON."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from rich.console import Console
from rich.table import Table

from .models import Prediction, Recommendation

TIER_STYLE = {"strong": "bold green", "play": "green", "lean": "yellow", "pass": "dim"}
TIER_ORDER = {"strong": 0, "play": 1, "lean": 2, "pass": 3}


def _fmt_price(price: float) -> str:
    return f"{int(price):+d}"


def _fmt_line(rec: Recommendation) -> str:
    if rec.line is None:
        return "ML"
    return f"{rec.line:+g}" if rec.market == "spread" else f"{rec.line:g}"


def _abbrev(name: str, width: int = 11) -> str:
    """Shorten a school name so a full slate fits an 80-column terminal."""
    if len(name) <= width:
        return name
    words = name.split()
    if len(words) > 1:
        short = " ".join(w[0] if len(w) > 3 else w for w in words[:-1]) + " " + words[-1]
        if len(short) <= width:
            return short
    return name[: width - 1] + "\u2026"


def picks_table(recs: Sequence[Recommendation], *, title: str = "Recommended bets") -> Table:
    """Compact terminal view. The full detail lives in the markdown/CSV output."""
    table = Table(title=title, header_style="bold", expand=False)
    table.add_column("Tier", no_wrap=True)
    table.add_column("Matchup", max_width=24, overflow="ellipsis")
    table.add_column("Bet", max_width=20, overflow="ellipsis")
    table.add_column("Price", justify="right", no_wrap=True)
    table.add_column("Book", max_width=10, overflow="ellipsis")
    table.add_column("Edge", justify="right", no_wrap=True)
    table.add_column("Pts", justify="right", no_wrap=True)
    table.add_column("Units", justify="right", no_wrap=True)

    for rec in sorted(recs, key=lambda r: (TIER_ORDER.get(r.tier, 9), -r.prob_edge)):
        style = TIER_STYLE.get(rec.tier, "")
        matchup = f"{_abbrev(rec.matchup.split(' @ ')[0].split(' vs ')[0])} @ " \
                  f"{_abbrev(rec.matchup.split(' @ ')[-1].split(' vs ')[-1])}" \
            if (" @ " in rec.matchup or " vs " in rec.matchup) else rec.matchup
        table.add_row(
            f"[{style}]{rec.tier}[/{style}]" if style else rec.tier,
            matchup,
            rec.selection,
            _fmt_price(rec.price),
            _abbrev(rec.book, 10),
            f"{rec.prob_edge * 100:+.1f}%",
            f"{rec.point_edge:+.1f}",
            f"{rec.stake_units:.2f}",
        )
    return table


def predictions_table(preds: Sequence[Prediction], limit: Optional[int] = None) -> Table:
    table = Table(title="Model projections", header_style="bold")
    table.add_column("Matchup", overflow="fold")
    table.add_column("Proj margin", justify="right")
    table.add_column("Fair spread", justify="right")
    table.add_column("Proj total", justify="right")
    table.add_column("Home win", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Sources", justify="right")

    ordered = sorted(preds, key=lambda p: abs(p.projected_margin), reverse=True)
    for pred in ordered[:limit] if limit else ordered:
        table.add_row(
            f"{pred.away_team} @ {pred.home_team}",
            f"{pred.projected_margin:+.1f}",
            f"{pred.fair_home_spread:+.1f}",
            f"{pred.projected_total:.1f}",
            f"{pred.home_win_prob * 100:.0f}%",
            f"{pred.confidence:.2f}",
            str(len(pred.rating_sources)),
        )
    return table


def render_console(recs: Sequence[Recommendation], console: Optional[Console] = None) -> None:
    console = console or Console()
    if not recs:
        console.print("[yellow]No bets clear the configured thresholds.[/yellow]")
        return
    console.print(picks_table(recs))
    total_units = sum(r.stake_units for r in recs)
    ev_units = sum(r.expected_value * r.stake_units for r in recs)
    console.print(
        f"\n[bold]{len(recs)} bets[/bold] · {total_units:.2f}u staked · "
        f"expected {ev_units:+.2f}u"
    )


def to_markdown(
    recs: Sequence[Recommendation],
    *,
    season: int,
    week: int,
    predictions: Optional[Sequence[Prediction]] = None,
) -> str:
    """A shareable card sheet for the week."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = [
        f"# College Football Picks — {season} Week {week}",
        "",
        f"_Generated {stamp}_",
        "",
    ]

    if not recs:
        out += ["No bets cleared the configured edge thresholds this week.", ""]
    else:
        staked = sum(r.stake_units for r in recs)
        ev = sum(r.expected_value * r.stake_units for r in recs)
        out += [
            f"**{len(recs)} bets · {staked:.2f} units staked · {ev:+.2f} units expected**",
            "",
            "| Tier | Matchup | Bet | Price | Book | Model | Market | Edge | Pts | Units |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
        for rec in sorted(recs, key=lambda r: (TIER_ORDER.get(r.tier, 9), -r.prob_edge)):
            out.append(
                f"| {rec.tier} | {rec.matchup} | {rec.selection} | {_fmt_price(rec.price)} "
                f"| {rec.book} | {rec.model_prob * 100:.1f}% | {rec.market_prob * 100:.1f}% "
                f"| {rec.prob_edge * 100:+.1f}% | {rec.point_edge:+.1f} | {rec.stake_units:.2f} |"
            )
        out.append("")

        notes = [(r, r.notes) for r in recs if r.notes]
        if notes:
            out += ["### Caveats", ""]
            for rec, rec_notes in notes:
                out.append(f"- **{rec.selection}** — {'; '.join(rec_notes)}")
            out.append("")

    if predictions:
        out += [
            "### Full projections",
            "",
            "| Matchup | Proj margin | Fair spread | Proj total | Home win | Confidence |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for pred in sorted(predictions, key=lambda p: p.game_id):
            out.append(
                f"| {pred.away_team} @ {pred.home_team} | {pred.projected_margin:+.1f} "
                f"| {pred.fair_home_spread:+.1f} | {pred.projected_total:.1f} "
                f"| {pred.home_win_prob * 100:.0f}% | {pred.confidence:.2f} |"
            )
        out.append("")

    out += [
        "---",
        "",
        "Model output, not betting advice. Stakes are fractional-Kelly units; "
        "one unit is 1% of bankroll by default.",
    ]
    return "\n".join(out)


def to_csv(recs: Sequence[Recommendation]) -> str:
    buffer = io.StringIO()
    fields = [
        "season", "week", "game_id", "matchup", "market", "side", "selection",
        "line", "price", "book", "model_prob", "market_prob", "prob_edge",
        "point_edge", "expected_value", "stake_units", "confidence", "tier",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for rec in recs:
        writer.writerow(rec.as_dict())
    return buffer.getvalue()


def to_json(recs: Sequence[Recommendation]) -> str:
    return json.dumps([r.as_dict() for r in recs], indent=2)


def render(recs: Sequence[Recommendation], fmt: str, *, season: int, week: int,
           predictions: Optional[Sequence[Prediction]] = None) -> str:
    if fmt == "markdown":
        return to_markdown(recs, season=season, week=week, predictions=predictions)
    if fmt == "csv":
        return to_csv(recs)
    if fmt == "json":
        return to_json(recs)
    raise ValueError(f"Unknown output format: {fmt}")
