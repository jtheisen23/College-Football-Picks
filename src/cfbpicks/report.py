"""Rendering: terminal tables, Markdown card sheets, CSV and JSON."""

from __future__ import annotations

import csv
import html as html_module
import io
import json
from dataclasses import dataclass
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




# --- HTML -------------------------------------------------------------------
#
# Colours come from a validated palette rather than taste. Tiers are an
# ordinal ramp -- one hue, monotone lightness -- because they encode a
# single ordered quantity (conviction), and the ramp inverts in dark mode
# so "strong" is always the step furthest from the surface. Every badge
# carries its text label too, so tier is never communicated by colour
# alone. Verified with the palette validator in both modes.

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --rule: #e1e0d9;
  --border: rgba(11,11,11,0.10);
  --tier-lean: #86b6ef;
  --tier-play: #2a78d6;
  --tier-strong: #184f95;
  --good: #006300;
  --bad: #d03b3b;
  --bar: #86b6ef;
  --warn: #fab219;
  --warn-bg: #fdf7e7;
  --warn-edge: #f0dfae;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --rule: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --tier-lean: #184f95;
    --tier-play: #2a78d6;
    --tier-strong: #86b6ef;
    --good: #0ca30c;
    --bad: #d03b3b;
    --bar: #184f95;
    --warn: #fab219;
    --warn-bg: #2a2517;
    --warn-edge: #4a4126;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --rule: #2c2c2a;
  --border: rgba(255,255,255,0.10);
  --tier-lean: #184f95;
  --tier-play: #2a78d6;
  --tier-strong: #86b6ef;
  --good: #0ca30c;
  --bad: #d03b3b;
  --bar: #184f95;
  --warn: #fab219;
  --warn-bg: #2a2517;
  --warn-edge: #4a4126;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1220px; margin: 0 auto; padding: 32px 20px 64px; }

header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
.stamp { color: var(--muted); font-size: 13px; }
.toggle {
  background: var(--surface); color: var(--ink-2); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer;
}
.toggle:hover { color: var(--ink); }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 24px 0 8px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.tile .label { color: var(--ink-2); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.tile .value { font-size: 26px; margin-top: 4px; letter-spacing: -0.02em; }
.tile .note { color: var(--muted); font-size: 12px; margin-top: 2px; }

h2 { font-size: 15px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-2); margin: 32px 0 10px; font-weight: 600; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 9px 10px; text-align: left; white-space: nowrap; }
th { color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 1px solid var(--rule); }
td { border-bottom: 1px solid var(--rule); }
tr:last-child td { border-bottom: none; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.matchup { white-space: normal; min-width: 180px; }
.bet { font-weight: 600; }
.good { color: var(--good); }
.bad { color: var(--bad); }
.dim { color: var(--muted); }

.badge {
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 12px; font-weight: 600; color: #fff;
}
.badge.lean { background: var(--tier-lean); color: #0b0b0b; }
.badge.play { background: var(--tier-play); }
.badge.strong { background: var(--tier-strong); }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .badge.lean { color: #fff; }
  :root:not([data-theme="light"]) .badge.strong { color: #0b0b0b; }
}
:root[data-theme="dark"] .badge.lean { color: #fff; }
:root[data-theme="dark"] .badge.strong { color: #0b0b0b; }

/* Edge magnitude, as a scan aid beside the number -- never instead of it. */
.bar { display: block; height: 6px; border-radius: 0 3px 3px 0; background: var(--bar); min-width: 2px; }
.barcell { width: 62px; padding-left: 0; }

details { margin-top: 8px; }
summary { cursor: pointer; color: var(--ink-2); font-size: 14px; padding: 6px 0; }
footer { margin-top: 40px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--rule); padding-top: 16px; }
.empty { padding: 28px; color: var(--ink-2); }

/* A standing caveat about the page as a whole -- sample data, a stale
   fetch, a source that failed. It qualifies everything below, so it sits
   above the title rather than among the content, and takes the warning
   status colour rather than the accent.

   This is deliberately loud. The demo fixtures invent matchups between
   real schools, so a board of sample picks looks exactly like a board of
   real ones. A caveat that can be skimmed past is not doing its job. */
.note-bar {
  display: flex; gap: 10px; align-items: flex-start;
  margin: 0 0 24px; padding: 14px 16px;
  background: var(--warn-bg); color: var(--ink);
  border: 1px solid var(--warn-edge); border-left: 4px solid var(--warn);
  border-radius: 0 10px 10px 0; font-size: 14px; line-height: 1.45;
}
.note-bar .mark { flex: none; font-size: 16px; line-height: 1.3; }
.note-bar strong { letter-spacing: 0.01em; }
"""

_TOGGLE_JS = """
(function () {
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem('cfbpicks-theme');
    if (saved) root.setAttribute('data-theme', saved);
  } catch (e) {}
  btn.addEventListener('click', function () {
    var dark = getComputedStyle(root).colorScheme.indexOf('dark') !== -1;
    var next = dark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('cfbpicks-theme', next); } catch (e) {}
  });
})();
"""


def _esc(value: object) -> str:
    return html_module.escape(str(value), quote=True)


def _signed(value: float, digits: int = 1, suffix: str = "") -> str:
    return f"{value:+.{digits}f}{suffix}"


def _tone(value: float) -> str:
    return "good" if value > 0 else "bad" if value < 0 else "dim"


def to_html(
    recs: Sequence[Recommendation],
    *,
    season: int,
    week: int,
    predictions: Optional[Sequence[Prediction]] = None,
    note: Optional[str] = None,
) -> str:
    """A self-contained page for the week's board.

    No external requests: it is a single file that opens from disk, works
    offline, and follows the reader's light/dark preference.

    ``note`` renders a caveat above the board — for marking sample data,
    a stale fetch, or a source that failed. Anything that qualifies every
    number on the page rather than one row of it.
    """
    stamp = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    ordered = sorted(recs, key=lambda r: (TIER_ORDER.get(r.tier, 9), -r.prob_edge))

    staked = sum(r.stake_units for r in recs)
    expected = sum(r.expected_value * r.stake_units for r in recs)
    edges = sorted(r.prob_edge for r in recs)
    median_edge = edges[len(edges) // 2] if edges else 0.0
    widest = max((abs(r.prob_edge) for r in recs), default=1.0) or 1.0

    parts: list[str] = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>CFB Picks — {season} Week {week}</title>",
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">",
    ]
    if note:
        parts.append(
            "<p class=\"note-bar\"><span class=\"mark\" aria-hidden=\"true\">\u26a0</span>"
            f"<span><strong>Heads up.</strong> {_esc(note)}</span></p>"
        )
    parts += [
        "<header><div>",
        f"<h1>College Football Picks</h1>",
        f"<div class=\"stamp\">{season} · Week {week} · generated {stamp}</div>",
        "</div>",
        "<button class=\"toggle\" id=\"theme-toggle\">Theme</button>",
        "</header>",
    ]

    parts += [
        "<div class=\"tiles\">",
        _tile("Bets", str(len(recs)), f"{sum(1 for r in recs if r.tier == 'strong')} strong"),
        _tile("Staked", f"{staked:.2f}u", "fractional Kelly"),
        _tile("Expected", _signed(expected, 2, "u"), "if the model is right"),
        _tile("Median edge", f"{median_edge * 100:.1f}%", "over the no-vig price"),
        "</div>",
    ]

    if not ordered:
        parts.append(
            "<div class=\"card\"><p class=\"empty\">No bets cleared the configured "
            "edge thresholds this week. That is a normal outcome, not a failure — "
            "the market is efficient most of the time.</p></div>"
        )
    else:
        parts += ["<h2>Recommended bets</h2>", "<div class=\"card\"><table><thead><tr>",
                  "<th>Tier</th><th>Matchup</th><th>Bet</th>",
                  "<th class=\"num\">Price</th><th>Book</th>",
                  "<th class=\"num\">Model</th><th class=\"num\">Market</th>",
                  "<th class=\"num\">Edge</th><th></th>",
                  "<th class=\"num\">Pts</th><th class=\"num\">Units</th>",
                  "</tr></thead><tbody>"]
        for rec in ordered:
            width = max(2, round(abs(rec.prob_edge) / widest * 56))
            note = "; ".join(rec.notes)
            parts.append(
                "<tr>"
                f"<td><span class=\"badge {_esc(rec.tier)}\">{_esc(rec.tier)}</span></td>"
                f"<td class=\"matchup\">{_esc(rec.matchup)}</td>"
                f"<td class=\"bet\">{_esc(rec.selection)}</td>"
                f"<td class=\"num\">{_fmt_price(rec.price)}</td>"
                f"<td>{_esc(rec.book)}</td>"
                f"<td class=\"num\">{rec.model_prob * 100:.1f}%</td>"
                f"<td class=\"num\">{rec.market_prob * 100:.1f}%</td>"
                f"<td class=\"num\">{rec.prob_edge * 100:+.1f}%</td>"
                f"<td class=\"barcell\"><span class=\"bar\" style=\"width:{width}px\"></span></td>"
                f"<td class=\"num\">{rec.point_edge:+.1f}</td>"
                f"<td class=\"num\" title=\"{_esc(note)}\">{rec.stake_units:.2f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")

        caveats = [r for r in ordered if r.notes]
        if caveats:
            parts.append("<details><summary>Caveats "
                         f"({len(caveats)})</summary><div class=\"card\"><table><tbody>")
            for rec in caveats:
                parts.append(
                    f"<tr><td class=\"bet\">{_esc(rec.selection)}</td>"
                    f"<td class=\"dim\" style=\"white-space:normal\">{_esc('; '.join(rec.notes))}</td></tr>"
                )
            parts.append("</tbody></table></div></details>")

    if predictions:
        parts += ["<h2>All projections</h2>",
                  "<details><summary>"
                  f"{len(predictions)} games</summary><div class=\"card\"><table><thead><tr>",
                  "<th>Matchup</th><th class=\"num\">Proj margin</th>",
                  "<th class=\"num\">Fair spread</th><th class=\"num\">Proj total</th>",
                  "<th class=\"num\">Home win</th><th class=\"num\">Confidence</th>",
                  "</tr></thead><tbody>"]
        for pred in sorted(predictions, key=lambda p: -abs(p.projected_margin)):
            low = " dim" if pred.confidence < 0.3 else ""
            parts.append(
                f"<tr><td class=\"matchup{low}\">{_esc(pred.away_team)} @ {_esc(pred.home_team)}</td>"
                f"<td class=\"num\">{pred.projected_margin:+.1f}</td>"
                f"<td class=\"num\">{pred.fair_home_spread:+.1f}</td>"
                f"<td class=\"num\">{pred.projected_total:.1f}</td>"
                f"<td class=\"num\">{pred.home_win_prob * 100:.0f}%</td>"
                f"<td class=\"num\">{pred.confidence:.2f}</td></tr>"
            )
        parts.append("</tbody></table></div></details>")

    parts += [
        "<footer>Model output, not betting advice. Stakes are fractional-Kelly "
        "units; one unit is 1% of bankroll by default. Edges are measured against "
        "the de-vigged market price at the quoted number.</footer>",
        f"</div><script>{_TOGGLE_JS}</script></body></html>",
    ]
    return "\n".join(parts)


def _tile(label: str, value: str, note: str = "") -> str:
    note_html = f"<div class=\"note\">{_esc(note)}</div>" if note else ""
    return (
        f"<div class=\"tile\"><div class=\"label\">{_esc(label)}</div>"
        f"<div class=\"value\">{_esc(value)}</div>{note_html}</div>"
    )




@dataclass
class BoardEntry:
    """One published week, for the index page."""

    season: int
    week: int
    filename: str
    bets: int
    staked: float
    generated: str


def to_index_html(entries: Sequence["BoardEntry"], *, note: Optional[str] = None) -> str:
    """A landing page listing every published board, newest first."""
    parts: list[str] = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>College Football Picks</title>",
        f"<style>{_CSS}{_INDEX_CSS}</style></head><body><div class=\"wrap\">",
    ]
    if note:
        parts.append(
            "<p class=\"note-bar\"><span class=\"mark\" aria-hidden=\"true\">\u26a0</span>"
            f"<span><strong>Heads up.</strong> {_esc(note)}</span></p>"
        )
    parts += [
        "<header><div><h1>College Football Picks</h1>",
        "<div class=\"stamp\">Model output, not betting advice</div></div>",
        "<button class=\"toggle\" id=\"theme-toggle\">Theme</button></header>",
    ]

    if not entries:
        parts.append(
            "<div class=\"card\"><p class=\"empty\">No boards published yet. "
            "Run <code>cfbpicks publish --week N</code>.</p></div>"
        )
    else:
        parts.append("<ul class=\"weeks\">")
        for entry in sorted(entries, key=lambda e: (e.season, e.week), reverse=True):
            parts.append(
                f"<li><a href=\"{_esc(entry.filename)}\">"
                f"<span class=\"wk\">{entry.season} · Week {entry.week}</span>"
                f"<span class=\"meta\">{entry.bets} bets · {entry.staked:.2f}u staked</span>"
                f"<span class=\"when\">{_esc(entry.generated)}</span></a></li>"
            )
        parts.append("</ul>")

    parts += [
        "<footer>Stakes are fractional-Kelly units; one unit is 1% of bankroll "
        "by default. Edges are measured against the de-vigged market price at the "
        "quoted number.</footer>",
        f"</div><script>{_TOGGLE_JS}</script></body></html>",
    ]
    return "\n".join(parts)


_INDEX_CSS = """
.weeks { list-style: none; margin: 24px 0 0; padding: 0; display: grid; gap: 10px; }
.weeks a {
  display: grid; grid-template-columns: 1fr auto; gap: 4px 16px; align-items: baseline;
  padding: 16px 18px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; text-decoration: none; color: var(--ink);
}
.weeks a:hover { border-color: var(--tier-play); }
.weeks a:focus-visible { outline: 2px solid var(--tier-play); outline-offset: 2px; }
.weeks .wk { font-size: 17px; font-weight: 600; }
.weeks .meta { color: var(--ink-2); font-size: 14px; text-align: right; }
.weeks .when { grid-column: 1 / -1; color: var(--muted); font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
"""


def render(recs: Sequence[Recommendation], fmt: str, *, season: int, week: int,
           predictions: Optional[Sequence[Prediction]] = None) -> str:
    if fmt == "markdown":
        return to_markdown(recs, season=season, week=week, predictions=predictions)
    if fmt == "html":
        return to_html(recs, season=season, week=week, predictions=predictions)
    if fmt == "csv":
        return to_csv(recs)
    if fmt == "json":
        return to_json(recs)
    raise ValueError(f"Unknown output format: {fmt}")
