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
# The board is read on a phone as often as a laptop, so it is built as a
# responsive grid rather than a table: stacked cards on a narrow screen,
# aligned columns on a wide one. A table cannot do that -- an eleven
# column table on a 390px screen hides most of itself behind a sideways
# scroll, which is where every number that matters lives.
#
# Colour is assigned by the job it does. Tiers encode one ordered
# quantity (conviction), so they take a single-hue ordinal ramp,
# validated in both modes against the surfaces the page actually renders
# on, and inverted for dark so "strong" is always furthest from the
# ground. Every badge carries its text label, so tier never depends on
# colour alone.

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Archivo:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
)

_CSS = """
:root {
  color-scheme: light;
  --paper: #f6f7f9;
  --surface: #ffffff;
  --ink: #101418;
  --ink-2: #55606c;
  --muted: #6b7681;
  --rule: #e6e9ed;
  --border: rgba(16,20,24,0.10);
  --tier-lean: #86b6ef;
  --tier-play: #2a78d6;
  --tier-strong: #184f95;
  --tier-lean-ink: #101418;
  --tier-play-ink: #ffffff;
  --tier-strong-ink: #ffffff;
  --good: #006300;
  --bad: #c0392b;
  --accent: #2a78d6;
  --warn: #fab219;
  --warn-bg: #fdf7e7;
  --warn-edge: #f0dfae;
  --track: #eef1f4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper: #0e1114;
    --surface: #171b1f;
    --ink: #f2f5f7;
    --ink-2: #b7c0c9;
    --muted: #8a94a0;
    --rule: #262c32;
    --border: rgba(255,255,255,0.10);
    --tier-lean: #184f95;
    --tier-play: #2a78d6;
    --tier-strong: #86b6ef;
    --tier-lean-ink: #ffffff;
    --tier-play-ink: #ffffff;
    --tier-strong-ink: #101418;
    --good: #3fbf5f;
    --bad: #f0736a;
    --accent: #6da7ec;
    --warn: #fab219;
    --warn-bg: #2a2517;
    --warn-edge: #4a4126;
    --track: #222831;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper: #0e1114;
  --surface: #171b1f;
  --ink: #f2f5f7;
  --ink-2: #b7c0c9;
  --muted: #8a94a0;
  --rule: #262c32;
  --border: rgba(255,255,255,0.10);
  --tier-lean: #184f95;
  --tier-play: #2a78d6;
  --tier-strong: #86b6ef;
  --tier-lean-ink: #ffffff;
  --tier-play-ink: #ffffff;
  --tier-strong-ink: #101418;
  --good: #3fbf5f;
  --bad: #f0736a;
  --accent: #6da7ec;
  --warn: #fab219;
  --warn-bg: #2a2517;
  --warn-edge: #4a4126;
  --track: #222831;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 400 15px/1.5 "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 16px 72px; }
@media (min-width: 700px) { .wrap { padding: 36px 28px 80px; } }

h1 {
  font-family: Archivo, ui-sans-serif, system-ui, sans-serif;
  font-weight: 700; font-size: clamp(24px, 5vw, 34px);
  letter-spacing: -0.02em; margin: 0; text-wrap: balance;
}
.stamp { color: var(--muted); font-size: 13px; margin-top: 6px; }
header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
.toggle {
  flex: none; background: var(--surface); color: var(--ink-2);
  border: 1px solid var(--border); border-radius: 999px;
  padding: 7px 14px; font: inherit; font-size: 13px; cursor: pointer;
}
.toggle:hover { color: var(--ink); border-color: var(--accent); }
.toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* Week switcher. Only rendered when there is more than one week. */
.weeknav { display: flex; gap: 8px; flex-wrap: wrap; margin: 20px 0 0; }
.weeknav a {
  padding: 7px 14px; border-radius: 999px; text-decoration: none;
  font-size: 14px; font-weight: 500; color: var(--ink-2);
  background: var(--surface); border: 1px solid var(--border);
}
.weeknav a:hover { border-color: var(--accent); color: var(--ink); }
.weeknav a[aria-current="page"] {
  background: var(--accent); border-color: var(--accent); color: #fff;
}
.weeknav a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.tiles {
  display: grid; gap: 10px; margin: 22px 0 0;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
@media (min-width: 700px) { .tiles { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 13px 15px; }
.tile .label {
  color: var(--muted); font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.07em;
}
.tile .value {
  font-family: Archivo, ui-sans-serif, system-ui, sans-serif;
  font-weight: 600; font-size: 25px; letter-spacing: -0.02em;
  margin-top: 3px; font-variant-numeric: tabular-nums;
}
.tile .sub { color: var(--muted); font-size: 12px; margin-top: 1px; }

h2 {
  font-family: Archivo, ui-sans-serif, system-ui, sans-serif;
  font-size: 13px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.09em; color: var(--muted); margin: 34px 0 12px;
}

/* --- the board -------------------------------------------------------
   One grid per bet. Narrow: a card, with each figure labelled. Wide: the
   same cells snap into shared columns under a header row. */
.board { display: grid; gap: 10px; }
.board-head { display: none; }
.bet {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 14px 15px;
  display: grid; gap: 12px;
  /* Badge takes only its own width; the game text gets the rest. The
     other way round pushes the text to the right-hand edge. */
  grid-template-columns: auto minmax(0, 1fr);
  align-items: start;
}
.bet .game { min-width: 0; }
.matchup { color: var(--ink-2); font-size: 13px; }
.pick {
  font-family: Archivo, ui-sans-serif, system-ui, sans-serif;
  font-weight: 600; font-size: 18px; letter-spacing: -0.01em; margin-top: 2px;
}
.at { color: var(--muted); font-weight: 400; }
.terms { color: var(--ink-2); font-size: 13px; margin-top: 3px; font-variant-numeric: tabular-nums; }
.terms .book { color: var(--muted); }

.figs {
  grid-column: 1 / -1;
  /* Five figures across even on a 390px screen; auto-fit rather than a
     fixed count so a narrow phone reflows instead of orphaning one. */
  display: grid; grid-template-columns: repeat(auto-fit, minmax(56px, 1fr));
  gap: 8px 12px; padding-top: 11px; border-top: 1px solid var(--rule);
}
.fig { min-width: 0; }
.fig .lab {
  display: block; color: var(--muted); font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.07em;
}
.fig .val {
  display: block; font-variant-numeric: tabular-nums; font-size: 14px;
  margin-top: 2px; white-space: nowrap;
}
.fig.edge .val { font-weight: 600; }
.pos { color: var(--good); }
.neg { color: var(--bad); }

.badge {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; white-space: nowrap;
}
.badge.lean { background: var(--tier-lean); color: var(--tier-lean-ink); }
.badge.play { background: var(--tier-play); color: var(--tier-play-ink); }
.badge.strong { background: var(--tier-strong); color: var(--tier-strong-ink); }

.caveat { grid-column: 1 / -1; color: var(--muted); font-size: 12px; }

@media (min-width: 860px) {
  .board { gap: 8px; }
  .board-head, .bet {
    display: grid;
    grid-template-columns: 92px minmax(0, 1fr) 78px 78px 88px 66px 74px;
    gap: 14px; align-items: center;
  }
  .board-head {
    padding: 0 15px 6px; color: var(--muted); font-size: 11px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
  }
  .board-head span:not(:nth-child(2)) { text-align: right; }
  .board-head span:first-child { text-align: left; }
  .bet { padding: 12px 15px; }
  .figs {
    display: contents;            /* the figures become sibling columns */
  }
  .fig { text-align: right; }
  .fig .lab {                     /* the label lives in the header row now */
    position: absolute; width: 1px; height: 1px;
    overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap;
  }
  .fig .val { font-size: 14px; margin-top: 0; }
  .caveat { grid-column: 2 / -1; margin-top: -4px; }
}

.note-bar {
  display: flex; gap: 10px; align-items: flex-start;
  margin: 0 0 22px; padding: 13px 15px;
  background: var(--warn-bg); color: var(--ink);
  border: 1px solid var(--warn-edge); border-left: 4px solid var(--warn);
  border-radius: 0 12px 12px 0; font-size: 14px; line-height: 1.45;
}
.note-bar .mark { flex: none; font-size: 16px; }

.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }
.empty { padding: 26px; color: var(--ink-2); margin: 0; }
details { margin-top: 10px; }
summary {
  cursor: pointer; color: var(--ink-2); font-size: 14px; padding: 8px 0;
  font-weight: 500;
}
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 9px 12px; text-align: left; white-space: nowrap; }
th {
  color: var(--muted); font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid var(--rule);
}
td { border-bottom: 1px solid var(--rule); }
tr:last-child td { border-bottom: none; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.dim { color: var(--muted); }

footer {
  margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 13px;
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
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


@dataclass
class BoardEntry:
    """One published week, for the index and the week switcher."""

    season: int
    week: int
    filename: str
    bets: int
    staked: float
    generated: str


def _esc(value: object) -> str:
    return html_module.escape(str(value), quote=True)


def _tone(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else ""


def _head(title: str) -> list[str]:
    return [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(title)}</title>",
        _FONTS,
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">",
    ]


def _note_bar(note: Optional[str]) -> list[str]:
    if not note:
        return []
    return [
        '<p class="note-bar"><span class="mark" aria-hidden="true">\u26a0</span>'
        f"<span><strong>Heads up.</strong> {_esc(note)}</span></p>"
    ]


def _week_nav(nav: Sequence["BoardEntry"], current: Optional[int]) -> list[str]:
    if len(nav) < 2:
        return []
    out = ['<nav class="weeknav" aria-label="Weeks">']
    for entry in sorted(nav, key=lambda e: (e.season, e.week)):
        mark = ' aria-current="page"' if entry.week == current else ""
        out.append(
            f'<a href="{_esc(entry.filename)}"{mark}>Week {entry.week}</a>'
        )
    out.append("</nav>")
    return out


def _fig(label: str, value: str, tone: str = "") -> str:
    klass = f"fig {label.lower()}" if label.lower() == "edge" else "fig"
    return (
        f'<div class="{klass}"><span class="lab">{_esc(label)}</span>'
        f'<span class="val {tone}">{_esc(value)}</span></div>'
    )


def to_html(
    recs: Sequence[Recommendation],
    *,
    season: int,
    week: int,
    predictions: Optional[Sequence[Prediction]] = None,
    note: Optional[str] = None,
    nav: Sequence["BoardEntry"] = (),
) -> str:
    """The week's board as one self-contained page.

    Readable on a phone: each bet is a card whose figures are labelled,
    and only on a wide screen do those cards snap into shared columns.

    ``note`` renders a caveat above everything -- for marking sample
    data, a stale fetch, or a source that failed.
    """
    stamp = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    ordered = sorted(recs, key=lambda r: (TIER_ORDER.get(r.tier, 9), -r.prob_edge))

    staked = sum(r.stake_units for r in recs)
    expected = sum(r.expected_value * r.stake_units for r in recs)
    edges = sorted(r.prob_edge for r in recs)
    median_edge = edges[len(edges) // 2] if edges else 0.0

    parts = _head(f"CFB Picks \u2014 {season} Week {week}")
    parts += _note_bar(note)
    parts += [
        "<header><div>",
        "<h1>College Football Picks</h1>",
        f'<div class="stamp">{season} \u00b7 Week {week} \u00b7 updated {stamp}</div>',
        "</div>",
        '<button class="toggle" id="theme-toggle" type="button">Theme</button>',
        "</header>",
    ]
    parts += _week_nav(nav, week)

    parts += [
        '<div class="tiles">',
        _tile("Bets", str(len(recs)), f"{sum(1 for r in recs if r.tier == 'strong')} strong"),
        _tile("Staked", f"{staked:.2f}u", "fractional Kelly"),
        _tile("Expected", f"{expected:+.2f}u", "if the model is right"),
        _tile("Median edge", f"{median_edge * 100:.1f}%", "over the no-vig price"),
        "</div>",
    ]

    if not ordered:
        parts += [
            "<h2>Recommended bets</h2>",
            '<div class="panel"><p class="empty">No bets cleared the configured '
            "edge thresholds this week. That is a normal outcome, not a failure "
            "\u2014 the market is efficient most of the time.</p></div>",
        ]
    else:
        parts += [
            "<h2>Recommended bets</h2>",
            '<div class="board">',
            '<div class="board-head" aria-hidden="true">'
            "<span>Tier</span><span>Bet</span><span>Model</span><span>Market</span>"
            "<span>Edge</span><span>Pts</span><span>Units</span></div>",
        ]
        for rec in ordered:
            away, _, home = rec.matchup.partition(" @ ")
            matchup = (
                f"{_esc(away)} <span class=\"at\">@</span> {_esc(home)}"
                if home else _esc(rec.matchup)
            )
            price = f"{int(rec.price):+d}"
            parts.append(
                '<article class="bet">'
                f'<div><span class="badge {_esc(rec.tier)}">{_esc(rec.tier)}</span></div>'
                f'<div class="game"><div class="matchup">{matchup}</div>'
                f'<div class="pick">{_esc(rec.selection)}</div>'
                f'<div class="terms">{price} <span class="book">at {_esc(rec.book)}</span></div></div>'
                '<div class="figs">'
                + _fig("Model", f"{rec.model_prob * 100:.1f}%")
                + _fig("Market", f"{rec.market_prob * 100:.1f}%")
                + _fig("Edge", f"{rec.prob_edge * 100:+.1f}%", _tone(rec.prob_edge))
                + _fig("Pts", f"{rec.point_edge:+.1f}")
                + _fig("Units", f"{rec.stake_units:.2f}")
                + "</div>"
                + (f'<p class="caveat">{_esc("; ".join(rec.notes))}</p>' if rec.notes else "")
                + "</article>"
            )
        parts.append("</div>")

    if predictions:
        parts += [
            "<h2>All projections</h2>",
            f"<details><summary>{len(predictions)} games</summary>",
            '<div class="panel scroll"><table><thead><tr>',
            '<th>Matchup</th><th class="num">Margin</th><th class="num">Fair spread</th>',
            '<th class="num">Total</th><th class="num">Home win</th><th class="num">Conf</th>',
            "</tr></thead><tbody>",
        ]
        for pred in sorted(predictions, key=lambda p: -abs(p.projected_margin)):
            low = ' class="dim"' if pred.confidence < 0.3 else ""
            parts.append(
                f"<tr><td{low}>{_esc(pred.away_team)} @ {_esc(pred.home_team)}</td>"
                f'<td class="num">{pred.projected_margin:+.1f}</td>'
                f'<td class="num">{pred.fair_home_spread:+.1f}</td>'
                f'<td class="num">{pred.projected_total:.1f}</td>'
                f'<td class="num">{pred.home_win_prob * 100:.0f}%</td>'
                f'<td class="num">{pred.confidence:.2f}</td></tr>'
            )
        parts += ["</tbody></table></div></details>"]

    parts += [
        "<footer>Model output, not betting advice. Stakes are fractional-Kelly "
        "units; one unit is 1% of bankroll by default. Edges are measured against "
        "the de-vigged market price at the quoted number.</footer>",
        f"</div><script>{_TOGGLE_JS}</script></body></html>",
    ]
    return "\n".join(parts)


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="tile"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div>{sub_html}</div>'
    )


def to_index_html(entries: Sequence["BoardEntry"], *, note: Optional[str] = None) -> str:
    """A landing page listing every published board, newest first."""
    parts = _head("College Football Picks")
    parts += _note_bar(note)
    parts += [
        "<header><div><h1>College Football Picks</h1>",
        '<div class="stamp">Model output, not betting advice</div></div>',
        '<button class="toggle" id="theme-toggle" type="button">Theme</button></header>',
    ]

    if not entries:
        parts.append(
            '<div class="panel" style="margin-top:22px"><p class="empty">'
            "No boards published yet. Run <code>cfbpicks publish</code>."
            "</p></div>"
        )
    else:
        parts.append('<div class="board" style="margin-top:22px">')
        for entry in sorted(entries, key=lambda e: (e.season, e.week), reverse=True):
            parts.append(
                f'<a class="bet" href="{_esc(entry.filename)}" '
                'style="text-decoration:none;color:inherit">'
                f'<div class="game"><div class="pick">Week {entry.week}</div>'
                f'<div class="matchup">{_esc(entry.generated)}</div></div>'
                '<div class="figs">'
                + _fig("Bets", str(entry.bets))
                + _fig("Staked", f"{entry.staked:.2f}u")
                + "</div></a>"
            )
        parts.append("</div>")

    parts += [
        "<footer>Stakes are fractional-Kelly units; one unit is 1% of bankroll "
        "by default. Edges are measured against the de-vigged market price at "
        "the quoted number.</footer>",
        f"</div><script>{_TOGGLE_JS}</script></body></html>",
    ]
    return "\n".join(parts)


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
