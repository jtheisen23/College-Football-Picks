# College Football Picks

A college football predictions engine. It pulls in games, power ratings
and sportsbook lines each week, projects every game, compares the
projection against what the market is actually offering, and recommends
the bets where the two disagree by enough to be worth a stake.

```
$ cfbpicks fetch --week 3
Fetched 62 games · 786 ratings · 2,232 quotes

$ cfbpicks picks --week 3
                            Recommended bets
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━┓
┃ Tier   ┃ Matchup             ┃ Bet        ┃ Price ┃ Book     ┃ Edge   ┃ Pts  ┃ Units ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━┩
│ strong │ Mississippi @ Texas │ Texas -5.5 │  -108 │ BetMGM   │ +11.3% │ +4.6 │  3.00 │
│ play   │ Iowa @ Penn State   │ Under 60   │  -110 │ FanDuel  │  +4.2% │ +2.8 │  1.85 │
│ lean   │ Duke @ Clemson      │ Duke +14   │  -110 │ DraftKings│ +2.4% │ +1.9 │  0.42 │
└────────┴─────────────────────┴────────────┴───────┴──────────┴────────┴──────┴───────┘
```

Try it with no API key at all:

```bash
pip install -e .
cfbpicks demo
```

## What it actually does

Most "model vs. line" tools compare a projection against the posted
number and call the difference an edge. That is wrong in three ways this
engine tries to get right.

**1. The posted number is not the market's opinion.** A book showing
`-3` at `-125/+105` is not offering a 3-point line; after removing the
vig it is really saying 3.4. Every book is de-vigged and converted into
the margin it implies, and the model is compared against *that*:

```
P(cover) = Φ((M + L) / σ)   ⟹   M = σ · Φ⁻¹(p) − L
```

**2. One rating system is a guess; five that disagree is information.**
SP+, SRS, Elo, Massey and Sagarin are rescaled onto a common points
scale and blended. How much they *disagree* is carried through as
uncertainty, and the model is shrunk toward the market in proportion to
it — so a game the systems can't agree on gets a smaller bet, or none.

**3. The price you get is not the consensus price.** Recommendations are
priced at the best available number and juice across your books, because
taking `-3 -105` instead of `-3.5 -115` is worth more over a season than
most model improvements.

Stakes are fractional-Kelly (quarter-Kelly by default), capped, and
adjusted for the real chance of a push on whole numbers.

## Setup

Needs Python 3.9 or newer — including the 3.9 that ships with macOS, so
there is nothing to install first.

```bash
git clone https://github.com/jtheisen23/College-Football-Picks
cd College-Football-Picks
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip     # editable installs need pip 21.3+
pip install -e .

export CFBD_API_KEY=...      # free: https://collegefootballdata.com/key
export ODDS_API_KEY=...      # optional: https://the-odds-api.com

cfbpicks init
cfbpicks providers           # check what's wired up
```

`cfbpicks providers` tells you which sources are ready and which are
missing a key.

## A normal week

```bash
cfbpicks fetch --week 3                       # games, ratings, lines
cfbpicks predict --week 3                     # projections, before prices
cfbpicks picks --week 3                       # the bets worth making
cfbpicks picks --week 3 -o reports/week3.md --format markdown

# after the games finish
cfbpicks grade --week 3
```

Everything is cached in SQLite, so re-running `picks` costs no API calls.

## Data sources

| Source | Supplies | Needs |
|---|---|---|
| **CollegeFootballData** | Schedules, results, SP+, SRS, Elo, betting lines | Free API key |
| **The Odds API** | Live prices across US books, for line shopping | Free API key |
| **Sagarin** | Ratings **and** his projected spreads/totals/moneylines | Nothing |
| **Massey / other experts** | Power ratings | Nothing |
| **Fixtures** | Bundled sample week for `cfbpicks demo` | Nothing |

CFBD alone is enough to run the whole engine — it supplies games,
ratings and lines. The Odds API adds live prices and line shopping,
which is where a lot of the real edge is.

### Sagarin

<http://sagarin.com/sports/cfsend.htm> publishes both a rating list and a
*Predictions with Totals and Moneylines* table — a full expert projection
for every game. Both are ingested, and the projections are blended into
the model's own number rather than treated as a market price.

The site has no API and blocks some hosts, so if a direct fetch fails,
save the page and import it:

```bash
cfbpicks import-sagarin ~/Downloads/cfsend.htm --week 3
```

The parser keys off the anchors Sagarin has always used (`=` for a
rating, `by` for a margin, capitals for the home team) rather than fixed
column positions, and reports how many rows it recovered so a layout
change is loud rather than silent.

### Massey and anything else

<https://masseyratings.com/ranks> renders its tables in the browser, so
save the finished page and drop it in `data/ratings/`:

```
data/ratings/massey-2026-w3.html
```

Any CSV, JSON or saved HTML with a team column and a rating column works
— see [`data/ratings/README.md`](data/ratings/README.md). This is the
general path for adding an expert: no code required.

```bash
cfbpicks import-ratings ~/Downloads/fpi.csv --source fpi --week 3
```

## Does it work?

Don't take the model's word for it. Grade it:

```bash
cfbpicks fetch --season 2024          # pull a completed season
cfbpicks backtest --season 2024
```

```
Backtest — 2024
  Bets:        412  (221-186-5)
  Win rate:    54.3%  (break-even at -110 is 52.4%)
  Profit:      +18.40u
  ROI:         +4.51%
  Avg CLV:     +0.71 pts
```

Two honest caveats:

- **Under a few hundred bets, ROI is mostly noise.** Closing line value
  is the leading indicator worth watching — beating the close is the only
  early evidence an edge is real.
- **A backtest is only as honest as its odds.** Grading uses whatever
  line snapshots are stored, so if lines were only ever fetched after
  kickoff, the result flatters itself. Fetch before games start.

## Tuning

Everything lives in [`config.yaml`](config.yaml), which is commented.
The knobs that matter most:

| Setting | Default | Effect |
|---|---|---|
| `min_prob_edge` | `0.02` | Raise it for fewer, better bets |
| `kelly_multiplier` | `0.25` | Lower is safer; full Kelly is far too swingy |
| `margin_sd` | `16.0` | Drives every win probability — change carefully |
| `home_field_advantage` | `2.2` | Modern FBS value, down from ~3 historically |
| `rating_weights` | see file | Relative trust in each expert |

Keep local overrides and API keys in `config.local.yaml`, which is
gitignored and takes precedence.

## Layout

```
src/cfbpicks/
  cli.py            command line entry points
  pipeline.py       fetch -> predict -> picks, orchestrated
  config.py         YAML + environment configuration
  storage.py        SQLite: games, ratings, quotes, picks, line history
  models.py         the domain objects
  backtest.py       grading and season replay
  report.py         terminal, Markdown, CSV, JSON output
  providers/        one module per data source
  ratings/blend.py  many rating systems -> one power number
  engine/
    predict.py      ratings + experts -> projected margin and total
    market.py       de-vigging, consensus, line shopping
    recommend.py    edge, expected value, Kelly staking
  util/odds.py      the probability maths everything rests on
  data/             packaged team aliases and the demo sample week
```

Adding a source means writing one `Provider` subclass and listing it in
`providers/__init__.py`. Nothing else changes.

## Tests

```bash
pytest
```

181 tests, no network required. The maths (de-vigging, cover
probability, Kelly, grading) is tested against known values; the parsers
are tested against saved sample pages.

## Caveats

This is a model, not betting advice. Sports betting has a negative
expected return for almost everyone, the edges here are small and
uncertain, and a backtest is not a promise. Bet only what you can lose,
and check that betting is legal where you are.

## Licence

MIT
