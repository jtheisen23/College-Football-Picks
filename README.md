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
cfbpicks rate                                 # fit the engine's own ratings
cfbpicks predict --week 3                     # projections, before prices
cfbpicks picks --week 3                       # the bets worth making
cfbpicks picks --week 3 -o reports/week3.md --format markdown

cfbpicks picks --week 3 -o reports/week3.html  # a board you can look at

cfbpicks snapshot                             # capture odds again later
cfbpicks lines --week 3                       # see what moved

# after the games finish
cfbpicks grade --week 3
```

Everything is cached in SQLite, so re-running `picks` costs no API calls.

## Seeing the week

The terminal table is fine for a glance. For something you can actually
sit with — or open on a phone — write the board out as a page:

```bash
cfbpicks picks --week 3 -o reports/week3.html
open reports/week3.html
```

One self-contained file: no network, no CDN, works offline and keeps
working. It follows your system light/dark setting and has a toggle.
Summary tiles up top (bets, units staked, expected units, median edge),
then every bet with its price, book, model and market probabilities, and
the edge between them. Full projections for the slate are behind a
collapsible section.

Tier badges use a single-hue ordinal ramp — validated for contrast in
both modes — and each badge carries its text label, so conviction is
never communicated by colour alone.

`-o something.html` picks the HTML renderer automatically; `.md`, `.csv`
and `.json` all still work, and `--format` overrides if you want.

## Closing line value

CLV is the one measurement that tells you early whether an edge is real.
Win rate takes hundreds of bets to say anything; beating the closing line
says it in weeks, because the closing line is the sharpest number the
market ever produces.

It requires a line captured *later* than your bet, so odds have to be
captured more than once. `cfbpicks snapshot` does that and reports what
moved:

```
$ cfbpicks snapshot
Captured 1,240 quotes

      Movement since last capture (18 lines)
  Game                              Market    Moved
  2026-03-alabama-at-georgia        spread     -2.0
  2026-03-iowa-at-penn_state        total      +1.5
```

Put it on a schedule — midweek and again near kickoff is plenty:

```cron
0 12 * * 3,6  cd /path/to/repo && .venv/bin/cfbpicks snapshot --quiet
```

`cfbpicks grade` then records CLV per bet automatically. Bet Georgia -6.5
on a game that closes -8.5 and that is +2.0 points of CLV, whether or not
the bet won — those are different questions, and a model with persistent
positive CLV and a losing month is in better shape than the reverse.

With a single capture, CLV is left null rather than being invented from
comparing a bet against itself.

## The engine's own ratings

Every external rating has a provenance problem. SP+ and SRS are
season-level, so for a finished season they encode the results you're
trying to predict. Sagarin and Massey publish only a current snapshot,
with no history. None of them can honestly backtest.

So the engine fits its own. `cfbpicks rate` runs a ridge regression on
scoring margin:

```
margin  ≈  rating[home] − rating[away] + home_field × (not neutral)
```

refit once per week from games played **strictly beforehand**, and
stamped with the week it's meant to price. That ordering is the whole
point: the fit has never seen the game it's about to be asked about, so
it's point-in-time by construction.

```
$ cfbpicks rate
Fitted 14 weekly snapshots · latest is week 14: 134 teams from 812 games,
HFA +2.31, residual sd 15.8

  Fitted residual sd is 15.8; config uses margin_sd 16.0.
```

Three details make it behave on real data:

- **Ridge regularisation.** In September a 2-0 team has beaten two
  opponents nobody has a read on yet. An L2 penalty pulls thin evidence
  toward average instead of letting it run wild, and keeps the normal
  equations solvable when the schedule graph is disconnected.
- **No margin capping, deliberately.** Clamping blowouts is standard for
  *ranking* systems, but it shrinks every coefficient, and this model
  outputs a point spread. Measured on simulated seasons, a 21-point cap
  left predicted margins 58% too small — enough to tip the model onto
  underdogs across the whole board.
- **A carried-forward prior.** Week 1 has nothing to fit on, so last
  season's final ratings come in at half strength.

Defaults were picked by sweep, not taste: `ridge: 2.0` gave both the best
held-out error and a calibration slope of ~1.02, where 6.0 over-shrank to
1.64 in September.

The `residual sd` it reports is an empirical read on `margin_sd`, the
number driving every win probability in the engine. If the two disagree,
believe the fit.

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

### The Odds API

CFBD tells you what the number is. The Odds API tells you what you can
actually bet it at, and where — which is where a real amount of the edge
lives. Taking `-3 -105` instead of `-3.5 -115` is worth more over a
season than most model improvements, and recommendations are always
priced at the best number and juice across your books.

```yaml
providers:
  odds_api:
    options:
      books: [draftkings, fanduel, betmgm]   # only ones you can bet
      min_quota_remaining: 25
```

Restrict `books` to accounts you actually hold. Shopping across books you
can't bet just invents edges you have no way to take.

**Watch the quota.** The free tier is ~500 credits a month and bills one
credit **per market per region** — so the three default markets in one
region cost 3 credits a call, not 1. A twice-weekly snapshot across a
full season fits comfortably, but only if nothing burns credits by
accident. Responses are cached, the balance is read off every response,
and below `min_quota_remaining` the provider refuses to call rather than
quietly spending your last credits. An exhausted balance is invisible
until the week you need a price and there isn't one.

Events are matched to the stored schedule by canonical name and kickoff
proximity, because the feed has its own spellings and no concept of a
week. Anything unmatched is reported rather than guessed at — a silently
dropped game is a game you never get a price on.

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
cfbpicks fetch --season 2025          # pull a completed season
cfbpicks backtest --season 2025
```

**The backtest actively refuses to lie to you**, which matters more than
any number it prints. Two forms of hindsight will otherwise manufacture
a spectacular fake edge, and both are guarded:

**Season-final ratings.** CFBD's SP+ and SRS endpoints are season-level:
ask for a finished season and you get the *end-of-year* number, which
already encodes the results of the games you're about to "predict." Used
naively this produces something like 58% ATS and +20% ROI — pure
leakage. Those sources are excluded from backtests, and if nothing
week-indexed remains the backtest refuses to run at all:

```
Backtest refused: Every stored rating for this season is season-final, so a
backtest would be predicting each game with a rating that already knows how it
ended. Fetch a week-indexed source (CFBD Elo is one) or re-run with
allow_final_ratings=True to see the contaminated number.
```

Two sources are safe to backtest with: the engine's own fitted ratings
(above) and CFBD's **Elo**, which is genuinely week-indexed.
`--allow-final-ratings` exists for exploration and stamps
`HINDSIGHT ENABLED` on the output.

The guard is verified by a test rather than by inspection: a synthetic
season is generated in which the sportsbook is given the *true* margin
of every game, and the backtest must fail to beat it. A model with no
hindsight cannot profit against an omniscient market, so any positive
ROI there means results are leaking into the ratings.

**Undefined closing line value.** CLV needs a line captured *later* than
the bet. With one odds snapshot per game, the "closing" line is the
bet's own line, and any CLV figure is line-shopping jitter. Rather than
print a number near zero that reads like a finding, the backtest says so:

```
Avg CLV:     n/a - odds were captured once, so there is no later line to
             compare against
```

To make CLV real, fetch odds during the week and again near kickoff.

What to watch, in order: **CLV** first — beating the close is the only
leading indicator that an edge is real. Then ROI per unit staked. Win
rate last; it's the number that looks most meaningful and tells you
least. Under a few hundred bets, treat ROI as noise.

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
  ratings/
    blend.py        many rating systems -> one power number
    regression.py   the engine's own point-in-time ratings
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
