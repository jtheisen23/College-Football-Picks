# Drop-folder for expert ratings

Anything you put here is picked up automatically by `cfbpicks fetch`. This
is the escape hatch for sources with no usable API — save the page or
export a spreadsheet, drop it in, and it becomes another input to the
blend.

## Naming

The filename supplies the source name, season and week:

```
massey-2026-w3.csv        -> source "massey",  2026, week 3
sagarin-2026-w3.html      -> source "sagarin", 2026, week 3
fpi-2026.csv              -> source "fpi",     2026, week 0
```

## Accepted formats

| Format | Requirement |
|---|---|
| `.csv` | A `team` column and a `rating` column |
| `.json` | A list of objects with `team` and `rating` keys |
| `.html` | Any saved page whose main table has team and rating columns |

Ratings should be **points above average** (a `+14.0` team beats a `+7.0`
team by 7 on a neutral field). Sources on another scale are recentred and
rescaled automatically during blending, so an absolute scale like
Sagarin's 70-90 range still works.

## Sources worth adding

- **Massey** — <https://masseyratings.com/ranks> and
  <https://masseyratings.com/cf/fbs/games>. The tables render in the
  browser, so save the finished page rather than fetching the URL.
- **Sagarin** — <http://sagarin.com/sports/cfsend.htm>. Use
  `cfbpicks import-sagarin` instead of this folder: it also reads his
  projected spreads, totals and moneylines, not just the ratings.

## Importing by hand

```bash
cfbpicks import-ratings ~/Downloads/massey.html --source massey --week 3
cfbpicks import-sagarin ~/Downloads/cfsend.htm --week 3
```
