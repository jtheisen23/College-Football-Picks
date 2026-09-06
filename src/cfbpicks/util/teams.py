"""Canonical team naming.

Every provider spells school names differently ("Ole Miss" vs
"Mississippi", "Miami (FL)" vs "Miami Hurricanes", "App State" vs
"Appalachian State"). If those aren't reconciled, a team silently ends up
with half its ratings and no market, so all names pass through here
before they touch storage.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

_PUNCT = re.compile(r"[^a-z0-9]+")

# Mascots that get appended by odds feeds and must be stripped to find the
# school. Only unambiguous ones — never strip a word that is part of a
# school name (e.g. "State", "Tech", "A&M").
_MASCOT_SUFFIXES = {
    "aggies", "badgers", "bearcats", "bears", "beavers", "bison", "blazers",
    "blue devils", "blue hens", "blue raiders", "bobcats", "boilermakers",
    "broncos", "browns", "bruins", "buccaneers", "buckeyes", "buffaloes",
    "bulldogs", "bulls", "cardinal", "cardinals", "cavaliers", "chanticleers",
    "chippewas", "cornhuskers", "cougars", "cowboys", "crimson tide",
    "cyclones", "demon deacons", "ducks", "eagles", "falcons", "fighting illini",
    "fighting irish", "flames", "gamecocks", "gators", "golden bears",
    "golden eagles", "golden flashes", "golden gophers", "golden hurricane",
    "governors", "green wave", "hawkeyes", "hokies", "horned frogs",
    "hoosiers", "hurricanes", "huskers", "huskies", "jaguars", "jayhawks",
    "keydets", "knights", "lobos", "longhorns", "lumberjacks", "mean green",
    "midshipmen", "miners", "minutemen", "mocs", "mountaineers", "mustangs",
    "nittany lions", "orange", "owls", "panthers", "pirates", "quakers",
    "raiders", "rainbow warriors", "ragin cajuns", "rams", "razorbacks",
    "rebels", "red raiders", "red wolves", "redbirds", "roadrunners",
    "rockets", "sooners", "spartans", "sun devils", "seminoles", "tar heels",
    "terrapins", "tigers", "trojans", "utes", "vandals", "volunteers",
    "warhawks", "wildcats", "wolfpack", "wolverines", "yellow jackets",
}

_BUILTIN_ALIASES: dict[str, str] = {
    # School-name variants that differ beyond punctuation.
    "ole miss": "Mississippi",
    "miami fl": "Miami",
    "miami florida": "Miami",
    "miami oh": "Miami (OH)",
    "miami ohio": "Miami (OH)",
    "miami redhawks": "Miami (OH)",
    "pitt": "Pittsburgh",
    "app state": "Appalachian State",
    "southern miss": "Southern Mississippi",
    "usc": "USC",
    "southern cal": "USC",
    "southern california": "USC",
    "ucf": "UCF",
    "central florida": "UCF",
    "usf": "South Florida",
    "smu": "SMU",
    "southern methodist": "SMU",
    "tcu": "TCU",
    "texas christian": "TCU",
    "byu": "BYU",
    "brigham young": "BYU",
    "lsu": "LSU",
    "louisiana state": "LSU",
    "utep": "UTEP",
    "utsa": "UTSA",
    "unlv": "UNLV",
    "uab": "UAB",
    "ul monroe": "Louisiana Monroe",
    "louisiana lafayette": "Louisiana",
    "ul lafayette": "Louisiana",
    "nc state": "NC State",
    "north carolina state": "NC State",
    "n c state": "NC State",
    "unc": "North Carolina",
    "fiu": "Florida International",
    "fau": "Florida Atlantic",
    "uconn": "Connecticut",
    "umass": "Massachusetts",
    "hawaii": "Hawai'i",
    "hawai i": "Hawai'i",
    "san jose state": "San José State",
    "san jos state": "San José State",
    "texas a m": "Texas A&M",
    "texas am": "Texas A&M",
    "bowling green": "Bowling Green",
    "central michigan": "Central Michigan",
    "western kentucky": "Western Kentucky",
    "middle tennessee": "Middle Tennessee",
    "middle tennessee state": "Middle Tennessee",
    "sam houston": "Sam Houston State",
    "penn": "Pennsylvania",
    "ohio state": "Ohio State",
    "washington state": "Washington State",
    "jacksonville state": "Jacksonville State",
    "kennesaw state": "Kennesaw State",
    "army": "Army",
    "army west point": "Army",
    "navy": "Navy",
    "air force": "Air Force",
}

# Abbreviations that expand before the alias lookup runs.
_TOKEN_EXPANSIONS = {
    "st": "state",
    "st.": "state",
    "u": "",
    "univ": "",
    "university": "",
}


def slugify(name: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    return _PUNCT.sub(" ", name.strip().lower()).strip()


def _expand_tokens(slug: str) -> str:
    parts = []
    tokens = slug.split()
    for i, token in enumerate(tokens):
        # "St" means "State" at the end of a name but "Saint" at the front.
        if token == "st" and i == 0:
            parts.append("saint")
            continue
        replacement = _TOKEN_EXPANSIONS.get(token, token)
        if replacement:
            parts.append(replacement)
    return " ".join(parts)


def _strip_mascot(slug: str) -> str:
    """Remove a trailing mascot, longest match first."""
    for suffix in sorted(_MASCOT_SUFFIXES, key=len, reverse=True):
        if slug.endswith(" " + suffix):
            trimmed = slug[: -(len(suffix) + 1)].strip()
            if trimmed:
                return trimmed
    return slug


@lru_cache(maxsize=1)
def _load_alias_file() -> dict[str, str]:
    """Merge the shipped alias file over the built-in map, if present."""
    aliases = dict(_BUILTIN_ALIASES)
    # The packaged file first, then a project-local one so a user can add
    # aliases without editing the installed package.
    for candidate in (
        Path(__file__).resolve().parent.parent / "data" / "reference" / "team_aliases.json",
        Path.cwd() / "data" / "reference" / "team_aliases.json",
    ):
        if not candidate.exists():
            continue
        try:
            extra = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        for key, value in extra.items():
            if not key.startswith("_"):
                aliases[slugify(key)] = value
    return aliases


def canonical_team(name: str) -> str:
    """Return the canonical school name for any provider's spelling."""
    if not name:
        return ""
    aliases = _load_alias_file()
    slug = slugify(name)

    for candidate in (slug, _strip_mascot(slug)):
        if candidate in aliases:
            return aliases[candidate]
        expanded = _expand_tokens(candidate)
        if expanded in aliases:
            return aliases[expanded]

    # No alias matched. Only rewrite the name when normalisation actually
    # changed something (a mascot was dropped, or "St" was expanded);
    # otherwise the provider's own spelling is left alone.
    stripped = _strip_mascot(slug)
    expanded = _expand_tokens(stripped)
    if expanded != stripped or stripped != slug:
        return _titlecase(expanded)
    # Sources such as Sagarin shout the home team in capitals. Recase it
    # rather than treating "TEXAS" and "Texas" as two programs; genuine
    # acronyms survive via _KNOWN_ACRONYMS.
    if is_shouted(name):
        return _titlecase(expanded)
    return name.strip()


def is_shouted(name: str) -> bool:
    """True when a name is written in ALL CAPS (and isn't just an acronym)."""
    letters = [c for c in name if c.isalpha()]
    if not letters:
        return False
    if not all(c.isupper() for c in letters):
        return False
    # A short all-caps token is probably a real acronym (LSU, TCU, BYU).
    return not (len(letters) <= 4 and " " not in name.strip())


def _titlecase(slug: str) -> str:
    small = {"of", "and", "at"}
    words = []
    for i, word in enumerate(slug.split()):
        if word.upper() in _KNOWN_ACRONYMS:
            words.append(word.upper())
        elif i > 0 and word in small:
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


_KNOWN_ACRONYMS = {
    "LSU", "TCU", "SMU", "BYU", "UCF", "USC", "USF", "UAB", "UNC", "UTEP",
    "UTSA", "UNLV", "FIU", "FAU", "UCLA", "VMI", "UTC",
}


def match_teams(name: str, universe: Iterable[str]) -> Optional[str]:
    """Best-effort match of ``name`` against a set of canonical names."""
    canonical = canonical_team(name)
    universe = list(universe)
    if canonical in universe:
        return canonical

    target = _expand_tokens(slugify(canonical))
    by_slug = {_expand_tokens(slugify(u)): u for u in universe}
    if target in by_slug:
        return by_slug[target]

    # Fall back to a unique prefix match, which handles "Louisiana" vs
    # "Louisiana Monroe" style truncation without guessing between two.
    hits = [full for slug, full in by_slug.items() if slug.startswith(target) or target.startswith(slug)]
    if len(hits) == 1:
        return hits[0]
    return None


def game_key(season: int, week: int, home: str, away: str) -> str:
    """Stable identifier for a game, independent of any provider's id."""
    return f"{season}-{week:02d}-{slugify(canonical_team(away)).replace(' ', '_')}-at-{slugify(canonical_team(home)).replace(' ', '_')}"
