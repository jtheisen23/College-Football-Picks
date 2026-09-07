"""Minimal HTML table extraction (stdlib only).

Publishers such as Massey render ratings and game projections as HTML
tables. Saving the page and parsing it here means no browser, no
JavaScript runtime and no scraping dependency — and it keeps working
when the site is behind an egress policy.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Optional

_WS = re.compile(r"\s+")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: Optional[list[list[str]]] = None
        self._row: Optional[list[str]] = None
        self._cell: Optional[list[str]] = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_WS.sub(" ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if any(c for c in self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table":
            if self._depth == 1 and self._table is not None:
                self.tables.append(self._table)
                self._table = None
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def extract_tables(html_text: str) -> list[list[list[str]]]:
    """Return every table as a list of rows of cell strings."""
    parser = _TableParser()
    parser.feed(html_text)
    parser.close()
    return parser.tables


def table_to_dicts(
    table: list[list[str]], headers: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Convert a table to row dicts, using the first row as headers."""
    if not table:
        return []
    if headers is None:
        headers = [h.strip().lower() for h in table[0]]
        body = table[1:]
    else:
        headers = [h.strip().lower() for h in headers]
        body = table
    out = []
    for row in body:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        out.append(dict(zip(headers, row)))
    return out


def find_table_with(html_text: str, *required_headers: str) -> list[dict[str, Any]]:
    """Find the first table whose header row contains all the given labels.

    Pages wrap their real content in layout tables, so picking by header
    content is far more durable than picking by index.
    """
    wanted = {h.lower() for h in required_headers}
    for table in extract_tables(html_text):
        if not table:
            continue
        header = {c.strip().lower() for c in table[0]}
        if wanted <= header:
            return table_to_dicts(table)
    return []


def largest_table(html_text: str) -> list[dict[str, Any]]:
    """Fall back to the biggest table on the page, which is usually the data."""
    tables = extract_tables(html_text)
    if not tables:
        return []
    return table_to_dicts(max(tables, key=len))
