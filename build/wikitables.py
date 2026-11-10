"""Read the data tables out of a Wikipedia article, using only the standard library.

S&P 500 membership has no free API, and Wikipedia's constituent table is the usual source. This runs
in `build/build_refdata.py` only, never at request time, so a parse break shows up at build time rather
than in front of a user, and `records()` raises instead of returning a short list.

Checked 2026-11-09: only the S&P 500 article still publishes its constituents as a wikitable. The
Nasdaq-100 and Dow Jones articles now carry annual-returns tables and no components table, so index
membership stops at `in_sp500`. Little is lost - every Dow 30 name and almost every Nasdaq-100 name
is already in the S&P 500, and the index ETFs are covered by the curated ETF families instead.

Deliberately not BeautifulSoup: one more dependency to ship for one page shape, where ~60 lines of
HTMLParser does the job and the failure mode is a visible row count, not silent wrong data.
"""

import re
from html.parser import HTMLParser

# A plausible US ticker as Wikipedia writes it: 'AAPL', 'BRK.B'. Excludes prose and footnote marks.
TICKER = re.compile(r"^[A-Z][A-Z.]{0,6}$")
ZERO_WIDTH = "\u200b"  # Wikipedia pads some ticker cells with it


class _Tables(HTMLParser):
    """Collects every <table class="...wikitable..."> as a list of rows of cell text."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._rows: list[list[str]] | None = None  # the table being collected, if any
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._open = 0  # <table> depth *within* the current wikitable, so nesting cannot desync us

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            if self._rows is not None:
                self._open += 1  # a nested table inside the one we are reading
            elif "wikitable" in dict(attrs).get("class", ""):
                self._rows, self._open = [], 1
        elif tag == "tr" and self._rows is not None and self._open == 1:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._rows is not None:
            self._open -= 1
            if self._open == 0:  # the wikitable itself closed, so flush it
                self.tables.append(self._rows)
                self._rows = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def tables(html: str) -> list[list[list[str]]]:
    """Every wikitable in `html`, each as rows of cell text."""
    parser = _Tables()
    parser.feed(html)
    return parser.tables


def records(html: str, key: str, min_rows: int) -> list[dict[str, str]]:
    """Rows of the largest wikitable having a `key` column, as dicts keyed by lowercased header.

    Rows whose `key` cell is not a plausible ticker are dropped, which removes footnote and total
    rows. Raises ValueError rather than returning a short list: a silently truncated index would
    quietly shrink the screening universe, which is far worse than a failed build.
    """
    best: list[dict[str, str]] = []
    for rows in tables(html):
        if len(rows) < min_rows:
            continue
        header = [re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_") for h in rows[0]]
        if key not in header:
            continue
        at = header.index(key)
        found = []
        for row in rows[1:]:
            if at >= len(row):
                continue
            cells = {h: row[i].replace(ZERO_WIDTH, "").strip() for i, h in enumerate(header) if i < len(row)}
            if TICKER.match(cells[key]):
                found.append(cells)
        if len(found) > len(best):
            best = found
    if len(best) < min_rows:
        raise ValueError(f"found only {len(best)} rows with a '{key}' column; the page layout changed")
    return best
