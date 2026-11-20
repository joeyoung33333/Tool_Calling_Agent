"""Market, news and filing fetchers (free, no API keys): Nasdaq, Yahoo Finance, SEC EDGAR, Google News.

Errors are DataError with a message for the model; headlines are never cached.
Prices are end of day: Nasdaq first, Yahoo Finance if Nasdaq fails (Cboe blocked this app on 2026-11-19).
"""

import functools
import re
import threading
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from sources import refdata

_NASDAQ = "https://api.nasdaq.com/api/quote"
_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart"
_NEWS = "https://news.google.com/rss/search"
_SEC = "https://data.sec.gov"
_SEC_TICKERS = "https://www.sec.gov/include/ticker.txt"
_SYMBOLS = "https://www.nasdaqtrader.com/dynamic/symdir"
# Sent to every feed, including SEC. Generic on purpose: no personal contact details anywhere.
_UA = "Mozilla/5.0 (compatible; converge-agent/1.0; Columbia IEOR-4570 class project)"

HISTORY_YEARS = 10  # what Nasdaq serves; config.MAX_LOOKBACK_DAYS matches it
_TICKER = re.compile(r"[A-Z][A-Z0-9.]{0,9}")  # 'NVDA', 'BRK.B'
# Indices this feed lacks, with the nearest thing it has: a bare "no prices" tells the model nothing.
_NO_DATA = {"NDX": "NDX is not on this feed. Use QQQ, the ETF that tracks the Nasdaq-100.",
            "VIX": "VIX is not on this feed. Use VIXY or VXX, ETFs that track VIX futures.",
            "SPX": "SPX is not on this feed. Use SPY, the ETF that tracks the S&P 500.",
            "RUT": "RUT is not on this feed. Use IWM, the ETF that tracks the Russell 2000."}
# Blocked, since some crypto names are also ETF tickers (BTC is a Grayscale fund).
_CRYPTO = {"BTC", "ETH", "XBT", "LTC", "SOL", "XRP", "DOGE", "ADA", "BITCOIN", "ETHEREUM"}
# Used by _symbol_directory(): directory rows that are not the common stock itself.
_NOT_STOCK = re.compile(r"warrant|\brights\b|\bunits?\b|preferred|\bnotes?\b", re.I)

_SEC_ITEMS = {
    "1.01": "Material agreement signed", "1.02": "Material agreement terminated", "1.03": "Bankruptcy",
    "2.01": "Acquisition or disposal completed", "2.02": "Earnings results", "2.03": "New debt or financial obligation",
    "2.05": "Restructuring or exit costs", "2.06": "Material impairment", "3.01": "Delisting notice",
    "3.02": "Unregistered share sale", "4.01": "Auditor change", "4.02": "Financial statements not reliable",
    "5.01": "Change in control", "5.02": "Executive or director change", "5.03": "Charter or bylaw change",
    "5.07": "Shareholder vote results", "7.01": "Regulation FD disclosure", "8.01": "Other event",
}
_SEC_FORMS = {"10-K": "Annual report", "10-Q": "Quarterly report"}


def eastern_now() -> datetime:
    """Now in US Eastern, the market's clock (UTC if the machine has no time-zone data)."""
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc)


class DataError(Exception):
    """A failure whose message is written for the model: what went wrong and what to try."""


# Process-wide: the feeds throttle per IP, and the fan-out multiplies bursts.
_SLOTS = threading.BoundedSemaphore(4)


def _fetch(url: str, **params) -> requests.Response:
    host = url.split("/")[2]
    try:
        with _SLOTS:
            r = requests.get(url, params=params, headers={"User-Agent": _UA}, timeout=15)
    except requests.RequestException as e:
        raise DataError(f"Could not reach {host} ({type(e).__name__}). Try again shortly.")
    if r.status_code == 404:
        raise DataError(f"{host} has no data for this request (HTTP 404). Check the symbol.")
    if r.status_code == 403:
        # A block or a rate limit, not a missing symbol: SEC and Cloudflare both answer 403.
        raise DataError(f"{host} refused the request (HTTP 403): it is blocking or rate limiting this app. "
                        "Try again later; the symbol itself may be fine.")
    if r.status_code >= 400:
        raise DataError(f"{host} returned HTTP {r.status_code}. Try again shortly.")
    return r


def _json(url: str, **params) -> dict:
    try:
        return _fetch(url, **params).json()
    except ValueError:
        raise DataError(f"{url.split('/')[2]} returned an unreadable response. Try again shortly.")


def clean(raw: str) -> str:
    """'$nvda' -> 'NVDA', 'brk-b' -> 'BRK.B'. Rejects company names, indices and unsupported crypto."""
    symbol = re.sub(r"[-/]USD$", "", (raw or "").strip().upper().lstrip("$^"))  # 'BTC-USD' -> 'BTC', so crypto is caught
    symbol = symbol.replace("-", ".")  # share classes: 'BRK-B' -> 'BRK.B'
    if not _TICKER.fullmatch(symbol):
        raise DataError(f"'{raw}' is not a ticker. Pass a symbol like 'NVDA' or 'SPY'.")
    if symbol in _NO_DATA:
        raise DataError(_NO_DATA[symbol])
    if symbol in _CRYPTO:
        raise DataError(
            f"Crypto is not supported, so '{raw}' is rejected (on US exchanges 'BTC' and 'ETH' are unrelated ETFs). "
            "For crypto exposure ask about an ETF such as IBIT (Bitcoin) or ETHA (Ethereum)."
        )
    return symbol


def _number(text) -> float | None:
    """'$1,234.50' -> 1234.5; 'N/A' or blank -> None."""
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def _nasdaq_bars(symbol: str, today: date) -> list[dict]:
    """Every daily bar Nasdaq serves for `symbol`, oldest first.

    Asks from a year before Nasdaq's ten-year limit up to today: windows ending in the past, and
    some starting near the limit (2016-11-21 on 2026-11-19), come back empty.
    """
    classes = ["etf", "stocks"] if refdata.is_etf(symbol) else ["stocks", "etf"]
    for asset_class in classes:
        # Rows are newest first: {"date": "11/19/2026", "open": "$352.91", ..., "volume": "40,906,860"}.
        d = _json(f"{_NASDAQ}/{symbol}/historical", assetclass=asset_class, limit=9999,
                  fromdate=(today - timedelta(days=365 * (HISTORY_YEARS + 1))).isoformat(), todate=today.isoformat())
        rows = ((d.get("data") or {}).get("tradesTable") or {}).get("rows") or []
        if rows:
            break
    else:
        raise DataError(f"Nasdaq has no prices for '{symbol}'. It may not be a US-listed stock or ETF; "
                        "pass a ticker such as 'NVDA' or 'SPY'.")
    bars = []
    for r in reversed(rows):
        o, h, l, c, v = (_number(r.get(k)) for k in ("open", "high", "low", "close", "volume"))
        # No volume means a stale repeat of the previous close, not a traded session (SPY on 2026-04-20).
        if None in (o, h, l, c, v):
            continue
        month, day, year = r["date"].split("/")
        bars.append({"date": f"{year}-{month}-{day}", "open": o, "high": h, "low": l, "close": c, "volume": v})
    return bars


def _yahoo_bars(symbol: str) -> list[dict]:
    """Every daily bar Yahoo Finance serves for `symbol` over HISTORY_YEARS, oldest first."""
    d = _json(f"{_YAHOO}/{symbol.replace('.', '-')}", range=f"{HISTORY_YEARS}y", interval="1d")
    results = (d.get("chart") or {}).get("result") or []
    if not results or not results[0].get("timestamp"):
        raise DataError(f"Yahoo Finance has no prices for '{symbol}'.")
    r = results[0]
    q = r["indicators"]["quote"][0]
    eastern, bars = eastern_now().tzinfo, []
    # Parallel arrays; a session's timestamp is its open. `close` is split-adjusted, not for dividends.
    for t, o, h, l, c, v in zip(r["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"]):
        if None in (o, h, l, c):
            continue
        day = datetime.fromtimestamp(t, eastern).strftime("%Y-%m-%d")
        bars.append({"date": day, "open": o, "high": h, "low": l, "close": c, "volume": float(v or 0)})
    return bars


PRICE_SOURCE = "Nasdaq (Yahoo Finance if Nasdaq fails)"


# One lock per symbol: lru_cache does not deduplicate concurrent misses.
_BAR_LOCKS: dict[str, threading.Lock] = {}
_BAR_LOCKS_GUARD = threading.Lock()


def published_bars(symbol: str) -> tuple[dict, ...]:
    """`_published_bars` with concurrent fetches of one symbol collapsed into one."""
    with _BAR_LOCKS_GUARD:
        lock = _BAR_LOCKS.setdefault(symbol, threading.Lock())
    now = eastern_now()
    closed = now.strftime("%H:%M") >= "16:15"
    with lock:  # after the close, refetched hourly in case the day's bar posts late
        return _published_bars(symbol, now.date(), closed, now.hour if closed else 0)


@functools.lru_cache(maxsize=192)
def _published_bars(symbol: str, day: date, closed: bool, hour: int) -> tuple[dict, ...]:
    """Every completed daily session for `symbol`, oldest first: Nasdaq, else Yahoo Finance.

    `day`, `closed` and `hour` are only the cache key. Call `published_bars()` for the lock.
    """
    s = clean(symbol)
    try:
        bars = _nasdaq_bars(s, day)
    except DataError as nasdaq_error:
        try:
            bars = _yahoo_bars(s)
        except DataError:
            raise nasdaq_error
    if bars and bars[-1]["date"] == day.isoformat() and not closed:
        bars = bars[:-1]  # end of day: today's bar is not final until the close prints
    return tuple(bars)


def price_history(symbol: str, start: date, end: date | None = None) -> list[dict]:
    """Completed daily bars (date, open, high, low, close, volume) from `start` to `end`, oldest first."""
    last = (end or date.today()).isoformat()
    return [b for b in published_bars(clean(symbol)) if start.isoformat() <= b["date"] <= last]


def news(query: str, start: date | None = None, end: date | None = None, limit: int = 20) -> list[dict]:
    """Google News headlines for a ticker or keywords, newest first. No dates means the last 7 days.

    Pass both dates or neither; pipeline.tools._optional_range validates model-supplied dates.
    """
    q = (query or "").strip()
    if not q:
        raise DataError("'query' is empty. Pass a ticker like 'NVDA' or keywords like 'Fed rate decision'.")
    if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", q):
        q += " stock"  # a bare ticker is ambiguous, so bias towards market news
    # Google's `before:` is exclusive, so add a day to include `end`.
    q += f" after:{start} before:{end + timedelta(days=1)}" if start else " when:7d"
    # Google News RSS, up to 100 items: each <item> carries title, link, pubDate and a <source url=...> publisher name.
    try:
        root = ET.fromstring(_fetch(_NEWS, q=q, hl="en-US", gl="US", ceid="US:en").content)
    except ET.ParseError:
        raise DataError("Google News returned an unexpected response (it may be rate-limiting). Try again shortly.")
    items, seen = [], set()
    for item in root.iterfind("./channel/item"):
        source, title = item.findtext("source") or "", item.findtext("title") or ""
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]  # Google appends the publisher to every title
        key = re.sub(r"\W+", " ", title.lower()).strip()
        if key in seen:  # syndicated copies of the same headline
            continue
        seen.add(key)
        published = parsedate_to_datetime(item.findtext("pubDate")).astimezone(timezone.utc)
        if start and not start <= published.date() <= end:  # Google's date filter is fuzzy at the edges
            continue
        items.append({"published": published.strftime("%Y-%m-%d %H:%M UTC"), "source": source, "title": title, "link": item.findtext("link")})
    return sorted(items, key=lambda i: i["published"], reverse=True)[:limit]


@functools.cache
def _sec_ciks() -> dict[str, int]:
    # SEC ticker.txt, one "ticker<TAB>cik" per line: "nvda\t1045810"
    lines = _fetch(_SEC_TICKERS).text.split()
    return {ticker.upper(): int(cik) for ticker, cik in zip(lines[::2], lines[1::2])}


# Cached: filings() is called once per symbol per episode, and each record is ~1 MB.
@functools.lru_cache(maxsize=32)
def _submissions(cik: int) -> dict:
    """One company's SEC submissions record; the filing arrays are parallel, newest first."""
    return _json(f"{_SEC}/submissions/CIK{cik:010d}.json")


def cik_for(symbol: str) -> int:
    """SEC CIK for a ticker: the reference table first, then SEC's ticker list. Raises for non-filers."""
    s = clean(symbol)
    # The reference table holds every S&P 500 stock; SEC writes share classes as BRK-B.
    cik = refdata.cik_for(s) or _sec_ciks().get(s.replace(".", "-"))
    if cik is None:
        raise DataError(
            f"'{symbol}' is not an SEC filer, so it has no filings. Indices never file, and most "
            "ETFs do not either; commodity trusts such as GLD do."
        )
    return cik


def filings(symbol: str, start: date | None = None, end: date | None = None, limit: int = 20) -> list[dict]:
    """SEC 8-K (events), 10-K and 10-Q for a US-listed company, newest first. No dates means the latest.

    Pass both dates or neither; pipeline.tools._optional_range is what validates model-supplied dates.
    """
    s = clean(symbol)
    cik = cik_for(s)
    recent = _submissions(cik)["filings"]["recent"]
    out = []
    for i, form in enumerate(recent["form"]):
        day = recent["filingDate"][i]
        if form not in ("8-K", "10-K", "10-Q"):
            continue
        if start and not start.isoformat() <= day <= end.isoformat():
            continue
        if form == "8-K":
            events = [_SEC_ITEMS[c] for c in recent["items"][i].split(",") if c in _SEC_ITEMS] or ["Other"]
        else:
            events = [_SEC_FORMS[form]]
        accession = recent["accessionNumber"][i].replace("-", "")
        out.append({
            "date": day,
            "form": form,
            "events": events,
            "report_date": recent["reportDate"][i] or None,
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{recent['primaryDocument'][i]}",
        })
        if len(out) == limit:
            break
    return out


@functools.cache
def _symbol_directory() -> list[tuple[str, str, bool]]:
    """(symbol, name, is_etf) for every listed stock and ETF, without test issues, warrants, rights, units or preferreds."""
    # Pipe-delimited; the symbol pattern skips each file's header, footer and preferreds ('JPM$J').
    rows = []
    for name, etf_column, test_column in (("nasdaqlisted", 6, 3), ("otherlisted", 4, 6)):
        for line in _fetch(f"{_SYMBOLS}/{name}.txt").text.splitlines()[1:]:
            f = line.split("|")
            if len(f) > 6 and _TICKER.fullmatch(f[0]) and f[test_column] != "Y" and not _NOT_STOCK.search(f[1]):
                rows.append((f[0], f[1], f[etf_column] == "Y"))
    return rows


def company_name(symbol: str) -> str | None:
    """Listed name of a stock or ETF, or None (indices are not in the directory)."""
    name = next((name for s, name, _ in _symbol_directory() if s == symbol), None)
    return name and re.sub(r"\s+(-\s+)?(New\s+)?(Common|Capital)\s+Stock\s*$", "", name)  # 'NVIDIA Corporation - Common Stock' -> 'NVIDIA Corporation'
