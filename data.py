"""Market and filing data fetchers (free, no API keys). Only dependency: requests.

Sources: Cboe (stocks, ETFs, indices - daily bars and quotes), SEC EDGAR (filings) and the Nasdaq
Trader symbol directory. US equities only.

Every function raises DataError on failure, with a message written for the model to act on.

Caching: reference data (published daily bars, the symbol directory, the CIK map, SEC submissions)
is cached for the life of the process because it does not change. Live quotes and headlines are
never cached.

    price(symbol)                          latest quote: price, bid/ask, volume, 30-day IV
    price_history(symbol, start, end)      daily bars, oldest first
    news(query, start, end)                headlines, newest first; no dates means the last 7 days
    filings(symbol, start, end)            SEC filings, newest first; no dates means the latest
    company_name(symbol)                   the listed name of a stock or ETF
    cik_for(symbol)                        SEC CIK for a ticker
    eastern_now()                          now in US Eastern, the market's clock
    clean(raw)                             '$nvda' -> 'NVDA'; rejects names and crypto
"""

import functools
import re
import threading
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

_CBOE = "https://cdn.cboe.com/api/global/delayed_quotes"
_NEWS = "https://news.google.com/rss/search"
_SEC = "https://data.sec.gov"
_SEC_TICKERS = "https://www.sec.gov/include/ticker.txt"
_SYMBOLS = "https://www.nasdaqtrader.com/dynamic/symdir"
# Sent to every feed, including SEC. Generic on purpose: no personal contact details anywhere.
_UA = "Mozilla/5.0 (compatible; converge-agent/1.0; Columbia IEOR-4570 class project)"

_TICKER = re.compile(r"[A-Z][A-Z0-9.]{0,9}")  # 'NVDA', 'BRK.B'
# Cboe prefixes index symbols with an underscore.
_INDICES = {"VIX", "SPX", "RUT"}
# Symbols this feed lacks, with the nearest thing it has: rejected up front, since a bare 403 tells the model nothing.
_NO_DATA = {"NDX": "NDX is not on this feed. Use QQQ, the ETF that tracks the Nasdaq-100."}
# Crypto is unsupported. Some double as ETF tickers (BTC is a Grayscale fund), so block them rather than pass a fund's price.
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
    """Now in US Eastern time, where the markets are (UTC if the machine has no time-zone data).

    Shared because both the price tape and the agent's instructions need today's market date.
    """
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc)


class DataError(Exception):
    """A failure whose message is written for the model: what went wrong and what to try."""


# Cboe throttles bursts, and the research fan-out multiplies them: two researchers screening the
# same sector is 2 x 21 symbols x SCREEN_THREADS requests arriving together, which is what turned a
# working screen into a wall of HTTP 429s. The cap is on this process as a whole rather than per
# caller, because the upstream limit is per IP and does not care how many agents we are running.
_SLOTS = threading.BoundedSemaphore(4)


def _fetch(url: str, **params) -> requests.Response:
    host = url.split("/")[2]
    try:
        with _SLOTS:
            r = requests.get(url, params=params, headers={"User-Agent": _UA}, timeout=15)
    except requests.RequestException as e:
        raise DataError(f"Could not reach {host} ({type(e).__name__}). Try again shortly.")
    if r.status_code in (403, 404):
        # Cboe answers 403, not 404, for a symbol it does not carry, so both mean "unavailable here".
        raise DataError(
            f"{host} has no data for this request: the symbol is not available on this feed. "
            "Use a US-listed stock, ETF or index ticker such as 'NVDA', 'SPY' or 'VIX'."
        )
    if r.status_code >= 400:
        raise DataError(f"{host} returned HTTP {r.status_code}. Try again shortly.")
    return r


def _json(url: str, **params) -> dict:
    try:
        return _fetch(url, **params).json()
    except ValueError:
        raise DataError(f"{url.split('/')[2]} returned an unreadable response. Try again shortly.")


def clean(raw: str) -> str:
    """'$nvda' -> 'NVDA', 'brk-b' -> 'BRK.B'. Rejects company names and unsupported crypto."""
    symbol = re.sub(r"[-/]USD$", "", (raw or "").strip().upper().lstrip("$^"))  # 'BTC-USD' -> 'BTC' so crypto is caught
    symbol = symbol.replace("-", ".")  # share classes: 'BRK-B' -> 'BRK.B'
    if not _TICKER.fullmatch(symbol):
        raise DataError(f"'{raw}' is not a ticker. Pass a symbol like 'NVDA', 'SPY' or 'VIX'.")
    if symbol in _NO_DATA:
        raise DataError(_NO_DATA[symbol])
    if symbol in _CRYPTO:
        raise DataError(
            f"Crypto is not supported, so '{raw}' is rejected (on US exchanges 'BTC' and 'ETH' are unrelated ETFs). "
            "For crypto exposure ask about an ETF such as IBIT (Bitcoin) or ETHA (Ethereum)."
        )
    return symbol


def _cboe(symbol: str) -> str:
    return f"_{symbol}" if symbol in _INDICES else symbol


def _current_price(d: dict) -> float:
    """After the close, current_price can be a late print, so use the official close that change_pct is based on."""
    session_over = d["last_trade_time"][11:16] >= "15:59"
    return d["close"] if session_over else d["current_price"]


def price(symbol: str) -> dict:
    s = clean(symbol)
    # Cboe quote fields: current_price, close, price_change, price_change_percent, bid, ask, open, high, low, volume, iv30, last_trade_time (ET).
    d = _json(f"{_CBOE}/quotes/{_cboe(s)}.json")["data"]
    return {
        "symbol": s,
        "price": _current_price(d),
        "open": d["open"],
        "change": d["price_change"],
        "change_pct": d["price_change_percent"],
        "bid": d["bid"],
        "ask": d["ask"],
        "high": d["high"],
        "low": d["low"],
        "volume": d["volume"],
        "iv30_pct": None if s in _INDICES else d["iv30"],  # for an index this would be the vol of the index itself
        "as_of": d["last_trade_time"],
        "source": "Cboe",
        "delay": "~15 min delayed",
    }


# One lock per symbol, so two threads wanting the same series do not both fetch it. lru_cache is
# thread-safe but does not deduplicate: concurrent callers all miss, all fetch, and all populate.
# That is exactly the pattern parallel researchers produce.
_BAR_LOCKS: dict[str, threading.Lock] = {}
_BAR_LOCKS_GUARD = threading.Lock()


def published_bars(symbol: str) -> tuple[dict, ...]:
    """`_published_bars` with duplicate concurrent fetches collapsed into one.

    The first caller for a symbol fetches and fills the cache; the rest wait on its lock and then
    read the cached value, so a screen run by four researchers costs one fetch per symbol, not four.
    """
    with _BAR_LOCKS_GUARD:
        lock = _BAR_LOCKS.setdefault(symbol, threading.Lock())
    with lock:
        return _published_bars(symbol)


@functools.lru_cache(maxsize=192)
def _published_bars(symbol: str) -> tuple[dict, ...]:
    """Every daily bar Cboe publishes for `symbol`, oldest first.

    Cached because a pair screen fetches each leg once and then every pair, episode and backtest
    re-reads the same series: 25 symbols answer 300 pairs. Bounded generously because several
    researchers screen different universes in one run and eviction would re-fetch what a sibling
    just paid for. Published bars never change, so this is safe for the life of the process; the
    live top-up in price_history() stays uncached.

    Call `published_bars()` instead of this directly - it adds the per-symbol lock that stops
    concurrent callers all missing the cache at once.
    """
    s = clean(symbol)
    # Cboe history rows: date, open, high, low, close, volume. Index rows hold strings ("16.150000") and omit the latest session.
    rows = _json(f"{_CBOE}/charts/historical/{_cboe(s)}.json")["data"]
    bars = [
        {
            "date": r["date"],
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"] or 0),
        }
        for r in rows
    ]
    return tuple(bars)


def price_history(symbol: str, start: date, end: date | None = None) -> list[dict]:
    """Daily bars (date, open, high, low, close, volume) from `start` to `end` (default today), oldest first."""
    s = clean(symbol)
    bars = list(published_bars(s))
    last = (end or date.today()).isoformat()
    if bars and last <= bars[-1]["date"]:
        return [b for b in bars if start.isoformat() <= b["date"] <= last]  # wholly historical
    latest = price(s)  # the published bars stop short of today, so top up from the live quote
    if not bars or latest["as_of"][:10] > bars[-1]["date"]:
        bars.append({
            "date": latest["as_of"][:10],
            "open": latest["open"],
            "high": latest["high"],
            "low": latest["low"],
            "close": latest["price"],
            "volume": latest["volume"],
        })
    return [b for b in bars if start.isoformat() <= b["date"] <= last]



def news(query: str, start: date | None = None, end: date | None = None, limit: int = 20) -> list[dict]:
    """Google News headlines for a ticker or keywords, newest first. No start/end means the last 7 days.

    Pass both dates or neither; tools._optional_range is what validates model-supplied dates.
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
    """One company's SEC submissions record.

    {"name": "LOCKHEED MARTIN CORP", "filings": {"recent": {"form": [...], "filingDate": [...]}}} -
    the filing arrays are parallel, newest first, and hold roughly the last 1000 filings.
    """
    return _json(f"{_SEC}/submissions/CIK{cik:010d}.json")


def cik_for(symbol: str) -> int:
    """SEC CIK for a ticker. Raises if the symbol is not an SEC filer (ETFs and indices are not)."""
    cik = _sec_ciks().get(clean(symbol).replace(".", "-"))  # SEC writes share classes as BRK-B
    if cik is None:
        raise DataError(
            f"'{symbol}' is not an SEC filer, so it has no filings. Indices never file. Most ETFs do "
            "not either, though trust-structured ones such as SPY and QQQ do."
        )
    return cik




def filings(symbol: str, start: date | None = None, end: date | None = None, limit: int = 20) -> list[dict]:
    """SEC 8-K (events), 10-K and 10-Q for a US-listed company, newest first. No dates means the latest.

    Pass both dates or neither; tools._optional_range is what validates model-supplied dates.
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
    # nasdaqlisted.txt: Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
    # otherlisted.txt:  ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
    # Both carry a header line and a trailing "File Creation Time" line; the symbol pattern skips those and preferreds ('JPM$J').
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
