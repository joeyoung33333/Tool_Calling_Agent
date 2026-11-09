"""Market-data fetchers (free, no API keys): Cboe (stocks, ETFs, indices, options), Google News (headlines),
SEC EDGAR (company filings), the Nasdaq Trader symbol directory (names to tickers) and US Treasury (bill rates).
Equities only.

Public API, keyed by `symbol` (e.g. 'NVDA', 'SPY', 'VIX') or `query`, with `start`/`end` dates for ranges:
    search_symbol(query)                   tickers for a company or fund name
    company_name(symbol)                   the listed name of a stock or ETF
    risk_free_rate(start, end)             average 3-month Treasury bill yield, % a year
    price(symbol)                          latest price snapshot
    price_history(symbol, start, end)      daily bars
    option_chain(symbol)                   latest option chain
    news(query, start, end)                headlines, newest first; no start/end means the last 7 days
    filings(symbol, start, end)            SEC filings, newest first; no start/end means the latest ones
"""

import csv
import functools
import re
import statistics
import xml.etree.ElementTree as ET
from datetime import date, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

_CBOE = "https://cdn.cboe.com/api/global/delayed_quotes"
_NEWS = "https://news.google.com/rss/search"
_SEC = "https://data.sec.gov"
_SEC_TICKERS = "https://www.sec.gov/include/ticker.txt"
_SYMBOLS = "https://www.nasdaqtrader.com/dynamic/symdir"
_TREASURY = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
# SEC asks clients to identify themselves
_UA = "Mozilla/5.0 (compatible; ticker-agent/1.0; Columbia IEOR-4570 class project)"

_TICKER = re.compile(r"[A-Z][A-Z0-9.]{0,9}")  # 'NVDA', 'BRK.B'
_INDICES = {"VIX", "SPX", "NDX", "RUT"}  # Cboe prefixes index symbols with an underscore
# Crypto is not supported. Some of these are also real ETF tickers (BTC is a Grayscale fund), so block them
# rather than let a fund's price pass as the coin's price.
_CRYPTO = {"BTC", "ETH", "XBT", "LTC", "SOL", "XRP", "DOGE", "ADA", "BITCOIN", "ETHEREUM"}

# Nicknames that are not in legal names; and directory rows that are not the stock itself
_NICKNAMES = {"google": "alphabet", "youtube": "alphabet", "facebook": "meta platforms"}
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


class DataError(Exception):
    """A failure whose message is written for the model: what went wrong and what to try."""


def _fetch(url: str, **params) -> requests.Response:
    host = url.split("/")[2]
    try:
        r = requests.get(url, params=params, headers={"User-Agent": _UA}, timeout=15)
    except requests.RequestException as e:
        raise DataError(f"Could not reach {host} ({type(e).__name__}). Try again shortly.")
    if r.status_code in (403, 404):  # Cboe answers 403 for a symbol it does not know
        raise DataError(f"{host} has no data for this request. If you gave a ticker, use a US-listed stock, ETF or index like 'NVDA', 'SPY' or 'VIX'. If you gave a company name, call search_symbol to find its ticker.")
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
        raise DataError(f"'{raw}' is not a ticker. Pass a symbol like 'NVDA', 'SPY' or 'VIX'. For a company name, call search_symbol first.")
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
    # Cboe quote: {"data": {"symbol": "NVDA", "current_price": 234.22, "close": 233.95, "price_change": 3.09,
    #   "price_change_percent": 1.3208, "bid": 234.22, "ask": 234.24, "high": 237.88, "low": 233.6,
    #   "volume": 135167817, "iv30": 28.966, "last_trade_time": "2026-10-02T15:59:59"}}   times are US Eastern
    d = _json(f"{_CBOE}/quotes/{_cboe(s)}.json")["data"]
    return {
        "symbol": s,
        "price": _current_price(d),
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


def price_history(symbol: str, start: date, end: date | None = None) -> list[dict]:
    """Daily bars (date, high, low, close, volume) from `start` to `end` (default today), oldest first."""
    s = clean(symbol)
    # Cboe history: {"data": [{"date": "2026-10-01", "open": 230.0, "high": 232.29, "low": 228.16, "close": 230.86,
    #   "volume": 98591411}, ...]}   index rows hold strings ("16.150000") and the latest session is missing
    rows = _json(f"{_CBOE}/charts/historical/{_cboe(s)}.json")["data"]
    bars = [{"date": r["date"], "high": float(r["high"]), "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"] or 0)} for r in rows]
    latest = price(s)  # add the latest session from the live quote
    if not bars or latest["as_of"][:10] > bars[-1]["date"]:
        bars.append({"date": latest["as_of"][:10], "high": latest["high"], "low": latest["low"], "close": latest["price"], "volume": latest["volume"]})
    last = (end or date.today()).isoformat()
    return [b for b in bars if start.isoformat() <= b["date"] <= last]


def option_chain(symbol: str) -> dict:
    s = clean(symbol)
    # Cboe options: {"data": {"current_price": 234.22, "close": 233.95, "last_trade_time": "2026-10-02T15:59:59",
    #   "options": [{"option": "NVDA261007C00235000", "bid": 2.19, "ask": 2.25, "iv": 0.2442, "delta": 0.4442,
    #   "open_interest": 2925.0, "volume": 120.0, "last_trade_price": 2.2}, ...]}}   option = root + YYMMDD + C/P + strike*1000
    d = _json(f"{_CBOE}/options/{_cboe(s)}.json")["data"]
    contracts = []
    for o in d["options"]:
        m = re.match(r"^.+?(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", o["option"])
        if m:
            contracts.append({
                "expiry": f"20{m[1]}-{m[2]}-{m[3]}",
                "type": "call" if m[4] == "C" else "put",
                "strike": int(m[5]) / 1000,
                "bid": o["bid"],
                "ask": o["ask"],
                "last": o["last_trade_price"],
                "iv_pct": round(o["iv"] * 100, 2),
                "delta": o["delta"],
                "open_interest": o["open_interest"],
                "volume": o["volume"],
            })
    return {"symbol": s, "spot": _current_price(d), "as_of": d["last_trade_time"], "contracts": contracts}


def news(query: str, start: date | None = None, end: date | None = None, limit: int = 20) -> list[dict]:
    """Google News headlines for a ticker or keywords, newest first. No start/end means the last 7 days."""
    if (start is None) != (end is None):
        raise DataError("Pass both start and end for a date range, or neither for the last 7 days.")
    q = (query or "").strip()
    if not q:
        raise DataError("'query' is empty. Pass a ticker like 'NVDA' or keywords like 'Fed rate decision'.")
    if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", q):
        q += " stock"  # a bare ticker is ambiguous, so bias towards market news
    # Google's `before:` is exclusive, so add a day to include `end`.
    q += f" after:{start} before:{end + timedelta(days=1)}" if start else " when:7d"
    # Google News RSS (up to 100 items): <item><title>NVIDIA Announces ... - NVIDIA Newsroom</title><link>https://news.google.com/rss/articles/CBMi...</link>
    #   <pubDate>Mon, 28 Sep 2026 17:02:51 GMT</pubDate><source url="https://nvidianews.nvidia.com">NVIDIA Newsroom</source></item>
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


def filings(symbol: str, start: date | None = None, end: date | None = None, limit: int = 20) -> list[dict]:
    """SEC 8-K (events), 10-K and 10-Q (reports) for a US-listed company, newest first. No start/end means the latest."""
    s = clean(symbol)
    if (start is None) != (end is None):
        raise DataError("Pass both start and end for a date range, or neither for the latest filings.")
    cik = _sec_ciks().get(s.replace(".", "-"))  # SEC writes share classes as BRK-B
    if cik is None:
        raise DataError(f"No SEC filings for '{s}'. Only companies file 8-Ks; ETFs and indices don't.")
    # SEC submissions: {"filings": {"recent": {"accessionNumber": ["0001045810-26-000123", ...], "filingDate": ["2026-08-26", ...],
    #   "form": ["10-Q", ...], "items": ["2.02,9.01", ...], "reportDate": ["2026-07-26", ...], "primaryDocument": [...]}}}   parallel arrays, newest first, ~1000 filings
    recent = _json(f"{_SEC}/submissions/CIK{cik:010d}.json")["filings"]["recent"]
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
def _bill_rates(year: int) -> dict[str, float]:
    """Date -> 3-month Treasury bill yield (% a year) for one year."""
    # Treasury par yield curve CSV, one file per year: 'Date,"1 Mo","3 Mo",...' then '10/07/2026,4.07,4.22,...'
    # (the columns differ between years, so read the "3 Mo" column by name). A restart refreshes the current year.
    text = _fetch(f"{_TREASURY}/{year}/all", type="daily_treasury_yield_curve", field_tdr_date_value=year, page="", _format="csv").text
    rates = {}
    for row in csv.DictReader(text.splitlines()):
        if row.get("3 Mo"):
            month, day, y = row["Date"].split("/")
            rates[f"{y}-{month}-{day}"] = float(row["3 Mo"])
    return rates


def risk_free_rate(start: date, end: date) -> float:
    """Average 3-month Treasury bill yield in % a year from `start` to `end`."""
    rates = [r for year in range(start.year, end.year + 1) for day, r in _bill_rates(year).items() if start.isoformat() <= day <= end.isoformat()]
    if not rates:
        raise DataError(f"No Treasury bill rates found from {start} to {end}.")
    return statistics.fmean(rates)


@functools.cache
def _symbol_directory() -> list[tuple[str, str, bool]]:
    """(symbol, name, is_etf) for every listed stock and ETF, without test issues, warrants, rights, units or preferreds."""
    # nasdaqlisted.txt: "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
    # otherlisted.txt:  "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"
    # Each starts with a header line and ends with a "File Creation Time" line; the symbol pattern skips that and preferreds ('JPM$J').
    rows = []
    for name, etf_column, test_column in (("nasdaqlisted", 6, 3), ("otherlisted", 4, 6)):
        for line in _fetch(f"{_SYMBOLS}/{name}.txt").text.splitlines()[1:]:
            f = line.split("|")
            if len(f) > 6 and _TICKER.fullmatch(f[0]) and f[test_column] != "Y" and not _NOT_STOCK.search(f[1]):
                rows.append((f[0], f[1], f[etf_column] == "Y"))
    return rows


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())  # 'JP Morgan' and 'JPMorgan' both become 'jpmorgan'


def company_name(symbol: str) -> str | None:
    """Listed name of a stock or ETF, or None (indices are not in the directory)."""
    name = next((name for s, name, _ in _symbol_directory() if s == symbol), None)
    return name and re.sub(r"\s+(-\s+)?(New\s+)?(Common|Capital)\s+Stock\s*$", "", name)  # 'NVIDIA Corporation - Common Stock' -> 'NVIDIA Corporation'


def search_symbol(query: str) -> dict:
    """Up to 5 tickers for a company or fund name, best first: an exact ticker, then names starting with the query, then names containing it."""
    key = _squash(_NICKNAMES.get(_squash(query), query))
    if not key:
        raise DataError("Give a company or fund name to search for, like 'Nvidia' or 'S&P 500 ETF'.")
    if key.upper() in _CRYPTO:
        clean(key)  # raises the crypto message; 'BTC' is a real ETF ticker, so it must not pass as bitcoin
    ticker = query.strip().upper().replace("-", ".")
    found = []
    for symbol, name, etf in _symbol_directory():
        squashed = _squash(name)
        if symbol == ticker or key in squashed:
            found.append(((symbol != ticker, not squashed.startswith(key), etf, len(squashed)), symbol, name, etf))  # stocks before ETFs, shorter first
    if not found:
        raise DataError(f"No US-listed stock or ETF matches '{query.strip()[:60]}'. Try fewer words, such as the company's main name. Non-US companies and crypto are not covered.")
    matches = [{"symbol": symbol, "name": name, "type": "ETF" if etf else "stock"} for _, symbol, name, etf in sorted(found)[:5]]
    return {"query": query.strip(), "matches": matches, "source": "Nasdaq Trader symbol directory"}
