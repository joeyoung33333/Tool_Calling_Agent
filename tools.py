"""The tools the harness can run, and the JSON that describes them to the model.

Naming: every tool takes `symbol` (or `query` for news) and ranges take `start`/`end` (YYYY-MM-DD).
get_price and get_price_history are separate because their output differs; news and filings are one tool each,
where no dates means the latest. Limits and unsupported items are checked here.
"""

import json
import math
import statistics
from datetime import date

import chart
import data
from data import DataError

MAX_HISTORY_DAYS = 3650  # longest price history or filings range
MAX_NEWS_DAYS = 14  # longest news range (Google News returns at most 100 items per query)
CHAIN_STRIKES = 11  # strikes shown around the money in get_option_chain
MAX_CHART_POINTS = 250  # longer ranges are thinned out so the chart stays small


def _date_range(start: str | None, end: str | None, max_days: int) -> tuple[date, date]:
    """Parse and validate a YYYY-MM-DD range; `end` defaults to today."""
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end) if end else date.today()
    except (TypeError, ValueError):
        raise DataError("Dates must be YYYY-MM-DD like '2026-07-06', and 'start' is required.")
    if last > date.today():
        raise DataError(f"'end' {last} is in the future. Use today ({date.today()}) or earlier.")
    if first > last:
        raise DataError(f"'start' {first} is after 'end' {last}.")
    if (last - first).days > max_days:
        raise DataError(f"The range is {(last - first).days} days but the limit is {max_days}. Use a later 'start'.")
    return first, last


def _optional_range(start: str | None, end: str | None, max_days: int) -> tuple[date | None, date | None]:
    """No dates means 'latest'. Otherwise a validated range, where `end` defaults to today."""
    if not start and not end:
        return None, None
    if not start:
        raise DataError("'end' needs a 'start'. Give both dates, or neither for the latest.")
    return _date_range(start, end, max_days)


def _history(symbol: str, start: str, end: str | None) -> tuple[str, date, list[dict]]:
    """Validated symbol, first date and daily bars, shared by get_price_history and get_chart."""
    first, last = _date_range(start, end, MAX_HISTORY_DAYS)
    s = data.clean(symbol)
    bars = data.price_history(s, first, last)
    if len(bars) < 2:
        raise DataError(f"Fewer than 2 daily bars for {s} from {first} to {last}. Widen the range.")
    return s, first, bars


def get_price_history(symbol: str, start: str, end: str | None = None) -> dict:
    s, first, bars = _history(symbol, start, end)
    closes = [b["close"] for b in bars]
    daily_returns = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    # Realized volatility needs at least two returns; 252 trading days a year.
    vol = statistics.stdev(daily_returns) * math.sqrt(252) * 100 if len(daily_returns) > 1 else None
    # The last 5 closes each carry their own day's move, so a one-day jump is never confused with the period return.
    recent = [
        {"date": bars[i]["date"], "close": bars[i]["close"], "change_pct": round((bars[i]["close"] / bars[i - 1]["close"] - 1) * 100, 2)}
        for i in range(max(1, len(bars) - 5), len(bars))
    ]
    out = {
        "symbol": s,
        "start": bars[0]["date"],
        "end": bars[-1]["date"],
        "first_close": closes[0],
        "last_close": closes[-1],
        "return_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
        "period_high": max(b["high"] for b in bars),
        "period_low": min(b["low"] for b in bars),
        "realized_vol_pct": round(vol, 2) if vol is not None else None,
        "last_5_closes": recent,
        "source": "Cboe",
        "delay": "daily bars, split-adjusted; latest session included",
    }
    if (date.fromisoformat(bars[0]["date"]) - first).days > 7:  # e.g. the symbol listed after `start`
        out["note"] = f"There is no data before {bars[0]['date']}, so the range starts there."
    return out


def get_chart(symbol: str, start: str, end: str | None = None) -> dict:
    s, _, bars = _history(symbol, start, end)
    step = math.ceil(len(bars) / MAX_CHART_POINTS)
    thinned = bars[::step]
    if thinned[-1] is not bars[-1]:
        thinned.append(bars[-1])  # always keep the last day
    title = f"{s} daily close, {bars[0]['date']} to {bars[-1]['date']}"
    return {
        "symbol": s,
        "start": bars[0]["date"],
        "end": bars[-1]["date"],
        "daily_bars": len(bars),
        "first_close": bars[0]["close"],
        "last_close": bars[-1]["close"],
        "return_pct": round((bars[-1]["close"] / bars[0]["close"] - 1) * 100, 2),
        "chart": "A daily closing-price line chart is now shown to the user.",
        "chart_svg": chart.line_chart_svg([(b["date"], b["close"]) for b in thinned], title, "Close"),  # for the UI, not the model
    }


def get_option_chain(symbol: str, expiry: str | None = None) -> dict:
    chain = data.option_chain(symbol)
    spot, as_of = chain["spot"], chain["as_of"]
    expiries = sorted({c["expiry"] for c in chain["contracts"] if c["expiry"] > as_of[:10]})
    if not expiries:
        raise DataError(f"No unexpired options found for {chain['symbol']}.")
    if expiry is None:
        expiry = expiries[0]
    elif expiry not in expiries:
        raise DataError(f"No {chain['symbol']} options expire on {expiry}. Available expiries: {', '.join(expiries[:10])}.")

    options = [c for c in chain["contracts"] if c["expiry"] == expiry]
    calls = {c["strike"]: c for c in options if c["type"] == "call"}
    puts = {c["strike"]: c for c in options if c["type"] == "put"}
    if not calls.keys() & puts.keys():
        raise DataError(f"No strikes with both a call and a put for {expiry}. Try another expiry: {', '.join(expiries[:10])}.")
    near = sorted(calls.keys() & puts.keys(), key=lambda k: abs(k - spot))[:CHAIN_STRIKES]  # closest to the money
    strike = near[0]  # at-the-money

    def mid(c):
        return (c["bid"] + c["ask"]) / 2 if c["ask"] else c["last"]

    def put_call_ratio(field):
        calls_total = sum(c[field] for c in options if c["type"] == "call")
        puts_total = sum(c[field] for c in options if c["type"] == "put")
        return round(puts_total / calls_total, 2) if calls_total else None

    def side(c):
        return {k: c[k] for k in ("bid", "ask", "iv_pct", "delta", "open_interest", "volume")}

    straddle = mid(calls[strike]) + mid(puts[strike])
    return {
        "symbol": chain["symbol"],
        "spot": spot,
        "expiry": expiry,
        "days_to_expiry": (date.fromisoformat(expiry) - date.fromisoformat(as_of[:10])).days,
        "atm_strike": strike,
        "atm_iv_pct": round((calls[strike]["iv_pct"] + puts[strike]["iv_pct"]) / 2, 2),
        "atm_straddle": round(straddle, 2),
        "implied_move_pct": round(straddle / spot * 100, 2),  # the straddle price as a % of spot
        "put_call_volume_ratio": put_call_ratio("volume"),
        "put_call_open_interest_ratio": put_call_ratio("open_interest"),
        "chain": [{"strike": k, "call": side(calls[k]), "put": side(puts[k])} for k in sorted(near)],
        "available_expiries": expiries[:10],
        "as_of": as_of,
        "source": "Cboe",
        "delay": "~15 min delayed",
    }


def _feed(key: str, value: str, items: list[dict], window: str, source: str, delay: str) -> dict:
    out = {key: value, "window": window, "items": items, "source": source, "delay": delay}
    if not items:
        out["note"] = "No results in this window. Try a wider window or, for news, different keywords."
    return out


def get_news(query: str, start: str | None = None, end: str | None = None) -> dict:
    first, last = _optional_range(start, end, MAX_NEWS_DAYS)
    window = f"{first} to {last}" if first else "last 7 days"
    delay = "headlines only, no article text; past dates carry a placeholder time, so treat them as date-only"
    return _feed("query", query, data.news(query, first, last), window, "Google News", delay)


def get_filings(symbol: str, start: str | None = None, end: str | None = None) -> dict:
    first, last = _optional_range(start, end, MAX_HISTORY_DAYS)
    s = data.clean(symbol)
    items = data.filings(s, first, last, limit=20 if first else 10)
    return _feed("symbol", s, items, f"{first} to {last}" if first else "latest 10", "SEC EDGAR", "official filings, same day")


# What the model sees: the "set notes" in the screenplay.
def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    parameters = {"type": "object", "properties": properties, "required": required}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


_SYMBOL = {"type": "string", "description": "Ticker, not a company name. E.g. 'NVDA', 'SPY', 'QQQ', 'VIX'."}
_QUERY = {"type": "string", "description": "A ticker like 'NVDA', or keywords like 'Fed rate decision' or 'AI chip export rules'."}
_START = {"type": "string", "description": "First date, YYYY-MM-DD, worked out from today's date (e.g. 90 days ago for 'last 90 days')."}
_END = {"type": "string", "description": "Last date, YYYY-MM-DD. Defaults to today."}
_EXPIRY = {"type": "string", "description": "Expiration date, YYYY-MM-DD. Omit for the nearest; if not listed, the error returns valid dates."}

TOOLS = [
    _tool(
        "get_price",
        "Latest price of ONE US stock, ETF or index (e.g. SPY, VIX): price, change %, bid/ask, day range, volume and "
        "30-day implied volatility. ~15 min delayed, or the last close when the market is shut. For past prices use "
        "get_price_history. No crypto or futures.",
        {"symbol": _SYMBOL},
        ["symbol"],
    ),
    _tool(
        "get_price_history",
        "Daily history of ONE US stock, ETF or index between two dates: return, high/low, realized volatility and the "
        "last 5 closes with each day's change %. return_pct covers the whole period; for one day's move use that "
        "day's change_pct (end the range on that day, start a week earlier). Prices are split-adjusted, so old ones "
        "can differ from old headlines.",
        {"symbol": _SYMBOL, "start": _START, "end": _END},
        ["symbol", "start"],
    ),
    _tool(
        "get_chart",
        "Show the user a daily closing-price line chart of ONE US stock, ETF or index between two dates. Call ONLY "
        "when the user asks for a chart, graph or plot. The image appears in the chat; you get back the first and "
        "last close and the return.",
        {"symbol": _SYMBOL, "start": _START, "end": _END},
        ["symbol", "start"],
    ),
    _tool(
        "get_option_chain",
        "Options on ONE US stock, ETF or index at one expiry: strikes around the money (bid/ask, implied vol, delta, "
        "open interest, volume), ATM implied vol, ATM straddle, the implied move % and put/call ratios. ~15 min "
        "delayed. Use for expected-move and volatility questions.",
        {"symbol": _SYMBOL, "expiry": _EXPIRY},
        ["symbol"],
    ),
    _tool(
        "get_news",
        "Google News headlines for a ticker or keywords, newest first (max 20): title, source, time, link. No dates "
        f"means the last 7 days. With start and end it returns that window (max {MAX_NEWS_DAYS} days) to find what "
        "could have caused a past move; for one day set both to that day. Headlines only, and past items have a date "
        "but no time. Judge relevance yourself: most headlines do not move prices.",
        {"query": _QUERY, "start": _START, "end": _END},
        ["query"],
    ),
    _tool(
        "get_filings",
        "Official SEC filings, newest first: 8-K events (earnings, executive changes, deals) and 10-K/10-Q reports, "
        "each with a filing date and a report_date (period covered, or event date). No dates means the latest 10; with "
        "start and end, up to 20 in that window, going back years. US-listed companies only, not ETFs or indices.",
        {"symbol": _SYMBOL, "start": _START, "end": _END},
        ["symbol"],
    ),
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {
    "get_price": data.price,
    "get_price_history": get_price_history,
    "get_chart": get_chart,
    "get_option_chain": get_option_chain,
    "get_news": get_news,
    "get_filings": get_filings,
}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}"})
    try:
        return json.dumps(TOOL_MAP[name](**args))
    except DataError as e:
        return json.dumps({"error": str(e)})
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{name} failed unexpectedly ({type(e).__name__}). Try different arguments."})
