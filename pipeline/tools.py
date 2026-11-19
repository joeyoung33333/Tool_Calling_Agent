"""The tools the agent can call.

Bodies are plain Python; the @function_tool docstrings are the descriptions the model reads.
"""

import json
import math
import statistics
from datetime import date, timedelta

from agents import RunContextWrapper, function_tool

from pipeline import chart
import config
from pipeline.runtime import RunContext
from quant import backtest, pairs, screen
from sources import data, refdata
from sources.data import DataError

MAX_HISTORY_DAYS = config.MAX_LOOKBACK_DAYS  # the price feed's reach, defined once
MAX_NEWS_DAYS = 14  # longest news range (Google News returns at most 100 items per query)
MAX_CHART_POINTS = 250  # longer ranges are thinned out so the chart stays small
ALSO_RANKED_SHOWN = 20  # the ranked tail is context, not reading; the full count is reported


# --- Shared helpers ---


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
    """Validated symbol, first date and daily bars; the one 'too few bars' guard."""
    first, last = _date_range(start, end, MAX_HISTORY_DAYS)
    s = data.clean(symbol)
    bars = data.price_history(s, first, last)
    if len(bars) < 2:
        raise DataError(f"Fewer than 2 daily bars for {s} from {first} to {last}. Widen the range.")
    return s, first, bars


def _period(symbol: str, bars: list[dict]) -> dict:
    """What every range tool reports: the days covered, the first and last close and the return."""
    first_close, last_close = bars[0]["close"], bars[-1]["close"]
    return {
        "symbol": symbol,
        "start": bars[0]["date"],
        "end": bars[-1]["date"],
        "first_close": first_close,
        "last_close": last_close,
        "return_pct": round((last_close / first_close - 1) * 100, 2),
    }


def _feed(key: str, value: str, items: list[dict], window: str, source: str, delay: str) -> dict:
    """The shape every headline/filing feed returns, including the 'nothing found' note."""
    out = {key: value, "window": window, "items": items, "source": source, "delay": delay}
    if not items:
        out["note"] = "No results in this window. Try a wider window or, for news, different keywords."
    return out


def _run(fn, ctx=None, **kwargs) -> str:
    """Run a tool body and return JSON, never raising. A `(summary, figure)` result sends the figure
    to the page via `ctx`; only the summary reaches the model.
    """
    try:
        out = fn(**kwargs)
        if isinstance(out, tuple):  # no tool's model payload is a tuple, so this is unambiguous
            out, figure = out
            # No context threaded in: drop the figure, not the answer.
            if ctx is not None and getattr(ctx, "context", None) is not None:
                label = out.get("symbol") or out.get("pair") or out.get("chart_label")
                agent = getattr(getattr(ctx, "agent", None), "name", "")
                ctx.context.draw(label, figure, agent)
        return json.dumps(out, default=str)
    except DataError as e:
        # A feed 429 counts toward the run's rate-limit stop, like a model 429.
        if "429" in str(e) and ctx is not None and getattr(ctx, "context", None) is not None:
            ctx.context.note_rate_limit()
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"This tool failed unexpectedly ({type(e).__name__}). Try different "
                           f"arguments once, then report the data as unavailable."})


# --- Tool bodies (plain Python, no SDK) ---


def _price_history(symbol: str, start: str, end: str | None = None) -> tuple[dict, dict]:
    s, first, bars = _history(symbol, start, end)
    closes = [b["close"] for b in bars]
    vol = pairs.annual_vol(pairs.returns(closes)) if len(bars) > 2 else None  # needs 2+ returns
    # Per-day moves, so a one-day jump is not mistaken for the period return.
    recent = [
        {
            "date": bars[i]["date"],
            "close": bars[i]["close"],
            "change_pct": round((bars[i]["close"] / bars[i - 1]["close"] - 1) * 100, 2),
        }
        for i in range(max(1, len(bars) - 5), len(bars))
    ]
    out = {
        **_period(s, bars),
        "period_high": max(b["high"] for b in bars),
        "period_low": min(b["low"] for b in bars),
        "realized_vol_pct": round(vol, 2) if vol is not None else None,
        "last_5_closes": recent,
        "source": "Cboe",
        "delay": "daily bars, split-adjusted; latest session included",
    }
    if (date.fromisoformat(bars[0]["date"]) - first).days > 7:  # e.g. the symbol listed after `start`
        out["note"] = f"There is no data before {bars[0]['date']}, so the range starts there."
    out["table"] = "A month-end price table is now shown to the user."
    return out, _month_end_table(s, bars)


def _month_end_table(symbol: str, bars: list[dict]) -> dict:
    """The last close of each month, as a table (the first month's change is blank)."""
    months: dict[str, dict] = {}
    for bar in bars:
        months[bar["date"][:7]] = bar  # later bars overwrite, so each month keeps its last
    rows, previous = [], None
    for month, bar in list(months.items())[-13:]:
        change = f"{(bar['close'] / previous - 1) * 100:+.2f}%" if previous else "-"
        rows.append([month, f"{bar['close']:,.2f}", change])
        previous = bar["close"]
    return chart.table(["Month", "Close", "Change"], rows, f"{symbol} month-end closes")


def _chart(symbol: str, start: str, end: str | None = None) -> tuple[dict, dict]:
    """Returns (summary for the model, Plotly figure spec for the page)."""
    s, _, bars = _history(symbol, start, end)
    step = math.ceil(len(bars) / MAX_CHART_POINTS)
    thinned = bars[::step]
    if thinned[-1] is not bars[-1]:  # identity: the last bar either is or is not already in the slice
        thinned.append(bars[-1])  # always keep the last day
    title = f"{s} daily close"  # no date range: it overflows a phone, and the x axis already says it
    summary = {
        **_period(s, bars),
        "daily_bars": len(bars),
        "chart": "A daily closing-price line chart is now shown to the user.",
        "source": "Cboe",
        "delay": "daily bars, split-adjusted; latest session included",
    }
    figure = chart.line(f"{s} close", [(b["date"], b["close"]) for b in thinned], title, "Close")
    return summary, figure


def _news(query: str, start: str | None = None, end: str | None = None) -> dict:
    first, last = _optional_range(start, end, MAX_NEWS_DAYS)
    window = f"{first} to {last}" if first else "last 7 days"
    delay = "headlines only, no article text; past dates carry a placeholder time, so treat them as date-only"
    return _feed("query", query, data.news(query, first, last), window, "Google News", delay)


def _filings(symbol: str, start: str | None = None, end: str | None = None) -> dict:
    first, last = _optional_range(start, end, MAX_HISTORY_DAYS)
    s = data.clean(symbol)
    items = data.filings(s, first, last, limit=20 if first else 10)
    window = f"{first} to {last}" if first else "latest 10"
    return _feed("symbol", s, items, window, "SEC EDGAR", "official filings, same day")


# --- Pair research ---


def _rules(entry_z: float | None, exit_z: float | None) -> backtest.Rules:
    """Backtest rules from config defaults plus overrides; a bad threshold raises DataError."""
    try:
        return backtest.Rules(
            entry_z=entry_z if entry_z is not None else config.ENTRY_Z,
            exit_z=exit_z if exit_z is not None else config.EXIT_Z,
            stop_z=config.STOP_Z,
            max_hold=config.MAX_HOLD_SESSIONS,
            cost_bps=config.COST_BPS,
            borrow_bps_annual=config.BORROW_BPS_ANNUAL,
            in_sample_frac=config.IN_SAMPLE_FRAC,
        )
    except ValueError as e:
        raise DataError(str(e)) from e


def _pair_series(a: str, b: str, lookback_days: int | None) -> tuple[str, str, list[dict], list[dict]]:
    """Validated symbols and their daily bars over the lookback window."""
    first, second = data.clean(a), data.clean(b)
    if first == second:
        raise DataError(f"{first} cannot be paired with itself. Give two different symbols.")
    days = screen.clamp_lookback(lookback_days)
    start = date.today() - timedelta(days=days)
    return first, second, data.price_history(first, start), data.price_history(second, start)


def _analyze_pair(
    symbol_a: str,
    symbol_b: str,
    lookback_days: int | None = None,
    entry_z: float | None = None,
    exit_z: float | None = None,
) -> tuple[dict, dict]:
    """Full workup of one pair: statistics, holdout backtest, divergence episodes, and a chart."""
    rules = _rules(entry_z, exit_z)  # before the fetches, so a bad threshold costs nothing
    a, b, hist_a, hist_b = _pair_series(symbol_a, symbol_b, lookback_days)
    dates, closes_a, closes_b = pairs.align(hist_a, hist_b)
    if len(dates) < config.MIN_OVERLAP_SESSIONS:
        raise DataError(
            f"{a} and {b} share only {len(dates)} sessions, and at least "
            f"{config.MIN_OVERLAP_SESSIONS} are needed. One of them may be newly listed."
        )
    cut = backtest.split(dates, rules)
    # In-sample only, matching screen.fit_pair, so it compares with the holdout.
    stats = pairs.summary(dates[:cut], closes_a[:cut], closes_b[:cut])
    # refdata is the table the screen prices, so the name matches the universe.
    out = {"pair": f"{a}/{b}", "symbol_a": a, "symbol_b": b,
           "name_a": refdata.name_for(a), "name_b": refdata.name_for(b), **stats,
           "measured_on": f"in-sample window, {dates[0]} to {dates[cut - 1]} ({cut} sessions)",
           "holdout_sessions": len(dates) - cut,
           # Here too: an untradeable pair returns before any backtest runs.
           "limitations": backtest.LIMITATIONS,
           "source": "Cboe", "delay": "daily bars, split-adjusted; price return only (no dividends)"}

    if not stats["tradeable"]:
        # Still chart it, rebuilt from the unrounded fit, not stats' display-rounded values.
        beta = pairs.hedge_ratio(closes_a[:cut], closes_b[:cut])
        spread = pairs.spread(closes_a, closes_b, beta)
        mean, sd = statistics.fmean(spread[:cut]), statistics.pstdev(spread[:cut])
        z = pairs.zscores(spread, mean, sd) if sd > pairs.FLAT else [0.0] * len(spread)
        out["current_z"], out["current_z_as_of"] = round(z[-1], 2), dates[-1]
        figure = chart.zscore_panel(list(zip(dates, z)), rules.entry_z, rules.exit_z,
                                    f"{a}/{b} spread z-score - not tradeable",
                                    note=_zscore_note(a, b, rules, stats["not_tradeable_because"]))
        return out, figure

    result = backtest.evaluate(hist_a, hist_b, rules=rules, trials=1)
    out["backtest"] = {k: result[k] for k in ("fit", "rules", "in_sample", "holdout", "limitations")
                       if k in result}
    out["recent_trades"] = result["trades"][-5:]

    # Scored against the in-sample fit the backtest traded on, so episodes, bands and trades agree.
    fit = backtest.fit(dates[:cut], closes_a[:cut], closes_b[:cut])
    z = pairs.zscores(pairs.spread(closes_a, closes_b, fit.beta), fit.mean, fit.sd)
    found = pairs.episodes(dates, z, config.DIVERGENCE_Z, config.DIVERGENCE_MIN_DAYS)
    # Today's z against the frozen fit, dated because the rest describes the in-sample window.
    out["current_z"], out["current_z_as_of"] = round(z[-1], 2), dates[-1]
    out["divergence_episodes"] = found[-config.MAX_EPISODES_RESEARCHED:]
    out["episode_count"] = len(found)
    if found:
        out["next_step"] = ("Call research_divergence for an episode's exact window to find what "
                            "caused it: a temporary dislocation or a broken relationship.")
    figure = chart.zscore_panel(list(zip(dates, z)), rules.entry_z, rules.exit_z,
                                f"{a}/{b} spread z-score", episodes=found, note=_zscore_note(a, b, rules))
    return out, figure


def _zscore_note(a: str, b: str, rules: backtest.Rules, refused: str | None = None) -> str:
    """The caption under a z-score panel: what the line is and when a trade opens and closes."""
    fitted = f"{rules.in_sample_frac:.0%}"
    what = (f"How far the spread {a} − β·{b} sits from its mean, in standard deviations, with β, "
            f"the mean and the standard deviation fitted on the first {fitted}.")
    if refused:
        return f"{what} Not traded: {refused.split(':')[0].rstrip('.')}. The bands are for reference."
    return (f"{what} A trade opens beyond ±{rules.entry_z:g} (dashed), closes inside "
            f"±{rules.exit_z:g} (dotted), and is cut at ±{rules.stop_z:g} or after "
            f"{rules.max_hold} sessions. Shaded spans stayed beyond ±{config.DIVERGENCE_Z:g} for "
            f"{config.DIVERGENCE_MIN_DAYS}+ sessions.")


def _screen_pairs(
    sector: str | None = None,
    etf_family: str | None = None,
    symbols: list[str] | None = None,
    top_k: int | None = None,
    lookback_days: int | None = None,
) -> tuple[dict, dict]:
    """Resolve a universe, fit every combination in it, rank them, and backtest the best.

    Returns (summary, figure): the ranked table is drawn for the page, never returned to the model.
    """
    try:
        universe = refdata.resolve_universe(sector=sector, etf_family=etf_family, symbols=symbols)
    except refdata.RefDataError as e:
        raise DataError(str(e)) from e  # these messages tell the model how to recover
    try:
        # screen.screen clamps this itself, so there is one clamp on the path.
        out = screen.screen(universe["symbols"], lookback_days=lookback_days, top_k=top_k)
    except screen.ScreenError as e:
        raise DataError(str(e)) from e  # same reasoning as resolve_universe above
    out["ranked"] = [{**c, "backtest": _compact(c["backtest"])} for c in out["ranked"]]
    # Copied before the pops below, which would otherwise empty it.
    reference = next((c["backtest"] for c in out["ranked"] if c["backtest"].get("tradeable")), {})
    shared = {"rules": reference.get("rules"), "limitations": backtest.LIMITATIONS}
    shared["deflation_note"] = (reference.get("holdout") or {}).get("deflation_note")
    for candidate in out["ranked"]:
        for key in ("rules", "limitations", "trials_considered"):
            candidate["backtest"].pop(key, None)
        (candidate["backtest"].get("holdout") or {}).pop("deflation_note", None)
    if len(out["also_ranked"]) > ALSO_RANKED_SHOWN:
        out["also_ranked_count"] = len(out["also_ranked"])
        out["also_ranked"] = out["also_ranked"][:ALSO_RANKED_SHOWN]
    summary = {"universe_sql": universe["sql"], "universe_note": universe.get("note"),
               "universe_truncated": universe.get("truncated", False),
               "universe_available": universe.get("available"), **out,
               "backtest_rules": shared.get("rules"),
               "limitations": shared.get("limitations"),
               "deflation_note": shared.get("deflation_note"),
               # Same precedence as resolve_universe, so an ignored junk `sector` cannot title it.
               "chart_label": ("selected symbols" if symbols else etf_family or sector or "universe"),
               "source": "Cboe daily bars; universe from the committed reference database",
               "detail": "Call analyze_pair on any pair above for its episodes, trades and chart."}
    return summary, _ranked_table(summary)


# Headline figures, in the order a reader scans them, each with the line printed under its name.
SCREEN_ROWS = [
    ("Pair", "A − β·B", lambda c, h: f"{c['a']}/{c['b']}"),
    ("Half-life", "days to ½ gap", lambda c, h: _cell(c.get("half_life_days"), "{:.1f}d")),
    ("Beta stab.", "0 = steady", lambda c, h: _cell(c.get("beta_stability"), "{:.2f}")),
    ("Corr", "of returns", lambda c, h: _cell(c.get("correlation"), "{:.2f}")),
    ("Trades", "round trips", lambda c, h: _cell(h.get("trades"), "{:d}")),
    ("Exposure", "% in trade", lambda c, h: _cell(h.get("exposure_pct"), "{:.0f}%")),
    ("Return", "after costs", lambda c, h: _cell(h.get("total_return_pct"), "{:+.2f}%")),
    ("Sharpe", "annualized", lambda c, h: _cell(h.get("sharpe"), "{:.2f}")),
    ("Deflated", "less luck", lambda c, h: _cell(h.get("deflated_sharpe"), "{:+.2f}")),
]


def _cell(value, fmt: str) -> str:
    """One table cell, or an em dash where the statistic does not exist."""
    return fmt.format(value) if value is not None else "-"


def _ranked_table(summary: dict) -> dict:
    """The backtested pairs as a table for the page; refused pairs carry their reason."""
    rows = []
    for candidate in summary.get("ranked", []):
        result = candidate.get("backtest") or {}
        if not result.get("tradeable"):
            # Widths follow SCREEN_ROWS, so adding a column cannot silently shorten a row.
            rows.append([f"{candidate['a']}/{candidate['b']}", "untradeable",
                         *["-"] * (len(SCREEN_ROWS) - 2)])
            continue
        holdout = result.get("holdout") or {}
        rows.append([build(candidate, holdout) for _, _, build in SCREEN_ROWS])
    fitted = f"{config.IN_SAMPLE_FRAC:.0%}"
    held = f"{1 - config.IN_SAMPLE_FRAC:.0%}"
    note = (f"Pairs are fitted and ranked on the first {fitted} of the window, then traded on the last "
            f"{held} with that fit unchanged: in beyond ±{config.ENTRY_Z:g}σ, out inside "
            f"±{config.EXIT_Z:g}σ, cut at ±{config.STOP_Z:g}σ or after {config.MAX_HOLD_SESSIONS} "
            f"sessions, net of costs. Deflated is the Sharpe less what the luckiest of the "
            f"{summary.get('trials_considered') or ''} pairs backtested would post by chance; "
            f"only a positive one counts.")
    return chart.table(
        [name for name, _, _ in SCREEN_ROWS],
        rows,
        f"{summary.get('chart_label', 'screen')}: {summary.get('pairs_screened', 0)} pairs screened, "
        f"top {len(rows)} backtested on the holdout",
        groups=[(f"IN-SAMPLE FIT · FIRST {fitted}", 4), (f"HOLDOUT BACKTEST · LAST {held}", 5)],
        subtitles=[sub for _, sub, _ in SCREEN_ROWS],
        note=note,
    )


def _compact(result: dict) -> dict:
    """A backtest without its per-session arrays: chart data, ~52k tokens per screen."""
    if not result.get("tradeable"):
        return result
    return {k: v for k, v in result.items() if k not in ("equity", "trades")}


def _research_divergence(symbol_a: str, symbol_b: str, start: str, end: str) -> dict:
    """News and filings for both legs over exactly one divergence window."""
    a, b = data.clean(symbol_a), data.clean(symbol_b)
    first, last = _date_range(start, end, MAX_HISTORY_DAYS)
    # News is capped shorter, so a long episode gets the slice from its *start*, where its cause is.
    news_to = min(last, first + timedelta(days=MAX_NEWS_DAYS))
    out = {"pair": f"{a}/{b}", "window": f"{first} to {last}", "legs": {}}
    for symbol in (a, b):
        leg = {"news_window": f"{first} to {news_to}",
               "news": data.news(symbol, first, news_to, limit=10)}
        try:
            leg["filings"] = data.filings(symbol, first, last, limit=10)
        except DataError as e:
            leg["filings"] = []
            leg["filings_note"] = str(e)
        out["legs"][symbol] = leg
    if news_to < last:
        out["note"] = (f"News is limited to {MAX_NEWS_DAYS} days, so headlines cover "
                       f"{first} to {news_to} - the start of the episode, where its cause is - "
                       f"while filings cover the whole window.")
    out["how_to_read"] = (
        "An 8-K item code states what happened and is dated, so it outweighs a headline. A merger, "
        "spin-off, guidance reset or auditor change suggests the relationship itself changed; "
        "sector news or a broad sell-off suggests a dislocation that should close."
    )
    out["source"] = "Google News, SEC EDGAR"
    return out


def _reference_schema() -> dict:
    """The reference table's columns and the real values of its categorical columns."""
    card = refdata.schema()
    return {
        **card,
        "enum_values": {c: refdata.enum_values(c) for c in ("gics_sector", "etf_family", "type")},
        "hint": ("screen_pairs takes a gics_sector or gics_sub_industry value directly as "
                 "`sector`; query_reference is for a universe that filter cannot express."),
        "source": "committed reference database (data/refdata.duckdb)",
    }


def _reference_query(sql: str) -> dict:
    """Run one read-only SELECT against the reference table."""
    try:
        return refdata.query(sql)
    except refdata.RefDataError as e:
        raise DataError(str(e))


# --- The tools the model sees: @function_tool builds each schema from signature and docstring ---


@function_tool
def get_price_history(
    ctx: RunContextWrapper[RunContext], symbol: str, start: str, end: str | None = None
) -> str:
    """Daily prices for one symbol: period return, high, low, realized volatility and the last five
    closes, plus a month-end price table shown to the user. The last bar is today's price, about 15
    minutes delayed, so this also answers "where is it trading now?".

    Args:
        symbol: Ticker, not a company name. E.g. 'NVDA', 'SPY', or the index 'VIX', 'SPX' or 'RUT'.
        start: First date, YYYY-MM-DD, worked out from today (e.g. 365 days ago for 'last year').
            At most 10 years back.
        end: Last date, YYYY-MM-DD. Defaults to today.
    """
    return _run(_price_history, ctx, symbol=symbol, start=start, end=end)


@function_tool
def get_chart(
    ctx: RunContextWrapper[RunContext], symbol: str, start: str, end: str | None = None
) -> str:
    """Draw a daily closing-price line chart for one symbol, shown to the user. Call it only when a
    chart is wanted; it also returns the period return.

    Args:
        symbol: Ticker to chart. E.g. 'AAPL'.
        start: First date, YYYY-MM-DD, worked out from today. At most 10 years back.
        end: Last date, YYYY-MM-DD. Defaults to today.
    """
    # `ctx` is excluded from the schema the model sees; _run routes the figure to the page with it.
    return _run(_chart, ctx, symbol=symbol, start=start, end=end)


@function_tool
def get_news(query: str, start: str | None = None, end: str | None = None) -> str:
    """News headlines for a ticker or keywords, newest first. Headlines only, no article text.

    Omit both dates for the last 7 days. A dated range is at most 14 days long.

    Args:
        query: A ticker like 'NVDA', or keywords like 'Fed rate decision'.
        start: First date, YYYY-MM-DD. Omit both dates for the last 7 days.
        end: Last date, YYYY-MM-DD. Defaults to today when `start` is given.
    """
    return _run(_news, query=query, start=start, end=end)


@function_tool
def get_filings(symbol: str, start: str | None = None, end: str | None = None) -> str:
    """SEC filings for a US-listed company, newest first: 8-K events with their item codes, plus
    10-K and 10-Q. An 8-K's item code says what actually happened, so a filing is stronger evidence
    than a headline.

    Indices never file these forms and most ETFs do not either; commodity trusts such as GLD do.
    Omit both dates for the latest 10 filings; a dated range returns up to 20.

    Args:
        symbol: Ticker of a company. E.g. 'LMT'.
        start: First date, YYYY-MM-DD. Omit both dates for the latest.
        end: Last date, YYYY-MM-DD. Defaults to today when `start` is given.
    """
    return _run(_filings, symbol=symbol, start=start, end=end)


@function_tool
def analyze_pair(
    ctx: RunContextWrapper[RunContext],
    symbol_a: str,
    symbol_b: str,
    lookback_days: int | None = None,
    entry_z: float | None = None,
    exit_z: float | None = None,
) -> str:
    """Test one pair: correlation, hedge ratio, half-life, beta stability, a holdout backtest and
    divergence episodes, plus a z-score chart of the spread shown to the user.

    The holdout is the figure that matters: the hedge ratio and bands are fitted on the earlier 70%
    of the window and reused unchanged on the later 30%, so a pair that is strong in-sample and weak
    out-of-sample was fitted, not discovered. Only one pair is tested, so its Sharpe is not
    deflated; use screen_pairs to compare several.

    Args:
        symbol_a: First leg. E.g. 'LMT'.
        symbol_b: Second leg. E.g. 'RTX'.
        lookback_days: History to use. Default 1095 (3 years); held between 400 and 3650.
        entry_z: Open a trade at this many standard deviations. Default 2.0; must be below 4.0,
            where a trade is stopped out.
        exit_z: Close it inside this many. Default 0.5; must be below entry_z.
    """
    return _run(_analyze_pair, ctx, symbol_a=symbol_a, symbol_b=symbol_b,
                lookback_days=lookback_days, entry_z=entry_z, exit_z=exit_z)


@function_tool
def screen_pairs(
    ctx: RunContextWrapper[RunContext],
    sector: str | None = None,
    etf_family: str | None = None,
    symbols: list[str] | None = None,
    top_k: int | None = None,
    lookback_days: int | None = None,
) -> str:
    """Screen every pair in a universe, rank them, and backtest the best few on the holdout. The
    ranked table is shown to the user.

    Every combination is fitted in Python on the earlier 70% of the window, so the ranking is
    arithmetic rather than judgement. Pairs with an unusable hedge ratio or a spread that does not
    mean-revert are rejected with a reason. Each backtested pair's Sharpe is deflated for the number
    of pairs backtested, so the screen, not a later analyze_pair call, decides which pairs are
    viable; analyze_pair afterwards adds one pair's episodes and trades.

    Pass exactly one of sector, etf_family or symbols. The universe is capped at 40 names and five
    GICS sectors are larger, so a result may carry universe_truncated=true with universe_available
    and a note. Then report a subset, not the sector.

    Args:
        sector: A GICS sector or sub-industry, as named in the reference table. E.g. 'Energy',
            'Semiconductors'.
        etf_family: A curated ETF family. E.g. 'nasdaq_leveraged', 'sector_spdr'.
        symbols: An explicit list of 2 to 40 tickers. E.g. ['LMT', 'RTX', 'NOC', 'GD'].
        top_k: How many of the best pairs to backtest. Default 8, maximum 20.
        lookback_days: History to use. Default 1095 (3 years); held between 400 and 3650.
    """
    return _run(_screen_pairs, ctx, sector=sector, etf_family=etf_family, symbols=symbols,
                top_k=top_k, lookback_days=lookback_days)


@function_tool
def research_divergence(symbol_a: str, symbol_b: str, start: str, end: str) -> str:
    """News and SEC filings for both legs over one divergence window, to explain why it happened.

    Call this with the exact start and end of an episode that analyze_pair reported. It answers the
    question the statistics cannot: was the gap a temporary dislocation that later closed, or did
    the relationship between the two names actually break? Filings cover the whole window; news
    covers its first 14 days, where the cause usually is.

    Args:
        symbol_a: First leg. E.g. 'LMT'.
        symbol_b: Second leg. E.g. 'RTX'.
        start: Episode start, YYYY-MM-DD.
        end: Episode end, YYYY-MM-DD.
    """
    return _run(_research_divergence, symbol_a=symbol_a, symbol_b=symbol_b, start=start, end=end)


@function_tool
def get_reference_schema() -> str:
    """The reference table's columns, with the real values of its sector, ETF-family and type
    columns.

    Call this before query_reference filters on one of those values, so the filter uses a value
    that exists rather than a guess. A name search with ILIKE does not need it.
    """
    return _run(_reference_schema)


@function_tool
def query_reference(sql: str) -> str:
    """Run one read-only SELECT against `instruments`, the reference table of symbols and sectors.

    Use it for counts and groupings of the universe, a company's ticker from its name, or a universe
    screen_pairs' sector filter cannot express, such as "ETFs with leverage above 2". One
    statement, no semicolons; at most 100 rows come back. The connection is read-only.

    Args:
        sql: A single SELECT, e.g. "SELECT symbol, name FROM instruments WHERE etf_family = 'bond'".
    """
    return _run(_reference_query, sql=sql)


# Shared by the analyst and the researchers; the Lead and Writer hold different sets.
ALL = [
    screen_pairs, analyze_pair, research_divergence, get_reference_schema, query_reference,
    get_price_history, get_chart, get_news, get_filings,
]
