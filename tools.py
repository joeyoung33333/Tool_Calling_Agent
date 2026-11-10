"""The tools the agent can call.

Two layers, deliberately separate:

- The private functions (`_price_history`, `_risk_metrics`, ...) are plain Python returning plain
  dicts. They can be called and tested without the SDK or a model.
- The `@function_tool` wrappers are what the model sees. Each is one line through `_run()`, and
  their docstrings *are* the tool descriptions the model reads, which is why they spell out
  argument formats and examples.

Naming: every tool takes `symbol` (or `query` for news), and ranges take `start`/`end` (YYYY-MM-DD).
Limits live in the constants below so a model cannot ask for an unbounded range.
"""

import json
import math
import statistics
from datetime import date, timedelta

from agents import RunContextWrapper, function_tool

import backtest
import chart
import config
import data
import pairs
import refdata
import screen
from data import DataError
from run_context import RunContext

MAX_HISTORY_DAYS = 3650  # longest price history or filings range
MAX_NEWS_DAYS = 14  # longest news range (Google News returns at most 100 items per query)
MAX_CHART_POINTS = 250  # longer ranges are thinned out so the chart stays small
ALSO_RANKED_SHOWN = 20  # the ranked tail is for context, not reading; the full count is reported


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
    """Validated symbol, first date and daily bars. Every price tool starts here, and both legs
    of a pair will go through it, so the 'too few bars' guard is in one place."""
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
    """Run a tool body and return JSON the model can always reason about.

    A tool must never raise: the model sees results, not exceptions, so a failure has to arrive as
    something it can act on.

    A body that also draws returns `(summary, figure)`. The figure goes to the page through `ctx`,
    and only the summary reaches the model, because a tool result is stored in the session and
    replayed on every later turn (see run_context). This is the only place that knows `ctx` exists.
    """
    try:
        out = fn(**kwargs)
        if isinstance(out, tuple):  # no tool's model payload is a tuple, so this is unambiguous
            out, figure = out
            # Drop the figure rather than the answer if no context was threaded in. Forgetting the
            # wrapper's ctx parameter is a one-character mistake that otherwise turns every call of
            # a drawing tool into "this tool failed unexpectedly (AttributeError)".
            if ctx is not None and getattr(ctx, "context", None) is not None:
                label = out.get("symbol") or out.get("pair") or out.get("chart_label")
                ctx.context.charts.append({"symbol": label, "figure": figure})
        return json.dumps(out, default=str)
    except DataError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"This tool failed unexpectedly ({type(e).__name__}). Try different arguments."})


# --- Tool bodies (plain Python, no SDK) ---


def _price_history(symbol: str, start: str, end: str | None = None) -> dict:
    s, first, bars = _history(symbol, start, end)
    closes = [b["close"] for b in bars]
    vol = pairs.annual_vol(pairs.returns(closes)) if len(bars) > 2 else None  # needs 2+ returns
    # Each of the last 5 closes carries its own day's move, so a one-day jump is never read as the period return.
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
    return out



def _chart(symbol: str, start: str, end: str | None = None) -> tuple[dict, dict]:
    """Returns (summary for the model, Plotly figure spec for the page).

    Kept separate on purpose: only the summary goes into the model's context. See run_context.
    """
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
    figure = chart.line([(f"{s} close", [(b["date"], b["close"]) for b in thinned])], title, "Close")
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


def _rules(entry_z: float | None, exit_z: float | None, in_sample_frac: float | None) -> backtest.Rules:
    """Backtest rules from the defaults in config, with any caller override applied."""
    return backtest.Rules(
        entry_z=entry_z if entry_z is not None else config.ENTRY_Z,
        exit_z=exit_z if exit_z is not None else config.EXIT_Z,
        stop_z=config.STOP_Z,
        max_hold=config.MAX_HOLD_SESSIONS,
        cost_bps=config.COST_BPS,
        borrow_bps_annual=config.BORROW_BPS_ANNUAL,
        in_sample_frac=in_sample_frac if in_sample_frac is not None else config.IN_SAMPLE_FRAC,
    )



def _pair_series(a: str, b: str, lookback_days: int | None) -> tuple[str, str, list[dict], list[dict]]:
    """Validated symbols and their daily bars over the lookback window."""
    first, second = data.clean(a), data.clean(b)
    if first == second:
        raise DataError(f"{first} cannot be paired with itself. Give two different symbols.")
    days = min(max(int(lookback_days or config.LOOKBACK_DAYS), 200), MAX_HISTORY_DAYS)
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
    a, b, hist_a, hist_b = _pair_series(symbol_a, symbol_b, lookback_days)
    dates, closes_a, closes_b = pairs.align(hist_a, hist_b)
    if len(dates) < config.MIN_OVERLAP_SESSIONS:
        raise DataError(
            f"{a} and {b} share only {len(dates)} sessions, and at least "
            f"{config.MIN_OVERLAP_SESSIONS} are needed. One of them may be newly listed."
        )
    rules = _rules(entry_z, exit_z, None)
    cut = backtest.split(dates, rules)
    # In-sample only, matching screen.fit_pair: these statistics describe the window the parameters
    # were fitted on, so they can be compared with the holdout rather than averaged into it.
    stats = pairs.summary(dates[:cut], closes_a[:cut], closes_b[:cut])
    out = {"pair": f"{a}/{b}", "symbol_a": a, "symbol_b": b, **stats,
           "measured_on": f"in-sample window, {dates[0]} to {dates[cut - 1]} ({cut} sessions)",
           "holdout_sessions": len(dates) - cut,
           "source": "Cboe", "delay": "daily bars, split-adjusted; price return only (no dividends)"}

    if not stats["tradeable"]:
        # Still chart it: seeing a spread that never comes back is the point being made. Rebuilt from
        # the unrounded fit, not from stats' display-rounded values.
        beta = pairs.hedge_ratio(closes_a[:cut], closes_b[:cut])
        spread = pairs.spread(closes_a, closes_b, beta)
        mean, sd = statistics.fmean(spread[:cut]), statistics.pstdev(spread[:cut])
        z = pairs.zscores(spread, mean, sd) if sd > pairs.FLAT else [0.0] * len(spread)
        out["current_z"], out["current_z_as_of"] = round(z[-1], 2), dates[-1]
        figure = chart.zscore_panel(list(zip(dates, z)), rules.entry_z, rules.exit_z,
                                    f"{a}/{b} spread z-score - not tradeable")
        return out, figure

    result = backtest.evaluate(hist_a, hist_b, rules=rules, trials=1)
    out["backtest"] = {k: result[k] for k in ("fit", "rules", "in_sample", "holdout", "limitations")}
    out["recent_trades"] = result["trades"][-5:]

    # Scored against the in-sample fit - the same numbers the backtest traded on - so the episodes,
    # the chart's bands and the trade table all describe one z series. Fitting this on the whole
    # window instead would move the episode dates and plot bands the strategy never used.
    fit = backtest.fit(dates[:cut], closes_a[:cut], closes_b[:cut])
    z = pairs.zscores(pairs.spread(closes_a, closes_b, fit.beta), fit.mean, fit.sd)
    found = pairs.episodes(dates, z, config.DIVERGENCE_Z, config.DIVERGENCE_MIN_DAYS)
    # Today, against the frozen fit - not the in-sample slice summary() measured over. The date is
    # stated because every other figure in this result describes the in-sample window.
    out["current_z"], out["current_z_as_of"] = round(z[-1], 2), dates[-1]
    out["divergence_episodes"] = found[-config.MAX_EPISODES_RESEARCHED:]
    out["episode_count"] = len(found)
    if found:
        out["next_step"] = ("Call research_divergence for an episode's exact window to find what "
                            "caused it: a temporary dislocation or a broken relationship.")
    figure = chart.zscore_panel(list(zip(dates, z)), rules.entry_z, rules.exit_z,
                                f"{a}/{b} spread z-score", episodes=found)
    return out, figure


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
        raise DataError(str(e)) from e  # these messages tell the model how to recover; keep them
    lookback = screen.clamp_lookback(lookback_days)
    try:
        out = screen.screen(universe["symbols"], lookback_days=lookback, top_k=top_k)
    except screen.ScreenError as e:
        raise DataError(str(e)) from e  # same reasoning as resolve_universe above
    out["ranked"] = [{**c, "backtest": _compact(c["backtest"])} for c in out["ranked"]]
    # The rules, the trial count and the caveats are identical for every pair, so state them once.
    # Copied before the pops below, which would otherwise empty the dict this points at.
    reference = next((c["backtest"] for c in out["ranked"] if c["backtest"].get("tradeable")), {})
    shared = {key: reference.get(key) for key in ("rules", "limitations")}
    # Same for the deflation note, which sits one level down in the holdout block.
    shared["deflation"] = (reference.get("holdout") or {}).get("deflation_note")
    for candidate in out["ranked"]:
        for key in ("rules", "limitations", "trials_considered"):
            candidate["backtest"].pop(key, None)
        (candidate["backtest"].get("holdout") or {}).pop("deflation_note", None)
    if len(out["also_ranked"]) > ALSO_RANKED_SHOWN:
        out["also_ranked_count"] = len(out["also_ranked"])
        out["also_ranked"] = out["also_ranked"][:ALSO_RANKED_SHOWN]
    summary = {"universe_sql": universe["sql"], "universe_note": universe.get("note"), **out,
               "backtest_rules": shared.get("rules"),
               "limitations": shared.get("limitations"),
               "deflation": shared.get("deflation"),
               "chart_label": sector or etf_family or "selected symbols",
               "source": "Cboe daily bars; universe from the committed reference database",
               "detail": "Call analyze_pair on any pair above for its episodes, trades and chart."}
    return summary, _ranked_table(summary)


# The screen's headline figures, in the order a reader scans them.
SCREEN_ROWS = [
    ("Pair", lambda c, h: f"{c['a']}/{c['b']}"),
    ("Half-life", lambda c, h: _cell(c.get("half_life_days"), "{:.1f}d")),
    ("Beta stab.", lambda c, h: _cell(c.get("beta_stability"), "{:.2f}")),
    ("Corr", lambda c, h: _cell(c.get("correlation"), "{:.2f}")),
    ("Trades", lambda c, h: _cell(h.get("trades"), "{:d}")),
    ("Exposure", lambda c, h: _cell(h.get("exposure_pct"), "{:.0f}%")),
    ("Return", lambda c, h: _cell(h.get("total_return_pct"), "{:+.2f}%")),
    ("Sharpe", lambda c, h: _cell(h.get("sharpe"), "{:.2f}")),
    ("Deflated", lambda c, h: _cell(h.get("deflated_sharpe"), "{:+.2f}")),
]


def _cell(value, fmt: str) -> str:
    """One table cell, or an em dash where the statistic does not exist."""
    return fmt.format(value) if value is not None else "-"


def _ranked_table(summary: dict) -> dict:
    """The backtested pairs as a table for the page, so the ranking is readable at a glance.

    The model already has these numbers in the tool result; this is the same data drawn, not extra
    data fetched. Pairs the arithmetic refused carry their reason instead of empty columns.
    """
    rows = []
    for candidate in summary.get("ranked", []):
        result = candidate.get("backtest") or {}
        if not result.get("tradeable"):
            rows.append([f"{candidate['a']}/{candidate['b']}", "not tradeable", "-", "-", "-", "-", "-", "-", "-"])
            continue
        holdout = result.get("holdout") or {}
        rows.append([build(candidate, holdout) for _, build in SCREEN_ROWS])
    return chart.table(
        [name for name, _ in SCREEN_ROWS],
        rows,
        f"{summary.get('chart_label', 'screen')}: {summary.get('pairs_screened', 0)} pairs screened, "
        f"top {len(rows)} backtested on the holdout",
    )


def _compact(result: dict) -> dict:
    """A backtest with its per-session arrays removed, for a result that carries many of them.

    A screen backtests up to 20 pairs, and each full result holds two equity curves of several
    hundred points. Returned whole, one screen was measured at 208,762 characters - about 52k
    tokens, which the session would then replay on every later turn. The metrics are what a model
    reasons over; the curves are chart data and belong in the run context, not the transcript.
    """
    if not result.get("tradeable"):
        return result
    return {k: v for k, v in result.items() if k not in ("equity", "trades")}


def _research_divergence(symbol_a: str, symbol_b: str, start: str, end: str) -> dict:
    """News and filings for both legs over exactly one divergence window."""
    a, b = data.clean(symbol_a), data.clean(symbol_b)
    first, last = _date_range(start, end, MAX_HISTORY_DAYS)
    # News is capped far shorter than filings, so a long episode gets the slice from its *start*.
    # The cause of a divergence is whatever opened it; the sessions where it closes are the ones
    # least likely to explain it, and taking those would blame every move on its own resolution.
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
        "hint": ("Resolve a sector with gics_sector, or gics_sub_industry for close substitutes. "
                 "Pass the symbols you find to screen_pairs."),
    }


def _reference_query(sql: str) -> dict:
    """Run one read-only SELECT against the reference table."""
    try:
        return refdata.query(sql)
    except refdata.RefDataError as e:
        raise DataError(str(e))


# --- The tools the model sees ---
#
# @function_tool builds each schema from the signature and docstring, replacing Project 1's hand-written TOOLS list.


@function_tool
def get_price_history(symbol: str, start: str, end: str | None = None) -> str:
    """Daily price history for one symbol: period return, high, low and realized volatility.

    Args:
        symbol: Ticker, not a company name. E.g. 'NVDA', 'SPY', 'QQQ', 'VIX'.
        start: First date, YYYY-MM-DD, worked out from today (e.g. 365 days ago for 'last year').
        end: Last date, YYYY-MM-DD. Defaults to today.
    """
    return _run(_price_history, symbol=symbol, start=start, end=end)



@function_tool
def get_chart(
    ctx: RunContextWrapper[RunContext], symbol: str, start: str, end: str | None = None
) -> str:
    """Draw a daily closing-price line chart, shown to the user. Call only when a chart is wanted.

    Args:
        symbol: Ticker to chart. E.g. 'AAPL'.
        start: First date, YYYY-MM-DD.
        end: Last date, YYYY-MM-DD. Defaults to today.
    """
    # `ctx` is excluded from the schema the model sees; _run sends the figure to the page through it.
    return _run(_chart, ctx, symbol=symbol, start=start, end=end)


@function_tool
def get_news(query: str, start: str | None = None, end: str | None = None) -> str:
    """News headlines for a ticker or keywords. Headlines only, no article text.

    Omit both dates for the last 7 days. A range is capped at 14 days.

    Args:
        query: A ticker like 'NVDA', or keywords like 'Fed rate decision'.
        start: First date, YYYY-MM-DD. Omit for the latest.
        end: Last date, YYYY-MM-DD. Omit for the latest.
    """
    return _run(_news, query=query, start=start, end=end)


@function_tool
def get_filings(symbol: str, start: str | None = None, end: str | None = None) -> str:
    """SEC filings for a US-listed company: 8-K events with their item codes, plus 10-K and 10-Q.

    Indices never file. Most ETFs do not either, though trust-structured ones such as SPY and QQQ
    file 10-Ks. Omit both dates for the latest 10 filings. An 8-K's item code says what actually
    happened, which makes a filing stronger evidence than a headline.

    Args:
        symbol: Ticker of a company. E.g. 'LMT'.
        start: First date, YYYY-MM-DD. Omit for the latest.
        end: Last date, YYYY-MM-DD. Omit for the latest.
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
    """Test one pair: correlation, hedge ratio, half-life, a holdout backtest and divergence episodes.

    The holdout is the figure that matters: the hedge ratio and bands are fitted on the earlier 70%
    of the window and reused unchanged on the later 30%, so a pair that is strong in-sample and weak
    out-of-sample was fitted, not discovered. Also draws the spread's z-score with its bands.

    Args:
        symbol_a: First leg. E.g. 'LMT'.
        symbol_b: Second leg. E.g. 'RTX'.
        lookback_days: History to use. Default 1095 (3 years); minimum 200.
        entry_z: Open a trade at this many standard deviations. Default 2.0.
        exit_z: Close it at this many. Default 0.5.
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
    """Screen EVERY pair in a universe and rank them, then backtest the best few.

    Give exactly one of sector, etf_family or symbols. Every combination is fitted in Python, so the
    ranking is arithmetic rather than judgement; pairs whose hedge ratio is negative or whose spread
    does not mean-revert are rejected with a reason. Call this before analyze_pair when the question
    is "which pairs", and use analyze_pair afterwards to go deeper on one of them.

    Args:
        sector: A GICS sector or sub-industry. E.g. 'Energy', 'Semiconductors'.
        etf_family: A curated ETF family. E.g. 'nasdaq_leveraged', 'sector_spdr'.
        symbols: An explicit list to screen. E.g. ['LMT', 'RTX', 'NOC', 'GD'].
        top_k: How many of the best pairs to backtest. Default 8, maximum 20.
        lookback_days: History to use. Default 1095 (3 years).
    """
    return _run(_screen_pairs, ctx, sector=sector, etf_family=etf_family, symbols=symbols,
                top_k=top_k, lookback_days=lookback_days)


@function_tool
def research_divergence(symbol_a: str, symbol_b: str, start: str, end: str) -> str:
    """News and SEC filings for both legs over one divergence window, to explain why it happened.

    Call this with the exact start and end of an episode that analyze_pair reported. It answers the
    question the statistics cannot: was the gap a temporary dislocation that later closed, or did
    the relationship between the two names actually break?

    Args:
        symbol_a: First leg. E.g. 'LMT'.
        symbol_b: Second leg. E.g. 'RTX'.
        start: Episode start, YYYY-MM-DD.
        end: Episode end, YYYY-MM-DD.
    """
    return _run(_research_divergence, symbol_a=symbol_a, symbol_b=symbol_b, start=start, end=end)


@function_tool
def get_reference_schema() -> str:
    """The reference table's columns and the real values of its sector and ETF-family columns.

    Call this before writing SQL with query_reference, so a filter is written against values that
    exist rather than guessed.
    """
    return _run(_reference_schema)


@function_tool
def query_reference(sql: str) -> str:
    """Run one read-only SELECT against `instruments`, the reference table of symbols and sectors.

    Use it to find a universe that resolve-by-sector cannot express, such as "ETFs with leverage
    above 2" or "the biggest GICS sub-industries". One statement, no semicolons, at most 100 rows.
    The connection is read-only, so no query can modify or reach outside the database.

    Args:
        sql: A single SELECT, e.g. "SELECT symbol, name FROM instruments WHERE etf_family = 'bond'".
    """
    return _run(_reference_query, sql=sql)


# Phase 3 gives each agent its own tool set, at which point this becomes one list per agent.
ALL = [
    screen_pairs, analyze_pair, research_divergence, get_reference_schema, query_reference,
    get_price_history, get_chart, get_news, get_filings,
]
